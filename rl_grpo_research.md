# RL Fine-Tuning with GRPO: Research Log

> **Goal**: RL fine-tune gpt-oss-120b on AIME math using Gemini 3 Pro as teacher/reward model.
> **Framework**: OpenRLHF 0.9.3 + Ray + DeepSpeed ZeRO-3 + vLLM.
> **Hardware**: 8x H100 80GB GPUs, 1TB boot NVMe, 1TB data NVMe, 5.9TB RAID-0 scratch.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Ray Cluster (8 GPUs)                        │
│                                                                     │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐  │
│  │  vLLM Rollout (4 GPU)│    │  Actor Training (4 GPU)          │  │
│  │  Engine 0: GPU 0,1   │    │  DeepSpeed ZeRO-3 + CPU offload │  │
│  │  Engine 1: GPU 2,3   │    │  GPU 4,5,6,7                    │  │
│  │  TP=2, MXFP4 native  │    │  LoRA rank=64, alpha=128        │  │
│  └──────────┬───────────┘    └──────────────┬───────────────────┘  │
│             │                                │                      │
│             │ ◄── LoRA weight sync ──────────┘                      │
│             │     (Ray object store,                                │
│             │      ~96 MB, ~6s)                                     │
│             │                                                       │
│             ▼                                                       │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐  │
│  │  Generate 8 samples  │    │  Gemini Reward Function          │  │
│  │  per prompt           │───▶│  0.6 * exact_match              │  │
│  │  (rollout_batch=8)   │    │  + 0.4 * reasoning_quality      │  │
│  └──────────────────────┘    │  Fallback: 3 models × 3 retries │  │
│                               └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| RL algorithm | GRPO (dr_grpo) | Group-relative advantages; no critic model needed |
| KL coefficient | 0 | Maximizes learning signal; risky for stability |
| Weight sync | Ray object store | NCCL rendezvous fails between actor and vLLM EngineCore subprocess |
| Param offload | CPU (ZeRO-3) | 234GB dequantized model / 4 GPUs = 58.5 GB/GPU; no room for activations |
| Grad checkpointing | Reentrant | Non-reentrant validates tensor shapes; ZeRO-3 offload changes shapes |
| Checkpoint format | LoRA-only (HF) | Full-model save was 203 GB; adapter_model.bin is 91 MB |
| Quantization | MXFP4 (inference) / bf16 (training) | vLLM uses native MXFP4; actor dequantizes to bf16 for LoRA |

### Model Details

- **Base model**: openai/gpt-oss-120b (120B params, MXFP4 quantized, ~30GB on disk)
- **Dequantized**: ~234GB bf16 (for training actor)
- **LoRA config**: rank=64, alpha=128, targets=q_proj,k_proj,v_proj,o_proj
- **Trainable params**: 288 adapter parameter tensors, ~96 MB total

---

## 2. Training Configuration

```bash
# train_grpo.sh — key hyperparameters
--advantage_estimator dr_grpo     # Group-relative, no variance normalization
--init_kl_coef 0                  # No KL penalty (aggressive)
--n_samples_per_prompt 8          # Group size for advantage estimation
--rollout_batch_size 8            # 8 prompts per rollout = 64 samples
--train_batch_size 8              # Full batch = 8 prompts
--micro_train_batch_size 1        # Gradient accumulation: 8 micro-steps
--max_len 4096                    # Total sequence length cap
--prompt_max_len 1024             # Prompt budget
--generate_max_len 3072           # Response budget
--actor_learning_rate 1e-6        # Peak LR after warmup
--lora_rank 64                    # LoRA rank
--lora_alpha 128                  # LoRA alpha (scaling = alpha/rank = 2.0)
--zero_stage 3                    # Full parameter partitioning
--offload                         # CPU parameter offloading
--adam_offload                    # CPU optimizer state offloading
--save_steps 10                   # Checkpoint every 10 steps
--max_ckpt_num 2                  # Keep last 2 checkpoints
--num_episodes 50                 # 50 passes over the dataset
```

### Training Data

- **AIME train**: 72 problems (competition math, very hard)
- **AIME eval**: 18 problems (held-out)
- **MATH Level 4-5**: 45 problems (alternate dataset, not used for training)

### Reward Function Design

```
reward = 0.6 * correctness + 0.4 * reasoning_quality

correctness:  1.0 if extracted answer matches ground truth (normalized), else 0.0
reasoning_quality:  Gemini 3 Pro rates solution quality on [0, 1] scale
                    Fallback chain: gemini-3.1-pro → gemini-3-pro → gemini-3-flash
                    If all fail: reward = correctness (pure EM, reasoning_quality=0)
```

**Rationale**: Pure EM (0/1) has high variance with 8 samples — many batches where all 8 are wrong. Adding reasoning quality provides gradient signal even when the answer is wrong (partial credit for good reasoning).

**Risk**: The model could learn to "please the Gemini judge" rather than solve problems. Must monitor correlation between reward and correctness over time.

---

## 3. Attempt History & Lessons Learned

### Attempt 19: Ray collective + NCCL sync
- **Problem**: `extra_logs` contained string values causing `torch.tensor([str])` crash
- **Fix**: Filter extra_logs to float/int only; add CUDA sync before NCCL group init
- **Lesson**: **Always validate data types in logging pipelines.** Non-numeric values in RL metrics cause silent crashes.

### Attempt 20: vLLM memory tuning
- **Problem**: OOM during actor training forward pass (8.79 GiB activation alloc, only 5 GB free)
- **Root cause**: ZeRO-3 partitions 234 GB / 4 GPUs = 58.5 GB/GPU for params alone
- **Lesson**: **ZeRO-3 alone is insufficient for 120B models on 4 GPUs.** Even with full partitioning, parameter memory dominates GPU capacity, leaving no room for activations.

### Attempt 21: CPU parameter offloading
- **Problem**: Solved OOM but hit `CheckpointError` during backward (tensor shape mismatch)
- **Root cause**: Non-reentrant gradient checkpointing validates tensor shapes; ZeRO-3 offload changes param shapes from `[2880]` (gathered) to `[0]` (partitioned)
- **Lesson**: **ZeRO-3 CPU offload requires reentrant gradient checkpointing.** The non-reentrant variant's metadata validation is incompatible with parameter shape changes during offload/gather cycles.

### Attempt 22: Reentrant gradient checkpointing
- **Problem**: Training completed episode 1 but weight sync failed (NCCL rendezvous timeout)
- **Lesson**: **Reentrant gradient checkpointing resolves the ZeRO-3 offload shape mismatch.**

### Attempt 23: Weight sync fix
- **Problem**: Ray collective NCCL rendezvous fails between actor and vLLM EngineCore subprocess workers
- **Fix**: Replaced NCCL broadcast with Ray object store (ray.put/ray.get)
- **Lesson**: **NCCL-based weight sync is fragile across Ray actor boundaries**, especially with vLLM v1's EngineCore subprocess architecture. Ray object store is slower (~6s vs ~1s) but reliable.
- **New problem**: Disk full — checkpoints consumed 422 GB on 1TB disk

### Attempt 24: Checkpoint disk fix
- **Problem**: PeftModel.save_pretrained() with ZeRO-3 writes full-model shards (203 GB) alongside adapter_model.bin (91 MB)
- **Fix**: Patch save_model to clean up pytorch_model-*.bin shards after saving adapter; disable DeepSpeed intermediate checkpoints
- **Lesson**: **PeftModel + ZeRO-3 save is broken by default.** Must explicitly clean up full-model shard files. Also: always set `--max_ckpt_num` and `--disable_ds_ckpt` to avoid disk runaway.
- **Result**: 17 steps completed. Training functional but lacked diagnostics.
  Correctness dropped to 9.4-14.1% at steps 16-17 once LR hit peak — could not
  diagnose root cause without gradient norms or advantage stats. Stopped to add monitoring.

### Attempt 25: Enhanced metrics + held-out eval (CURRENT)
- **Problem**: Attempt 24 showed loss spikes (act_loss=2.69) and sudden correctness
  drops (step 16: 9.4%) without diagnostic data to determine if these were random
  bad batches, LR-shock, or systematic instability
- **Fix**: Added 3 new patches for comprehensive monitoring:
  - `metrics_actor.patch`: gradient norm, advantage stats, entropy distribution
  - `metrics_experience.patch`: reward distribution, high-reward-but-wrong detection
  - `metrics_trainer.patch`: per-step JSONL to /mnt/scratch, sample output logging
  - Built-in held-out eval every 10 steps via `--eval_dataset`
  - Entropy logging via `--entropy_loss_coef 0`
  - Checkpoints/output on /mnt/data (1TB), metrics on /mnt/scratch (5.9TB)
- **Lesson**: **Instrument before you train.** Without grad norms, advantage stats,
  and held-out eval, you can observe problems but not diagnose them.
- **Status**: Starting...

### Summary: What Broke and What Fixed It

| Problem | Root Cause | Fix | Attempts |
|---------|-----------|-----|----------|
| String in tensor | Non-numeric extra_logs | Filter to float/int | 19 |
| Actor OOM | 234GB / 4 GPUs | CPU param offload (`--offload`) | 20→21 |
| Backward shape error | Non-reentrant checkpointing + ZeRO-3 | `--gradient_checkpointing_use_reentrant` | 21→22 |
| NCCL rendezvous timeout | Ray actor→vLLM subprocess boundary | Ray object store sync | 22→23 |
| Disk full (422 GB) | PeftModel + ZeRO-3 shard leak | Patch save_model + `--disable_ds_ckpt` | 23→24 |

---

## 4. Training Metrics — Attempt 24

### 4.1 Per-Step Summary (Global Steps 1–17)

| Step | policy_loss | reward | correctness | reasoning_q | response_len | truncated | ppo_clip | ppo_kl | actor_lr |
|------|------------|--------|-------------|-------------|-------------|-----------|----------|--------|----------|
| 1 | 0.025 | 0.500 | 39.1% | 0.665 | 2652 | 60.9% | 0.192 | +0.020 | 3.7e-8 |
| 2 | 0.072 | 0.540 | 45.3% | 0.670 | 2567 | 53.1% | 0.238 | +0.022 | 1.1e-7 |
| 3 | 0.119 | 0.349 | 23.4% | 0.521 | 2603 | 70.3% | 0.331 | +0.019 | 1.9e-7 |
| 4 | 0.204 | 0.522 | 43.8% | 0.649 | 2626 | 54.7% | 0.286 | -0.068 | 2.6e-7 |
| 5 | 0.255 | 0.373 | 21.9% | 0.605 | 2738 | 65.6% | 0.324 | -0.039 | 3.3e-7 |
| 6 | 0.101 | 0.505 | 39.1% | 0.677 | 2504 | 57.8% | 0.244 | -0.004 | 4.1e-7 |
| 7 | 0.141 | 0.351 | 18.8% | 0.596 | 3029 | 84.4% | 0.323 | +0.021 | 4.8e-7 |
| 8 | 0.090 | **0.704** | **64.1%** | **0.800** | 2397 | 34.4% | 0.254 | -0.029 | 5.6e-7 |
| 9 | 0.060 | 0.470 | 37.5% | 0.613 | 2780 | 64.1% | 0.224 | +0.006 | 6.3e-7 |
| 10 | 0.068 | 0.550 | 45.3% | 0.695 | 2407 | 59.4% | 0.191 | -0.025 | 7.0e-7 |
| 12 | 0.235 | 0.591 | 51.6% | 0.704 | 2530 | 46.9% | 0.243 | +0.019 | 8.5e-7 |
| 13 | 0.066 | 0.452 | 35.9% | 0.591 | 2795 | 64.1% | 0.234 | +0.017 | 9.3e-7 |
| 14 | 0.055 | 0.396 | 29.7% | 0.544 | 2463 | 62.5% | 0.237 | -0.004 | 9.9e-7 |
| 15 | 0.102 | **0.696** | **57.8%** | **0.873** | 2626 | 40.6% | 0.241 | +0.011 | **1.0e-6** |
| 16 | 0.073 | **0.235** | **9.4%** | 0.447 | 3001 | **85.9%** | 0.313 | -0.005 | 1.0e-6 |
| 17 | 0.143 | **0.271** | **14.1%** | 0.466 | 3012 | **85.9%** | 0.330 | +0.009 | 1.0e-6 |

### 4.2 Trend Analysis

**Reward**: Noisy. Steps 1-5 avg: 0.457, Steps 6-10 avg: 0.516 (+13%), but steps 16-17 crashed to 0.235-0.271.

**Correctness**: High variance due to small batches (64 samples) on hard AIME problems. Range 9.4%–64.1%. Steps 16-17 showed sharp collapse (9.4%, 14.1%) coinciding with LR reaching peak (1e-6).

**Policy loss**: NOT expected to decrease monotonically in GRPO. This is a policy gradient loss (advantage-weighted log-prob shift), not supervised cross-entropy. Oscillation around 0.06–0.25 is normal.

**Response length**: Averaging ~2600 tokens. Steps 16-17 both hit 3001-3012 with 85.9% truncation — model started generating near-max-length responses that fail to reach final answers.

**ppo_clip_ratio**: 19-33% of updates are clipped. Steps 16-17 showed higher clip ratios (0.313, 0.330) indicating more aggressive updates at peak LR.

**ppo_kl**: Small oscillation -0.07 to +0.02. With `init_kl_coef=0` there's no penalty pushing back. This means the model is drifting slowly but unconstrained from the reference policy.

**Key finding**: Steps 16-17 show a pattern consistent with LR-shock: the learning rate hit its peak (1e-6) at step 15, and the next two steps showed 85.9% truncation, <15% correctness, and elevated clip ratios. Without gradient norms and advantage stats, we cannot determine if this was caused by a single bad gradient update or a systematic instability. This motivated the restart with enhanced monitoring (attempt 25).

### 4.3 Micro-Batch Level Observations

At step 11, a loss spike appeared: `act_loss=2.69` (10x normal range). This recovered to normal within 2 micro-batches. Likely caused by a single high-advantage sample dominating the batch. Without gradient norm logging, we can't confirm whether this caused a destructive parameter update.

### 4.4 Infrastructure Metrics

- **Step time**: ~12 minutes per global step (experience generation + training + weight sync)
- **Experience generation**: ~1.5 min (vLLM inference, 64 samples)
- **Training epoch**: ~8 min (16 micro-batches × 29s each)
- **Weight sync**: ~6s per sync (288 params, 96 MB via Ray object store)
- **GPU memory**: vLLM 68GB/80GB, Actor 25-29GB/80GB (rest CPU offloaded)
- **Disk**: Stable at 100GB used / 868GB free (checkpoint: 91MB per save)
- **Gemini API**: Intermittent 429 RESOURCE_EXHAUSTED (burst of 64 concurrent requests exceeds quota). Fallback chain resolves most failures.

---

## 5. Baseline Scores

| Model | Dataset | Metric | Score |
|-------|---------|--------|-------|
| Teacher (Gemini 3 Pro) | AIME eval (18) | Exact Match | 0% |
| Teacher (Gemini 3 Pro) | MATH45 eval (18) | Exact Match | 27.8% |
| Base Student (gpt-oss-120b) | AIME eval (18) | pass@1 | TBD |
| RL Student (step 10) | AIME eval (18) | pass@1 | TBD (pending eval) |

**Note**: Teacher scores 0% on AIME because AIME problems require deep multi-step reasoning that exceeds Gemini's capability at temperature=0. The teacher's value is in its reasoning quality assessment (reward signal), not in solving AIME directly.

---

## 6. Diagnostic Framework for GRPO

### 6.1 What to Monitor (Priority Order)

**Tier 1 — Non-negotiable**:
1. **Held-out Exact Match** (AIME eval set, periodic). Ground truth signal.
2. **Train batch correctness** (per step). Proxy when eval isn't running.
3. **Reward** (mean per step). Composite signal including reasoning quality.
4. **Answer parse success rate** (`has_answer` metric).

**Tier 2 — Stability diagnostics**:
5. **ppo_kl** (approximate KL from importance ratios). Drift indicator.
6. **ppo_clip_ratio**. How aggressively the policy is updating.
7. **Gradient norm** (requires patch). Detect spikes.
8. **Advantage stats** (mean/std/max, requires patch). Variance indicator.

**Tier 3 — Collapse detection**:
9. **Response length** (mean/p90). Length collapse = model degeneration.
10. **Token entropy** (requires patch). Diversity collapse indicator.
11. **Repetition rate**. 3-gram repeat ratio.
12. **Truncation rate** (`response_clip_ratio`). Already logged.

**Tier 4 — Reward integrity**:
13. **Reward vs correctness correlation** (per-step rolling window).
14. **High-reward-but-wrong rate**. Reward hacking indicator.
15. **Gemini fallback rate**. Judge reliability.

### 6.2 Tuning Playbook

| Scenario | Symptoms | Likely Cause | Actions |
|----------|----------|-------------|---------|
| **Flat learning** | KL tiny, entropy stable, EM flat | Updates too conservative | Increase LR, decrease KL coef, increase group size |
| **Instability** | KL spikes, entropy drops, EM worsens | Updates too aggressive | Decrease LR, increase KL coef, add grad clipping, clip advantages |
| **Reward hacking** | Reward ↑ but EM ↔ or ↓ | Policy pleases judge, not task | Reweight to correctness-dominant, add judge redundancy |
| **Loss spikes** | Rare huge act_loss values | One extreme-advantage sample | Lower LR, add grad clip, add advantage clip, increase batch size |
| **Length collapse** | Response length drops sharply | Degeneration | Add length penalty, check entropy, add KL constraint |
| **Truncation ↑** | >80% responses truncated | Model rambling | Increase max_len or add length penalty in reward |

### 6.3 Current Diagnosis (Attempt 24, Step 10)

Based on the tuning playbook:
- **Scenario**: Mostly "Flat learning" (Scenario 1). KL is near zero, LR still in warmup, no clear EM improvement yet.
- **Risk**: With `init_kl_coef=0`, we have no guardrail against policy drift. If/when learning rate reaches peak (1e-6), instability could emerge.
- **Micro-batch spikes**: `act_loss=2.69` at step 11 suggests Scenario 4 (extreme advantage on a single batch). Without grad norm data, can't assess impact.
- **Truncation**: 34-84% range, averaging ~60%. Many responses hitting the 3072 token limit. Not alarming for AIME (long reasoning is expected) but worth monitoring.

---

## 7. Patches Applied to OpenRLHF 0.9.3

All patches are in `patches/` and applied via `bash apply_patches.sh`.

| Patch | What It Does | Why |
|-------|-------------|-----|
| `actor.patch` | MXFP4 dequantization + ZeRO-3 delayed init | Base model uses MXFP4 format; must dequantize to bf16 for training |
| `ppo_actor.patch` | CUDA sync before NCCL group init + Ray collective logging | Prevents stale CUDA errors from crashing NCCL communicator creation |
| `experience_maker.patch` | Filter extra_logs to float-only | String values in extra_logs crash torch.tensor() |
| `save_model.patch` | Clean up full-model shards after LoRA save + CPU offload fix | PeftModel.save_pretrained() + ZeRO-3 writes 203 GB of unnecessary shard files |
| `metrics_actor.patch` | Grad norm, advantage stats, entropy distribution | Diagnose loss spikes and detect policy collapse |
| `metrics_experience.patch` | Reward distribution stats, high-reward-but-wrong rate | Detect reward hacking and understand reward variance |
| `metrics_trainer.patch` | Per-step JSONL + sample output logging to /mnt/scratch | Persistent metrics for post-hoc analysis and presentations |

---

## 8. Lessons for Future RL Fine-Tuning

### 8.1 Infrastructure Lessons

1. **Memory hierarchy matters more than raw GPU count.** With 120B params, even 4x80GB GPUs aren't enough without CPU offloading. Plan for: params on CPU, one layer at a time on GPU.

2. **Checkpoint size can silently kill training.** PeftModel + ZeRO-3 saves full model shards (~200 GB) alongside the tiny LoRA adapter (~90 MB). Always verify checkpoint size on first save.

3. **NCCL is fragile across process boundaries.** Ray actors, vLLM subprocess workers, and DeepSpeed all have their own NCCL expectations. When they conflict, fall back to simpler communication (Ray object store, shared memory).

4. **Gradient checkpointing mode must match your training strategy.** ZeRO-3 + CPU offload changes tensor shapes dynamically; only reentrant checkpointing tolerates this.

5. **Disk planning**: Budget 2x model size for inference (MXFP4 → bf16 dequant), plus checkpoint overhead. For 120B: ~30 GB model + 234 GB dequant + checkpoints.

### 8.2 Training/Algorithm Lessons

1. **GRPO policy loss does NOT decrease monotonically.** Unlike supervised loss, policy gradient loss oscillates around zero. Watch reward and correctness instead.

2. **Small batches on hard problems = high variance.** 8 AIME problems × 8 samples = 64 responses per step. With AIME's low solve rate (~20-40%), many batches have mostly-wrong answers, giving noisy advantage estimates.

3. **LR warmup matters.** With warmup ratio 0.03, the first ~27 steps (of 900 total) are warmup. Don't judge training effectiveness before warmup completes.

4. **`init_kl_coef=0` is aggressive.** No constraint on how far the policy drifts from the initial model. Fine for short training runs, risky for long ones. Consider starting with 0.001-0.01.

5. **Response truncation is informative.** High truncation (>70%) means the model is trying to reason longer than `generate_max_len` allows. Either increase the limit or add a length penalty.

### 8.3 Reward Function Lessons

1. **Pure exact-match (0/1) has extreme variance with small groups.** Adding a continuous reasoning quality signal (0.4 weight) smooths gradient estimates.

2. **LLM judges (Gemini) have rate limits.** 64 concurrent reward calls can exceed API quotas. Use fallback chains and graceful degradation.

3. **Monitor reward-correctness correlation.** If reward increases but correctness doesn't, the model may be learning to game the reasoning quality judge rather than solve problems.

4. **Fallback scoring changes the reward distribution.** When Gemini fails, reward = correctness only (no reasoning quality component). This creates a bimodal reward distribution that could confuse learning.

### 8.4 What We'd Do Differently Next Time

1. **Add held-out eval from the start.** Run AIME eval every N steps to catch reward hacking early. Training metrics alone are insufficient.

2. **Log gradient norms and advantage stats.** Would have immediately diagnosed the act_loss=2.69 spike.

3. **Start with small KL coefficient (0.001).** Provides a stability guardrail without significantly limiting learning.

4. **Use larger eval dataset.** 18 AIME problems is too small for reliable EM estimates. Consider mixing in MATH Level 5.

5. **Pre-compute base model performance.** Should have baseline pass@1 and pass@8 numbers before starting training.

---

## 9. Held-Out Evaluation Results

> Will be populated as checkpoints are evaluated.

| Checkpoint | AIME pass@1 | AIME pass@8 | Notes |
|-----------|-------------|-------------|-------|
| Base model | TBD | TBD | Pre-training baseline |
| Step 10 | TBD | TBD | First checkpoint, still in LR warmup |
| Step 20 | TBD | TBD | |
| Final | TBD | TBD | |

---

## 10. Open Questions

1. **Is 72 AIME problems enough training data?** With 50 episodes × 72 problems = 3600 training iterations, the model sees each problem ~50 times. Risk of overfitting to AIME formats vs learning general math reasoning.

2. **Should we add KL regularization?** Currently `init_kl_coef=0`. The model is free to drift arbitrarily from the base policy. May be fine for short runs but risky for 50 episodes.

3. **Is the Gemini reasoning quality signal helpful or harmful?** It provides smoother gradients than pure EM, but could incentivize "judge-pleasing" reasoning patterns rather than correct solutions.

4. **Optimal group size?** Currently `n_samples_per_prompt=8`. Larger groups (16, 32) give better advantage estimates but cost more compute per step. With AIME's low solve rate, larger groups may be critical.

5. **Should we use MATH Level 4-5 instead of (or in addition to) AIME?** AIME may be too hard for meaningful learning signal. The teacher (Gemini) scores 0% on AIME but 28% on MATH — suggesting MATH problems are in the "learnable" difficulty range.

---

*Last updated: 2026-02-20, Attempt 24 stopped at Step 17, Attempt 25 starting*
