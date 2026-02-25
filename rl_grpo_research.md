# RL Fine-Tuning with GRPO: Research Log

> **Goal**: Improve math reasoning via DR-GRPO with pure exact-match reward. Multi-model exploration: gpt-oss-120b (Phase 1) → gpt-oss-20b (Phase 2, A26-A30) → qwen2.5-14b (next).
> **Framework**: OpenRLHF 0.9.3 + Ray + DeepSpeed ZeRO-3 + vLLM.
> **Hardware**: 8x H100 80GB GPUs, 1TB boot NVMe, 1TB data NVMe, 5.9TB RAID-0 scratch.

---

## 0. Current State & Pivot History

### Current Status (2026-02-25)

- **RL on gpt-oss-20b showed zero measurable effect** (§11.26): paired McNemar test on OOD-1000 gives Δ=−0.40pp, p=0.783, 95% CI [−2.5, +1.7]. The 118 discordant problems (61 regressions, 57 improvements) confirm RL changes outputs but without directional improvement.
- **Pivoting to qwen2.5-14b** (§11.28): model sweep identified qwen2.5-14b at 67.0% OOD-1000 (33% headroom) with 99.2% boxed rate — near-perfect EM reward signal.
- **Current recipe**: DR-GRPO, pure EM reward, eps_clip=0.1, micro_train_batch_size=2, LoRA r=32.
- **Key baselines**: gpt-oss-20b 73.8% OOD-1000 (no suffix) / 83.5% (with suffix); qwen2.5-14b 67.0%.
- **Next steps**: Stage 2 eval (ID-200, AIME-18) for qwen2.5-14b, then RL experiment with paired McNemar test.
- **SOP**: See §11.24 for experiment protocol. **Consolidated lessons**: See §12.

### Pivot 1: gpt-oss-120b → gpt-oss-20b (Phase 1 → Phase 2, A25 → A26)

- **Why**: 120b training was painfully slow (~13 min/step), required 4 GPUs for training + 4 for inference, CPU offloading, and complex weight sync. Attempts 19–25 were consumed by infrastructure bugs (OOM, NCCL, checkpoint disk, LoRA sync). Once training finally worked (A24–25), lack of diagnostics made it impossible to interpret results. The 120b setup allowed ~2 experiments per week.
- **Decision**: Use 20b for rapid iteration — 3–6x faster per step, fits on fewer GPUs, allows more experimental cycles to perfect the training recipe (reward function, LR schedule, KL, eps_clip). Plan was to validate the recipe on 20b, then apply to 120b for the final run.
- **Trade-off accepted**: Lower AIME baseline (28% vs 40%), but still in the "learnable" zone.

### Pivot 2: gpt-oss-20b → model sweep / qwen2.5-14b (A30 → model sweep)

- **Why**: After 5 training attempts on gpt-oss-20b (A26–A30, ~200 steps each, ~50 GPU-hours total), paired McNemar test against the base model showed **zero measurable effect** (Δ=−0.40pp, p=0.783, 95% CI [−2.5, +1.7]). The 118 discordant problems (61 regressions, 57 improvements) confirmed RL was changing outputs but without directional improvement.
- **Root cause hypothesis**: Insufficient headroom. gpt-oss-20b scores 73.8% OOD-1000 (83.5% with boxed suffix), leaving only 16–26% of problems in the "learning zone." With binary EM reward, most rollouts either all succeed (no gradient signal) or all fail (model can't solve it, no useful contrast). The effective learning zone — problems the model sometimes gets right and sometimes wrong — is too narrow for GRPO to exploit.
- **Decision**: Sweep candidate models targeting 60–70% OOD-1000 baseline (30–40% headroom). qwen2.5-14b landed at 67.0% with 99.2% boxed rate (near-perfect EM reward signal), half the params of 32b variant. Selected as next RL target.
- **Trade-off accepted**: Switching models means the infrastructure patches (MXFP4 dequant, MoE dtype casting) may not apply. But qwen2.5-14b is a standard dense model — simpler training setup expected.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Ray Cluster (8 GPUs)                        │
│                                                                     │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐  │
│  │  vLLM Rollout (4 GPU)│    │  Actor Training (4 GPU)          │  │
│  │  Engine 0: GPU 0,1   │    │  DeepSpeed ZeRO-3               │  │
│  │  Engine 1: GPU 2,3   │    │  GPU 4,5,6,7                    │  │
│  │  TP=2                │    │  LoRA rank=32, alpha=64          │  │
│  └──────────┬───────────┘    └──────────────┬───────────────────┘  │
│             │                                │                      │
│             │ ◄── LoRA weight sync ──────────┘                      │
│             │     (collective_rpc delta,                            │
│             │      base-weight caching)                             │
│             │                                                       │
│             ▼                                                       │
│  ┌──────────────────────┐    ┌──────────────────────────────────┐  │
│  │  Generate 8 samples  │    │  Pure Exact-Match Reward          │  │
│  │  per prompt           │───▶│  reward = 1 if \boxed{} correct  │  │
│  │  (rollout_batch=16)  │    │          0 otherwise              │  │
│  └──────────────────────┘    │  Zero API cost, deterministic    │  │
│                               └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| RL algorithm | GRPO (dr_grpo) | Group-relative advantages; no critic model needed |
| KL coefficient | 0.001 | Non-zero for stability (learned from 120b instability at kl=0) |
| eps_clip | 0.1 | Half of OpenRLHF default; eliminates late-run gradient spikes (§11.16) |
| Weight sync | LoRA delta via collective_rpc | NCCL rendezvous fails; ray.get() deadlocks in compiled DAG; base-weight caching fixes accumulation bug (§11.19) |
| Param offload | CPU (ZeRO-3) — 120b only | 234GB dequantized / 4 GPUs; 20b fits without offload |
| Grad checkpointing | Reentrant | Non-reentrant validates tensor shapes; ZeRO-3 offload changes shapes |
| Checkpoint format | LoRA-only (HF) | Full-model save was 203 GB; adapter_model.bin is 91 MB |
| Quantization | MXFP4 (inference) / bf16 (training) | vLLM uses native MXFP4; actor dequantizes to bf16 for LoRA |

### Model Details

- **Current target**: qwen2.5-14b (Qwen/Qwen2.5-14B-Instruct, 14B dense, 67.0% OOD-1000)
- **Previous**: openai/gpt-oss-20b (21B MoE, 73.8% OOD-1000 — insufficient headroom)
- **Phase 1**: openai/gpt-oss-120b (120B MoE, MXFP4 quantized, ~30GB on disk, ~234GB bf16 dequantized)
- **LoRA config**: rank=32, alpha=64, targets=q_proj,k_proj,v_proj,o_proj (was rank=64, alpha=128 for 120b)
- **Trainable params**: ~48 MB total (rank=32)

---

## 2. Training Configuration

> **Archive (Phase 1 — gpt-oss-120b).** Current config: see §11.8.

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

### Attempt 25: Enhanced metrics + LoRA weight sync rewrite (completed)
- **Problem 1**: Attempt 24 lacked diagnostic metrics (no grad norms, advantage stats)
- **Problem 2**: LoRA weight sync from actor→vLLM required 4 sub-attempts to fix:
  - **Sub-attempt A**: `ray.get()` inside vLLM compiled DAG worker → **deadlock**.
    vLLM v1 with Ray TP uses compiled DAGs; `ray.get()` blocks the DAG worker thread.
  - **Sub-attempt B**: Pass lora_state directly through `collective_rpc` → **crash**.
    `ValueError: too many dimensions 'str'`. vLLM's msgspec serializer converts
    torch tensors to string representations, not lists.
  - **Sub-attempt C**: Convert tensors to `(shape, dtype, list)` tuples before
    `collective_rpc`, reconstruct on worker side → **crash**. `TypeError:
    load_weights() got an unexpected keyword argument 'add_to_existing'`. GptOss
    model doesn't support `add_to_existing=True`.
  - **Sub-attempt D (working)**: Use save-load-add pattern for TP-safe delta
    application. Save `param.data.clone()`, call `load_weights(delta)` to correctly
    shard for TP, then `param.data.add_(saved)`. **Success — 18s sync time.**
- **Fixes applied**:
  - 3 new metrics patches (grad norm, advantage stats, reward distribution, JSONL)
  - 2 new vLLM patches: `vllm_engine.patch` + `vllm_worker.patch` (LoRA sync)
  - Built-in held-out eval every 10 steps via `--eval_dataset`
  - Entropy logging via `--entropy_loss_coef 0`
- **Lessons**:
  - **vLLM compiled DAGs prohibit `ray.get()` inside workers.** Must pass data
    through `collective_rpc` args, not via Ray object store indirection.
  - **msgspec (vLLM's serializer) cannot handle torch tensors.** Must convert to
    primitive types: `(list(shape), str(dtype), tensor.tolist())`.
  - **Not all vLLM models support `load_weights(add_to_existing=True)`.** The
    save-load-add pattern is universally compatible with TP sharding.
  - **Instrument before you train.** Without grad norms, advantage stats,
    and held-out eval, you can observe problems but not diagnose them.
- **Status**: Completed. Infrastructure validated; moved to Phase 2 (gpt-oss-20b).

### Attempt 26: NuminaMath EM-GRPO on gpt-oss-20b
- **Problem 1**: GPU allocation hang — `init_kl_coef=0.001` creates ref model; default `ref_num_gpus_per_node=8` conflicted with 4 vLLM + 4 actor GPUs.
- **Fix**: `--colocate_actor_ref --ref_num_gpus_per_node 4`.
- **Problem 2**: MoE dtype mismatch — 192 gate/router params remained float32 after MXFP4 dequant, crashing ZeRO-3 optimizer step.
- **Fix**: Cast all params to bf16 before `deepspeed.zero.Init()`.
- **Result**: 60 steps completed, step 30 best (8/18 AIME = 44.4%), then degraded. LR was <6% of target due to `num_episodes` misconfiguration (500K actual steps vs ~40 expected).
- **Lesson**: **MXFP4 MoE models need dtype casting before ZeRO-3** (#7). **Colocate actor and ref model when GPU budget is tight** (#8).

### Attempt 27: eps_clip 0.2→0.1 + OOD probe eval
- **Problem**: Same `num_episodes` bug — 125K actual steps, near-zero LR for entire run.
- **Result**: Stopped at step 46. OOD stable 64–69% but effectively at near-zero LR. Spikes still present (grad=294 at step 9).
- **Lesson**: **Understand `num_episodes` before launching** (#13). Always compute `max_steps` and verify LR schedule.

### Attempt 28: Step-count fix (400-prompt pool, num_episodes=1)
- **Problem**: `max_steps=200` counts gradient steps, not global steps. Each global step = 8 gradient steps → only 25 global steps completed.
- **Result**: LR reached target (99.4% at step 3). Largest-ever spike: grad=2497 at step 16. Only 2 eval points — insufficient for A/B comparison.
- **Lesson**: **`max_steps` counts gradient steps, not global steps** (#14). For N global steps, need `pool_size = N × rollout_batch_size` with `num_episodes=1`.

### Attempt 29 A/B: Clean eps_clip comparison (3200 prompts, 200 global steps)
- **Setup**: A=eps_clip=0.2, B=eps_clip=0.1. Same pool, same seed, 200 global steps each.
- **Result**: OOD equivalent (~64.5% avg). But B has 3x fewer spikes, 4.4x smaller max grad, no spikes after step 71 (A has spikes through step 164). p99 ratio tails shrink with eps_clip=0.1, grow with 0.2.
- **Lesson**: **eps_clip=0.1 eliminates late-run gradient spikes without sacrificing generalization** (#15).

### Attempt 30: micro_train_batch_size 1→2
- **Result**: 200 steps, 18% faster (114s vs 140s/step), no OOM, same OOD (~64.6%). Pipeline regression check: **PASS** (exact match in-training vs post-hoc after LoRA accumulation bug fix).
- **Bug found & fixed**: LoRA weight accumulation in vLLM worker — `apply_lora_delta()` added full delta to already-delta'd weight. Fix: cache base weights, reconstruct each sync (#17).
- **Lesson**: **Doubling micro_train_batch_size (1→2) gives ~18% speedup with no regression** (#19). **After fixing a pipeline bug, always run a quantitative regression check** (#20).

### Summary: What Broke and What Fixed It

| Problem | Root Cause | Fix | Attempts |
|---------|-----------|-----|----------|
| String in tensor | Non-numeric extra_logs | Filter to float/int | 19 |
| Actor OOM | 234GB / 4 GPUs | CPU param offload (`--offload`) | 20→21 |
| Backward shape error | Non-reentrant checkpointing + ZeRO-3 | `--gradient_checkpointing_use_reentrant` | 21→22 |
| NCCL rendezvous timeout | Ray actor→vLLM subprocess boundary | Ray object store sync | 22→23 |
| Disk full (422 GB) | PeftModel + ZeRO-3 shard leak | Patch save_model + `--disable_ds_ckpt` | 23→24 |
| LoRA sync deadlock | `ray.get()` inside compiled DAG worker | Pass data via `collective_rpc` args instead | 25 |
| Tensor serialization | msgspec can't serialize torch.Tensor | Convert to `(shape, dtype_str, list)` tuples | 25 |
| `load_weights(add_to_existing)` | Not supported by GptOss model | Save-load-add pattern: clone param, load delta, add back | 25 |
| GPU allocation hang | `colocate_actor_ref` needed with `init_kl_coef>0` | `--colocate_actor_ref --ref_num_gpus_per_node 4` | 26 |
| MoE dtype mismatch | MXFP4 dequant leaves gates as float32 | Cast all params to bf16 before ZeRO-3 | 26 |
| `num_episodes` miscalc | 50K × 8 / 16 × N = huge step count | Use small pool + `num_episodes=1` | 26→27→28 |
| Gradient vs global steps | `max_steps` counts `optimizer.step()` not rollout cycles | 1 global step = `n_samples` gradient steps | 28 |
| LoRA weight accumulation | `apply_lora_delta` added full delta to already-delta'd weight | Cache base weights, reconstruct each sync | 30 |

---

## 4. Training Metrics — Attempt 24

> **Archive.** Current training data: see §11.10 (A26), §11.16 (A29), §11.21 (A30).

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

## 4b. Training Metrics — Attempt 25

> **Archive.** Current training data: see §11.10 (A26), §11.16 (A29), §11.21 (A30).

### 4b.1 Per-Step Summary (in progress)

| Step | policy_loss | reward | correctness | reasoning_q | response_len | truncated | ppo_clip | ppo_kl | grad_norm | entropy | actor_lr |
|------|------------|--------|-------------|-------------|-------------|-----------|----------|--------|-----------|---------|----------|
| 1 | 0.982 | 0.552 | 43.8% | 0.723 | 2525 | 45.3% | 0.281 | +0.029 | 302.0 | 1.593 | 3.7e-8 |
| 2 | 0.110 | 0.398 | 28.1% | 0.572 | 2681 | 68.8% | 0.282 | +0.045 | 191.5 | 1.571 | 1.1e-7 |
| 3 | 0.087 | 0.609 | 50.0% | 0.773 | 2437 | 51.6% | 0.241 | +0.064 | 29.3 | 1.494 | 1.9e-7 |
| 4 | 0.176 | 0.458 | 35.9% | 0.605 | 2648 | 62.5% | 0.240 | -0.059 | 166.5 | 1.453 | 2.6e-7 |

### 4b.2 Early Observations (Steps 1-4)

**Gradient norm**: 302 → 191 → **29** → 167. Highly variable. Step 3's 29.3 is much
lower (closer to `max_norm=1.0`), possibly because that batch had lower loss. Steps
1, 2, 4 all had >100x clipping, meaning effective updates are severely damped.

**Entropy**: Declining slowly: 1.593 → 1.571 → 1.494 → 1.453. This is a slight
downward trend but not alarming yet. If it drops below ~1.0, that's a concern. Will
monitor for continued decline.

**Correctness**: 43.8% → 28.1% → 50.0% → 35.9%. Average ~39.5%. Extreme variance due
to only 1 prompt per step. Cannot distinguish signal from noise at this scale.

**`advantage_std: 0.0` — not a bug.** Investigated and confirmed this is expected:
with `micro_train_batch_size=1`, each training micro-batch contains 1 sample. Since
`gamma=1.0` and the reward is placed at the last token, `get_cumulative_returns`
assigns the same value to ALL tokens in a response. So within a single sample, all
token-level advantages are identical → std = 0. The advantages ARE different between
samples (driven by `group_reward_std: 0.15-0.19`), which is what drives learning.

**Truncation**: 45% → 69% → 52% → 63%. Averaging ~57%. Similar to attempt 24.

**KL**: 0.029 → 0.045 → 0.064 → -0.059. The sign flip at step 4 is notable — negative
KL means the current policy assigns HIGHER probability to some actions than the
reference policy did. With `init_kl_coef=0`, there's no penalty for this.

**Key concern (RESOLVED)**: The teacher appeared to score 0% on AIME, suggesting
capability inversion. This was a configuration bug — Gemini 3 Pro with
`thinking_level=high` + 16k output tokens scores **100%** on AIME eval. See Section 5.1
for details. The reward function must be updated to enable thinking mode.

### 4b.3 Timing (Attempt 25)

- **Per step**: ~13 min (generation ~1:44, reward ~1:00, forward ~1:35, training ~8:00, sync ~0:18)
- **Per episode**: ~2 hours (9 global steps × 72 prompts / 8 per batch)
- **Total (50 episodes)**: ~100 hours (~4.2 days)
- **First held-out eval**: Step 10 (~episode 2, ~4 hours in)
- **Training started**: 2026-02-21 01:23 UTC

### 4b.4 LoRA Weight Sync Details

The working LoRA sync pipeline (after 4 sub-attempts):
1. Actor collects 288 LoRA A/B params via ZeRO-3 AllGather (0.2s, 95.6 MB)
2. Actor calls `ray.put(lora_state)` → ObjectRef in Ray object store
3. Engine resolves ObjectRef, converts tensors to `(shape, dtype_str, list)` tuples
4. Engine dispatches via `collective_rpc("apply_lora_delta", args=(serializable, scaling))`
5. Each TP worker reconstructs tensors from tuples
6. Worker computes `delta = scaling * B @ A` for each LoRA pair (scaling=2.0)
7. Worker applies delta using save-load-add pattern:
   - `saved = param.data.clone()`
   - `load_weights([(name, delta)])` — correctly shards delta for TP
   - `param.data.add_(saved)` — adds sharded delta to original weight
8. Total time: ~18s (dominated by tensor serialization and load_weights calls)

---

## 5. Baseline Scores

> **Phase 1 baselines. For current baselines (OOD-1000), see §11.25–§11.28.**

| Model | Dataset | Metric | Score | Date | Config |
|-------|---------|--------|-------|------|--------|
| Teacher (Gemini 3 Pro) | AIME eval (18) | Exact Match | **100.0%** (18/18) | 2026-02-21 | thinking_level=high, max_output_tokens=16384 |
| Teacher (Gemini 3 Pro) | AIME eval (18) | Exact Match | **0.0%** (0/18) | 2026-02-21 | no thinking, max_output_tokens=4096 |
| Teacher (Gemini 3 Pro) | MATH45 eval (30) | Exact Match | **20.0%** (6/30) | 2026-02-21 | no thinking, max_output_tokens=4096 |
| Base Student (gpt-oss-120b) | AIME train (72) | pass@1 (estimate) | **~39.5%** (steps 1-4 avg) | 2026-02-21 | generate_max_len=3072 |
| **Base Student (gpt-oss-20b)** | **AIME eval (18)** | **pass@1** | **27.8%** (5/18) | 2026-02-21 | max_tokens=3072, temp=0.0 |
| **Base Student (gpt-oss-20b)** | **MATH45 eval (18)** | **pass@1** | **44.4%** (8/18) | 2026-02-21 | max_tokens=3072, temp=0.0 |

### 5.1 Teacher Evaluation — Root Cause of 0% AIME Score

**Initial finding**: Gemini 3 Pro scored 0/18 on AIME with our default eval_baseline.py
configuration (no thinking mode, 4096 max output tokens). This contradicted published
benchmarks reporting ~95% AIME accuracy.

**Root cause**: Two configuration errors were responsible:
1. **Missing thinking mode**: Gemini 3 Pro requires `thinking_level=high` (via
   `ThinkingConfig`) to activate extended reasoning. Without it, the model gives
   surface-level answers without deep mathematical reasoning.
2. **Insufficient output tokens**: With `max_output_tokens=4096`, the model's reasoning
   + JSON output frequently exceeds the limit. The structured JSON response gets
   truncated mid-string, causing parse failures. 7 out of 10 responses failed to parse
   at 4096 tokens. Of the 7 truncated responses, 4 had the correct answer visible in
   the truncated text — the model was solving the problems but we couldn't read the answer.

**Fix**: Using `thinking_level=high` + `max_output_tokens=16384` → **18/18 correct (100%)**.

**Detailed results** (`eval_gemini_aime.py`, structured output with JSON schema):
| Problem | Gold | Pred | Correct |
|---------|------|------|---------|
| AIME2025-01 | 104 | 104 | Y |
| AIME2025-02 | 371 | 371 | Y |
| AIME2025-03 | 204 | 204 | Y |
| AIME2025-04 | 25 | 25 | Y |
| AIME2025-05 | 39 | 39 | Y |
| AIME2025-06 | 396 | 396 | Y |
| AIME2025-07 | 73 | 73 | Y |
| AIME2025-08 | 23 | 23 | Y |
| AIME2025-09 | 259 | 259 | Y |
| AIME2025-10 | 16 | 16 | Y |
| AIME2025-11 | 468 | 468 | Y |
| AIME2025-12 | 82 | 82 | Y |
| AIME2025-13 | 244 | 244 | Y |
| AIME2025-14 | 80 | 80 | Y |
| AIME2025-15 | 240 | 240 | Y |
| AIME2025-16 | 113 | 113 | Y |
| AIME2025-17 | 79 | 79 | Y |
| AIME2025-18 | 896 | 896 | Y |

**Lesson**: Gemini 3 Pro is a highly capable AIME solver when configured correctly.
The teacher is NOT weaker than the student — the 0% score was a configuration bug.

### 5.2 Implications for Reward Function

**The teacher-student capability inversion is resolved.** With `thinking_level=high`,
Gemini 3 Pro achieves 100% on AIME — far exceeding the student's ~40%. This means:
1. The `reasoning_quality` signal from Gemini is now meaningful — it's coming from a
   model that can actually solve these problems.
2. The current reward function (`0.6 * correctness + 0.4 * reasoning_quality`) is
   reasonable IF we configure Gemini with thinking mode in the reward function.
3. However, enabling thinking mode in the reward function will significantly increase
   API costs and latency (each reward call will take longer due to extended reasoning).

**Action required**: ~~Update `reward_func.py` to use `thinking_level=high` and
`max_output_tokens=16384` when calling Gemini for reasoning quality assessment.~~
**Superseded**: reward function switched to pure EM — no Gemini dependency (§11.7).

### 5.3 Student Baseline

#### gpt-oss-120b (proxy baseline from training steps 1-4)

Steps 1-4 training-set correctness gives a rough estimate of base model ability on AIME:
- Step 1: 43.8%, Step 2: 28.1%, Step 3: 50.0%, Step 4: 35.9%
- Average: **~39.5%** (but high variance — only 1 prompt per step)
- These generations used the base model weights (LR still in warmup, effective
  updates are negligible due to 100-300x gradient clipping)

#### gpt-oss-20b (proper baseline via eval_baseline.py + vLLM)

Evaluated with vLLM 0.15.1 on 1x H100, `max_tokens=3072`, `temperature=0.0`:

**AIME eval (18 problems)**: **5/18 correct (27.8%)**
- Correct: #6 (396), #10 (16), #11 (468), #13 (244), #17 (79)
- 9 problems had empty extracted answers — model often fails to use `\boxed{}`
  format or generates responses without reaching a final answer
- The 20b model solves fewer AIME problems than 120b (~28% vs ~40%)

**MATH45 eval (18 problems)**: **8/18 correct (44.4%)**
- Stronger on MATH Level 4-5 than on AIME, as expected (MATH is easier)
- Some failures due to answer format mismatch (e.g., `40\text{ cm}` vs `40`)

#### gpt-oss-20b vs 120b: Student Model Choice

| Factor | gpt-oss-120b | gpt-oss-20b |
|--------|-------------|-------------|
| Total params | 117B (5.1B active) | 21B (3.6B active) |
| AIME baseline | ~39.5% (proxy) | 27.8% (measured) |
| MATH45 baseline | TBD | 44.4% |
| GPU for inference | 2x H100 (TP=2) | 1x H100 |
| GPU for training | 4x H100 (ZeRO-3 + offload) | TBD (likely 1-2x H100) |
| Training speed | ~13 min/step | TBD (significantly faster) |
| Disk footprint | ~30 GB (MXFP4) | ~11 GB (MXFP4) |

**Trade-off**: gpt-oss-20b enables faster experimentation (more attempts to tune
reward function, hyperparameters, etc.) at the cost of a lower AIME baseline (28% vs
40%). The 20b model's AIME performance is still in the "learnable" zone (20-70%).

**Recommendation**: Use gpt-oss-20b for rapid iteration to perfect the training recipe
(reward function, learning rate, KL coefficient, etc.), then apply the validated recipe
to gpt-oss-120b for the final training run.

### 5.4 Action Items
- ~~[ ] Run proper gpt-oss-120b baseline with eval_baseline.py~~ — Abandoned; pivoted away from 120b
- [x] For future training runs: add `--eval_steps 1` to force step-0 eval — Done in A27+
- [x] Update `reward_func.py` to use Gemini thinking mode — Superseded: switched to pure EM (§11.7)
- [x] Decide on student model — Resolved: used 20b for A26–A30, now pivoting to qwen2.5-14b (§11.27)

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

### 6.3 Historical Diagnosis (Attempt 24, Step 10)

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
| `ratio_logging_loss.patch` | Log-ratio clamping + ratio tail stats in PolicyLoss.forward | Diagnose gradient spikes: log both raw and clamped log_ratio_max, ratio_max, p99 |
| `vllm_engine.patch` | `apply_lora_update` + `update_weight_from_ref` on LLMRayActor | Dispatches LoRA delta and per-weight sync to vLLM workers via collective_rpc |
| `vllm_worker.patch` | `apply_lora_delta` + `update_weight_from_ray_ref` on WorkerWrap | Reconstructs serialized tensors, computes delta=B@A, applies via save-load-add |

---

## 8. Lessons for Future RL Fine-Tuning

### 8.1 Infrastructure Lessons

1. **Memory hierarchy matters more than raw GPU count.** With 120B params, even 4x80GB GPUs aren't enough without CPU offloading. Plan for: params on CPU, one layer at a time on GPU.

2. **Checkpoint size can silently kill training.** PeftModel + ZeRO-3 saves full model shards (~200 GB) alongside the tiny LoRA adapter (~90 MB). Always verify checkpoint size on first save.

3. **NCCL is fragile across process boundaries.** Ray actors, vLLM subprocess workers, and DeepSpeed all have their own NCCL expectations. When they conflict, fall back to simpler communication (Ray object store, shared memory).

6. **vLLM compiled DAGs block `ray.get()`.** vLLM v1 with Ray TP uses compiled DAGs internally. Worker methods called via `collective_rpc` run inside the DAG execution thread. Calling `ray.get()` there causes a deadlock because the thread cannot yield. Pass all data through `collective_rpc` args instead.

7. **vLLM's msgspec serializer cannot handle torch tensors.** Convert to `(shape_list, dtype_string, data_list)` tuples before passing through `collective_rpc`. Workers reconstruct with `torch.tensor(data, dtype=dtype).reshape(shape)`.

8. **TP-safe weight delta application**: Not all vLLM models support `load_weights(add_to_existing=True)`. The universally compatible pattern is: save original param, call `load_weights(delta)` to shard correctly for TP, then add saved original back.

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

> Moved to §11.17 (post-hoc A29), §11.23 (3-checkpoint comparison), §11.25 (OOD-1000 baseline).

---

## 10. Open Questions

1. **Is 72 AIME problems enough training data?** [Superseded — switched to 50K NuminaMath pool (§11.3)] With 50 episodes × 72 problems = 3600 training iterations, the model sees each problem ~50 times. Risk of overfitting to AIME formats vs learning general math reasoning.

2. **Should we add KL regularization?** [Resolved — using `init_kl_coef=0.001` since A26] Currently `init_kl_coef=0`. The model is free to drift arbitrarily from the base policy. May be fine for short runs but risky for 50 episodes.

3. **Is the Gemini reasoning quality signal helpful or harmful?** [Superseded — switched to pure EM reward (§11.7)] It provides smoother gradients than pure EM, but could incentivize "judge-pleasing" reasoning patterns rather than correct solutions.

4. **Optimal group size?** Currently `n_samples_per_prompt=8`. Larger groups (16, 32) give better advantage estimates but cost more compute per step. With AIME's low solve rate, larger groups may be critical.

5. **Should we use MATH Level 4-5 instead of (or in addition to) AIME?** [Superseded — training on NuminaMath, eval on OOD-1000 (§11.24)] AIME may be too hard for meaningful learning signal. The teacher (Gemini) scores 0% on AIME but 20% on MATH — suggesting MATH problems are in the "learnable" difficulty range.

6. **Teacher-student capability inversion (RESOLVED).** Previously appeared that the
   teacher scored 0% on AIME while the student scored ~40%. Root cause: Gemini was not
   configured with thinking mode (`thinking_level=high`) and had insufficient output tokens
   (4096 vs needed 16384). With correct configuration, Gemini 3 Pro scores **100%** on
   AIME eval (18/18). The teacher is far more capable than the student. The reward function
   needs to be updated to enable thinking mode. See Section 5.1 for full analysis.

7. **gpt-oss-20b as student model?** [Resolved — used 20b for A26–A30; now pivoting to qwen2.5-14b (§11.27)] With only 21B total params (3.6B active), gpt-oss-20b
   would train much faster, fit on fewer GPUs, and allow more experimental iterations.
   However, its baseline AIME performance is unknown and likely lower than 120b's ~40%.
   Need to evaluate before deciding. Key trade-off: faster iteration vs lower ceiling.

---

### 8.5 Gemini API Lessons

1. **Gemini 3 Pro requires `thinking_level=high` for hard math.** Without thinking mode,
   the model gives surface-level answers and scores 0% on AIME. With thinking mode, it
   scores 100%. This is not documented prominently in the API docs.

2. **`max_output_tokens` must account for reasoning + response.** With thinking mode, the
   model produces long reasoning traces. At 4096 tokens, 70% of responses were truncated
   mid-JSON, causing parse failures. 16384 tokens was sufficient for all 18 AIME problems.

3. **`thinking_level` vs `thinking_budget`**: The correct parameter for Gemini 3 is
   `thinking_level` (values: "low", "high") set via `ThinkingConfig`. The older
   `thinking_budget` parameter did not activate thinking in our tests. Always use
   `thinking_level` with `ThinkingConfig`.

4. **Structured output + thinking**: Combining JSON schema enforcement
   (`response_mime_type="application/json"`, `response_json_schema=...`) with thinking
   mode works correctly. The model reasons internally, then outputs valid JSON.

---

*Last updated: 2026-02-25. Model sweep complete (§11.28): qwen2.5-14b selected as next RL target (67% OOD-1000, 33% headroom). RL on gpt-oss-20b showed zero measurable effect (§11.26). A30 step 130 pinned as Baseline-SOTA (73.5% OOD-1000).*

---

## 11. Phase 2: EM-GRPO Experiments

### 11.1 Decision: Student Model = gpt-oss-20b

Selected gpt-oss-20b for rapid iteration. Key trade-offs:

| Factor | gpt-oss-120b | gpt-oss-20b |
|--------|-------------|-------------|
| Total / active params | 117B / 5.1B | 21B / 3.6B |
| AIME pass@1 (3k tokens) | ~39.5% (proxy) | 27.8% |
| AIME pass@1 (20k, high) | TBD | **50.0% (9/18)** |
| MATH45 pass@1 (3k) | TBD | 44.4% |
| GPU for inference | 2×H100 (TP=2) | 1×H100 |
| Disk footprint | ~30 GB | ~11 GB |

### 11.2 Updated Baselines (gpt-oss-20b)

| Dataset | Config | Score | Notes |
|---------|--------|-------|-------|
| AIME eval (18) | max_tokens=3072, temp=0 | 5/18 (27.8%) | 9 empty answers — format/truncation issue |
| AIME eval (18) | reasoning=high, max_tokens=20000 | **9/18 (50.0%)** | All wrong answers hit 20k token limit |
| MATH45 eval (18 subset) | max_tokens=3072, temp=0 | 8/18 (44.4%) | Some format mismatch |
| Apex Shortlist (48) | reasoning=medium, max_tokens=20k | **TBD** | Running |

Key finding: the 3072-token baseline (27.8%) was bottlenecked by truncation and reasoning
budget. With `reasoning=high` and 20k tokens, the model reaches 50% on AIME. This means
the model already has substantial capability — the question is whether SFT is needed at all,
or if we can go straight to RL.

### 11.3 NuminaMath-1.5 Dataset Analysis

**Full dataset**: 896,215 examples.

**Filter pipeline** (for integer-only, verifiable problems):

| Stage | Remaining | Removed |
|-------|-----------|---------|
| Start | 896,215 | — |
| question_type == "math-word-problem" | 631,522 | 264,693 (29.5%) |
| Clean integer answer | ~286k | ~345k |
| Exclude AMC/AIME sources | ~285k | ~920 |
| Require valid problem + solution | ~265k | ~18k |
| Deduplicate by problem hash | ~264k | ~1,264 |
| Remove too-short / empty-solution | ~265,128 | ~962 |

**Token length distribution** (50k sample, gpt-oss-20b tokenizer):

| Metric | Prompt | Solution | Combined |
|--------|--------|----------|----------|
| p50 | 58 | 278 | 344 |
| p90 | 129 | 599 | 711 |
| p95 | 167 | 772 | 905 |
| p99 | 278 | 1,234 | 1,371 |
| p99.5 | 354 | 1,471 | 1,608 |
| max | 1,580 | 4,379 | 4,624 |

**Critical finding**: NuminaMath solutions are very short. `max_seq_len=4096` has
effectively 0% truncation. The plan's discussion of 8k/16k/32k is unnecessary for this
dataset. This also means SFT will be very fast (short sequences = large effective batch size).

**Source distribution** (after all filters, 265k examples):

| Source | Count | % |
|--------|-------|---|
| orca_math | ~93k | 35% |
| synthetic_math | ~74k | 28% |
| cn_k12 | ~40k | 15% |
| olympiads | ~36k | 14% |
| metamath | ~8k | 3% |
| aops_forum | ~6k | 2% |

**Integer answer range**: 84% in [0, 999]. Some extreme outliers exist.

### 11.4 Data Prepared

| File | Size | Purpose |
|------|------|---------|
| data/sft_train.jsonl | 5,000 | Micro-SFT (only if format compliance check fails) |
| data/sft_dev.jsonl | 500 | Dev set for SFT eval |
| data/sft_rl_pool.jsonl | 50,000 | RL training pool (no overlap with SFT) |
| data/apex_shortlist.jsonl | 48 | Apex Shortlist eval (primary) |

All splits are deduped by problem hash. No overlap between SFT train, dev, and RL pool.

### 11.5 MathArena Apex Shortlist 2025 Inspection

48 problems from international math competitions (USA TST, HMMT, IMO Shortlist, etc.).

**Answer type breakdown**:
- ~30 problems: clean integer answers (evaluable with exact match)
- ~5 problems: fractions (\frac{a}{b})
- ~13 problems: symbolic/parametric (n-1, 4N^3+..., etc.) — not evaluable with EM

For our eval, we use the ~35 problems with integer or fraction answers.

### 11.6 Format Compliance Check — SFT Decision

**Method**: 20 MATH Level 4-5 problems, reasoning=medium, max_tokens=8192, temp=0.

**Results**:

| Metric | Value |
|--------|-------|
| \boxed{} in content | 18/20 (90%) |
| \boxed{} in reasoning | 12/20 (60%) |
| \boxed{} anywhere | **19/20 (95%)** |
| Parsed via \boxed{} | 19/20 (95%) |
| Parsed via last_number fallback | 1/20 (5%) |
| Parse failed | 0/20 (0%) |
| Truncated (hit 8k limit) | 2/20 (10%) |

**The single non-\boxed{} result (problem 13) was caused by truncation at 8k tokens**, not
format non-compliance. When the model has enough token budget, it consistently produces
`\boxed{}`.

**Decision: SKIP SFT. Go directly to RL with pure exact-match reward.**

The model's native format compliance is 95%+ when not truncated. SFT would be wasted
compute for this model. The key insight: gpt-oss-20b is already instruction-tuned and
reliably produces `\boxed{}` — the earlier 27.8% baseline (at 3072 tokens) was bottlenecked
by truncation, not format compliance.

### 11.7 Reward Function: Pure Exact-Match

**Decision**: Use pure exact-match reward (no Gemini judge) for the first RL run.

**`reward_func_em.py`**:
- Extract answer from `\boxed{...}` (fallback: last number)
- Normalize and compare to ground truth
- reward = 1 if correct, 0 otherwise
- Zero API cost, zero latency, deterministic

**Rationale**: The Gemini quality signal in the original reward function (0.4 weight for
reasoning quality) was giving partial credit to wrong answers, which incentivizes judge-
pleasing rather than correctness. Pure EM eliminates this failure mode. If reward sparsity
becomes a problem (baseline solve rate < 5%), we can add curriculum or Gemini quality
scoring for correct-only answers later.

### 11.8 RL Training Configuration (gpt-oss-20b)

**File**: `train_grpo_20b.sh`

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | openai/gpt-oss-20b | Faster iteration than 120b |
| Data | NuminaMath RL pool (50k) | Integer-only, deduped, no AIME overlap |
| Reward | Pure exact-match | No Gemini dependency |
| Actor GPUs | 4 | 84GB bf16 / 4 = 21GB/GPU, fits without offload |
| vLLM GPUs | 4 (2 engines × TP=2) | Fast rollout generation |
| ZeRO stage | 3 | Parameter partitioning across 4 GPUs |
| CPU offload | **No** (84GB / 4 = 21GB fits comfortably) | Add if OOM |
| LoRA rank | 32 | Smaller model → smaller rank (attention-only) |
| LoRA alpha | 64 | Scaling = alpha/rank = 2.0 |
| LR | 5e-7 | Conservative for smaller model |
| KL coef | 0.001 | Non-zero for stability (learned from 120b instability) |
| n_samples_per_prompt | 8 | Group size for advantage estimation |
| rollout_batch_size | 16 | Larger than 120b (faster model) |
| generate_max_len | 4096 | Start conservative, increase if truncation high |
| prompt_max_len | 1024 | NuminaMath prompts are short (p99.5 = 354 tokens) |
| num_episodes | 20 | Start with 20, extend if learning |
| Eval | AIME eval (18 problems) every 10 steps | Held-out exact-match accuracy |
| Checkpoints | /mnt/data (1TB NVMe) | Avoid boot disk space issues |
| Logs | /mnt/scratch (5.9TB RAID) | Persistent metrics |

**Changes from 120b training**:
1. No CPU parameter offload (model fits in GPU memory)
2. No Gemini API calls (pure EM reward)
3. Larger training pool (50k vs 72 problems)
4. Smaller LoRA rank (32 vs 64)
5. Lower LR (5e-7 vs 1e-6)
6. Non-zero KL coefficient (0.001 vs 0)

### 11.9 Apex Shortlist Baseline

**Status**: Partial results (killed when training started, 27/39 problems attempted).

| Metric | Value |
|--------|-------|
| Total problems | 48 |
| Evaluable (integer/fraction) | 39 |
| Skipped (symbolic/parametric) | 9 |
| Attempted before kill | 27 |
| Correct | **2/27 (7.4%)** |
| Config | reasoning=medium, max_tokens=20000 |

Correctly solved: `imosl-2024-n7` (59), `Taiwan TST 2025 #3 Quiz 1C` (74).

Apex Shortlist is significantly harder than AIME (~7% vs 28% baseline). Full re-eval needed after training on a saved checkpoint.

### 11.10 Attempt-26 Training Progress

**Status**: Running (PID 198948, started 2026-02-21 15:08)

#### Bugs Fixed During Launch

1. **GPU allocation hang**: `init_kl_coef=0.001` creates a reference model. Default `ref_num_gpus_per_node=8` conflicted with 4 vLLM + 4 actor GPUs. Fix: `--colocate_actor_ref --ref_num_nodes 1 --ref_num_gpus_per_node 4`.

2. **DeepSpeed ZeRO-3 dtype mismatch**: 192 MoE gate/router parameters remained float32 after MXFP4 dequantization, causing `TypeError: output tensor must have the same type as input tensor` during optimizer step. Fix: explicit dtype casting to bf16 before `deepspeed.zero.Init()` in `actor.py`. Patch updated.

#### Training Metrics (Steps 1-34)

| Step | Reward | Correct | Boxed | Loss | GradNorm | Entropy | KL | ClipR | Trunc | RespLen |
|------|--------|---------|-------|------|----------|---------|------|-------|-------|---------|
| 1 | 0.391 | 0.391 | 0.523 | 0.025 | 12.75 | 1.324 | 0.000 | 0.067 | 0.070 | 1207 |
| 2 | 0.445 | 0.445 | 0.695 | 0.021 | 5.07 | 1.266 | 0.000 | 0.084 | 0.211 | 1792 |
| 3 | 0.586 | 0.586 | 0.625 | 0.014 | 9.43 | 1.246 | 0.000 | 0.060 | 0.188 | 1515 |
| 4 | 0.531 | 0.531 | 0.609 | 0.017 | 1.45 | 1.228 | 0.000 | 0.111 | 0.141 | 1642 |
| 5 | 0.438 | 0.438 | 0.438 | 0.025 | 5.01 | 1.375 | 0.000 | 0.137 | 0.414 | 2383 |
| 6 | 0.648 | 0.648 | 0.484 | 0.024 | 14.57 | 1.198 | -0.001 | 0.144 | 0.125 | 1199 |
| 7 | 0.523 | 0.523 | 0.688 | 0.021 | 13.55 | 1.270 | 0.000 | 0.175 | 0.102 | 1241 |
| 8 | 0.594 | 0.594 | 0.695 | 0.024 | 4.38 | 1.338 | 0.000 | 0.189 | 0.125 | 1506 |
| 9 | 0.461 | 0.461 | 0.633 | 0.007 | 0.72 | 1.311 | 0.000 | 0.182 | 0.203 | 1660 |
| 10 | 0.516 | 0.516 | 0.664 | 0.019 | 1.69 | 1.241 | -0.001 | 0.197 | 0.039 | 1192 |
| 11 | 0.438 | 0.438 | 0.359 | 0.013 | 1.85 | 1.439 | 0.001 | 0.178 | 0.336 | 2247 |
| 12 | 0.633 | 0.633 | 0.523 | 0.054 | 11.35 | 1.252 | -0.003 | 0.209 | 0.062 | 845 |
| 13 | 0.719 | 0.719 | 0.586 | 0.039 | 22.34 | 1.331 | -0.003 | 0.197 | 0.125 | 1429 |
| 14 | 0.430 | 0.430 | 0.539 | 0.007 | 1.01 | 1.472 | 0.000 | 0.184 | 0.266 | 2138 |
| 15 | 0.586 | 0.586 | 0.453 | 0.014 | 4.43 | 1.380 | -0.002 | 0.207 | 0.148 | 1443 |
| 16 | 0.406 | 0.406 | 0.734 | 0.015 | 1.36 | 1.302 | 0.000 | 0.181 | 0.180 | 1656 |
| 17 | 0.531 | 0.531 | 0.539 | 0.037 | 18.54 | 1.250 | -0.001 | 0.201 | 0.078 | 992 |
| 18 | 0.664 | 0.664 | 0.656 | 0.020 | 3.12 | 1.260 | 0.000 | 0.193 | 0.133 | 1456 |
| 19 | 0.477 | 0.477 | 0.461 | 0.014 | 3.33 | 1.306 | 0.002 | 0.189 | 0.219 | 1629 |
| 20 | 0.438 | 0.438 | 0.484 | 0.045 | 21.79 | 1.333 | -0.001 | 0.187 | 0.266 | 1929 |
| 21 | 0.234 | 0.234 | 0.531 | 0.022 | 8.02 | 1.338 | -0.001 | 0.201 | 0.188 | 1522 |
| 22 | 0.617 | 0.617 | 0.539 | 0.054 | **70.29** | 1.320 | -0.009 | 0.227 | 0.016 | 815 |
| 23 | 0.531 | 0.531 | 0.562 | 0.028 | 9.06 | 1.413 | -0.001 | 0.201 | 0.219 | 1952 |
| 24 | 0.430 | 0.430 | 0.594 | 0.025 | 5.30 | 1.253 | 0.001 | 0.216 | 0.094 | 1121 |
| 25 | 0.633 | 0.633 | 0.625 | 0.049 | 35.13 | 1.281 | -0.002 | 0.212 | 0.117 | 1215 |
| 26 | 0.562 | 0.562 | 0.477 | 0.026 | 5.53 | 1.322 | 0.000 | 0.216 | 0.070 | 1069 |
| 27 | 0.398 | 0.398 | 0.688 | 0.032 | 50.14 | 1.322 | -0.001 | 0.191 | 0.031 | 1548 |
| 28 | 0.555 | 0.555 | 0.500 | 0.018 | 2.92 | 1.355 | -0.003 | 0.217 | 0.172 | 1361 |
| 29 | 0.531 | 0.531 | 0.773 | 0.014 | 2.78 | 1.272 | 0.000 | 0.179 | 0.188 | 1848 |
| 30 | 0.406 | 0.406 | 0.625 | 0.026 | 3.86 | 1.256 | 0.000 | 0.193 | 0.117 | 1392 |
| 31 | 0.438 | 0.438 | 0.539 | 0.042 | 6.66 | 1.291 | -0.002 | 0.191 | 0.188 | 1601 |
| 32 | 0.398 | 0.398 | 0.656 | 0.156 | **280.92** | 1.366 | 0.001 | 0.184 | 0.156 | 1773 |
| 33 | 0.516 | 0.516 | 0.664 | 0.010 | 1.45 | 1.319 | -0.001 | 0.196 | 0.195 | 1588 |
| 34 | 0.523 | 0.523 | 0.484 | 0.069 | **69.57** | 1.339 | 0.000 | 0.200 | 0.234 | 1740 |
| 35 | 0.523 | 0.523 | 0.375 | 0.022 | 12.75 | 1.327 | 0.015 | 0.202 | 0.000 | 1621 |
| 36 | 0.547 | 0.547 | 0.703 | 0.014 | 1.26 | 1.313 | 0.002 | 0.195 | 0.000 | 1512 |
| 37 | 0.523 | 0.523 | 0.594 | 0.014 | 3.91 | 1.387 | 0.004 | 0.178 | 0.000 | 2180 |
| 38 | 0.633 | 0.633 | 0.609 | 0.024 | 19.92 | 1.269 | 0.006 | 0.189 | 0.000 | 1435 |
| 39 | 0.602 | 0.602 | 0.688 | 0.025 | 7.02 | 1.309 | 0.011 | 0.197 | 0.000 | 1393 |
| 40 | 0.516 | 0.516 | 0.516 | 0.019 | 3.24 | 1.318 | -0.002 | 0.190 | 0.000 | 1376 |
| 41 | 0.625 | 0.625 | 0.531 | 0.022 | 5.63 | 1.310 | -0.012 | 0.199 | 0.000 | 1498 |
| 42 | 0.664 | 0.664 | 0.711 | 0.011 | 3.10 | 1.238 | -0.013 | 0.196 | 0.000 | 1344 |
| 43 | 0.461 | 0.461 | 0.539 | 0.016 | 3.13 | 1.314 | 0.006 | 0.187 | 0.000 | 1757 |
| 44 | 0.469 | 0.469 | 0.789 | 0.016 | 3.64 | 1.278 | 0.001 | 0.193 | 0.000 | 1325 |
| 45 | 0.453 | 0.453 | 0.594 | 0.023 | 5.24 | 1.304 | -0.000 | 0.187 | 0.000 | 1564 |
| 46 | 0.547 | 0.547 | 0.352 | 0.019 | 4.07 | 1.336 | 0.012 | 0.209 | 0.000 | 1029 |
| 47 | 0.695 | 0.695 | 0.570 | 0.203 | **429.64** | 1.337 | 0.002 | 0.207 | 0.000 | 1049 |
| 48 | 0.602 | 0.602 | 0.508 | 0.014 | 1.91 | 1.335 | 0.011 | 0.206 | 0.000 | 1087 |
| 49 | 0.562 | 0.562 | 0.586 | 0.018 | 3.77 | 1.291 | 0.005 | 0.185 | 0.000 | 1442 |
| 50 | 0.477 | 0.477 | 0.555 | 0.016 | 1.89 | 1.293 | -0.002 | 0.190 | 0.000 | 1381 |
| 51 | 0.562 | 0.562 | 0.570 | 0.052 | 29.19 | 1.319 | 0.002 | 0.186 | 0.000 | 1740 |
| 52 | 0.477 | 0.477 | 0.586 | 0.022 | **86.06** | 1.190 | -0.000 | 0.173 | 0.000 | 1699 |

#### AIME 2024 Evaluation Trajectory

| Step | pass@8 (problems) | pass@1 | Notes |
|------|-------------------|--------|-------|
| Baseline | 27.8% (5/18) | — | Pre-training |
| 10 | **38.9% (7/18)** | 32.6% | +2 problems |
| 20 | **38.9% (7/18)** | 35.4% | Peak pass@1 |
| 30 | **44.4% (8/18)** | 34.7% | **Best checkpoint** (+3 problems) |
| 40 | **38.9% (7/18)** | 27.8% | Regression; pass@1 back to baseline |
| 50 | **38.9% (7/18)** | 27.1% | Continued decline; pass@1 below baseline |
| 60 | **33.3% (6/18)** | 28.5% | pass@8 now only +1 problem over baseline |

Config: temp=0.0, n=8 (`nondet_proxy_pass@8`, not true pass@k — see Section 11.10.1).

**Analysis**: Step 30 was the peak (+3 AIME problems over baseline). Steps 40 and 50 both regressed to 7/18 (pass@8). More concerning is the pass@1 trajectory: it peaked at step 20 (35.4%), then declined steadily through step 50 (27.1%) — now *below* the pre-training baseline (27.8%). This suggests the model is becoming less reliable per-sample even as it occasionally reaches correct answers for 7 problems. The gradient norm spikes (especially the 429x spike at step 47) may have contributed by introducing noise into LoRA parameters despite clipping.

**Rolling averages (training correctness)**:
| Steps | Correct (avg) | Boxed (avg) | Loss (avg) | Max GradNorm | Entropy (avg) |
|-------|---------------|-------------|------------|--------------|---------------|
| 1-10 | 0.513 | 0.605 | 0.020 | 14.6 | 1.280 |
| 11-20 | 0.532 | 0.534 | 0.026 | 22.3 | 1.332 |
| 21-30 | 0.490 | 0.591 | 0.029 | 70.3 | 1.313 |
| 31-40 | 0.522 | 0.583 | 0.040 | 280.9 | 1.324 |
| 41-50 | 0.555 | 0.573 | 0.036 | 429.6 | 1.303 |

Training correctness shows a slight upward trend (0.51 → 0.56) but AIME eval does not confirm this — the generalization gap is widening. Maximum gradient norm spikes are escalating each decade (14.6 → 22.3 → 70.3 → 280.9 → 429.6), though the model recovers immediately each time. The escalating spikes combined with declining eval performance suggest the current configuration has reached its useful training horizon.

**Conclusion**: Attempt-26 has passed its useful training horizon. Step 60 eval (6/18 pass@8) confirms continued degradation. **Step 30 is locked as the best checkpoint** for reporting and as the fallback baseline. Post-hoc probe_set_200 evaluation on step 30 is still planned to confirm.

**Best checkpoint**: `global_step30_hf` (8/18 AIME = 44.4%, +3 over baseline).

#### Observations (Steps 1-34)

**Positive signals (steps 1-30):**
1. **Training correctness improved**: Step 1 reward=0.391 → rolling average ~0.50 by steps 20-34. Model is solving more training problems.
2. **AIME eval improved**: 7/18 (38.9%) at step 10, 8/18 (44.4%) at step 30, up from 5/18 (27.8%) baseline. Real generalization through step 30.
3. **Entropy stable**: Range 1.19-1.47 across all 52+ steps, no diversity collapse.
4. **KL near zero**: Policy is updating but not drifting catastrophically from reference.
5. **Truncation manageable**: Mostly <25%, `generate_max_len=4096` is adequate.
6. **Peak correctness 0.719** (step 13): Model can solve ~72% of easy NuminaMath problems in a good batch.

**Concerns:**
1. **Gradient norm spikes escalating**: Full spike history (grad_norm > 10):
   - Steps 1-10: max 14.6 (4 spikes)
   - Steps 11-20: max 22.3 (5 spikes)
   - Steps 21-30: max 70.3 (3 spikes)
   - Steps 31-40: max 280.9 (4 spikes)
   - Steps 41-50: max **429.6** (1 spike, step 47)
   - Steps 51-52: max 86.1 (2 spikes in 2 steps)
   These are *pre-clipping* values (`max_norm=1.0` is active). Model recovers within 1 step after each spike. But the escalating trend correlates with declining eval performance (see Section 11.12 for theory).
2. **Train/eval divergence**: Training correctness improved (rolling avg 0.51 → 0.56) while AIME pass@1 degraded (35.4% peak → 27.1%). The model is overfitting to the NuminaMath training distribution.
3. **No monotonic correctness trend**: Oscillates 0.23-0.72 with high variance. Normal for RL but makes individual step values unreliable — use rolling averages or probe sets.
4. **`has_boxed` inconsistent**: Ranges 0.35-0.79 with no clear improvement. Format compliance is not being reliably learned.

**Root cause analysis for gradient spikes:**
- `max_norm=1.0` is already active -- spikes are *contained* by gradient clipping.
- The logged `grad_norm` is the pre-clipping norm. A spike of 280 means the raw gradient was 280x the clip threshold, but the actual update used norm=1.0.
- **Spikes do NOT correlate with advantage outliers.** Advantage stats (max, min, std) are identical at spike and non-spike steps (adv_max ≈ 0.011 in both cases).
- **Spikes correlate with policy_loss magnitude.** Step 32: loss=0.156 (vs normal 0.01-0.02), grad_norm=280.9. This means individual tokens have extreme log-probability ratios (large shift from reference policy), not extreme advantages.
- Likely cause: with `micro_train_batch_size=1` and gradient accumulation across 16 micro-batches, a single prompt with extreme token-level ratios can dominate the accumulated gradient. Clipping happens post-accumulation, so the spike is visible in pre-clip grad_norm.
- Model recovers immediately (step after each spike has normal grad_norm), confirming clipping is working.
- **Implication**: Advantage clipping would NOT fix these spikes. The correct interventions target the ratio (lower eps_clip, tighter PPO clipping) or the accumulation (per-micro-batch gradient clipping).
- No NaN/Inf observed in any metric.

#### Actions Taken

1. **Grad clipping already enabled** (`--max_norm 1.0`): Confirmed in training config. The spikes are logged as pre-clipping values; actual updates are bounded. No additional action needed for current run.

2. **Reward function instrumented** (`reward_func_em.py`): Added three new logging fields for next run:
   - `parse_method`: 2.0=boxed, 1.0=last_number fallback, 0.0=none (numeric for tensor compatibility)
   - `boxed_in_final`: 1.0 if last `\boxed{}` is in final 20% of response, 0.0 otherwise
   - `truncated_response`: 1.0 if response appears truncated (heuristic: long + no boxed, or ends mid-sentence)

   These will diagnose whether `has_boxed` failures are from: missing boxed entirely, boxed only in reasoning, or truncation preventing final output.

3. **Probe sets created** (two complementary sets for different signals):
   - **ID probe** (`data/probe_set_200.jsonl`): 200 problems from the RL training pool (seed=42). Measures in-distribution learning / stability. Use for early stopping.
   - **OOD probe** (`data/probe_set_200_ood.jsonl`): 202 problems from held-out sources (170 MATH Level 4-5 + 32 Apex Shortlist). Verified disjoint from training pool by hash. Measures generalization. Use for checkpoint selection alongside AIME.

   **Rationale**: ID probe tells you if the model is learning at all; OOD probe tells you if that learning generalizes. Attempt-26 showed train correctness improving (0.51 → 0.56) while AIME eval degraded (35.4% → 27.1% pass@1) — an ID-only probe would have missed this divergence.

4. **Rolling average tracking**: Step-to-step correctness oscillation (0.2-0.7) is normal in RL. For trend analysis, use rolling averages over 10 steps rather than individual step values. Full run rolling averages:
   - Steps 1-10: 0.513, Steps 11-20: 0.532, Steps 21-30: 0.490, Steps 31-40: 0.522, Steps 41-50: 0.555
   - Training correctness shows slight upward trend but AIME eval does not confirm — classic train/eval divergence.

#### Actions Planned for Next Run (Attempt-27)

1. **Ratio tail logging (implemented, ready to apply)**:
   - `ratio_logging_loss.patch`: Log-ratio clamping (`[-20, 20]` before `exp()`) + ratio tail stats per micro-batch (`log_ratio_max`, `log_ratio_min`, `log_ratio_raw_max`, `log_ratio_raw_min`, `ratio_max`, `log_ratio_abs_p99`, `tokens_in_batch`)
   - Both raw (pre-clamp) and clamped log_ratio_max are logged, so we can see how extreme tails really are
   - `ppo_actor.patch`: Reads ratio stats from PolicyLoss and logs to training metrics JSONL

2. **Spike sample logging (implemented, ready to apply)**:
   - When `grad_norm > 50`, dumps prompt hashes (SHA-256) to `spike_log.jsonl` along with grad_norm, policy_loss, and ratio stats
   - Enables post-hoc replay: find which prompts cause spikes, inspect token-level ratios

3. **Eval config fix + metrics labeling**: Three clearly separated eval modes:

   | Metric Name | Temp | n | What It Measures | Use Case |
   |-------------|------|---|------------------|----------|
   | `greedy_pass@1` | 0.0 | 1 | Deterministic single-sample accuracy | Fast checkpoint comparison (primary) |
   | `sampling_pass@8` | 0.6 | 8 | Majority-vote accuracy with diversity | More reliable final evaluation |
   | `nondet_proxy_pass@8` | 0.0 | 8 | vLLM batching nondeterminism fragility | Diagnostics only (gap = fragile solutions) |

   Attempt-26 used `nondet_proxy_pass@8` (temp=0, n=8) which conflates model quality with vLLM nondeterminism. Attempt-27 should use `greedy_pass@1` for fast evals and `sampling_pass@8` for final checkpoint comparison.

4. **Dual probe set eval**: Use both probe sets for checkpoint selection:
   - **ID probe** (`probe_set_200.jsonl`): Early stopping signal. If ID accuracy drops, training is unstable.
   - **OOD probe** (`probe_set_200_ood.jsonl`, 170 MATH + 32 Apex): Primary generalization signal. Checkpoint selected by OOD probe EM + AIME sanity.
   - **AIME** (18 problems): Sanity check only. Too small (1 problem = 5.6% swing) for reliable trend detection.

5. **Stability improvements** (based on root cause analysis):
   - **Lower `eps_clip`** from 0.2 → 0.1 to tighten PPO ratio clipping. This directly targets the root cause (extreme log-prob ratios). The escalating spike pattern (14.6 → 22.3 → 70.3 → 280.9 → 429.6 max grad_norm per decade) confirms the current eps_clip=0.2 allows too much policy shift.
   - **Per-micro-batch gradient clipping**: Worth exploring — clip gradients after each micro-batch backward pass, not just after the full accumulation. This prevents a single extreme micro-batch from dominating the accumulated gradient.
   - Advantage clipping NOT prioritized (advantages are normal during spikes).

6. **Hyperparameter escalation ladder** (change one variable at a time):
   - **First**: Tighten eps_clip 0.2 → 0.1 (directly targets ratio outliers)
   - **If spikes persist**: Add KL coefficient 0.01 as seatbelt
   - **If OOD probe is flat with tighter clipping**: Increase LR from 5e-7 → 1e-6
   - **If ratio tail logs confirm extreme tails even with eps_clip=0.1**: Lower LR instead

7. **Training horizon**: Based on attempt-26, step 30 is the sweet spot. Cap at ~30-40 steps with early stopping triggered by OOD probe degradation (2 consecutive eval drops).

### 11.10.1 Attempt-27 Configuration

**Script**: `train_grpo_20b_a27.sh`

**Single training change**: `eps_clip` 0.2 → 0.1

All other training hyperparameters are identical to attempt-26. Instrumentation-only changes (do not affect training dynamics):

| Parameter | Attempt-26 | Attempt-27 | Type |
|-----------|------------|------------|------|
| `eps_clip` | 0.2 | **0.1** | **Training** |
| `actor_learning_rate` | 5e-7 | 5e-7 | — |
| `micro_train_batch_size` | 1 | 1 | — |
| `n_samples_per_prompt` | 8 | 8 | — |
| `generate_max_len` | 4096 | 4096 | — |
| `rollout_batch_size` | 16 | 16 | — |
| `train_batch_size` | 16 | 16 | — |
| `lora_rank / alpha` | 32 / 64 | 32 / 64 | — |
| `init_kl_coef` | 0.001 | 0.001 | — |
| `max_norm` | 1.0 | 1.0 | — |
| `eval_dataset` | AIME (18) | OOD probe (202) | Instrumentation |
| `eval metric` | `nondet_proxy_pass@8` | `greedy_pass@1` | Instrumentation |
| `eval_steps` | 10 | 5 | Instrumentation |
| `save_steps` | 10 | 5 | Instrumentation |
| `num_episodes` | 20 | 5 (125K actual steps — see §11.13) | Instrumentation |

**Eval schedule**:
- **Built-in** (every 5 steps): `greedy_pass@1` on OOD probe (202 MATH+Apex problems, `temp=0, n=1`)
- **Post-hoc** (on saved checkpoints): AIME (18 problems) + ID probe (200 problems) — run manually after training
- **Early stopping**: If OOD `greedy_pass@1` drops for 2 consecutive evals (10 steps), stop training

**New instrumentation active** (via patches):
- Ratio tail logging: `log_ratio_max`, `log_ratio_raw_max`, `ratio_max`, `log_ratio_abs_p99` per step
- Spike sample logging: prompt hashes dumped to `spike_log.jsonl` when `grad_norm > 50`

**Output paths** (separate from attempt-26):
- Checkpoints: `/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a27`
- Metrics: `/mnt/scratch/rft_metrics_20b_a27/training_metrics.jsonl`
- Spike log: `/mnt/scratch/rft_metrics_20b_a27/spike_log.jsonl`

**Provenance**:
- Git commit: `e734d56` (branch: `attempt-26-grpo-20b-em`)
- Key patch SHAs (sha256):
  - `ratio_logging_loss.patch`: `3f84df906ac7...`
  - `ppo_actor.patch`: `0acfe5d0270e...`
- OpenRLHF version: 0.9.3
- vLLM version: 0.15.1

### 11.11 Lessons Learned (Shareable)

**For others doing GRPO RL fine-tuning with MoE models:**

1. **Grad norm spikes come from ratio outliers, not advantage outliers**: With GRPO + binary reward, we observed grad_norm spikes up to 430x the clip threshold. Initial hypothesis was advantage outliers, but data showed advantage stats are identical at spike and non-spike steps (adv_max ≈ 0.011 in both cases). The actual cause: individual tokens with extreme log-probability ratios (policy shifted far from reference). This means advantage clipping won't fix spikes — tighter PPO ratio clipping (`eps_clip`) or per-micro-batch gradient clipping are the correct interventions.

2. **Spikes escalate over training but are individually harmless**: Over 52 steps, max grad_norm per decade escalated: 14.6 → 22.3 → 70.3 → 280.9 → 429.6. The model recovered within 1 step after every spike (post-clipping). However, the escalation pattern correlates with declining eval performance — suggesting that even clipped spikes inject noise that accumulates in LoRA parameters over many steps. The spike itself doesn't destroy the model, but the trend indicates the policy is drifting into regions where extreme ratios become more common.

3. **Train/eval divergence is the key signal to stop**: Training correctness improved steadily (rolling avg 0.51 → 0.56) while held-out AIME pass@1 degraded (35.4% peak → 27.1%, below pre-training baseline). This is classic overfitting: the model learns to exploit the training distribution (NuminaMath) without generalizing to harder held-out problems (AIME). **An ID-only probe set would miss this.** Use separate ID and OOD probes.

4. **Separate ID and OOD probe sets for different signals**:
   - **ID probe** (from training pool): measures learning/stability. Good for early stopping on catastrophic collapse.
   - **OOD probe** (MATH + Apex, disjoint from training pool): measures generalization. Good for checkpoint selection.
   - **Tiny held-out set** (AIME, N=18): sanity check only. 1 problem = 5.6% swing — too noisy for trends.

5. **Pre-clipping grad_norm is the metric to log, not post-clipping**: Logging the raw gradient norm before clipping tells you *how aggressive* the update would have been. Large values don't mean the model is damaged — they mean clipping is working. But the magnitude trend over training tells you whether the policy is diverging.

6. **Step-to-step correctness is too noisy for trend detection**: With batch size 16 and 8 samples per prompt, each step evaluates ~128 samples from a random subset. One step can show 0.23, the next 0.72. Use 10-step rolling averages or a fixed probe set for reliable progress tracking.

7. **MXFP4 MoE models need dtype casting before ZeRO-3**: MXFP4 dequantization leaves router/gate parameters as float32 while other parameters become bf16. ZeRO-3's all-gather requires uniform dtype. Cast all parameters to bf16 before `deepspeed.zero.Init()`.

8. **Colocate actor and ref model when GPU budget is tight**: With `--colocate_actor_ref`, both share the same GPUs. The ref model only does forward passes (no gradients), so memory overhead is minimal with ZeRO-3.

9. **Eval config matters: clearly label your metrics**: Three distinct eval modes exist and should never be conflated:
   - `greedy_pass@1` (temp=0, n=1): deterministic single-sample accuracy
   - `sampling_pass@8` (temp>0, n=8): majority-vote accuracy with diversity
   - `nondet_proxy_pass@8` (temp=0, n=8): measures vLLM batching nondeterminism, NOT true pass@k. The gap between this and greedy_pass@1 reveals "fragile" solutions that depend on batch composition rather than model capability.

10. **`has_boxed` is a format compliance proxy, not a reward signal**: Track it to detect if the model is losing the ability to produce `\boxed{}`, but don't over-optimize for it. The real signal is held-out eval accuracy.

11. **KL ≈ 0 is expected with small `init_kl_coef` and early training**: With `init_kl_coef=0.001`, the KL penalty is negligible. The policy changes slowly with LR=5e-7 and LoRA rank=32. KL will grow as training progresses. If it stays near zero after 50+ steps, the LR may be too low.

12. **The training horizon is shorter than you think**: With LoRA rank=32, LR=5e-7, and 50k training problems, the best checkpoint was at step 30 (~3 episodes through the data). Steps 40-50 showed declining generalization despite improving training metrics. Over-training LoRA adapters is easy — they have limited capacity and quickly memorize the training distribution. Plan for ~30-step runs and evaluate frequently.

13. **Understand `num_episodes` before launching**: In OpenRLHF, `total_steps = len(prompts) * n_samples_per_prompt // train_batch_size * num_episodes * max_epochs`. With 50K prompts, `n_samples=8`, `train_batch=16`, and `num_episodes=5`, this yields **125,000 steps** — not ~40. The LR warmup (5% of total = 6,250 steps) means the model barely trains in the first 50 steps. Both A26 and A27 were effectively at near-zero LR for their entire observed runs. Always compute `max_steps` before launching and verify the LR schedule reaches meaningful values within your intended step budget.

14. **`max_steps` counts gradient steps, not global steps**: OpenRLHF's `max_steps` formula counts **optimizer.step()** calls (gradient steps). Each "global_step" in the training log represents one full rollout-then-train cycle, which contains `n_samples_per_prompt * rollout_batch_size / train_batch_size` gradient steps. With `n_samples=8`, `rollout_batch=16`, `train_batch=16`: each global step = **8 gradient steps**. So `max_steps=200` = only **25 global steps**, not 200. The LR scheduler advances per gradient step (8 times per global step), so the full cosine schedule completes across those 25 global steps. For N global steps, you need `pool_size = N * rollout_batch_size` (with `num_episodes=1`), or equivalently `num_episodes = N * rollout_batch_size / pool_size`.

### 11.12 Theory: Why Gradient Spikes Escalate

Based on attempt-26 data (52 steps), we propose the following mechanism for escalating gradient norm spikes in GRPO with LoRA:

**The positive feedback loop:**

```
1. Policy shifts slightly from reference on some tokens
   → log_ratio = log π(a|s) - log π_ref(a|s) grows
2. Some prompts have tokens where the shift is large
   → ratio = exp(log_ratio) becomes extreme (>>1 or <<1)
3. Extreme ratios × advantages = large surrogate loss
   → grad_norm spikes (pre-clipping)
4. Clipping bounds the update, but the policy has still shifted
   → On next encounter with similar prompts, ratios are even larger
5. Go to step 2 with larger ratios → escalating spikes
```

**Why LoRA amplifies this**: LoRA's low-rank structure means a small number of parameters control a large number of token log-probabilities. A gradient update that shifts LoRA weights to improve one prompt can have outsized effects on token distributions for other prompts. This is a form of catastrophic interference at the token level.

**Why `micro_train_batch_size=1` amplifies this**: Each micro-batch gradient comes from a single prompt's 8 samples. If that prompt has extreme ratios, its gradient dominates that micro-batch. With gradient accumulation across 16 micro-batches, one extreme prompt contributes 1/16 of the accumulated gradient — but if its gradient norm is 100x the others, it effectively contributes ~86% of the final gradient before clipping.

**Interventions that target the root cause**:
- **Tighter `eps_clip`** (0.2 → 0.1): Limits how much the ratio can contribute to the loss. Directly caps the feedback loop at step 2.
- **Log-ratio clamping** ([-20, 20]): Already implemented. Prevents numerical overflow but doesn't prevent the ratios from being large within the clamped range.
- **Per-micro-batch gradient clipping**: Clips after each micro-batch backward, not just after accumulation. Prevents a single extreme prompt from dominating.
- **Lower LR**: Slows step 1 (policy shifts less per update), reducing the growth rate of the feedback loop.
- **KL penalty**: Adds a cost for policy divergence, explicitly penalizing large log_ratios. Acts as a soft constraint on step 1.

**Interventions that do NOT help**:
- **Advantage clipping**: Advantages are identical at spike and non-spike steps. The problem is the ratio, not the advantage.
- **Larger batch size** (without per-micro-batch clipping): More micro-batches dilute the extreme one, but with `micro_train_batch_size=1` and gradient accumulation, the extreme prompt's gradient is still fully computed and accumulated before clipping.

### 11.13 Attempt-27 Results

**Run parameters**: eps_clip=0.1 (single variable change from A26), OOD probe eval (greedy_pass@1), eval_steps=5.

**Critical finding — `num_episodes` misconfiguration**:
The `num_episodes=5` parameter with 50K prompts in `sft_rl_pool.jsonl` computes to:

```
max_steps = len(prompts) * n_samples_per_prompt // train_batch_size * num_episodes * max_epochs
         = 50000 * 8 // 16 * 5 * 1
         = 125,000 steps
```

The original estimate of "~40 steps" was incorrect. With `lr_warmup_ratio=0.05`, warmup alone is 6,250 steps. At step 46, the LR was at 2.91e-8 — only 5.8% of the target 5e-7. The model was effectively training at near-zero LR for the entire observed run. Run was manually stopped at step 46.

**Note**: Attempt-26 also had this misconfiguration (num_episodes=20 with same 50K pool → 500,000 total steps, stopped manually at step 60). Both A26 and A27's results should be interpreted as very-early-training behavior.

**OOD Eval trajectory (greedy_pass@1 on 202 MATH+Apex problems)**:

| Step | OOD greedy_pass@1 | Notes |
|------|-------------------|-------|
| 5    | 65.72% | |
| 10   | 65.84% | |
| 15   | 64.17% | |
| 20   | 66.52% | |
| 25   | 65.97% | |
| 30   | 64.17% | |
| 35   | **68.87%** | Best |
| 40   | **68.87%** | Best (tie) |
| 45   | 65.84% | |

**Key metrics table** (selected steps):

| Step | policy_loss | grad_norm | ratio_max | p99 | correct | Notes |
|------|------------|-----------|-----------|-----|---------|-------|
| 1    | 0.053 | 10.6 | 74 | 1.96 | 0.602 | |
| 9    | 0.286 | 294.1 | 822 | 1.92 | 0.570 | **SPIKE** |
| 15   | 0.017 | 4.2 | 947 | 2.01 | 0.633 | |
| 21   | 0.130 | 232.9 | 247 | 2.23 | 0.609 | **SPIKE** |
| 23   | 0.027 | 14.4 | 1043 | 1.90 | 0.531 | Highest ratio_max |
| 35   | 0.013 | 3.1 | 309 | 1.72 | 0.672 | Best eval step |
| 40   | 0.280 | 139.7 | 2112 | 1.58 | 0.414 | **SPIKE** + highest ratio |
| 41   | 0.043 | 171.4 | 97 | 1.93 | 0.492 | Residual from step 40 |
| 46   | 0.018 | 3.0 | 34 | 1.72 | 0.680 | |

**LR progression** (all values from warmup phase):
- Step 1: 3.0e-10, Step 10: 6.1e-9, Step 20: 1.2e-8, Step 30: 1.9e-8, Step 40: 2.5e-8, Step 46: 2.9e-8

**Spike log analysis**: 60 entries, all with identical prompt hash `2f444de84f582bf0` — confirms timing bug (grad_norm trigger fires on wrong micro-batch). Bug was fixed for future runs (trigger on per-micro-batch log_ratio_raw_max > 5.0 or policy_loss > 0.15 instead).

**Observations**:

1. **OOD eval stable**: No degradation trend through 46 steps (range: 64.2%–68.9%). This contrasts with A26's degradation (44.4% → 27.8% on AIME). However, interpretation is limited because the LR was near-zero — the model barely changed from the SFT baseline.

2. **Spikes still occur at near-zero LR**: Major spikes at steps 9 (grad=294), 21 (grad=233), 40 (grad=140) despite LR being <3e-8. The ratio_max values (822, 247, 2112) are comparable to A26 but occur at 50x lower LR. This suggests the spike mechanism operates even with minimal policy drift.

3. **log_ratio_abs_p99 stable**: Consistently in 1.6–2.2 range across all steps. The mass of the ratio distribution is well-behaved; spikes are driven by extreme outlier tokens in the tail.

4. **eps_clip=0.1 effect unclear**: Can't isolate the eps_clip effect from the LR effect. Both A26 and A27 were in early warmup when evaluated. A clean comparison requires matching LR schedules or running to similar effective training budgets.

**Checkpoints saved**: Steps 5, 10, 15, 20, 25, 30, 35, 40 at `/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a27/`

**Implication for future attempts**: The `num_episodes` parameter must be calibrated to the dataset size. With 50K prompts:
- For ~40 steps: need `num_episodes=1` + limit rollouts, or use a smaller subset
- For a proper training run: need to decide on a target step count and set episodes accordingly
- Formula: `total_steps = pool_size * n_samples // train_batch_size * num_episodes`

### 11.14 Spike Log Bug Fix

**Bug**: The original spike logging triggered on `grad_norm > 50`. DeepSpeed's `get_global_grad_norm()` returns a stale value from the previous optimizer step. This caused:
1. All spike entries had the same prompt hash (`2f444de84f582bf0`) — the hash of the first micro-batch of the step AFTER the spike
2. `training_metrics` grad_norm is the AVERAGE across 32 micro-batches, while spike_log grad_norm is the per-micro-batch value from DeepSpeed cache — explaining the discrepancy (spike_log shows 2334 vs training_metrics shows 294 for the same step)

**Fix** (applied to installed `ppo_actor.py`, regenerated `ppo_actor.patch`):
- Trigger on per-micro-batch `log_ratio_raw_max > 5.0` (ratio > 148x) or `policy_loss > 0.15` (vs normal 0.01–0.03)
- These values are fresh from the current forward pass, so the prompt hash correctly identifies the triggering micro-batch
- Records `trigger` field ("ratio" or "loss") for diagnostics
- Fix takes effect in next training run (A28+); A27 used the old trigger

### 11.15 Attempt 28 A/B Design: eps_clip Comparison

**Goal**: Clean A/B comparison of `eps_clip=0.2` (default) vs `eps_clip=0.1` (tighter clipping) with a correct LR schedule that actually reaches meaningful values.

**Root cause of A26/A27 failure**: `num_episodes` with 50K prompts produced 125K–500K total steps. With `lr_warmup_ratio=0.05`, warmup was 6,250–25,000 steps. Runs were stopped at step 46–60 — still in early warmup at <6% of target LR. No meaningful training occurred.

**Fix**: Use a 400-prompt subset (`sft_rl_pool_400.jsonl`) with `num_episodes=1` to yield 200 gradient steps (= **25 global steps**). Warmup is 10 gradient steps (~1.25 global steps), reaching 100% of target LR by global step 2. **Note**: This was insufficient for a meaningful A/B comparison — see results below and Lesson #14.

**Scripts**: `train_grpo_20b_a28a.sh` (eps_clip=0.2), `train_grpo_20b_a28b.sh` (eps_clip=0.1)

**Pre-flight check**: `preflight_lr.sh` runs before each launch and aborts if LR@step20 < 30% of target.

| Parameter | 28A | 28B | Notes |
|-----------|-----|-----|-------|
| `eps_clip` | **0.2** | **0.1** | Only training variable |
| `prompt_data` | pool_400 | pool_400 | Same 400 prompts, seed=42 |
| `num_episodes` | 1 | 1 | = 200 gradient steps = **25 global steps** |
| `seed` | 42 | 42 | Fixed for reproducibility |
| `actor_learning_rate` | 5e-7 | 5e-7 | |
| `lr_warmup_ratio` | 0.05 | 0.05 | = 10 warmup gradient steps (~1.25 global steps) |
| `eval_steps` | 10 | 10 | |
| `save_steps` | 10 | 10 | |
| `eval_dataset` | OOD (202) | OOD (202) | greedy_pass@1 |

**LR schedule** (both runs identical, **gradient step** numbering, 8 per global step):
```
grad_step   1 (global ~0.1):  10% of target
grad_step  10 (global ~1.3): 100% of target (peak)
grad_step  50 (global ~6.3):  91% of target
grad_step 100 (global ~12.5): 59% of target
grad_step 150 (global ~18.8): 25% of target
grad_step 200 (global  25.0): 10% of target (min_lr)
```
**Actual LR at global steps** (observed in A28A):
```
global_step  1: 37.5% (avg of grad steps 1-8)
global_step  3: 99.4% (peak)
global_step 10: 75.9%
global_step 20: 21.5%
global_step 25: 10.1% (min_lr, final step)
```

**Pool subset provenance**:
- Source: `sft_rl_pool.jsonl` (50K prompts)
- Selection: reservoir sampling, seed=42, k=400, sorted by SHA-256
- SHA256: `1d09bb26d12774ea770b304bed08b329ca7629c8344dd67dd9f4f7b64e930a0b`
- Verified: 0 overlap with OOD probe (202) and AIME eval (18)

**Success criteria** (original, before step-count discovery):
1. ~~LR reaches >90% of target by step 10 (verified by preflight)~~ ✅ Achieved by global step 3
2. ~~Both runs complete 200 steps without crashes~~ ❌ Runs complete in 25 global steps, not 200
3. ~~OOD eval shows measurable difference between eps_clip=0.2 vs 0.1~~ ❌ Only 2 eval points — insufficient
4. ~~Spike pattern (frequency, magnitude) differs between runs~~ ⚠️ Spikes observed but no comparison yet
5. ~~If OOD degrades in either run, compare degradation onset step~~ ❌ Too few eval points

**Output paths**:
- 28A: `/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a28a/`, `/mnt/scratch/rft_metrics_20b_a28a/`
- 28B: `/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a28b/`, `/mnt/scratch/rft_metrics_20b_a28b/`

**Run order**: Sequential (not parallel — only 8 GPUs). Run 28A first, then 28B.

**Early-stop rule**: If OOD greedy_pass@1 drops for 2 consecutive evals (20 steps) by >= 0.5 pp each, stop and take the best checkpoint so far.

**Provenance (28A)**:
- Git commit: `fb83c91` (branch: `attempt-26-grpo-20b-em`)
- Training script SHA256: `222f1ee489d60ebd...`
- Pool subset SHA256: `1d09bb26d12774ea...`
- Patch SHAs:
  - `ratio_logging_loss.patch`: `3f84df906ac7...`
  - `ppo_actor.patch`: `804ced547a73...`
  - `vllm_engine.patch`: `9f31cc7ad02f...`
  - `vllm_worker.patch`: `5d56735cedb9...`
- OpenRLHF: 0.9.3, vLLM: 0.15.1
- Seed: 42
- PID: 734645, launched 2026-02-21 21:03

**Instrumentation verified** (step 1): All 17 required fields present in training_metrics.jsonl, including `actor_lr` (logged every step). LR at step 1 = 1.875e-7 (37.5% of target) — already 6x higher than A27 ever reached.

#### 28A Final Results (25 global steps — completed, not 200)

**CRITICAL DISCOVERY**: The run completed at 25 global steps, not the expected 200. `max_steps=200` counts **gradient steps** (optimizer.step() calls), not global steps. Each global step has 8 gradient steps (`n_samples_per_prompt * rollout_batch_size / train_batch_size = 8*16/16 = 8`). So 200 gradient steps / 8 = 25 global steps. The prompt dataloader exhausted all 400 prompts in 25 rollout batches (`400 / rollout_batch_size=16 = 25`). The LR scheduler completed its full cosine schedule in just 25 global steps. See **Lesson #14**.

**OOD Eval (greedy_pass@1)**:

| Step | OOD greedy_pass@1 | LR (% of target) |
|------|-------------------|-------------------|
| 10   | 64.79% | 75.9% |
| 20   | 64.60% | 21.5% |

No eval at step 25 (eval_steps=10, not triggered at final step). No significant degradation in 2 eval points.

**Training metrics (all 25 steps)**:

| Step | loss | grad_norm | LR | ratio_max | p99 | correct | Notes |
|------|------|-----------|-----|-----------|-----|---------|-------|
| 1 | 0.022 | 6.0 | 1.88e-7 | 84 | 1.85 | 0.648 | |
| 2 | 0.013 | 2.2 | 4.84e-7 | 26 | 1.86 | 0.688 | LR near peak |
| 3 | 0.018 | 7.2 | 4.97e-7 | 94 | 1.81 | 0.484 | LR at peak |
| 4 | 0.023 | 13.7 | 4.90e-7 | 228 | 1.84 | 0.500 | |
| 5 | 0.024 | 3.1 | 4.80e-7 | 72 | 1.96 | 0.570 | |
| 6 | 0.048 | 18.4 | 4.66e-7 | 144 | 2.04 | 0.484 | |
| 7 | 0.017 | 2.6 | 4.48e-7 | 74 | 1.81 | 0.508 | |
| 8 | 0.023 | 2.5 | 4.28e-7 | 59 | 1.78 | 0.500 | |
| 9 | 0.033 | 5.4 | 4.05e-7 | 38 | 1.89 | 0.703 | |
| 10 | 0.018 | 14.1 | 3.80e-7 | 555 | 1.88 | 0.516 | |
| 11 | 0.059 | 13.9 | 3.52e-7 | 65 | 2.27 | 0.648 | |
| 12 | 0.016 | 26.4 | 3.24e-7 | 95 | 2.21 | 0.711 | |
| 13 | 0.016 | 3.0 | 2.95e-7 | 69 | 1.90 | 0.523 | |
| 14 | 0.015 | 10.0 | 2.65e-7 | 154 | 1.78 | 0.398 | |
| 15 | 0.013 | 1.9 | 2.35e-7 | 100 | 1.55 | 0.320 | |
| **16** | **1.677** | **2497.5** | 2.06e-7 | 387 | 2.07 | 0.609 | **SPIKE — largest ever** |
| 17 | 0.011 | 1.7 | 1.79e-7 | 51 | 1.69 | 0.812 | Recovery |
| 18 | 0.038 | 8.5 | 1.53e-7 | 88 | 1.90 | 0.516 | |
| 19 | 0.028 | 12.1 | 1.29e-7 | **15471** | 1.81 | 0.344 | Extreme ratio tail |
| 20 | 0.030 | 13.7 | 1.08e-7 | 156 | 2.02 | 0.586 | |
| 21 | 0.015 | 6.1 | 8.94e-8 | 57 | 1.62 | 0.430 | |
| 22 | 0.011 | 2.3 | 7.43e-8 | 144 | 1.86 | 0.594 | |
| 23 | 0.025 | 27.8 | 6.27e-8 | 74 | 1.85 | 0.398 | |
| 24 | 0.031 | 6.4 | 5.48e-8 | 99 | 1.82 | 0.508 | |
| **25** | **0.130** | **214.9** | 5.07e-8 | 95 | 1.98 | 0.594 | **SPIKE** |

**Key observations**:

1. **First real training dynamics**: Step 16 produced grad_norm=2497 — 8.5x larger than the worst A26 spike (294). This is what happens when LR actually reaches meaningful levels. The step 16 spike occurred at LR=2.06e-7 (41% of target).

2. **Ratio tails scale with LR**: Step 19 showed ratio_max=15,471 — an order of magnitude beyond anything seen in A26/A27. At real learning rates, the policy drifts faster from reference, producing extreme ratio outliers on rare tokens.

3. **Model recovers from spikes**: Step 17 (immediately after the worst spike) shows loss=0.011, grad=1.7 — perfectly healthy. The gradient clipping (max_norm=1.0) prevents catastrophic updates, but the underlying policy drift that caused the spike remains.

4. **p99 remains stable through spikes**: log_ratio_abs_p99 stayed in 1.5–2.3 range even at step 16 (p99=2.07) and step 19 (p99=1.81). This confirms spikes are driven by extreme tail tokens, not a broad shift in the ratio distribution.

5. **Spike log hash bug persists**: All 154 spike entries show the same prompt hash `2f444de84f582bf0`. The trigger types are correctly differentiated (69 ratio, 85 loss), but the prompt identification still fails — likely a bug in extracting prompt tokens from padded sequences (system prompt + padding dominates the hash).

6. **Second spike at step 25**: grad_norm=214.9, loss=0.130. Smaller than step 16 but still elevated. Occurred at LR=5.07e-8 (10.1%, at min_lr). Suggests residual policy drift from earlier updates continues to produce ratio outliers even at minimum learning rate.

7. **Reward stability**: Average reward=0.544 across all 25 steps (0.541 excluding step 16). No clear upward or downward trend — too few steps to distinguish signal from noise with 128-sample batches.

8. **Only 2 checkpoints saved**: Steps 10 and 20 (save_steps=10). No final checkpoint or eval at step 25.

9. **Run too short for meaningful A/B comparison**: 25 global steps with only 2 eval points is insufficient to detect eps_clip effects. Need longer runs for A28B.

**Rolling averages**:
   - Steps 1–10: loss=0.024, correct=0.560, grad=7.5
   - Steps 11–20: loss=0.190, correct=0.547, grad=258.9 (dominated by step 16 spike)
   - Steps 21–25: loss=0.046, correct=0.505, grad=51.5

**Conclusion**: A28A confirms the LR schedule fix works (LR reached 99.4% of target at step 3, vs <6% in A26/A27). However, 25 global steps is too few to draw conclusions about eps_clip effectiveness or training convergence. The step-count miscalculation (gradient steps vs global steps) must be fixed before running A28B.

#### Step-Count Fix for A28B and Beyond

**Problem**: `max_steps=200` in the formula counts gradient steps, but each global step has 8 gradient steps. With 400 prompts and `rollout_batch_size=16`, the prompt pool is exhausted in 25 global steps.

**Options for ~200 global steps**:

| Option | Pool size | num_episodes | Global steps | Pros | Cons |
|--------|-----------|-------------|-------------|------|------|
| A | 3,200 | 1 | 200 | Each prompt seen once | Need new pool, overlap check |
| B | 400 | 8 | 200 | Reuse existing pool | Each prompt seen 8 times — memorization risk |
| C | 1,600 | 2 | 200 | Moderate pool | 2 passes; need new pool |

**Recommendation**: Option A (3,200-prompt pool, num_episodes=1) avoids memorization risk and gives the most diverse training signal. Reservoir sample from the 50K pool with a new seed (or extend the existing 400-prompt pool to 3,200).

**Preflight update done**: `preflight_lr.sh` now displays both gradient steps and global steps, and accepts `RBS` (rollout_batch_size) parameter.

### 11.16 Attempt 29 A/B Design: eps_clip Comparison (corrected step count)

**Goal**: Same as A28 — clean A/B comparison of `eps_clip=0.2` vs `0.1` — but with correct step count (200 global steps instead of 25).

**What changed from A28**: Pool size increased from 400 to **3,200 prompts** (`sft_rl_pool_3200.jsonl`). Each prompt seen once (`num_episodes=1`). This yields `3200 / 16 = 200` global steps and `3200 * 8 / 16 = 1600` gradient steps.

**Scripts**: `train_grpo_20b_a29a.sh` (eps_clip=0.2), `train_grpo_20b_a29b.sh` (eps_clip=0.1)

| Parameter | 29A | 29B | Notes |
|-----------|-----|-----|-------|
| `eps_clip` | **0.2** | **0.1** | Only training variable |
| `prompt_data` | pool_3200 | pool_3200 | Same 3,200 prompts, seed=42 |
| `num_episodes` | 1 | 1 | Each prompt seen once |
| `global_steps` | **200** | **200** | = 1,600 gradient steps |
| `seed` | 42 | 42 | Fixed for reproducibility |
| `actor_learning_rate` | 5e-7 | 5e-7 | |
| `lr_warmup_ratio` | 0.05 | 0.05 | = 80 gradient steps (~10 global steps) |
| `eval_steps` | 10 | 10 | = 20 eval points |
| `save_steps` | 10 | 10 | = 20 checkpoints |
| `eval_dataset` | OOD (202) | OOD (202) | greedy_pass@1 |

**LR schedule** (verified by preflight_lr.sh):
```
global_step  10: 100% of target (peak, warmup complete)
global_step  50:  91%
global_step 100:  59%
global_step 150:  25%
global_step 200:  10% (min_lr)
```

**Pool subset provenance**:
- Source: `sft_rl_pool.jsonl` (50K prompts)
- Selection: reservoir sampling, seed=42, k=3200, sorted by SHA-256
- SHA256: `92b5a983eb343d6627d84f8db79238a9981f3965539acbd920dc87846b905ff2`
- Verified: 0 overlap with OOD probe (202) and AIME eval (18)
- 30 prompts overlap with A28's 400-prompt pool (expected, same source pool + seed)

**Success criteria**:
1. Both runs complete 200 global steps without crashes
2. OOD eval shows measurable difference between eps_clip=0.2 vs 0.1
3. Spike pattern (frequency, magnitude) differs between runs
4. If OOD degrades in either run, compare degradation onset step

**Early-stop rule**: If OOD greedy_pass@1 drops for 3 consecutive evals (30 global steps) by >= 0.5 pp each, stop and take the best checkpoint.

**Run order**: Sequential (not parallel — only 8 GPUs). Run 29A first, then 29B.

#### 29A Final Results (200 global steps, eps_clip=0.2)

**Run completed**: 2026-02-22 05:26 – 14:25 (~9 hours). All 200 global steps, 20 evals, 20 checkpoints.

**OOD Eval (greedy_pass@1)**:

| Step | OOD % | Δ from step 10 |
|------|-------|----------------|
| 10 | 64.60 | baseline |
| 20 | 63.56 | -1.04 |
| 30 | 63.80 | -0.80 |
| 40 | 62.50 | -2.10 |
| 50 | **65.78** | +1.18 |
| 60 | 64.11 | -0.49 |
| 70 | 65.22 | +0.62 |
| 80 | 63.61 | -0.99 |
| 90 | 63.24 | -1.36 |
| 100 | **65.78** | +1.18 |
| 110 | 62.44 | -2.16 |
| 120 | 64.85 | +0.25 |
| 130 | 65.41 | +0.81 |
| 140 | 64.17 | -0.43 |
| 150 | 64.48 | -0.12 |
| 160 | 64.42 | -0.18 |
| 170 | 65.35 | +0.75 |
| 180 | 65.90 | +1.30 |
| 190 | 65.10 | +0.50 |
| **200** | **65.97** | **+1.37 ← BEST** |

**Summary**: OOD eval oscillates in a 3.5pp band (62.4–66.0%) with no sustained degradation. Best checkpoint is the final one (step 200, 65.97%). The slight upward trend in the second half suggests the model is still learning at step 200, but the signal is noisy.

**Gradient spikes** (8 events, all with grad_norm > 100):

| Step | grad_norm | policy_loss | LR (% target) |
|------|-----------|-------------|----------------|
| 37 | 898.9 | 0.614 | 95.8% |
| 57 | 1444.0 | 0.142 | 87.4% |
| 111 | 142.2 | 0.055 | 50.9% |
| 125 | 604.1 | 0.020 | 40.8% |
| 133 | 1015.9 | 0.329 | 35.3% |
| 154 | 423.9 | 0.925 | 22.7% |
| 155 | 1590.1 | 0.177 | 22.2% |
| 164 | 264.6 | 0.166 | 18.0% |

Spikes span steps 37–164 (peak and mid-decay LR). Worst was step 155 (grad=1590). Model recovers within 1–2 steps every time.

**Rolling averages (20-step windows)**:
- Steps 1–20: reward=0.539, grad=6.6
- Steps 21–40: reward=0.530, grad=54.9
- Steps 41–60: reward=0.537, grad=84.8
- Steps 61–80: reward=0.512, grad=8.2
- Steps 81–100: reward=0.530, grad=12.7
- Steps 101–120: reward=0.523, grad=13.8
- Steps 121–140: reward=0.493, grad=90.9
- Steps 141–160: reward=0.523, grad=113.8
- Steps 161–180: reward=0.522, grad=29.4
- Steps 181–200: reward=0.513, grad=13.7

**KL drift**: 0.0000 → -0.0395 (gradual, not pathological)

**Checkpoints**: 20 saved at `/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a29a/global_step{10..200}_hf/`

**Provenance**:
- Git commit: post-`fb83c91` (A29 scripts added)
- Pool: `sft_rl_pool_3200.jsonl` (SHA256=`92b5a983eb343d66...`)
- PID: 1155951, launched 2026-02-22 05:26, completed 2026-02-22 14:25

#### 29B Final Results (200 global steps, eps_clip=0.1)

**Run completed**: 2026-02-22 14:28 – 23:23 (~9 hours). All 200 global steps, 20 evals, 20 checkpoints.

**OOD Eval (greedy_pass@1)**:

| Step | OOD % | Δ from step 10 |
|------|-------|----------------|
| 10 | 65.78 | baseline |
| 20 | 63.37 | -2.42 |
| 30 | 62.62 | -3.16 |
| 40 | 63.80 | -1.98 |
| 50 | 61.76 | -4.02 |
| 60 | 65.28 | -0.50 |
| 70 | 65.10 | -0.68 |
| **80** | **67.08** | **+1.30 ← BEST** |
| 90 | 65.72 | -0.06 |
| 100 | 64.11 | -1.67 |
| 110 | 63.86 | -1.93 |
| 120 | 63.30 | -2.48 |
| 130 | 63.55 | -2.23 |
| 140 | 64.48 | -1.30 |
| 150 | 64.11 | -1.67 |
| 160 | 67.02 | +1.24 |
| 170 | 65.22 | -0.56 |
| 180 | 63.30 | -2.48 |
| 190 | 63.92 | -1.86 |
| 200 | 64.67 | -1.12 |

**Gradient spikes** (3 events, all with grad_norm > 100):

| Step | grad_norm | policy_loss | LR (% target) |
|------|-----------|-------------|----------------|
| 3 | 129.5 | 0.111 | 24.7% |
| 60 | 241.9 | 0.036 | 85.8% |
| 71 | 360.8 | 0.245 | 80.6% |

**No spikes after step 71** — the last 129 steps are completely spike-free.

**KL drift**: 0.0000 → -0.0526

**Checkpoints**: 20 saved at `/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a29b/global_step{10..200}_hf/`

**Provenance**:
- PID: 1876917, launched 2026-02-22 14:28, completed 2026-02-22 23:23

#### A29 A/B Comparison: eps_clip=0.2 vs 0.1

**OOD Generalization — No meaningful difference**:
- Overall avg: A=64.51%, B=64.40% (Δ=-0.11pp)
- Tail avg (steps 150–200): A=65.20%, B=64.71% (Δ=-0.50pp)
- Best checkpoint: A=65.97% (step 200), B=67.08% (step 80)
- Both oscillate in ~4pp band with no sustained degradation

**Stability — Clear winner: eps_clip=0.1**:

| Metric | A (eps_clip=0.2) | B (eps_clip=0.1) | Ratio |
|--------|-----------------|-----------------|-------|
| Spike count | 8 | **3** | 2.7x fewer |
| Max grad_norm | 1590 | **361** | 4.4x smaller |
| Mean spike grad | 798 | **244** | 3.3x smaller |
| Last spike step | 164 | **71** | Stops 93 steps earlier |
| p99 ratio trend | 1.89→1.92 ↑ | **1.86→1.82 ↓** | Tails shrinking |

**Mechanism**: Tighter clipping (0.1 vs 0.2) limits how much a single extreme-ratio token can contribute to the surrogate loss gradient. This breaks the positive feedback loop: policy drift → extreme ratios → large gradients → more drift. With eps_clip=0.1, ratio tails actually shrink over training (p99: 1.86→1.82), while eps_clip=0.2 sees them grow (1.89→1.92).

**Policy dynamics**:
- Reward: identical (A=0.522, B=0.521)
- KL: B drifts slightly more (-0.053 vs -0.040) — tighter clipping constrains gradient magnitude but doesn't prevent policy movement, just makes it smoother
- Mean policy loss (non-spike): B=0.0041, A=0.0061

**Verdict**: `eps_clip=0.1` is the better default for this setup. OOD generalization is equivalent, but training stability is dramatically better. Spikes are 3x rarer, 4x smaller, and cease entirely after step 71 (vs continuing through step 164 with eps_clip=0.2).

**Lesson #15**: `eps_clip=0.1` (half the OpenRLHF default of 0.2) eliminates late-run gradient spikes without sacrificing generalization. For LoRA GRPO with binary reward and small batch sizes (micro_train_batch_size=1), tighter clipping is advisable because individual extreme-ratio tokens dominate the per-micro-batch gradient.

### 11.17 Post-Hoc Evaluation (A29 Checkpoints)

**Eval methodology**: Merge LoRA adapter into base model with PEFT (CPU), save merged weights (MXFP4 expert weights preserved), serve with vLLM (TP=2), evaluate via OpenAI-compatible API with greedy decoding (temperature=0, max_tokens=4096). Sequential per-problem inference.

**Eval sets**:
- **ID probe** (N=200): reservoir-sampled from training pool (3200 prompts), integer-label math problems
- **OOD probe** (N=202): held-out math problems, zero overlap with training pool
- **AIME** (N=18): competition-level problems, zero overlap with training pool

**Results**:

| Checkpoint | ID (N=200) | OOD (N=202) | AIME (N=18) |
|---|---|---|---|
| a29a_step200 (eps=0.2) | **53.0%** | **68.3%** | 33.3% |
| a29b_step80 (eps=0.1) | 48.5% | 60.4% | 50.0% |
| a29b_step200 (eps=0.1) | 50.5% | 62.9% | 38.9% |

**Statistical significance** (SE ≈ sqrt(p(1-p)/N)):
- ID (N=200): SE≈3.5%, 95% CI ≈ ±6.9pp → range 48.5-53.0% overlaps within ~1.3 SE
- OOD (N=202): SE≈3.3%, 95% CI ≈ ±6.5pp → a29a (68.3%) vs a29b_step200 (62.9%) Δ=5.4pp ≈ 1.2 SE
- AIME (N=18): SE≈11.8%, 95% CI ≈ ±23pp → all three within noise

**Observations**:
1. **No checkpoint separates with statistical significance** — all ID and OOD differences are within ~1.5 SE
2. **OOD > ID across all checkpoints** — suggests the ID probe (in-distribution from training pool) may be harder than the OOD probe, or the model learns general math reasoning rather than memorizing training examples
3. **a29a_step200 is numerically best on both ID and OOD** — but the advantage is not statistically significant
4. **B step80 vs B step200**: OOD improved from 60.4% → 62.9% (continued training helps slightly), ID improved from 48.5% → 50.5%, but AIME dropped from 50.0% → 38.9% (noise on N=18)
5. **AIME is too noisy** to inform decisions at N=18

**Checkpoint selection**: Per decision rule (prefer B step200 unless B step80 clearly better on ID/AIME) → **a29b_step200** is the selected checkpoint. While a29a_step200 has higher point estimates, the advantage is not statistically significant (Δ≈1.2 SE on OOD), and a29b has dramatically better training stability. The stability advantage is a systematic effect (not noise), while the eval accuracy difference is plausibly noise.

**Lesson #16**: Post-hoc evaluation on N=200 probe sets has SE≈3.4%, giving 95% CI of ±6.7pp. Distinguishing checkpoints within ~5pp requires either larger eval sets or multiple-seed evaluation. At current sample sizes, in-training OOD eval (which runs every 10 steps over 20 evaluations) provides more signal through averaging than a single post-hoc greedy eval.

### 11.18 Attempt-30 Design

**Goal**: Test `micro_train_batch_size=2` (doubled from 1) to reduce per-step gradient variance, which could:
- Further reduce spike frequency/severity
- Allow the optimizer to make more consistent updates
- Potentially improve OOD generalization by smoothing the loss landscape

**Configuration changes from A29B baseline (eps_clip=0.1)**:
- `micro_train_batch_size`: 1 → **2**
- All other hyperparameters identical to A29B

**Memory check needed**: micro_train_batch_size=2 doubles the per-GPU memory for activations during the training forward/backward pass. With the current setup (LoRA r=32, 1.8B active params, MoE 20B total, MXFP4 quantization), this needs profiling to confirm it fits in GPU memory.

**Eval plan**:
- Monitor OOD eval every 10 steps (as in A29)
- Compare spike frequency/severity with A29B
- Post-hoc eval on best checkpoint if OOD shows improvement

**Decision criteria**:
- If spikes drop further AND OOD improves → micro_train_batch_size=2 is the better default
- If spikes drop but OOD flat → marginal improvement, consider increasing horizon (400 global steps with 6400-prompt pool)
- If OOM → fall back to micro_train_batch_size=1 and try gradient accumulation or reduced max_len

### 11.19 Bug Fix: LoRA Weight Accumulation in vLLM Worker

**Bug discovered**: Post-hoc eval showed a ~6-7pp mismatch vs in-training eval on the same OOD probe set (e.g., a29b_step80: 67.1% in-training vs 60.4% post-hoc). With greedy decoding on N=202, this should be near-zero.

**Root cause**: Our `apply_lora_delta()` in `patches/vllm_worker.patch` had an accumulation bug. Each call computed `delta = scaling * B @ A` (the full LoRA contribution) and added it to the current weight — which already contained previous deltas. After N weight syncs:

```
param = W_base + delta_0 + delta_1 + ... + delta_N  (BUG)
param = W_base + delta_N                             (CORRECT)
```

Since `delta_N` is the complete LoRA correction (not an increment), the correct behavior is to replace the previous delta, not accumulate.

**Impact**:
1. **In-training eval**: Evaluated an accumulated-delta model that doesn't correspond to any saved checkpoint. All in-training OOD eval numbers from A29 (and earlier runs using this path) are unreliable.
2. **Rollout generation**: vLLM generated training samples using the accumulated model, creating a distribution mismatch between the generation policy and the actor policy. Effect is likely minor with small LR/LoRA, but introduces an uncontrolled variable.
3. **Post-hoc eval**: Uses PEFT merge directly (no `apply_lora_delta`), so post-hoc results are correct.
4. **Saved checkpoints**: PEFT adapters are saved from the actor model (which has correct LoRA weights), so all saved checkpoints are correct.

**Fix**: Cache the original base weight shard on the first call to `apply_lora_delta()`. On each subsequent call, reconstruct `param = base_shard + sharded(delta)` instead of accumulating. Added `self._base_weights` dict (~640 MB per vLLM worker GPU for this model's attention weights).

**Lesson #17**: When implementing LoRA weight sync to a separate inference engine, distinguish between "full delta" (the complete LoRA contribution `scaling * B @ A`) and "incremental delta" (the change from the previous step). If sending the full delta, the receiver must subtract the old delta before adding the new one, or cache the base weights and reconstruct from scratch. The original code treated a full delta as if it were an incremental delta.

**Lesson #18**: Always validate in-training eval against an independent post-hoc eval pipeline on a subset of checkpoints. Discrepancies exceeding the expected noise (SE ≈ sqrt(p(1-p)/N)) indicate a pipeline bug.

### 11.20 Profile: micro_train_batch_size=2

**Setup**: 5 global steps (40 gradient steps) on 80-prompt pool, all other config identical to A29B (eps_clip=0.1). Script: `profile_micro2.sh`.

**Results**:

| Step | Time (s) | Grad Norm | Reward | Tokens/batch | log_ratio_abs_p99 |
|------|----------|-----------|--------|-------------|-------------------|
| 1 | (warmup) | 5.16 | 0.547 | 3547 | 1.71 |
| 2 | 125.9 | 1.77 | 0.352 | 4489 | 1.77 |
| 3 | 104.1 | 39.06 | 0.500 | 2067 | 2.09 |
| 4 | 114.7 | 171.40 | 0.398 | 2990 | 1.99 |
| 5 | 113.5 | 7.03 | 0.609 | 2788 | 1.89 |

**Key findings**:
- **No OOM**: micro_train_batch_size=2 fits comfortably in GPU memory
- **Zero cache flush warnings**: Improved vs micro=1 (A29B had cache flush warnings at steps 1 and 5-6)
- **~18% faster per global step**: avg 114.6s/step (micro=2) vs ~140s/step (micro=1). Fewer micro-batch passes (8 vs 16) per gradient step reduces kernel launch overhead
- **Grad norms**: One spike (171 at step 4) on 80 prompts — too few steps to compare with A29B, but step 4 spike suggests spikes are still possible with micro=2
- **Verdict**: micro_train_batch_size=2 is viable. Proceed with full attempt-30.

**Lesson #19**: Doubling micro_train_batch_size from 1→2 in this setup does not cause OOM and actually reduces training time by ~18% (fewer micro-batch forward/backward passes, better GPU utilization). The previous cache flush warnings with micro=1 disappear, suggesting micro=1 was suboptimal for GPU memory management.

### 11.21 Attempt-30: micro_train_batch_size=2 (Complete)

**Status**: Complete. 200 global steps, 2026-02-23 06:21 → 13:54 (7.4 hours wall time).

**Config**: Identical to A29B except micro_train_batch_size: 1 → 2. Script: `train_grpo_20b_a30.sh`.

**Step timing**: median 114s/step (~18% faster than A29B's ~140s), confirming profile results.

**OOD Eval Trajectory (N=202, greedy@1)**:

| Step | A30 (FIXED) | A29B (BUGGY) | Delta |
|------|-------------|--------------|-------|
| 10 | 63.4% | 65.8% | -2.4pp |
| 20 | 63.6% | 63.4% | +0.2pp |
| 30 | 66.2% | 62.6% | +3.6pp |
| 40 | 62.1% | 63.8% | -1.7pp |
| 50 | 66.1% | 61.8% | +4.4pp |
| 60 | 64.2% | 65.3% | -1.1pp |
| 70 | 63.5% | 65.1% | -1.6pp |
| 80 | 64.7% | 67.1% | -2.4pp |
| 90 | 62.8% | 65.7% | -2.9pp |
| 100 | 66.4% | 64.1% | +2.3pp |
| 110 | 64.4% | 63.9% | +0.5pp |
| 120 | 62.9% | 63.3% | -0.4pp |
| 130 | 66.5% | 63.5% | +3.0pp |
| 140 | 63.9% | 64.5% | -0.6pp |
| 150 | 65.3% | 64.1% | +1.2pp |
| 160 | 65.7% | 67.0% | -1.4pp |
| 170 | 64.2% | 65.2% | -1.0pp |
| 180 | 65.2% | 63.3% | +1.9pp |
| 190 | 64.7% | 63.9% | +0.8pp |
| 200 | 65.5% | 64.7% | +0.9pp |

**Full-run summary**:
- A30 mean OOD: **64.57%** (range 62.1–66.5%)
- A29B mean OOD: 64.40% (range 61.8–67.1%)
- Delta: +0.17pp (not significant)

Note: A29B used BUGGY rollout generation and eval (accumulated LoRA weights). A30 is the first run with correct pipeline. Direct comparison is confounded by the bug fix.

**Training metrics by quarter**:

| Quarter | Reward | Grad Norm | Loss | Entropy | Spikes>50 |
|---------|--------|-----------|------|---------|-----------|
| Steps 1-50 | 0.539 | 12.89 | 0.027 | 1.307 | 2 |
| Steps 51-100 | 0.513 | 11.30 | 0.009 | 1.317 | 2 |
| Steps 101-150 | 0.537 | 20.27 | 0.014 | 1.315 | 3 |
| Steps 151-200 | 0.504 | 21.23 | 0.014 | 1.316 | 1 |

- **Reward**: Stable at ~0.52, no significant trend
- **Entropy**: Flat at ~1.31 (no collapse)
- **Grad norm spikes (>50)**: 8/200 (4.0%), comparable to A29B
- **Step time**: median 114s (18% faster than A29B)

### 11.22 Pipeline Regression Check: PASS

**Test**: Compare in-training eval with post-hoc eval (PEFT merge + vLLM) on the same checkpoint.

**Checkpoint**: A30 step 10 (`/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a30/global_step10_hf`)

**Results**:
- In-training eval:  **63.37%** (128/202)
- Post-hoc eval:     **63.37%** (128/202)
- Difference:        **0.0 problems** (exact match)

**Verdict**: **PASS**. The LoRA accumulation fix is fully validated. In-training eval curves are trustworthy going forward.

This is a critical milestone: for the first time, in-training eval and post-hoc eval produce identical results. Previously, the accumulation bug caused 2-7pp inflation in in-training numbers (growing with training steps).

**Lesson #20**: After fixing a pipeline bug, always run a quantitative regression check (same checkpoint, both eval paths, identical settings) to confirm the fix works in the actual training pipeline — not just in isolation. The unit test (`test_lora_fix.py`) validated correctness on a single layer; this regression check validates the full end-to-end pipeline.

### 11.23 Post-Hoc Eval: 3-Checkpoint Comparison and micro=2 Decision

**Purpose**: Compare micro_train_batch_size=1 (A29B) vs micro_train_batch_size=2 (A30) with post-hoc eval to decide whether to promote micro=2 as default.

**Checkpoints evaluated** (all merged via PEFT merge_and_unload + saved to HF format):
- `a29b_step200`: A29B final checkpoint (micro=1, LoRA bug present during training but merge is correct)
- `a30_step130`: A30 best step by in-training OOD (66.52%, micro=2, fixed pipeline)
- `a30_step200`: A30 final checkpoint (micro=2, fixed pipeline)

**Post-hoc eval** (vLLM TP=2, greedy, max_tokens=4096, identical settings for all):

| Model | OOD (N=202) | ID (N=200) | AIME (N=18) |
|-------|-------------|------------|-------------|
| a29b_step200 (micro=1) | 62.87% (127) | 51.00% (102) | 38.89% (7) |
| a30_step130 (micro=2, best) | **65.35% (132)** | **52.50% (105)** | **38.89% (7)** |
| a30_step200 (micro=2, final) | 61.88% (125) | 49.50% (99) | 33.33% (6) |

**Statistical notes** (SE ≈ sqrt(p(1-p)/N) at p≈0.65):
- OOD (N=202): SE ≈ 3.4%, 95% CI ≈ ±6.6pp
- ID (N=200): SE ≈ 3.4%, 95% CI ≈ ±6.7pp
- AIME (N=18): SE ≈ 11%, 95% CI ≈ ±22pp

All pairwise differences are within sampling noise. No statistically significant difference between micro=1 and micro=2.

**Observations**:
1. **A30 step 130 > A30 step 200**: The best in-training checkpoint (step 130) outperforms the final checkpoint in post-hoc eval across all metrics. This suggests cosine LR over-decays in the final quarter (steps 150-200), potentially locking in suboptimal weights.
2. **micro=2 is not worse**: A30 step 130 (micro=2) is numerically better than A29B step 200 (micro=1) on all three eval sets (+2.5pp OOD, +1.5pp ID, tied AIME). While within noise, there is zero evidence of regression.
3. **Practical benefits of micro=2**: ~18% faster per step (114s vs 140s), zero cache flush warnings, no OOM risk.

**Decision**: **micro_train_batch_size=2 promoted to default** for all future runs. Rationale:
- No regression on any metric (and slight numerical improvement)
- Meaningful speed improvement (~18%)
- Better GPU memory behavior (no cache flush warnings)
- The differences are within statistical noise at N=200, reinforcing the need for OOD-1000 (N=1000, SE≈1.5%) to detect meaningful effects in future experiments.

**Lesson #21**: When comparing hyperparameter changes, post-hoc eval on multiple checkpoints (best + final) with identical settings is essential. In-training eval curves are noisy (±3pp step-to-step), and comparing single checkpoints at the same step number can be misleading. Compare best-vs-best and final-vs-final to separate the hyperparameter effect from checkpoint selection noise.

### 11.24 Experiment SOP v2.1-final: STOP vs DEBUG Decision Framework

**Purpose**: Prevent self-deception and wasted compute by defining mechanical criteria for when to declare success (STOP-1) vs when to switch to structured debugging (STOP-3). v2.1-final adds staged Gate-1a/1b, pool-variance STOP-3b clause, pipeline integrity preflight, and calibrated early-stop thresholds.

#### 0) Baselines (lock these down for A31+)

Maintain two baselines, evaluated with the **same** eval pipeline + decoding:

- **Baseline-S0 (Step-0 Base Model)**: `openai/gpt-oss-20b` with no LoRA, evaluated once per attempt to guard against silent eval/pipeline drift. Stored as `eval/baseline_s0_ood1000.json`.
- **Baseline-SOTA (Pinned Best Checkpoint)**: current best checkpoint (initially `A30_step130`), pinned by path. Update this pin **only** when a new attempt passes STOP-1 Gate-1b + Gate-2. Stored as `eval/baseline_sota_ood1000.json`.

#### 1) Constants and Statistical Reporting

**Primary metric**: OOD-1000 greedy_pass@1 (temp=0, n=1)
- Dataset: `data/probe_set_1000_ood.jsonl` (SHA256: `5700782fa9a6b5f0...`)
- Parsing: `reward_func_em.py` (boxed → last_number fallback → integer normalization)
- SE ≈ 1.5% at p≈0.65, 95% CI ≈ ±3.0pp

**Secondary**: ID-200 greedy_pass@1 (stability/regression check)

**Sanity only**: AIME-18 (SE≈11%, do NOT use for decisions)

**Canonical reporting (required for every Primary comparison)**:
- `Δ = acc(new) - acc(baseline)` (percentage points)
- Discordant counts: `b = # (baseline correct, new wrong)`, `c = # (baseline wrong, new correct)`
- McNemar exact p-value on (b, c)
- Recommended: paired bootstrap 95% CI for Δ (resample problems with replacement)
- All computed from paired per-problem record: `eval/paired_records_{baseline}_{new}_ood1000.jsonl`
  - Format: one line per problem: `{problem_hash, y_baseline, y_new, output_baseline, output_new}`

#### 2) STOP-1: Performance Win (staged gates)

**Gate-1a (Exploration gate: "worth confirming")**:
- OOD-1000 Δ ≥ +2.0pp AND one of:
  - McNemar exact p < 0.10, OR
  - Paired bootstrap CI lower bound > -0.5pp
- If Gate-1a passes → launch confirmation run with different seed
- Rationale: prevents burning a second seed on clearly-flat runs

**Gate-1b (Win gate: "claimable effect")**:
- OOD-1000 Δ ≥ +3.0pp AND McNemar exact p < 0.05
- Bootstrap CI lower bound > 0 (if computed)

**Gate-2 (Replication)**:
- Rerun with different seed (same pool SHA256, same config)
- STOP-1 passes only if:
  - Both seeds have Δ > 0
  - Mean Δ across seeds ≥ +3.0pp (or ≥ +4.0pp for "gold win")
- ID regression guardrail:
  - ID-200 must not regress by > -3.0pp vs Baseline-S0
  - Exception: if OOD-1000 mean Δ ≥ +6.0pp, allow up to -5.0pp ID regression (must be explicitly reported)

If STOP-1 passes: freeze attempt as release candidate, start writeup.

#### 3) Checkpoint Selection (K=3 fixed, no p-hacking)

Evaluate exactly **K=3 checkpoints** per run on OOD-1000:
1. **Final** (last step)
2. **Mid** (predefined, e.g., step 100 for 200-step run or step 200 for 400-step run)
3. **Best-by-monitor** (best smoothed OOD-202 during training, 3-point moving average)

Always report all K results. Do not evaluate 20 checkpoints and pick the max. If more resolution is needed, pre-register the additional checkpoints before looking at Primary results and report K.

#### 4) Early Stopping Within a Run

Training-time OOD-202 is for monitoring/early stop only, not publishable claims.

- Use smoothed OOD-202 (3-eval moving average)
- **Patience**: 5 eval points (e.g., if eval_steps=10 → patience ≈ 50 global steps)
- **Minimum improvement threshold**: +1.0pp over best-so-far smoothed OOD-202
- If no improvement ≥ threshold within patience → stop and proceed to post-hoc OOD-1000 eval on K=3 checkpoints
- Rationale: OOD-202 SE≈3.4%, so +0.5pp is within single-problem noise. +1.0pp (≈2 problems) with smoothing is a better balance.

#### 5) STOP-3: Diminishing Returns (two flavors)

**STOP-3a (Same-pool stagnation)**:
- 2 consecutive single-variable attempts with **same pool SHA256** show:
  - OOD-1000 Δ < +1.0pp AND CI includes 0 (or McNemar not significant)
- Use a 3rd run only if results are in the gray zone (+1pp to +3pp, ambiguous CI)

**STOP-3b (Pool-variance stagnation)**:
- For experiments that necessarily change the pool (e.g., pool size 3200→6400):
  - Under the same recipe (same hyperparams, only pool subsample differs), 2 different pool subsamples (different SHA256, same size and selection method) both show:
    - OOD-1000 Δ < +1.0pp AND CI includes 0
- Rationale: distinguishes "no effect" from "pool sensitivity"; prevents infinite running where STOP-3a never triggers due to forced pool changes

When STOP-3 triggers (either a or b) → enter Debug Workflow (fault tree).

#### 6) Pipeline Integrity Preflight

**Mandatory** if an attempt changes anything in: eval path, parsing, vLLM weight-sync, LoRA application, decoding, or metrics serialization.

Before running the full attempt:
1. Pick one checkpoint (or step-0 base) and one dataset (OOD-202 is fine)
2. Run in-training eval and post-hoc eval with identical settings
3. Requirement: results must match exactly (or differ only by an explicitly documented nondeterminism source)

This is separate from the parser ceiling check and prevents pipeline inflation/deflation bugs (cf. §11.22 LoRA accumulation bug).

#### 7) Parser Ceiling Check

After each run, before entering the fault tree:
1. Sample 50 "incorrect" OOD-1000 cases from the best checkpoint
2. Manually verify: answer extraction correctness (boxed/last-number), normalization (fractions, negatives, units)
3. If mis-parse rate > 5% → fix parser/format first, re-evaluate, before attributing lack of gains to the model
4. Track parse_method distribution (boxed vs last_number vs none) per run

#### 8) Debug Workflow: Fault Tree

When STOP-3 triggers, classify the bottleneck and choose the next single-variable intervention.

**A) Reward Sparsity / Credit Assignment**

*Symptoms*: Average reward flat; pass@1 doesn't move; "fragile" behavior.

*Diagnostics*:
- For a fixed sample of training prompts, measure pass@1 vs pass@8-any-correct
- If pass@8 >> pass@1 → reliability problem, sparse positive signal

*Interventions* (choose one):
- Increase n_samples_per_prompt (8 → 16) to increase positive-sample density
- Add lightweight shaping bonuses (e.g., +0.05 for boxed_in_final) while keeping incorrect=0
- Last resort: judge-based quality shaping (higher complexity/cost)

**B) LoRA Capacity**

*Symptoms*: OOD and ID flat despite stable training dynamics (spikes controlled, ratio tails stable).

*Diagnostics*:
- Compare rank=32 across multiple runs; check if learning signals plateau early
- Check if LoRA weight norms are saturating

*Interventions*:
- Increase LoRA rank (32 → 64, with alpha=128 to maintain ratio)
- Later: expand target modules beyond attention-only

**C) Data Distribution / Difficulty Mismatch**

*Symptoms*: Training correctness improves but OOD-1000 does not (or worsens).

*Diagnostics*:
- Bucket OOD-1000 by source/topic and report per-bucket accuracy
- Compare to training pool distribution (coverage gap)

*Interventions*:
- Adjust training pool toward OOD distribution (more hard/targeted categories)
- Implement curriculum or difficulty-based sampling

**D) RL Dynamics / Schedule**

*Symptoms*: Ratio tails shrink to near-1 quickly; learning stalls in later steps.

*Diagnostics*:
- Track LR over global steps; confirm effective LR window overlaps training horizon
- Track ratio_p99 trend, KL trend, update magnitudes
- Check if peak performance step correlates with LR level

*Interventions*:
- Increase min_lr (0.1×target → 0.2×target) — directly addresses A30 observation (peak at step 130/200)
- Adjust KL coefficient (init_kl_coef) — separate A/B

#### 9) Deliverables Per Attempt (one-page summary)

1. Recipe diff (single variable changed)
2. Training pool SHA256 + probe set SHA256
3. Decoding settings (temp, n, max_tokens)
4. Primary (OOD-1000) baselines + new checkpoint metrics (K=3)
5. Paired stats: Δ, b/c, McNemar p-value (+ bootstrap CI if available)
6. Stability metrics: spike count, max grad_norm, ratio tail indicators
7. Parser ceiling check (50-case manual review)
8. Decision: **STOP-1** (Gate-1a/1b/2 status) / **Continue** / **STOP-3a/3b** / **DEBUG** (selected bottleneck class)

#### 10) Operational Rules

1. **One variable per run.** If you change pool size, do not change LR/KL/rank.
2. **Same eval pipeline.** OOD-1000 post-hoc eval selects checkpoints; OOD-202 is in-run monitoring only.
3. **Full provenance.** Git SHA, all dataset SHA256s, seeds, exact commands — in every run summary.
4. **No data leakage.** Verify disjoint-by-hash (pool vs OOD-1000, pool vs ID-200) before every run.
5. **Save model outputs.** Full generation text for all OOD-1000 problems (not just correct/incorrect) for post-hoc analysis.
6. **Pipeline preflight.** After any eval/pipeline change, run regression check before full attempt.

### 11.25 OOD-1000 Baseline Eval (Establishing Baseline-SOTA)

**Purpose**: Evaluate existing checkpoints on OOD-1000 to establish Baseline-SOTA per SOP v2.1-final §0.

**Checkpoints evaluated** (all already merged at `/mnt/scratch/merged_models/`):
- `a29b_step200`: A29B final (micro=1, LoRA bug during training, merge is correct)
- `a30_step130`: A30 best-by-inrun-OOD202 (micro=2, fixed pipeline)
- `a30_step200`: A30 final (micro=2, fixed pipeline)

**Eval settings**: greedy (temp=0, n=1), max_tokens=4096, vLLM TP=2, same parsing as `reward_func_em.py`.

**Results**:

| Model | OOD-1000 | ID-200 | AIME-18 |
|-------|----------|--------|---------|
| a29b_step200 | 73.70% (737/1000) | 51.00% (102/200) | 38.89% (7/18) |
| a30_step130 | 73.50% (735/1000) | 52.50% (105/200) | 38.89% (7/18) |
| a30_step200 | 73.60% (736/1000) | 50.00% (100/200) | 44.44% (8/18) |

**Analysis**:
- **All three checkpoints are statistically indistinguishable on OOD-1000.** Max pairwise Δ = 0.2pp (2 problems). SE ≈ 1.5%, so the 95% CI (±3.0pp) is much larger than any observed difference.
- The earlier OOD-202 result (a30_step130 = 65.35% vs a29b_step200 = 62.87%, Δ = +2.5pp) was **noise** — confirmed by OOD-1000 showing Δ = -0.2pp. This validates the decision to build OOD-1000: the smaller probe was giving misleading signals.
- ID-200 is similarly flat (range 50.0%–52.5%, within SE).
- AIME-18 shows a30_step200 at 44.4% vs 38.9%, but this is 1 problem difference (N=18, SE=11%) — pure noise.

**Baseline-SOTA pin**: **a30_step130** at 73.50% OOD-1000, 52.50% ID-200.
- Rationale: from the fixed pipeline (no LoRA accumulation bug), micro=2 default, and the best-by-monitor checkpoint
- All future runs compare against this baseline

**Baseline-S0**: Evaluated in §11.26 below — **RL has zero measurable effect**.

**Provenance**:
- OOD-1000: `data/probe_set_1000_ood.jsonl` SHA256: `5700782fa9a6b5f0e01ce60f0e239d25ec0e8a49954f656d4cb9bd37c42f264b`
- ID-200: `data/probe_set_200.jsonl` SHA256: `cee48574208d948163401a9421bb01963726ca0680d810dd2acf69e819b16384`
- AIME-18: `data/aime_eval.jsonl` SHA256: `856a54bf509acf5824893ac0929facf51408cc7cd40b88c12e9d620e1cfd55b0`
- Full results: `/mnt/scratch/posthoc_eval_ood1000_baseline.json`

### 11.26 Baseline-S0 vs Baseline-SOTA: Does RL Help At All?

**Purpose**: Answer the fundamental question — does RL training produce any measurable improvement over the raw base model? This is a prerequisite before investing in further recipe tuning (A31+).

**Method**: Evaluated both models on OOD-1000, ID-200, and AIME-18 using identical settings (greedy, temp=0, n=1, max_tokens=4096, same parser). Per-problem results saved with content-hash keys for paired comparison. Script: `eval_baseline_s0.py`.

- **Baseline-S0**: `openai/gpt-oss-20b` — raw base model, no LoRA adapter
- **Baseline-SOTA**: `/mnt/scratch/merged_models/a30_step130` — best RL checkpoint (A30, step 130, merged)

**Results**:

| Model | OOD-1000 | ID-200 | AIME-18 |
|-------|----------|--------|---------|
| Baseline-S0 (no LoRA) | 73.80% (738/1000) | 51.50% (103/200) | 33.33% (6/18) |
| Baseline-SOTA (A30 s130) | 73.40% (734/1000) | 51.00% (102/200) | 27.78% (5/18) |
| Δ (SOTA − S0) | **−0.40pp** | −0.50pp | −5.56pp |

**Paired Statistics (OOD-1000)**:
- N (paired problems): 1000
- Discordants: b=61 (S0 correct, SOTA wrong), c=57 (S0 wrong, SOTA correct)
- Net improvement: c − b = −4 problems
- McNemar exact p-value: **0.783** (far from significant)
- Bootstrap 95% CI for Δ: [−2.50pp, +1.70pp]
- SOP Gate check: **Neither Gate-1a nor Gate-1b passes** (Δ is negative)

**Paired Statistics (ID-200)**:
- N (paired problems): 200
- Discordants: b=11 (S0 correct, SOTA wrong), c=10 (S0 wrong, SOTA correct)
- Net improvement: c − b = −1 problem
- McNemar exact p-value: **1.000**
- Bootstrap 95% CI for Δ: [−5.00pp, +4.00pp]

**Diagnostic Analysis**:

*Bucket breakdown (OOD-1000)*:
- MATH45 subset (N=968): Base=76.2%, RL=75.8%, Δ=−0.4pp — same story
- Competition subset (N=32): **0% for both models** — neither model solves any competition problem
- The competition problems are too hard for this model; all signal is in MATH45

*Parse-method distribution*:
- Both models: ~71% boxed, ~12% last_number, ~17% fail
- No meaningful difference in parsing behavior between base and RL

*50-sample parser audit* (all 118 discordant cases examined):
- 10/118 (8.5%) confirmed parser errors — all from last_number fallback grabbing trailing incidental numbers when the model used `**bold**` instead of `\boxed{}`
- 8/118 (6.8%) ambiguous
- 100/118 (84.7%) genuine model differences
- **Parser errors are symmetric**: 5 hurt S0, 5 hurt SOTA → net impact on Δ is zero
- Root cause: models sometimes format answers as `**N**` instead of `\boxed{N}`, and the last-number fallback grabs a trailing digit from context (e.g., "base 7", "7 seats")
- Conclusion: parser noise does not bias the S0 vs SOTA comparison

**Interpretation**:

RL training (A29A through A30, ~200 gradient steps of DR-GRPO with exact-match reward) has produced **zero measurable improvement** on OOD-1000. The 61/57 discordant split is consistent with random noise (p=0.78). The RL model is neither better nor worse — the LoRA adapter is effectively doing nothing detectable on held-out problems.

This means all previous "improvements" observed in-training (reward curves going up, OOD-202 fluctuations) were either:
1. Overfitting to the training pool (without OOD generalization)
2. Noise in small eval sets (OOD-202 SE≈3.4%)
3. Some combination of both

**Implications per SOP fault tree** (§11.24 §8):

Before proceeding with A31+ recipe changes, we need to classify the bottleneck:

- **(A) Reward Sparsity / Signal**: With exact-match binary reward and greedy decoding already at 73.8%, the base model answers ~74% of problems correctly. The reward signal may be too sparse to drive learning — most rollouts either all get it right or all get it wrong, leaving few informative contrasts for GRPO advantages.

- **(B) LoRA Capacity**: rank=64 with only attention QKV projections. The adapter may not have enough capacity to modify the model's reasoning, or may need to target additional modules (MLP, output projections).

- **(C) Data Distribution**: The training pool may not contain enough problems in the difficulty range where the model can learn (currently wrong ~26% of the time on MATH45, and 100% wrong on competition).

- **(D) RL Dynamics**: Peak performance at step 130/200 suggests the LR schedule may overshoot. With KL coefficient = 0, there's no regularization pulling back toward the base model.

The most concerning signal is that the discordant count (b+c=118, or 12.2% of problems) is substantial — RL *does* change which problems the model gets right, but the net effect is zero. This suggests the RL signal exists but is directionless.

**Next steps** (ordered by diagnostic value):
1. **Pass@k headroom check**: Generate k=8 samples on problems where base model greedy is wrong (~262 problems). If pass@k is near 0%, the model genuinely cannot solve these and reward signal is absent. If pass@k is substantial, RL should be able to exploit it.
2. **Training reward curve deep-dive**: Check if training pool accuracy actually improved during RL (it should have). If it did, the issue is OOD generalization (overfitting). If it didn't, the issue is RL dynamics.
3. **Discordant analysis**: Examine the 118 discordant problems (61 regressions + 57 improvements) for patterns — are regressions concentrated in specific topics?

**Lesson #23**: Always evaluate the base model (Baseline-S0, no LoRA) before declaring RL success. In-training metrics (reward curves, in-run OOD monitors) can be misleading — they may reflect noise (small eval sets) or overfitting (training pool accuracy going up without OOD transfer). The only trustworthy signal is a paired comparison on a large held-out set with proper statistical tests. In this case, 30+ hours of RL training on gpt-oss-20b with DR-GRPO produced zero detectable improvement on OOD-1000 (Δ=−0.40pp, p=0.78).

**Provenance**:
- Script: `eval_baseline_s0.py`
- Full results: `/mnt/scratch/baseline_s0_eval/baseline_comparison.json`
- Per-problem results: `/mnt/scratch/baseline_s0_eval/s0_ood1000.jsonl`, `sota_ood1000.jsonl`
- Paired records: `/mnt/scratch/baseline_s0_eval/paired_records_s0_sota_ood1000.jsonl`

### 11.27 Strategic Pivot: Base Model Sweep for RL Headroom

**Motivation**: §11.26 established that RL training on gpt-oss-20b produces zero measurable improvement. The most likely explanation is **insufficient headroom**: the base model already scores 73.8% on OOD-1000 (greedy@1), leaving only ~26% of problems where RL could potentially improve performance. With binary EM reward, most rollouts either all succeed (reward=1, no gradient signal) or all fail (the model genuinely cannot solve the problem, so reward=0 with no useful contrast). The effective "learning zone" — problems the model sometimes gets right and sometimes wrong — may be too narrow for GRPO to exploit.

**Hypothesis**: A model with lower baseline accuracy (target: ~60–70% OOD-1000) would have more problems in the learning zone, providing richer reward signal for RL. If RL still shows zero improvement on such a model, the issue is elsewhere (reward function, RL algorithm, LoRA capacity). If RL works, we confirm that headroom was the bottleneck with gpt-oss-20b.

**gpt-oss-20b reference baseline** (from §11.25–11.26, no prompt suffix):

| Eval Set | Accuracy | Parse: boxed% | last_number% | fail% |
|----------|----------|---------------|-------------|-------|
| OOD-1000 | 73.80% (738/1000) | ~71% | ~12% | ~17% |
| ID-200 | 51.50% (103/200) | — | — | — |
| AIME-18 | 33.33% (6/18) | — | — | — |

Note: gpt-oss-20b was evaluated *without* the boxed instruction suffix. For apples-to-apples comparison with Instruct models, gpt-oss-20b will also be re-evaluated with the same suffix (OOD-1000 only) in the sweep.

**Candidate models** (first wave, Instruct variants, no SFT/RL):

| Model | HF ID | Params | Why |
|-------|-------|--------|-----|
| Llama 3.1 8B Instruct | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 8B | Smallest; likely lowest accuracy → most headroom |
| Qwen 2.5 14B Instruct | `Qwen/Qwen2.5-14B-Instruct` | 14B | Mid-range; Qwen2.5 known for strong math |
| Qwen 2.5 32B Instruct | `Qwen/Qwen2.5-32B-Instruct` | 32B | Largest; may be near gpt-oss-20b level |

**Selection criteria**:
1. Primary: OOD-1000 greedy@1 in ~60–70% range (more headroom than 73.8%)
2. Secondary: Higher boxed% and lower parse_fail% (less sparse RL reward signal)
3. Tie-breaker: Smaller model preferred (faster RL iteration, lower GPU cost)

**Eval protocol**:
- Same OOD-1000 / ID-200 / AIME-18 pipeline
- Greedy decoding (temp=0, n=1), max_tokens=4096, max_model_len=8192
- Prompt: problem text + concise boxed suffix (`"Please reason step by step but keep it concise, and put your final answer within \boxed{...}."`)
- Track per-problem: correct, parse_method, finish_reason, source bucket (MATH vs competition)
- Report: overall accuracy, MATH/competition bucket accuracy, parse breakdown, truncation rate
- Script: `eval_model_sweep.py`

**Expected outcome**: Identify the model with optimal headroom for EM-GRPO. Then run a quick RL experiment on that model to definitively answer "does RL help at all?" with a clean paired McNemar test.

### 11.28 Model Sweep Stage 1 Results: OOD-1000 (§11.27 execution)

**Date**: 2026-02-25

**Method**: Evaluated all 4 models on OOD-1000 (1000 problems) with identical settings: greedy decoding (temp=0, n=1), max_tokens=4096, max_model_len=8192, strengthened BOXED_SUFFIX prompt. Script: `eval_model_sweep.py --stage 1`.

**BOXED_SUFFIX** (applied to all models):
> "Please reason step by step but keep it concise, and put your final answer within \boxed{...}. In the final line, output ONLY \boxed{<integer>} and nothing else."

**Results**:

| Model | OOD-1000 | MATH(968) | Comp(32) | Boxed% | LN% | Fail% | Trunc% |
|-------|----------|-----------|----------|--------|-----|-------|--------|
| gpt-oss-20b | 83.50% (835/1000) | 86.26% | 0.00% | 86.2 | 0.0 | 13.8 | 14.2 |
| llama3.1-8b | 49.00% (490/1000) | 50.62% | 0.00% | 77.5 | 22.5 | 0.0 | 28.8 |
| **qwen2.5-14b** | **67.00% (670/1000)** | 69.11% | 3.13% | 99.2 | 0.8 | 0.0 | 30.8 |
| qwen2.5-32b | 69.30% (693/1000) | 71.49% | 3.13% | 99.2 | 0.8 | 0.0 | 15.6 |

**Key findings**:

1. **BOXED_SUFFIX adds ~10pp to gpt-oss-20b**: 83.5% with suffix vs 73.8% without (§11.26). This is a prompt-engineering gain, not a model capability gain. The suffix forces structured output, reducing parse failures from ~17% to ~14% and improving answer extraction. This also means the true headroom gap on gpt-oss-20b was even smaller than §11.26 suggested — with the same prompt that RL models would use, the base model is at 83.5%, not 73.8%.

2. **Both Qwen models land in the 60–70% target range**: qwen2.5-14b at 67.0% and qwen2.5-32b at 69.3%. These provide 30–33% of problems in the "model gets it wrong" zone, compared to only 16.5% for gpt-oss-20b with BOXED_SUFFIX.

3. **Qwen models have near-perfect parsing**: 99.2% boxed rate, 0% parse failures. This means the EM reward signal will be clean — almost no problems lost to formatting noise. Contrast with gpt-oss-20b (13.8% parse failures) and llama3.1-8b (22.5% last-number fallback).

4. **llama3.1-8b is too weak**: At 49.0%, over half the problems are wrong. While this gives maximum headroom, the model may lack the capability to solve these problems even with RL — too far below the "learning zone." Also has 22.5% last-number fallback and 28.8% truncation.

5. **Competition problems remain near-zero for all models**: 0/32 for gpt-oss-20b and llama, 1/32 for both Qwen models. These problems are genuinely hard and unlikely to yield RL signal.

6. **Truncation concern for qwen2.5-14b**: 30.8% truncation rate is high (vs 15.6% for qwen2.5-32b). The 14b model generates longer reasoning chains that hit the 4096-token max_tokens limit. This could suppress answer quality — some correct solutions may get cut off before reaching `\boxed{}`. For RL training, may need to increase max_tokens or the model may naturally learn shorter outputs under RL pressure.

**Selection per §11.27 criteria**:

| Criterion | qwen2.5-14b | qwen2.5-32b | Winner |
|-----------|-------------|-------------|--------|
| OOD-1000 in 60–70% | 67.0% ✓ | 69.3% ✓ | 14b (more headroom) |
| Boxed% | 99.2% | 99.2% | Tie |
| Fail% | 0.0% | 0.0% | Tie |
| Model size | 14B | 32B | 14b (cheaper RL) |
| Truncation | 30.8% ⚠ | 15.6% | 32b |

**Recommendation**: **qwen2.5-14b** is the primary candidate. It has 3.3pp more headroom than qwen2.5-32b, half the parameter count (faster RL iterations, lower GPU cost), and identical parsing quality. The truncation concern (30.8%) is worth monitoring but may resolve naturally under RL: if the model learns to output shorter, more focused reasoning, truncation should decrease. If truncation proves problematic, qwen2.5-32b is the backup.

**Lesson #24**: The BOXED_SUFFIX prompt is a confound — it adds ~10pp to gpt-oss-20b accuracy. Always use identical prompts when comparing models, and use the same prompt for both baseline evaluation and RL training. The "no-suffix" baseline from §11.25–11.26 was comparing apples to oranges with the RL models that were trained with a suffix.

**Provenance**:
- Script: `eval_model_sweep.py`
- Full results: `/mnt/scratch/model_sweep/sweep_stage1_summary.json`
- Per-problem results: `/mnt/scratch/model_sweep/{model}_ood1000.jsonl`

### 11.29 Model Sweep Stage 2: ID-200 + AIME-18 + Truncation Sensitivity

**Date**: 2026-02-25

#### Stage 2 Results (ID-200 + AIME-18)

**Method**: Evaluated qwen2.5-14b and qwen2.5-32b on ID-200 and AIME-18 with identical settings to Stage 1 (greedy, temp=0, max_tokens=4096, BOXED_SUFFIX). Script: `eval_model_sweep.py --stage 2 --models qwen2.5-14b qwen2.5-32b`.

**Results (combined with Stage 1 OOD-1000)**:

| Model | OOD-1000 | ID-200 | AIME-18 | Boxed% (all sets) |
|-------|----------|--------|---------|-------------------|
| **qwen2.5-14b** | **67.0%** (670/1000) | **69.5%** (139/200) | 16.7% (3/18) | 99–100% |
| qwen2.5-32b | 69.3% (693/1000) | 72.5% (145/200) | 16.7% (3/18) | 99–100% |

**Observations**:

1. **ID-200 tracks OOD-1000**: qwen2.5-14b at 69.5% and qwen2.5-32b at 72.5% — the ~2.5pp gap between models is consistent across both eval sets. ID-200 is a viable secondary guardrail for detecting overfitting vs OOD generalization during RL.

2. **AIME-18 is identical**: Both models solve exactly 3/18 AIME problems (16.7%). As expected, AIME is sanity-only — too small (N=18) for statistical discrimination and too hard for either model to score meaningfully.

3. **Parsing remains perfect**: 100% boxed rate on both ID-200 and AIME-18 for both Qwen models. Zero parse failures across all eval sets.

#### Truncation Sensitivity Check (qwen2.5-14b)

**Method**: Evaluated qwen2.5-14b on the first 200 OOD-1000 problems with max_tokens=4096 vs max_tokens=2048 (same BOXED_SUFFIX, same server instance). Script: `eval_trunc_sensitivity.py`.

**Results**:

| Setting | Accuracy | Boxed% | Trunc% (finish_reason=length) |
|---------|----------|--------|-------------------------------|
| max_tokens=4096 | 57.0% (114/200) | 98.0% | 2.0% |
| max_tokens=2048 | 60.0% (120/200) | 98.5% | 1.5% |
| **Δ (2048 − 4096)** | **+3.0pp** | +0.5 | −0.5 |

**Paired analysis (N=200)**:
- Both correct: 108
- Only 4096 correct: 6
- Only 2048 correct: 12
- Both wrong: 74

**Key insight**: The 30.8% "truncation rate" reported in Stage 1 (§11.28) was from the heuristic `is_truncated()` function (long output + no boxed OR ends mid-sentence), **not** from actually hitting the token limit. Only 2.0% of responses actually have `finish_reason=length`. The heuristic was misleadingly high.

More importantly, **max_tokens=2048 is not just viable — it's slightly better** (+3.0pp, with 12 vs 6 discordants favoring 2048). The model doesn't productively use the extra 2048 tokens. Shorter max_tokens forces more concise reasoning without sacrificing accuracy.

**Decision**: Standardize **max_tokens=2048** as the canonical setting for qwen2.5-14b going forward. This halves per-rollout token generation during RL training, directly reducing compute cost.

#### Final Selection

**Primary RL candidate: qwen2.5-14b** (confirmed)

| Property | Value |
|----------|-------|
| OOD-1000 baseline | 67.0% (330 wrong → learning zone) |
| ID-200 baseline | 69.5% (secondary guardrail) |
| AIME-18 baseline | 16.7% (sanity only) |
| Boxed parse rate | 99–100% (clean EM reward) |
| Parse failures | 0% |
| Canonical max_tokens | 2048 |
| TP requirement | 2 GPUs |
| Backup model | qwen2.5-32b (69.3% OOD, 72.5% ID) |

**Next steps**:
1. Set up RL training pipeline for qwen2.5-14b with EM-GRPO (max_tokens=2048)
2. Run a single clean RL experiment (100–200 steps)
3. Evaluate with paired McNemar test on OOD-1000 (paired records + CI)
4. Only run confirmation seed if Gate-1a/1b looks promising

**Lesson #25**: Measure truncation by `finish_reason`, not by heuristic response-length checks. The heuristic `is_truncated()` (no boxed + long output) conflates "model wrote a long response without boxing" with "model was cut off by token limit." In this case, the heuristic reported 30.8% truncation while actual token-limit truncation was only 2.0%. Halving max_tokens from 4096 to 2048 had no negative effect on accuracy.

**Provenance**:
- Stage 2: `/mnt/scratch/model_sweep/sweep_stage2_summary.json`, `{model}_{id200,aime}.jsonl`
- Truncation: `/mnt/scratch/model_sweep/trunc_sensitivity/trunc_sensitivity_summary.json`
- Scripts: `eval_model_sweep.py`, `eval_trunc_sensitivity.py`

### 11.30 Attempt Q1 Design: qwen2.5-14b EM-GRPO (RL-vs-noRL Test)

**Date**: 2026-02-25

**Goal**: Definitively answer "does RL (DR-GRPO with EM reward) produce measurable OOD improvement?" using a model with sufficient headroom. This is the clean paired test that gpt-oss-20b could not provide (§11.26: zero effect at 73.8% baseline).

#### Model & Baseline

| Property | Value |
|----------|-------|
| Model | `Qwen/Qwen2.5-14B-Instruct` (14B dense, bf16) |
| OOD-1000 baseline | 67.0% (670/1000) — 33% error zone |
| ID-200 baseline | 69.5% (139/200) |
| AIME-18 baseline | 16.7% (3/18) |
| Boxed parse rate | 99–100% |
| Canonical max_tokens | 2048 (§11.29 truncation study) |

#### Training Configuration (Q1 vs A30)

| Parameter | A30 (gpt-oss-20b) | Q1 (qwen2.5-14b) | Rationale |
|-----------|-------------------|-------------------|-----------|
| `--pretrain` | `openai/gpt-oss-20b` | `Qwen/Qwen2.5-14B-Instruct` | New model |
| `--generate_max_len` | 4096 | 2048 | §11.29: no accuracy loss |
| `--max_len` | 5120 | 3072 | prompt(1024) + gen(2048) |
| `--prompt_data` | `sft_rl_pool_3200.jsonl` | `sft_rl_pool_3200_boxed.jsonl` | BOXED_SUFFIX appended |
| `--eval_dataset` | `probe_set_200_ood.jsonl` | `probe_set_200_ood_boxed.jsonl` | BOXED_SUFFIX appended |
| `--vllm_num_engines` | 2 | 4 | TP=1 → 4 engines |
| `--vllm_tensor_parallel_size` | 2 | 1 | 14B bf16 (~28GB) fits 1 GPU |

**Unchanged** (proven across 30 experiments):
- Algorithm: `dr_grpo`, `eps_clip=0.1`, `init_kl_coef=0.001`
- LoRA: `rank=32`, `alpha=64`, targets `q_proj k_proj v_proj o_proj` (Qwen2.5 uses same names)
- Batch: `rollout_batch_size=16`, `train_batch_size=16`, `micro_train_batch_size=2`
- Sampling: `n_samples_per_prompt=8`
- LR: `5e-7` cosine decay, `warmup_ratio=0.05`, `min_lr=5e-8`
- `--colocate_actor_ref`, `--ref_num_nodes 1`, `--ref_num_gpus_per_node 4`
- 200 global steps (1600 gradient steps), save/eval every 10
- Seed: 42

**GPU allocation (8x H100 80GB)**:
```
GPUs 0-3: vLLM rollout — 4 engines × TP=1 (14B bf16 ~28GB + KV cache, fits)
GPUs 4-7: Actor training (ZeRO-3) + co-located ref model
```

**Output paths**:
```
SAVE_PATH="/mnt/data/rft_output/qwen2.5-14b-grpo-q01"
CKPT_PATH="/mnt/data/rft_checkpoints/qwen2.5-14b-grpo-q01"
METRICS_LOG_DIR="/mnt/scratch/rft_metrics_qwen14b_q01"
SAMPLES_LOG_DIR="/mnt/scratch/rft_samples_qwen14b_q01"
```

**Step count** (verified by preflight_lr.sh):
```
gradient_steps = 3200 * 8 / 16 * 1 = 1600
grad_per_global = 8 * 16 / 16 = 8
global_steps = 1600 / 8 = 200
warmup: 80 gradient steps (~10 global steps)
```

#### Guardrails Checklist

Six guardrails (G0–G5 + G6) organized into three execution phases:

**Phase A — Pre-training (must pass before training starts)**:

| Gate | Name | Method | PASS Criteria |
|------|------|--------|---------------|
| G0 | Prompt/Chat-Template Parity | Tokenize 5 samples via train path (HF `apply_chat_template`) vs eval path (vLLM chat API). Compare `input_ids`. | All 5 samples: exact `input_ids` match |
| G2 | Parser/Reward Ceiling | Run baseline qwen14b on 200 OOD with BOXED_SUFFIX through chat template | boxed ≥ 98%, fail = 0%, length ≤ 5% |
| G4 | Pass@k Headroom | Sample k=8 (temp=0.6) on ~330 problems where greedy@1 is wrong | pass@8 ≥ 15% (GRPO has signal) |

**Phase B — Smoke run (first 10 global steps)**:

| Gate | Name | Method | PASS Criteria |
|------|------|--------|---------------|
| G3 | vLLM Memory/Throughput | Watch for OOM, engine restarts, step time blow-up | No crashes, stable step time |
| G5 | Runtime Assertions | Check metrics at step 10: LR, reward mean, grad_norm | LR ≈ 100% target; reward neither ~0 nor ~1 |
| G1 | Pipeline Integrity | Compare in-training eval vs post-hoc eval on same step-10 checkpoint + OOD-202 | 0 discordants (exact match) |

**Phase C — Post-training evaluation**:

| Gate | Name | Method | PASS Criteria |
|------|------|--------|---------------|
| G6 | Paired McNemar Test | Eval K=3 checkpoints (step 100, 200, best-by-monitor) on OOD-1000. Paired comparison vs noRL baseline. | Gate-1b: Δ≥+3pp, p<0.05 → win. Gate-1a: Δ≥+2pp, p<0.10 → second seed. |

#### Execution Order

```
Phase A:
  1. prepare_rl_data_qwen.py → sft_rl_pool_3200_boxed.jsonl + probe_set_200_ood_boxed.jsonl (with SHA256)
  2. verify_prompt_parity.py → G0 PASS/FAIL
  3. eval_model_sweep.py --limit 200 → G2 PASS/FAIL
  4. eval_passk_headroom.py → G4 go/no-go

Phase B:
  5. apply_patches.sh
  6. preflight_lr.sh
  7. train_grpo_qwen14b_q01.sh (first 10 steps) → G3 + G5
  8. Pipeline integrity check (step 10 checkpoint) → G1

Phase C:
  9. Full training (200 steps)
  10. Post-hoc eval (K=3 checkpoints, OOD-1000) → G6
```

#### Recovery After Crash

If machine crashes during training:
1. Check latest checkpoint in `/mnt/data/rft_checkpoints/qwen2.5-14b-grpo-q01/`
2. Resume with `bash train_grpo_qwen14b_q01.sh --load_checkpoint <path>`
3. If no checkpoint survived: re-run from Phase B step 5
4. If crash during Phase A: re-run from step 1 (data files in `data/` are git-tracked)

#### Scripts Created for Q1

| Script | Purpose |
|--------|---------|
| `prepare_rl_data_qwen.py` | Append BOXED_SUFFIX to training pool + eval set, compute SHA256 |
| `verify_prompt_parity.py` | G0: tokenized prompt parity between train and eval paths |
| `eval_passk_headroom.py` | G4: pass@8 on wrong problems (go/no-go gate) |
| `train_grpo_qwen14b_q01.sh` | Training launch script (adapted from A30) |

---

## 12. Consolidated Lessons

All lessons from §11 in one numbered list, grouped by category, with back-references to the originating section.

### Infrastructure

**#7** (§11.11, A26): MXFP4 MoE models need dtype casting before ZeRO-3. MXFP4 dequantization leaves router/gate parameters as float32 while other parameters become bf16. ZeRO-3's all-gather requires uniform dtype. Cast all parameters to bf16 before `deepspeed.zero.Init()`.

**#8** (§11.11, A26): Colocate actor and ref model when GPU budget is tight. With `--colocate_actor_ref`, both share the same GPUs. The ref model only does forward passes (no gradients), so memory overhead is minimal with ZeRO-3.

**#17** (§11.19, A30): When implementing LoRA weight sync to a separate inference engine, distinguish between "full delta" (the complete LoRA contribution `scaling * B @ A`) and "incremental delta" (the change from the previous step). If sending the full delta, the receiver must subtract the old delta before adding the new one, or cache the base weights and reconstruct from scratch.

**#19** (§11.20, A30): Doubling `micro_train_batch_size` from 1→2 does not cause OOM and actually reduces training time by ~18% (fewer micro-batch forward/backward passes, better GPU utilization). The previous cache flush warnings with micro=1 disappear.

**#20** (§11.22, A30): After fixing a pipeline bug, always run a quantitative regression check (same checkpoint, both eval paths, identical settings) to confirm the fix works in the actual training pipeline — not just in isolation. The unit test validated a single layer; the regression check validates the full end-to-end pipeline.

### RL Training Dynamics

**#1** (§11.11, A26): Grad norm spikes come from ratio outliers, not advantage outliers. Advantage clipping won't fix spikes — tighter `eps_clip` or per-micro-batch gradient clipping are the correct interventions.

**#2** (§11.11, A26): Spikes escalate over training but are individually harmless. The model recovers within 1 step after each spike (post-clipping). However, the escalation correlates with declining eval performance — suggesting even clipped spikes inject noise that accumulates in LoRA parameters.

**#5** (§11.11, A26): Pre-clipping `grad_norm` is the metric to log, not post-clipping. Large values don't mean the model is damaged — they mean clipping is working. But the magnitude trend over training tells you whether the policy is diverging.

**#6** (§11.11, A26): Step-to-step correctness is too noisy for trend detection. With batch size 16 and 8 samples per prompt, each step evaluates ~128 samples from a random subset. Use 10-step rolling averages or a fixed probe set for reliable progress tracking.

**#11** (§11.11, A26): KL ≈ 0 is expected with small `init_kl_coef` and early training. With `init_kl_coef=0.001`, the KL penalty is negligible. If KL stays near zero after 50+ steps, the LR may be too low.

**#12** (§11.11, A26): The training horizon is shorter than you think. With LoRA rank=32 and 50K training problems, the best checkpoint was at step 30 (~3 episodes). Over-training LoRA adapters is easy — they have limited capacity and quickly memorize the training distribution.

**#15** (§11.16, A29): `eps_clip=0.1` (half the OpenRLHF default of 0.2) eliminates late-run gradient spikes without sacrificing generalization. For LoRA GRPO with binary reward and small batch sizes (`micro_train_batch_size=1`), tighter clipping is advisable because individual extreme-ratio tokens dominate the per-micro-batch gradient.

### Evaluation & Metrics

**#3** (§11.11, A26): Train/eval divergence is the key signal to stop. Training correctness improved steadily while held-out pass@1 degraded below pre-training baseline. An ID-only probe set would miss this. Use separate ID and OOD probes.

**#4** (§11.11, A26): Separate ID and OOD probe sets for different signals. ID probe (from training pool) measures learning/stability for early stopping. OOD probe (disjoint from pool) measures generalization for checkpoint selection. Tiny held-out set (AIME, N=18) is sanity check only.

**#9** (§11.11, A26): Eval config matters — clearly label your metrics. Three distinct modes: `greedy_pass@1` (temp=0, n=1), `sampling_pass@8` (temp>0, n=8), and `nondet_proxy_pass@8` (temp=0, n=8). Never conflate them.

**#10** (§11.11, A26): `has_boxed` is a format compliance proxy, not a reward signal. Track it to detect if the model is losing `\boxed{}` ability, but don't over-optimize for it. The real signal is held-out eval accuracy.

**#16** (§11.17, A29): Post-hoc eval on N=200 probe sets has SE≈3.4%, giving 95% CI of ±6.7pp. Distinguishing checkpoints within ~5pp requires larger eval sets or multiple-seed evaluation. In-training OOD eval (many evaluation points, averaged) can provide more signal.

**#18** (§11.19, A30): Always validate in-training eval against an independent post-hoc eval pipeline on a subset of checkpoints. Discrepancies exceeding expected noise (SE ≈ sqrt(p(1-p)/N)) indicate a pipeline bug.

**#21** (§11.23, A30): When comparing hyperparameter changes, post-hoc eval on multiple checkpoints (best + final) with identical settings is essential. In-training eval curves are noisy (±3pp step-to-step). Compare best-vs-best and final-vs-final to separate the hyperparameter effect from checkpoint selection noise.

**#23** (§11.26, A30): Always evaluate the base model (Baseline-S0, no LoRA) before declaring RL success. In-training metrics can be misleading — they may reflect noise or overfitting. The only trustworthy signal is a paired comparison on a large held-out set with proper statistical tests. In this case, 30+ hours of RL on gpt-oss-20b produced zero detectable improvement (Δ=−0.40pp, p=0.78).

**#24** (§11.28, model sweep): The BOXED_SUFFIX prompt is a confound — it adds ~10pp to gpt-oss-20b accuracy. Always use identical prompts when comparing models, and use the same prompt for both baseline evaluation and RL training.

**#25** (§11.29, model sweep): Measure truncation by `finish_reason`, not by heuristic response-length checks. The heuristic conflates "model wrote a long response without boxing" with "model was cut off by token limit." Halving max_tokens from 4096 to 2048 had no negative effect on qwen2.5-14b accuracy.

### Step Counting & Configuration

**#13** (§11.11, A26–A27): Understand `num_episodes` before launching. In OpenRLHF, `total_steps = len(prompts) * n_samples_per_prompt // train_batch_size * num_episodes * max_epochs`. With 50K prompts and num_episodes=5, this yields 125,000 steps — not ~40. Always compute `max_steps` before launching and verify the LR schedule reaches meaningful values within your intended step budget.

**#14** (§11.11, A28): `max_steps` counts gradient steps, not global steps. Each global step contains `n_samples_per_prompt * rollout_batch_size / train_batch_size` gradient steps. For N global steps, you need `pool_size = N * rollout_batch_size` with `num_episodes=1`.
