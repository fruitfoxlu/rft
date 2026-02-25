# Training Guidelines: RL Fine-Tuning with GRPO

> Reference document for running RL fine-tuning on this system.
> Framework: OpenRLHF 0.9.3 + Ray + DeepSpeed ZeRO-3 + vLLM 0.15.1.
> Hardware: 8x H100 80GB GPUs.

---

## 1. Code Management

All project code lives in a single repository:

```
/home/rlu/Code/rft/
```

**Key files and their purposes:**

| File | Purpose |
|------|---------|
| `train_grpo_20b.sh` | Training launch script — Attempt 26 config |
| `train_grpo_20b_a27.sh` | Training launch script — Attempt 27 config (eps_clip=0.1) |
| `train_grpo_20b_a28a.sh` | A/B experiment — eps_clip=0.2 baseline (200 steps, 400-prompt pool) |
| `train_grpo_20b_a28b.sh` | A/B experiment — eps_clip=0.1 tighter clipping (200 steps, 400-prompt pool) |
| `preflight_lr.sh` | Pre-flight LR schedule check — aborts if LR@step20 < 30% target |
| `reward_func_em.py` | Pure exact-match reward function with diagnostic fields |
| `prepare_sft_data.py` | NuminaMath-1.5 data filtering and preparation |
| `apply_patches.sh` | Applies all patches to installed site-packages |
| `rl_grpo_research.md` | Research log documenting all attempts and findings |
| `patches/*.patch` | Patches for OpenRLHF and vLLM |
| `data/sft_rl_pool.jsonl` | RL training data (50k problems) |
| `data/sft_rl_pool_400.jsonl` | 400-prompt subset for A28 A/B (seed=42, deterministic) |
| `data/sft_train.jsonl` | SFT data (5k problems, only if needed) |
| `data/aime_eval.jsonl` | AIME 2024 held-out eval (18 problems) |
| `data/probe_set_200.jsonl` | ID probe set (200 problems from training pool, seed=42) |
| `data/probe_set_200_ood.jsonl` | OOD probe set (170 MATH + 32 Apex, disjoint from training) |

**Rules:**

- Use `git` for all version control. Every change should be tracked.
- **Never edit files directly in `/opt/conda/.../site-packages/`.** All modifications to third-party libraries (OpenRLHF, vLLM) must go through the patch system (see Section 3).
- Keep code, configs, and small data files in `/home/rlu/Code/rft`. Large outputs go to `/mnt/data` or `/mnt/scratch` (see Section 4).

---

## 2. Git Commit & Branch Strategy

### Branch Naming

For every training attempt, create a branch with the naming convention:

```
attempt-XX-<short-desc>
```

Examples:

```bash
git checkout -b attempt-26-grpo-20b-em
git checkout -b attempt-27-grpo-20b-higher-lr
```

### Tagging

Tag the commit before training begins so you can always return to the exact state:

```bash
git tag attempt-26-start
```

### Workflow

1. **Before training:** Commit all relevant files (training script, reward function, patches, config changes). Use descriptive commit messages that explain *why*, not just *what*.
2. **Start training:** Tag `attempt-XX-start` on the commit that represents the training configuration.
3. **After training:** Record results, metrics, and lessons learned in `rl_grpo_research.md`. Commit the updated research log.
4. **If changing approach:** Create a new branch for the next attempt.

### Commit Message Style

```
Fix vLLM engine patches: use collective_rpc for TP-safe weight updates

- Replace NCCL broadcast with collective_rpc dispatch
- Convert tensors to (shape, dtype, list) tuples for msgspec compatibility
- Use save-load-add pattern for TP-correct delta application
```

Focus on the "why" in the first line. Use bullet points for details. Reference the attempt number if relevant.

---

## 3. Third-Party Code Patches

### Overview

We maintain patches against OpenRLHF 0.9.3 and vLLM v0.15.1 in the `patches/` directory. These fix bugs, add monitoring, and enable features needed for our training pipeline.

### Applying Patches

```bash
cd /home/rlu/Code/rft
bash apply_patches.sh
```

This script automatically finds the installed `openrlhf` site-packages directory and applies all `patches/*.patch` files using `patch -p1 --forward`. Already-applied patches are skipped with a warning.

**Always run `apply_patches.sh` after:**
- Reinstalling or upgrading OpenRLHF (`pip install openrlhf==0.9.3`)
- Reinstalling or upgrading vLLM
- Setting up a new environment

### Current Patches and Their Purposes

| Patch | Target | What It Does |
|-------|--------|-------------|
| `actor.patch` | `openrlhf/models/actor.py` | Adds MXFP4 dequantization support (detects `quant_method=mxfp4` in model config, applies `Mxfp4Config(dequantize=True)`). Delays DeepSpeed ZeRO-3 init until after model loading. Forces `device_map="cpu"` for MXFP4+ZeRO-3 to prevent GPU OOM. |
| `ppo_actor.patch` | `openrlhf/trainer/ray/ppo_actor.py` | Adds CUDA synchronization before NCCL group init to prevent stale CUDA errors. Implements Ray object store weight sync path (`--vllm_sync_with_ray`). Adds fast LoRA-only weight sync (`_broadcast_lora_to_vllm`) that transfers only adapter params (~96 MB) instead of all model params. Reads ratio tail diagnostics from PolicyLoss and logs to metrics. Dumps prompt hashes to `spike_log.jsonl` when `grad_norm > 50`. |
| `experience_maker.patch` | `openrlhf/trainer/ppo_utils/experience_maker.py` | Filters `extra_logs` to only `int`/`float` values before converting to `torch.tensor()`. Prevents crashes from string values in logging pipelines. Wraps reward/score conversion in try/except. |
| `save_model.patch` | `openrlhf/utils/deepspeed/deepspeed.py` | Cleans up full-model shard files (`pytorch_model-*.bin`, `model-*.safetensors`) after LoRA save. PeftModel + ZeRO-3 incorrectly writes ~203 GB of shard files alongside the ~91 MB adapter. Also wires the `--offload` CLI flag to the DeepSpeed train config. |
| `offload_cli.patch` | `openrlhf/cli/train_ppo_ray.py` | Adds `--offload` argument to the CLI for CPU parameter offloading with ZeRO-3. |
| `metrics_actor.patch` | `openrlhf/trainer/ray/ppo_actor.py` | Logs gradient norms (from DeepSpeed engine), advantage statistics (mean/std/max/min), and entropy distribution (mean/p10/p90) per training step. |
| `metrics_experience.patch` | `openrlhf/trainer/ppo_utils/experience_maker.py` | Computes reward distribution stats (mean/std/min/max/p90) and high-reward-but-wrong rate per batch for reward hacking detection. |
| `metrics_trainer.patch` | `openrlhf/trainer/ppo_trainer.py` | Writes per-step metrics to JSONL files at `$METRICS_LOG_DIR/training_metrics.jsonl`. Saves sample model outputs every `save_steps` to `$SAMPLES_LOG_DIR/samples_stepN.jsonl`. |
| `vllm_engine.patch` | `openrlhf/trainer/ray/vllm_engine.py` | Adds `update_weight_from_ref()` (single-weight sync via Ray object store) and `apply_lora_update()` (batch LoRA delta dispatch via `collective_rpc`). Handles tensor-to-tuple serialization for msgspec compatibility. |
| `vllm_worker.patch` | `openrlhf/trainer/ray/vllm_worker_wrap.py` | Adds `update_weight_from_ray_ref()` (single-weight restore from object store) and `apply_lora_delta()` (LoRA delta application). Implements the save-load-add pattern for TP-safe weight updates: clone original param, call `load_weights(delta)` for correct TP sharding, then add original back. |
| `ratio_logging_loss.patch` | `openrlhf/models/loss.py` | Adds log-ratio clamping ([-20, 20]) for numerical safety. Stores per-micro-batch ratio tail diagnostics on `PolicyLoss` instance: `log_ratio_max`, `log_ratio_min`, `ratio_max`, `log_ratio_abs_p99`, `tokens_in_batch`. These are read by `ppo_actor.py` and logged to metrics JSONL. |

### Creating New Patches

If you need to modify site-packages code:

1. Copy the original file to a temporary location.
2. Make your changes to the site-packages file for testing.
3. Generate the patch:

```bash
diff -u original_file modified_file > patches/my_fix.patch
```

4. Adjust the patch paths to use `a/openrlhf/...` and `b/openrlhf/...` format (for `-p1` stripping).
5. Test with `bash apply_patches.sh`.
6. Commit the patch file to git.

---

## 4. Disk Usage

### Storage Layout

| Mount Point | Size | Purpose | Examples |
|-------------|------|---------|----------|
| `/mnt/data` | 1 TB NVMe | Model checkpoints, training outputs, large persistent data | `/mnt/data/rft_output/`, `/mnt/data/rft_checkpoints/` |
| `/mnt/scratch` | 5.9 TB RAID-0 (16×375G NVMe) | HF cache, merged models, metrics, logs, temporary files | `/mnt/scratch/hf/`, `/mnt/scratch/merged_models/`, `/mnt/scratch/rft_metrics_20b/` |
| `/home/rlu/Code/rft` | Boot disk (1TB) | Code, configs, small data files | Scripts, patches, `data/*.jsonl` |

### Critical Rules

- **Always save training checkpoints to `/mnt/data`.** Training checkpoints can be huge -- even LoRA-only checkpoints with ZeRO-3 can temporarily write hundreds of GBs if the save_model patch is not applied.
- **Use `--max_ckpt_num` to limit checkpoint storage.** Set to 2-3 to keep only the most recent checkpoints:

```bash
--save_path /mnt/data/rft_output/gpt-oss-20b-grpo \
--ckpt_path /mnt/data/rft_checkpoints/gpt-oss-20b-grpo \
--max_ckpt_num 3 \
--disable_ds_ckpt
```

- **Use `--disable_ds_ckpt`** to prevent DeepSpeed from writing its own intermediate checkpoint files (which are redundant with HF LoRA checkpoints and waste disk space).
- **Monitor disk usage** during training:

```bash
df -h /mnt/data /mnt/scratch
du -sh /mnt/data/rft_checkpoints/*/
```

### Disk Size Reference

| Item | Size |
|------|------|
| gpt-oss-20b (MXFP4 on disk) | ~11 GB |
| gpt-oss-20b (dequantized bf16) | ~84 GB |
| gpt-oss-120b (MXFP4 on disk) | ~30 GB |
| gpt-oss-120b (dequantized bf16) | ~234 GB |
| LoRA adapter checkpoint | ~91 MB |
| PeftModel+ZeRO-3 shard leak (without patch) | ~203 GB |

---

## 5. GPU Usage

### Hardware

8x H100 80GB GPUs, exclusively allocated to this project.

### GPU Allocation by Model Size

**For gpt-oss-20b (21B total params, 3.6B active MoE):**

| Role | GPUs | Config | Memory |
|------|------|--------|--------|
| Actor (training) | GPU 4, 5, 6, 7 | DeepSpeed ZeRO-3, NO CPU offload | 84 GB bf16 / 4 = 21 GB/GPU, ~59 GB free for activations |
| vLLM (rollout) | GPU 0, 1, 2, 3 | 2 engines x TP=2, MXFP4 native | ~68 GB/80 GB used |

**For gpt-oss-120b (117B total params, 5.1B active MoE):**

| Role | GPUs | Config | Memory |
|------|------|--------|--------|
| Actor (training) | GPU 4, 5, 6, 7 | DeepSpeed ZeRO-3 + CPU offload (`--offload --adam_offload`) | 234 GB bf16 / 4 = 58.5 GB/GPU, requires CPU offload |
| vLLM (rollout) | GPU 0, 1, 2, 3 | 2 engines x TP=2, MXFP4 native | ~68 GB/80 GB used |

### vLLM Engine Configuration

vLLM uses TP=2 with 2 separate engines for optimal throughput:

```bash
--vllm_num_engines 2 \
--vllm_tensor_parallel_size 2 \
--vllm_gpu_memory_utilization 0.85
```

This gives 2 independent inference engines, each spanning 2 GPUs, allowing parallel generation.

### Monitoring

Watch for OOM errors with:

```bash
nvidia-smi
watch -n 5 nvidia-smi
```

Key signs of trouble:
- GPU memory utilization >95% on actor GPUs (OOM imminent)
- GPU utilization at 0% for extended periods (possible deadlock)
- CUDA errors in training logs (check `/mnt/scratch/train_20b.log`)

### Memory-Saving Techniques

- **Gradient checkpointing** (`--gradient_checkpointing`): Recomputes activations during backward pass to save memory. Use reentrant mode with ZeRO-3 offload (`--gradient_checkpointing_use_reentrant`).
- **ZeRO-3** (`--zero_stage 3`): Partitions model parameters, gradients, and optimizer states across all actor GPUs.
- **CPU offload** (`--offload --adam_offload`): Moves parameters and optimizer states to CPU RAM. Required for 120B model, not needed for 20B.
- **LoRA**: Only trains adapter parameters (~96 MB), not the full model.

---

## 6. Hyperparameter Tuning Approach

### Philosophy

Start with conservative hyperparameters and tune gradually. Change one variable at a time. Document every change and its effect in `rl_grpo_research.md`.

### Key Hyperparameters

| Parameter | 20B Default | 120B Default | Notes |
|-----------|-------------|-------------|-------|
| `--actor_learning_rate` | `5e-7` | `1e-6` | Start low. If learning is flat after 20 episodes, increase. |
| `--init_kl_coef` | `0.001` | `0` (risky) | Low value allows exploration while providing stability guardrail. 0 = no KL constraint (aggressive). |
| `--lora_rank` | `32` | `64` | Smaller model -> smaller rank. |
| `--lora_alpha` | `64` | `128` | Scaling factor = alpha/rank = 2.0 in both cases. |
| `--n_samples_per_prompt` | `8` | `8` | Group size for GRPO advantage estimation. Larger = better variance estimates but more compute. |
| `--rollout_batch_size` | `16` | `8` | Number of prompts per rollout. Total samples = rollout_batch_size x n_samples_per_prompt. |
| `--train_batch_size` | `16` | `8` | Full training batch size (in prompts). |
| `--micro_train_batch_size` | `1` | `1` | Gradient accumulation. Effective accumulation steps = train_batch_size / micro_train_batch_size. |
| `--eps_clip` | `0.1` | `0.1` | PPO ratio clipping range [1-eps, 1+eps]. Lower = more conservative updates. Attempt-26 used 0.2 (default), attempt-27 tightened to 0.1 to reduce gradient spikes from extreme log-probability ratios. |
| `--generate_max_len` | `4096` | `3072` | Max tokens for generation. Increase if truncation rate >70%. |
| `--prompt_max_len` | `1024` | `1024` | NuminaMath prompts are short (p99.5 = 354 tokens). |
| `--lr_warmup_ratio` | `0.05` | `0.03` | Fraction of total steps used for LR warmup. |
| `--num_episodes` | `20` | `50` | Number of passes over the dataset. Start with 20, extend if model is still learning. |

### LoRA Configuration

- **Target modules**: Attention-only (`q_proj, k_proj, v_proj, o_proj`) for MoE compatibility. Do NOT use `all-linear` with MoE models -- it targets expert FFN layers, which can cause instability.
- **Rank/alpha**: Use scaling factor of 2.0 (alpha = 2 x rank). Common choices: rank=32/alpha=64 for 20B, rank=64/alpha=128 for 120B.

```bash
--lora_rank 32 \
--lora_alpha 64 \
--target_modules q_proj k_proj v_proj o_proj
```

### Memory and Compute Settings

```bash
--zero_stage 3 \                              # Parameter sharding across GPUs
--gradient_checkpointing \                     # Recompute activations to save memory
--gradient_checkpointing_use_reentrant \        # Required for ZeRO-3 + CPU offload
--param_dtype bf16 \                           # bf16 for training precision
--entropy_loss_coef 0 \                        # Log entropy without penalizing it
```

### Tuning Playbook

| Scenario | Symptoms | Actions |
|----------|----------|---------|
| Flat learning | KL tiny, entropy stable, reward/correctness flat | Increase LR, decrease KL coef, increase group size |
| Instability | KL spikes, entropy drops, correctness worsens | Decrease LR, increase KL coef, add grad clipping |
| Reward hacking | Reward up but eval correctness flat or down | Reweight reward toward correctness, add judge redundancy |
| Loss/grad spikes | `policy_loss` 3-10x normal, `grad_norm` 100-400x `max_norm` | Root cause is **extreme log-probability ratios** (policy shifted far from reference on specific tokens), NOT advantage outliers. Fix: tighten `eps_clip` (0.2→0.1), add log-ratio clamping, or lower LR. Advantage clipping does NOT help. See `rl_grpo_research.md` Section 11.12 for theory. |
| Length collapse | Response length drops sharply | Add length penalty, check entropy, add KL constraint |
| High truncation | >70-80% responses truncated | Increase `--generate_max_len` or add length penalty in reward |

---

## 7. Metrics Collecting & Monitoring

### Metrics Output Locations

| Location | Content |
|----------|---------|
| `$METRICS_LOG_DIR` (`/mnt/scratch/rft_metrics_20b/`) | `training_metrics.jsonl` -- per-step training metrics |
| `$SAMPLES_LOG_DIR` (`/mnt/scratch/rft_samples_20b/`) | `samples_stepN.jsonl` -- model output samples every `save_steps` |
| `$SPIKE_LOG_PATH` (`$METRICS_LOG_DIR/spike_log.jsonl`) | Per-spike prompt hashes, grad_norm, ratio stats (when `grad_norm > 50`) |
| `/mnt/scratch/train_20b.log` | Full stdout/stderr training log |
| `/var/tmp/ray/session_*/logs/` | Ray worker logs (actor, vLLM engines) |

### Setting Up Logging

The training script sets up environment variables:

```bash
export METRICS_LOG_DIR="/mnt/scratch/rft_metrics_20b"
export SAMPLES_LOG_DIR="/mnt/scratch/rft_samples_20b"
mkdir -p "$METRICS_LOG_DIR" "$SAMPLES_LOG_DIR"
```

**IMPORTANT — Pre-flight LR check**: Before any training launch, verify the LR schedule will reach meaningful values within the intended step budget:

```bash
POOL=$(wc -l < data/sft_rl_pool_400.jsonl) NS=8 TBS=16 EP=1 WARMUP=0.05 LR=5e-7 bash preflight_lr.sh
```

This models the actual OpenRLHF scheduler (linear warmup + `cosine_with_min_lr`, `min_lr = 0.1 * target_lr`). Aborts if LR@step20 < 30% of target. Total steps = `pool_size * n_samples_per_prompt // train_batch_size * num_episodes`.

Run training with unbuffered output:

```bash
PYTHONUNBUFFERED=1 bash train_grpo_20b.sh 2>&1 | tee /mnt/scratch/train_20b.log
```

### Key Metrics to Monitor

**Tier 1 -- Ground Truth Signal:**

| Metric | What It Tells You | Healthy Range |
|--------|-------------------|---------------|
| `reward_mean` | Average reward per step | Should trend upward over time |
| `correctness` | Fraction of correct answers (exact match) | 20-60% for AIME-difficulty; should improve |
| `has_boxed` | Fraction of responses with `\boxed{}` | Should stay >90% |
| AIME eval (held-out) | True generalization signal | Evaluated every 10 steps |

**Tier 2 -- Stability Diagnostics:**

| Metric | What It Tells You | Warning Signs |
|--------|-------------------|---------------|
| `grad_norm` | Gradient magnitude before clipping | >100x `max_norm` = heavy clipping, damped updates |
| `ppo_kl` | KL divergence from reference policy | Large spikes or sustained negative values |
| `ppo_clip_ratio` | Fraction of clipped policy updates | >40% = very aggressive updates |
| `advantage_mean` / `advantage_std` | Advantage distribution | std near 0 within micro-batch is expected with micro_train_batch_size=1 |
| `entropy_mean` | Token-level entropy | Dropping below 1.0 signals diversity collapse |
| `log_ratio_raw_max` | Max log(π/π_ref) before clamping | >5 = tokens where policy diverged significantly from reference |
| `log_ratio_max` | Max log(π/π_ref) after [-20,20] clamp | Difference from raw_max indicates clamping is active |
| `ratio_max` | Max exp(log_ratio) | >10 = extreme importance sampling ratio |
| `log_ratio_abs_p99` | 99th percentile of |log_ratio| | Tracks the distribution tail, not just the single max |

**Tier 3 -- Collapse Detection:**

| Metric | What It Tells You | Warning Signs |
|--------|-------------------|---------------|
| Response length (mean) | Whether model is degenerating | Sharp drop = length collapse |
| Truncation rate | Fraction of responses hitting max length | >80% sustained = model rambling |
| `high_reward_but_wrong` | Reward hacking indicator | Any nonzero value with pure EM is impossible, but relevant if using LLM judge |

### Evaluation Metrics

**Three eval modes** (never conflate — always label which was used):

| Metric Name | Temp | n | What It Measures | Use Case |
|-------------|------|---|------------------|----------|
| `greedy_pass@1` | 0.0 | 1 | Deterministic single-sample accuracy | Fast checkpoint comparison (primary) |
| `sampling_pass@8` | 0.6 | 8 | Majority-vote accuracy with diversity | More reliable final evaluation |
| `nondet_proxy_pass@8` | 0.0 | 8 | vLLM batching nondeterminism | Diagnostics only (gap vs greedy_pass@1 = fragile solutions) |

**Eval datasets** (three levels):

| Dataset | Size | Type | Purpose |
|---------|------|------|---------|
| `probe_set_200_ood.jsonl` | 202 | OOD (MATH+Apex) | Primary generalization signal, checkpoint selection |
| `probe_set_200.jsonl` | 200 | ID (from training pool) | Stability / early stopping signal |
| `aime_eval.jsonl` | 18 | OOD (AIME 2024) | Sanity check only (too small for trends: 1 problem = 5.6%) |

- **Built-in eval** (automatic, every `--eval_steps`): Use OOD probe with `greedy_pass@1`
- **Post-hoc eval** (manual, on saved checkpoints): AIME + ID probe
- **Early stopping**: If OOD `greedy_pass@1` drops for 2 consecutive evals
- Always report which eval mode and dataset was used; never compare metrics across different modes

### Checking Logs

```bash
# Training progress
tail -f /mnt/scratch/train_20b.log

# Metrics JSONL
tail -1 /mnt/scratch/rft_metrics_20b/training_metrics.jsonl | python -m json.tool

# Ray worker logs (for debugging vLLM or actor issues)
ls /var/tmp/ray/session_*/logs/
tail -100 /var/tmp/ray/session_latest/logs/worker-*.err
```

### Post-Training Analysis

After each training attempt, update `rl_grpo_research.md` with:
- Per-step metrics table (policy_loss, reward, correctness, grad_norm, entropy, etc.)
- Trend analysis (is the model learning? any instability?)
- Evaluation results on held-out AIME and MATH
- Lessons learned and next steps

---

## 8. Training Data Selection

### Source Dataset

**NuminaMath-1.5** (`AI-MO/NuminaMath-1.5`) from HuggingFace. Full dataset: 896,215 examples.

### Filter Pipeline

The script `prepare_sft_data.py` applies the following filters:

| Filter Stage | Remaining | Purpose |
|--------------|-----------|---------|
| Start | 896,215 | Full NuminaMath-1.5 |
| `question_type == "math-word-problem"` | ~631k | Focus on solvable word problems |
| Clean integer answer | ~286k | Enable exact-match reward (no symbolic/parametric) |
| Exclude AMC/AIME sources | ~285k | Keep eval set separate from training data |
| Require valid problem + solution | ~285k | Filter garbage/empty entries |
| Deduplicate by problem content hash (MD5) | ~264k | Prevent memorization of repeated problems |
| Remove too-short or empty-solution | ~264k | Quality filter |

### Data Splits (No Overlap)

| File | Count | Purpose |
|------|-------|---------|
| `data/sft_rl_pool.jsonl` | 50,000 | RL training pool (primary) |
| `data/sft_train.jsonl` | 5,000 | SFT data (only used if format compliance < 95%) |
| `data/sft_dev.jsonl` | 500 | Dev set for SFT evaluation |
| `data/aime_eval.jsonl` | 18 | AIME 2024 held-out eval |

All splits are verified to have zero overlap by problem hash.

### Data Format

**RL pool** (`sft_rl_pool.jsonl`): Each line is a JSON object with `input` (problem text) and `label` (integer answer). The model generates the solution; the reward function checks the answer.

```json
{"input": "Find the sum of all integers n such that ...", "label": "42"}
```

**SFT data** (`sft_train.jsonl`): Includes `output` (solution with `\boxed{answer}`) for supervised fine-tuning.

### Regenerating Data

```bash
cd /home/rlu/Code/rft
python prepare_sft_data.py --sft-size 5000 --rl-pool-size 50000 --seed 42
```

### Key Design Decisions

- **Integer-only answers**: Enables clean exact-match reward with no ambiguity. Fractions, expressions, and symbolic answers are excluded.
- **Exclude AMC/AIME**: These problems are used for evaluation. Including them in training would invalidate eval metrics.
- **Require `\boxed{}` in solutions**: The SFT data ensures solutions end with `\boxed{answer}` format, which the model learns to produce.
- **50k RL pool**: Large enough for 20+ episodes without excessive repetition. With 50k problems and 16 prompts per step, each problem is seen ~6 times in 20 episodes.

---

## 9. Evaluation Methodology

### Primary Evaluation: AIME 2024

- **Dataset**: 18 AIME 2024 problems (held out from training)
- **Frequency**: Every 10 training steps (`--eval_steps 10`)

**Eval configs** (see Section 7 for the full three-metric framework):

```bash
# Recommended: OOD probe with greedy_pass@1 (fast, deterministic, primary for checkpointing)
--eval_dataset data/probe_set_200_ood.jsonl \
--eval_steps 5 \
--eval_temperature 0.0 \
--eval_n_samples_per_prompt 1

# Alternative: AIME with sampling_pass@8 (for final checkpoint comparison)
--eval_dataset data/aime_eval.jsonl \
--eval_steps 10 \
--eval_temperature 0.6 \
--eval_n_samples_per_prompt 8
```

**Do not combine temp=0 with n>1** — greedy produces near-identical samples, wasting compute. Attempt-26 used temp=0/n=8 (`nondet_proxy_pass@8`) which was confusing. Future runs use `greedy_pass@1` for built-in eval.

### Secondary Evaluation: MATH Subset

- **Dataset**: 45 MATH Level 4-5 problems
- **Method**: Single-sample exact match (pass@1)
- **Purpose**: Broader evaluation beyond competition math

### Baselines

| Model | Dataset | Score | Config |
|-------|---------|-------|--------|
| gpt-oss-20b | AIME eval (18) | 27.8% (5/18) | max_tokens=3072, temp=0.0 |
| gpt-oss-20b | AIME eval (18) | **50.0%** (9/18) | reasoning=high, max_tokens=20000 |
| gpt-oss-20b | MATH18_debug (18 problems) | 44.4% (8/18) | max_tokens=3072, temp=0.0 |
| gpt-oss-120b | AIME train (proxy) | ~39.5% | generate_max_len=3072, steps 1-4 avg |

### Evaluation Best Practices

- Always record the baseline before training starts. Use `--eval_steps 1` if you want a step-0 eval.
- Compare checkpoints against the baseline, not against each other (to detect regression).
- 18 AIME problems is a small eval set -- expect variance. A change of 1 problem = 5.6% swing.
- For final evaluation, also run on Apex Shortlist (48 international competition problems).

---

## 10. Reward Function Design

### Pure Exact-Match Reward

The reward function (`reward_func_em.py`) implements a simple binary reward:

| Condition | Reward |
|-----------|--------|
| Extracted answer matches ground truth | **1.0** |
| Extracted answer does not match | **0.0** |

**Properties**: No LLM judge, no Gemini dependency, deterministic, zero API cost, zero latency.

### Answer Extraction Pipeline

1. **Strip Harmony tokens**: If the response contains gpt-oss channel-based output markers (`<|channel|>`, `<|message|>`, etc.), extract the final message content.
2. **Try `\boxed{...}`**: Find the last `\boxed{...}` in the response and extract its contents (handles nested braces).
3. **Fallback -- last number**: If no `\boxed{}` found, extract the last number-like token (`-?\d[\d,]*(?:\.\d+)?`).

### Answer Normalization

Before comparison, both model answer and ground truth are normalized:

- Strip enclosing `$...$` or `\(...\)`
- Remove trailing periods
- Remove LaTeX wrappers: `\text{}`, `\textbf{}`, `\mathrm{}`, `\mathbf{}`
- Normalize fractions: `\dfrac` and `\tfrac` -> `\frac`
- Collapse whitespace
- Remove commas from numbers (e.g., `1,000` -> `1000`)
- Remove leading zeros (e.g., `042` -> `42`)
- Numeric comparison with tolerance (1e-6) as final fallback

### Extra Logging

The reward function returns `extra_logs` with:
- `correctness`: 1.0 or 0.0
- `has_answer`: 1.0 if an answer was extracted, 0.0 if empty
- `has_boxed`: 1.0 if `\boxed{}` was found in the response
- `parse_method`: 2.0=boxed, 1.0=last_number fallback, 0.0=none (numeric for tensor compatibility — the experience_maker drops string values)
- `boxed_in_final`: 1.0 if last `\boxed{}` is in the final 20% of the response, 0.0 otherwise
- `truncated_response`: 1.0 if response appears truncated (heuristic: long + no boxed, or ends mid-sentence)

### Why Pure EM (Not LLM Judge)

The original reward function used a weighted combination: `0.6 * correctness + 0.4 * reasoning_quality` (scored by Gemini 3 Pro). This was abandoned because:

1. Partial credit for wrong answers incentivizes judge-pleasing over correctness.
2. Gemini API has rate limits (64 concurrent calls can exceed quotas).
3. Fallback scoring (when Gemini fails) creates a bimodal reward distribution.
4. Pure EM is deterministic -- no noise from judge variability.

If reward sparsity becomes a problem (baseline solve rate <5%), consider adding curriculum (easy problems first) or Gemini quality scoring for correct-only answers.

---

## 11. Training Pipeline (RL with GRPO)

### Framework Stack

```
OpenRLHF 0.9.3
  + Ray (distributed orchestration)
  + DeepSpeed ZeRO-3 (parameter sharding for training)
  + vLLM v0.15.1 (fast inference for rollouts)
```

### Algorithm: DR-GRPO

**DR-GRPO** (DeepSeek-style Group Relative Policy Optimization):
- No critic model needed (unlike PPO).
- Advantages are computed relative to the group: for each prompt, generate `n_samples_per_prompt` responses, compute rewards, and normalize within the group.
- Advantage for sample `i` = `(reward_i - mean(rewards)) / std(rewards)`.

### Training Flow

```
1. ROLLOUT: vLLM generates n_samples_per_prompt responses per prompt
   (rollout_batch_size prompts x n_samples_per_prompt = total samples)
           |
           v
2. REWARD: reward_func_em.py scores each response (exact match: 1.0 or 0.0)
           |
           v
3. ADVANTAGE: Group-relative normalization within each prompt's samples
           |
           v
4. POLICY UPDATE: DeepSpeed ZeRO-3 trains LoRA adapters using clipped
   policy gradient loss (advantage-weighted log-probability shift)
           |
           v
5. WEIGHT SYNC: Updated LoRA weights transferred to vLLM engines
   (~96 MB via Ray object store + collective_rpc, ~18s per sync)
           |
           v
6. REPEAT from step 1
```

### LoRA Fine-Tuning

- **Why LoRA**: Memory efficiency (only ~96 MB of trainable params vs 84+ GB full model) and MoE compatibility (attention-only targets avoid disrupting expert routing).
- **Base model format**: MXFP4 quantized on disk, dequantized to bf16 for training. vLLM uses native MXFP4 for inference.
- **Weight sync**: After each training step, LoRA adapter weights are synced from actor to vLLM. The delta is computed as `scaling * B @ A` and applied using the save-load-add pattern for TP-safe application.

### Running Training

```bash
cd /home/rlu/Code/rft

# 1. Apply patches
bash apply_patches.sh

# 2. Prepare data (if not already done)
python prepare_sft_data.py

# 3. Start training
PYTHONUNBUFFERED=1 bash train_grpo_20b.sh 2>&1 | tee /mnt/scratch/train_20b.log

# Override any parameter:
PYTHONUNBUFFERED=1 bash train_grpo_20b.sh --num_episodes 50 --actor_learning_rate 1e-6 2>&1 | tee /mnt/scratch/train_20b.log
```

### Timing Estimates (gpt-oss-20b)

| Phase | Duration |
|-------|----------|
| Per step (generation + reward + training + sync) | ~10-13 min |
| Per episode (steps = dataset_size / rollout_batch_size) | ~2 hours |
| 20 episodes | ~40 hours (~1.7 days) |
| 50 episodes | ~100 hours (~4.2 days) |

---

## 12. Decision Framework

### SFT Decision

**SFT is only needed if format compliance (`\boxed{}` rate) is below 95%.**

gpt-oss-20b has been measured at 95% `\boxed{}` compliance when given sufficient token budget. Therefore, SFT was skipped and training proceeds directly to RL.

To re-check format compliance:
1. Run 20 MATH Level 4-5 problems with `reasoning=medium, max_tokens=8192, temp=0`.
2. Measure `\boxed{}` rate separately:
   - **Final answer format**: Does the response contain `\boxed{}`? (this is what matters for reward)
   - **Reasoning quality**: Is the chain-of-thought coherent? (secondary, track separately)
3. If final answer `\boxed{}` rate <95%, run SFT on `data/sft_train.jsonl` (5k examples) first.

### Reward Function Decision

**Start with pure exact-match reward before adding any LLM judge.**

- Pure EM is deterministic, fast, and eliminates reward hacking.
- If pure EM shows clear improvement on held-out AIME within 20 episodes, keep it.
- If reward is too sparse (solve rate <5%, many zero-reward batches), consider:
  1. Adding curriculum (start with easier problems, gradually increase difficulty).
  2. Adding Gemini quality scoring for correct-only answers (no partial credit for wrong ones).
  3. Increasing `n_samples_per_prompt` for better advantage estimates.

### Early Smoke-Test Criteria

Do not wait 20 episodes to detect problems. Check these within the first 2-3 episodes (~6 steps):

| Check | Healthy | Action If Failing |
|-------|---------|-------------------|
| `reward_mean` | Not stuck at 0 or 1 | If 0: data/reward bug. If 1: too-easy problems. |
| `has_boxed` | >50% and not dropping | If dropping: consider SFT phase |
| `grad_norm` | Non-zero, <100 | If 0: frozen weights. If >100: LR too high. |
| `advantage_std` | Growing from 0 step-by-step | If stuck at 0 beyond step 1: advantage computation bug |
| `truncated` | <50% | If >50%: increase `generate_max_len` |
| `ppo_clip_ratio` | <40% | If >40% from the start: LR too high |

If any of these fail early, stop training and fix before burning compute.

### Diagnosing No Improvement

If EM RL shows no improvement after 5 episodes (not 20 -- check sooner for LR tuning):

1. **Check gradient norms**: If consistently >100x `max_norm`, the model is barely updating. Increase LR.
2. **Check truncation rate**: If >80%, increase `generate_max_len`. The model may be solving problems but not reaching the answer.
3. **Check `has_boxed` rate**: If dropping, the model is losing format compliance. Consider a small SFT phase.
4. **Increase `n_samples_per_prompt`**: Going from 8 to 16 gives better advantage estimates (more signal per step).
5. **Increase LR**: From 5e-7 to 1e-6, then to 2e-6 if still flat.
6. **Add quality signal**: Use Gemini to score reasoning quality on correct-only answers.

### Monitoring for Overfitting / Reward Hacking

**Train/eval divergence** = training correctness improves but held-out eval degrades. This is the primary signal to stop training.

Signs to watch for:
- Training correctness rolling average trending up while OOD probe `greedy_pass@1` is flat or declining.
- `reward_mean` trending up but AIME eval score flat or declining.
- `high_reward_but_wrong` rate increasing (only relevant with LLM judge, not pure EM).
- Gradient norm spikes escalating over time (max per decade increasing = policy drifting into unstable regions).

**Attempt-26 example**: Training correctness improved (rolling avg 0.51 → 0.56) while AIME pass@1 degraded (35.4% peak at step 20 → 27.1% at step 50, below pre-training baseline). An ID-only probe set would have missed this divergence — the OOD probe is essential.

Response: Use separate ID and OOD probe sets. If OOD probe drops for 2 consecutive evals, stop training. The best checkpoint is usually earlier than you think (attempt-26 peaked at step 30 out of 60+).

---

## 13. Spare Machine (h100-8-5) — Eval & Diagnostics

### Overview

h100-8-5 is a drop-in spare for running eval/diagnostics in parallel with training on the primary (h100-8-4). It mirrors the primary's environment exactly.

| Item | Primary (h100-8-4) | Spare (h100-8-5) |
|------|---------------------|-------------------|
| Internal IP | `10.202.0.2` | `10.202.0.3` |
| Repo | `~/Code/rft` | `~/Code/rft` |
| Conda env | `rft` | `rft` |
| `/mnt/data` | 1TB NVMe | 1TB NVMe |
| `/mnt/scratch` | 5.9TB RAID-0 (16×375G) | 5.9TB RAID-0 (16×375G) |
| GPUs | 8× H100 80GB | 8× H100 80GB |

### SSH Access (from primary)

```bash
ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3
```

### Model Sync Strategy: Cache-First (Approach A)

Base models are downloaded once from HF and cached on `/mnt/scratch/hf/`. RL checkpoints are LoRA adapters (~50-200MB) — copy only adapters via `rsync`, then merge locally on spare. Merged outputs are cached under `/mnt/scratch/merged_models/`.

| Artifact | Size | How Synced | Location on Spare |
|----------|------|------------|-------------------|
| HF base models | 10-60GB | Auto-download (HF cache) | `/mnt/scratch/hf/` |
| LoRA adapters | 50-200MB | `rsync` from primary | `/mnt/data/rft_checkpoints/` |
| Merged models | 20-40GB | Built locally (`eval_posthoc.py`) | `/mnt/scratch/merged_models/` |
| Code + datasets | <25MB | `git bundle` (repo is not pushed) | `~/Code/rft/` |

**Sync commands:**

```bash
# Sync code (git bundle for unpushed commits)
cd ~/Code/rft
git bundle create /tmp/rft-sync.bundle origin/attempt-26-grpo-20b-em..HEAD
scp -i ~/.ssh/google_compute_engine /tmp/rft-sync.bundle rlu@10.202.0.3:/tmp/rft-sync.bundle
ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3 'cd ~/Code/rft && git fetch /tmp/rft-sync.bundle HEAD:refs/heads/bundle-tmp && git merge bundle-tmp --ff-only && git branch -d bundle-tmp'

# Sync data (gitignored files)
rsync -avz -e "ssh -i ~/.ssh/google_compute_engine" ~/Code/rft/data/ rlu@10.202.0.3:~/Code/rft/data/

# Sync a LoRA adapter
rsync -avz -e "ssh -i ~/.ssh/google_compute_engine" \
  /mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a30/global_step130_hf/ \
  rlu@10.202.0.3:/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a30/global_step130_hf/
```

**Verification:**

```bash
# Compare SHA, datasets, packages
LOCAL_SHA=$(git rev-parse HEAD)
SPARE_SHA=$(ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3 'cd ~/Code/rft && git rev-parse HEAD')
[ "$LOCAL_SHA" = "$SPARE_SHA" ] && echo "OK" || echo "MISMATCH"
```

### Task Dispatch

**Standard pattern** (tmux session on spare, logs to `/mnt/scratch/eval_spare/logs/`):

```bash
SSH="ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3"

# Example 1: Model sweep (OOD-1000, all 4 models)
$SSH 'tmux new-session -d -s eval-sweep "cd ~/Code/rft && conda activate rft && python eval_model_sweep.py --stage 1 2>&1 | tee /mnt/scratch/eval_spare/logs/sweep_stage1.log"'

# Example 2: Paired OOD-1000 (base vs RL)
$SSH 'tmux new-session -d -s eval-ood1000 "cd ~/Code/rft && conda activate rft && python eval_baseline_s0.py 2>&1 | tee /mnt/scratch/eval_spare/logs/paired_ood1000.log"'

# Example 3: Post-hoc eval with merged checkpoint
$SSH 'tmux new-session -d -s eval-posthoc "cd ~/Code/rft && conda activate rft && python eval_posthoc.py --model a30_step130=/mnt/scratch/merged_models/a30_step130 2>&1 | tee /mnt/scratch/eval_spare/logs/posthoc.log"'
```

**Monitoring:**

```bash
$SSH 'tmux list-sessions'                                           # list jobs
$SSH -t 'tmux attach -t eval-sweep'                                 # attach (Ctrl-B D to detach)
$SSH 'tail -30 /mnt/scratch/eval_spare/logs/sweep_stage1.log'       # tail log
$SSH 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'  # GPU status
```

### Environment Variables (on spare)

Set in `~/.bashrc` and `~/.profile`:

```bash
export HF_HOME=/mnt/scratch/hf
export TRANSFORMERS_CACHE=/mnt/scratch/hf/transformers
export HF_DATASETS_CACHE=/mnt/scratch/hf/datasets
export VLLM_CACHE_DIR=/mnt/scratch/vllm_cache
```

### Patch Sync

All 11 OpenRLHF patches are applied by copying the already-patched files from the primary's site-packages. If you reinstall openrlhf on the spare, re-sync patches:

```bash
# On primary: create tarball of patched files
SITE_PKG=$(conda run -n rft python -c 'import openrlhf; import os; print(os.path.dirname(os.path.dirname(openrlhf.__file__)))')
cd $SITE_PKG && tar czf /tmp/patched_openrlhf.tar.gz openrlhf/models/actor.py openrlhf/models/loss.py openrlhf/trainer/ray/ppo_actor.py openrlhf/trainer/ray/vllm_engine.py openrlhf/trainer/ray/vllm_worker_wrap.py openrlhf/trainer/ppo_utils/experience_maker.py openrlhf/trainer/ppo_trainer.py openrlhf/cli/train_ppo_ray.py openrlhf/utils/deepspeed/deepspeed.py

# Copy and extract on spare
scp -i ~/.ssh/google_compute_engine /tmp/patched_openrlhf.tar.gz rlu@10.202.0.3:/tmp/
ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3 "SITE_PKG=\$(conda run -n rft python -c 'import openrlhf; import os; print(os.path.dirname(os.path.dirname(openrlhf.__file__)))') && cd \$SITE_PKG && tar xzf /tmp/patched_openrlhf.tar.gz"
```

---

*This document should be updated as the training recipe evolves. All experimental results belong in `rl_grpo_research.md`.*
