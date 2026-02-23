# RL Fine-Tuning Pipeline for gpt-oss-20b

RL fine-tuning of gpt-oss-20b (20B MoE, 1.8B active params, MXFP4) on competition math using DR-GRPO with pure exact-match reward.

## Architecture

```
8x A30 24GB GPUs
├── vLLM Rollout (GPU 0-3): 2 engines × TP=2, MXFP4 native inference
└── Actor Training (GPU 4-7): DeepSpeed ZeRO-3, CPU param/optimizer offload, LoRA
```

- **RL Algorithm**: DR-GRPO (group-relative advantages, no critic)
- **Model**: gpt-oss-20b — 20B total params, 1.8B active (MoE with MXFP4 quantization)
- **LoRA**: rank=32, alpha=64, targets=q_proj,k_proj,v_proj,o_proj (~46 MB adapter)
- **Weight Sync**: Actor → vLLM via Ray object store (LoRA-only, ~6s per sync)
- **Reward**: Pure exact-match (1.0 if correct, 0.0 otherwise). No LLM judge.
- **Training Pool**: 3,200 NuminaMath-1.5 problems (integer-answer, filtered from 50k pool)
- **Eval**: OOD probe (1000 problems, MATH Level 4-5 + competition), ID probe (200), AIME (18)

## Quick Start

### Prerequisites

```bash
pip install openrlhf==0.9.3 kernels
```

### Setup

```bash
# Apply patches to OpenRLHF
bash apply_patches.sh

# Prepare training data (NuminaMath-1.5 → filtered integer-answer pool)
python filter_numinamath.py

# Run baseline evaluation
python eval_baseline.py --auto-symlink
```

### Train

```bash
# Latest training script (A30 config, micro_train_batch_size=2)
bash train_grpo_20b_a30.sh
```

Key hyperparameters (from A30):
- `actor_learning_rate`: 5e-7 (cosine decay, min_lr=5e-8)
- `n_samples_per_prompt`: 8
- `rollout_batch_size`: 16
- `train_batch_size`: 16
- `micro_train_batch_size`: 2
- `eps_clip`: 0.1
- `generate_max_len`: 4096
- `num_episodes`: 1 (single pass over pool → 200 global steps)

### Evaluate

Post-hoc evaluation of a trained checkpoint:

```bash
# Merge LoRA adapter into base model (CPU-only)
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained('base_model_path', torch_dtype='auto')
model = PeftModel.from_pretrained(model, 'checkpoint_path')
model = model.merge_and_unload()
model.save_pretrained('merged_output_path')
"

# Evaluate with vLLM
python eval_posthoc.py --model_path merged_output_path --eval_set data/probe_set_1000_ood.jsonl
```

## Project Structure

```
.
├── train_grpo_20b_a30.sh       # Latest training script (A30, micro=2)
├── train_grpo_20b_a29b.sh      # A29B training script (micro=1, eps_clip=0.1)
├── train_grpo_20b.sh           # Base 20B training template
├── reward_func_em.py           # Pure exact-match reward function
├── filter_numinamath.py        # NuminaMath-1.5 → integer-answer training pool
├── build_ood_1000.py           # Build OOD-1000 probe set (reproducible)
├── eval_posthoc.py             # Post-hoc evaluation via vLLM
├── eval_regression_check.py    # Pipeline regression check (in-training vs post-hoc)
├── eval_baseline.py            # Baseline evaluation (pre-training)
├── eval_apex.py                # Apex Shortlist evaluation
├── test_lora_fix.py            # LoRA accumulation bug fix regression test
├── preflight_lr.sh             # Verify LR schedule before training
├── monitor.sh                  # Live training monitor
├── apply_patches.sh            # Apply all patches to OpenRLHF
├── patches/                    # OpenRLHF patches (see below)
├── data/                       # Training and eval datasets (gitignored)
│   ├── sft_rl_pool.jsonl             # Full training pool (50k problems)
│   ├── sft_rl_pool_3200.jsonl        # Active training subset (3,200 problems)
│   ├── probe_set_1000_ood.jsonl      # OOD eval probe (1000 problems, SE≈1.5%)
│   ├── probe_set_200_ood.jsonl       # Legacy OOD probe (202 problems)
│   ├── probe_set_200.jsonl           # ID eval probe (200 problems, from pool)
│   ├── aime_eval.jsonl               # AIME 2024 held-out (18 problems)
│   ├── math45_train.jsonl            # MATH Level 4-5 train (5,224)
│   ├── math45_eval.jsonl             # MATH Level 4-5 eval (1,306)
│   └── apex_shortlist.jsonl          # International competition problems (48)
└── rl_grpo_research.md         # Full research log (30 attempts, 21 lessons)
```

## Patches

OpenRLHF 0.9.3 patches for the 20B MoE training setup:

| Patch | Purpose |
|-------|---------|
| `actor.patch` | Filter non-numeric values from extra_logs to prevent tensor crashes |
| `experience_maker.patch` | Wire custom reward function fields |
| `ppo_actor.patch` | Replace NCCL weight sync with Ray object store; LoRA-only fast sync |
| `save_model.patch` | Clean up full-model shards from PeftModel + ZeRO-3 saves |
| `offload_cli.patch` | Add `--offload` flag for CPU parameter offloading |
| `ratio_logging_loss.patch` | Log per-step ratio statistics and tail clipping |
| `vllm_engine.patch` | vLLM engine compatibility for MXFP4 models |
| `vllm_worker.patch` | vLLM worker patches for weight sync |
| `metrics_actor.patch` | Log gradient norms, advantage distribution, token entropy |
| `metrics_experience.patch` | Log reward distribution stats |
| `metrics_trainer.patch` | Write per-step JSONL metrics and sample outputs to disk |

## Reward Function

The reward function (`reward_func_em.py`) uses pure exact-match scoring:

```
reward = 1.0 if extracted_answer == ground_truth else 0.0
```

- Extracts the last `\boxed{...}` expression from model output
- Compares numerically against ground truth integer label
- Deterministic, zero API cost, no LLM judge dependency

## Evaluation

Three evaluation tiers:

| Probe | N | SE (p=0.65) | Purpose |
|-------|---|-------------|---------|
| OOD-1000 | 1000 | ±1.5% | Primary metric — MATH L4-5 + competitions, disjoint from training pool |
| ID-200 | 200 | ±3.4% | Stability check — sampled from training pool |
| AIME-18 | 18 | ±11% | Sanity check only — too noisy for decisions |

All evaluation uses greedy decoding (temperature=0), max_tokens=4096, via vLLM with TP=2.

## Current Results (Attempt 30)

Best checkpoint: **A30 step 130** (post-hoc eval):

| Metric | Base Model | A30 Best (step 130) |
|--------|-----------|-------------------|
| OOD (N=202) | ~58% | 65.35% |
| ID (N=200) | — | 52.50% |
| AIME (N=18) | — | 38.89% |

Training: 200 global steps, ~114s/step on 8x A30, 7.4 hours total.

## Research Log

See [rl_grpo_research.md](rl_grpo_research.md) for the full research log covering:
- 30 training attempts with root cause analysis
- Architecture decisions and rationale
- 21 lessons learned (LoRA accumulation bug, micro batch sizing, eval methodology, etc.)
- Training metrics tables and diagnostic framework

## Key Lessons

1. **LoRA weight accumulation bug**: Syncing adapter weights to vLLM without resetting base causes silent drift — fixed by caching base weights and reconstructing each sync
2. **micro_train_batch_size=2 is optimal**: ~18% faster, no OOM, no regression vs micro=1
3. **Cosine LR over-decays in final quarter**: Best checkpoint is often at step 130/200, not final — consider raising min_lr
4. **Pipeline regression check is essential**: Compare in-training eval with post-hoc eval on same checkpoint to validate correctness
5. **N=200 is too noisy for decisions**: SE≈3.4% means ~6pp differences needed for significance — use OOD-1000 (SE≈1.5%)
6. **ZeRO-3 + CPU offload needs reentrant gradient checkpointing**
7. **Ray object store is more reliable than NCCL for cross-actor weight sync**
