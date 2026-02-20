# RL Fine-Tuning Pipeline for gpt-oss-120b

RL fine-tuning of gpt-oss-120b on competition math (AIME) using GRPO with Gemini as the reward model.

## Architecture

```
8x H100 80GB GPUs
├── vLLM Rollout (GPU 0-3): 2 engines × TP=2, MXFP4 native inference
└── Actor Training (GPU 4-7): DeepSpeed ZeRO-3, CPU param/optimizer offload, LoRA
```

- **RL Algorithm**: GRPO (`dr_grpo`) — group-relative advantages, no critic needed
- **Model**: 120B params (MXFP4 quantized, dequantized to bf16 for training)
- **LoRA**: rank=64, alpha=128, targets=q_proj,k_proj,v_proj,o_proj (~96 MB adapter)
- **Weight Sync**: Actor → vLLM via Ray object store (LoRA-only, ~6s per sync)
- **Reward**: `0.6 * exact_match + 0.4 * gemini_reasoning_quality`

## Quick Start

### Prerequisites

```bash
pip install openrlhf==0.9.3 google-genai kernels
```

Google Cloud auth must be configured for Vertex AI (Gemini reward function):
```bash
gcloud auth application-default login
```

### Setup

```bash
# Prepare AIME train/eval datasets
python prepare_data.py

# Create data symlinks and run baseline evaluation
python eval_baseline.py --auto-symlink

# Apply patches to OpenRLHF
bash apply_patches.sh
```

### Train

```bash
bash train_grpo.sh
```

Key hyperparameters can be overridden:
```bash
bash train_grpo.sh --num_episodes 100 --actor_learning_rate 5e-7
```

### Evaluate

After training, compare base model vs. RL-trained model:
```bash
python eval_rl.py --checkpoint_path /mnt/data/rft_output/gpt-oss-120b-grpo
```

## Project Structure

```
.
├── train_grpo.sh           # Main training script (all hyperparameters)
├── reward_func.py          # Gemini-based reward function (EM + reasoning quality)
├── prepare_data.py         # Download and format AIME datasets
├── eval_baseline.py        # Baseline evaluation (teacher + student pre-training)
├── eval_rl.py              # Post-training RL model evaluation
├── monitor.sh              # Live training monitor
├── apply_patches.sh        # Apply all patches to OpenRLHF
├── patches/                # OpenRLHF patches
│   ├── actor.patch                 # Fix extra_logs type filtering
│   ├── experience_maker.patch      # Reward function integration
│   ├── ppo_actor.patch             # Ray object store weight sync + LoRA-only sync
│   ├── save_model.patch            # Fix PeftModel + ZeRO-3 checkpoint saving
│   ├── offload_cli.patch           # Add --offload CLI arg for CPU param offload
│   ├── metrics_actor.patch         # Grad norm, advantage stats, entropy logging
│   ├── metrics_experience.patch    # Reward distribution stats
│   └── metrics_trainer.patch       # JSONL metrics + sample output logging
├── data/                   # Training and eval datasets (gitignored)
├── notes/                  # Per-attempt notes
└── rl_grpo_research.md     # Detailed research log with metrics and lessons learned
```

## Patches

This project patches OpenRLHF 0.9.3 to support the 120B model training setup. Patches are applied via `bash apply_patches.sh` after `pip install openrlhf==0.9.3`.

| Patch | Purpose |
|-------|---------|
| `actor.patch` | Filter non-numeric values from extra_logs to prevent tensor crashes |
| `experience_maker.patch` | Wire custom reward function fields (correctness, reasoning_quality) |
| `ppo_actor.patch` | Replace NCCL weight sync with Ray object store; add LoRA-only fast sync |
| `save_model.patch` | Clean up full-model shards from PeftModel + ZeRO-3 saves (203GB → 91MB) |
| `offload_cli.patch` | Add `--offload` flag for CPU parameter offloading |
| `metrics_actor.patch` | Log gradient norms, advantage distribution, token entropy |
| `metrics_experience.patch` | Log reward distribution stats and high-reward-but-wrong detection |
| `metrics_trainer.patch` | Write per-step JSONL metrics and sample outputs to disk |

## Reward Function

The reward function (`reward_func.py`) combines exact-match correctness with Gemini-judged reasoning quality:

```
reward = 0.6 * correctness + 0.4 * reasoning_quality
```

- **Correctness**: Binary (0/1) — extracts boxed answer and compares to ground truth
- **Reasoning quality**: Gemini rates solution on [0, 1] scale
- **Fallback chain**: gemini-3.1-pro → gemini-3-pro → gemini-3-flash
- **Failure mode**: If all Gemini calls fail, falls back to pure exact-match (reasoning_quality=0)

## Monitoring

Training logs per-step metrics to JSONL files:
- `/mnt/scratch/rft_metrics/training_metrics.jsonl` — all numeric metrics per step
- `/mnt/scratch/rft_samples/samples_stepN.jsonl` — model outputs at checkpoint steps

Key metrics tracked: policy_loss, grad_norm, advantage stats, entropy, reward distribution, correctness, truncation rate, ppo_clip_ratio, ppo_kl.

Held-out evaluation runs every 10 steps on 18 AIME problems (pass@1 and pass@8).

## Research Log

See [rl_grpo_research.md](rl_grpo_research.md) for the full research log including:
- Architecture decisions and rationale
- Training configuration details
- Attempt-by-attempt history with root cause analysis
- Training metrics tables
- Diagnostic framework and tuning playbook
- Lessons learned for future RL fine-tuning projects

## Key Lessons

1. **ZeRO-3 alone is insufficient for 120B on 4 GPUs** — CPU param offload required
2. **ZeRO-3 + CPU offload requires reentrant gradient checkpointing** — non-reentrant validates shapes that change during offload/gather
3. **NCCL weight sync is fragile across Ray actor boundaries** — Ray object store is slower but reliable
4. **PeftModel + ZeRO-3 save writes full-model shards by default** — must clean up post-save
5. **Instrument before you train** — without grad norms, advantage stats, and held-out eval, you can observe problems but not diagnose them
