# Attempt 25

## Hypothesis
- Attempt 24 demonstrated training runs stably (16+ steps, checkpoint saving works)
- But lacks diagnostic metrics to understand loss spikes (act_loss=2.69) and
  step 16 performance collapse (9.4% correctness, 86% truncation)
- Adding comprehensive monitoring will enable principled tuning decisions
- Built-in held-out eval every 10 steps gives ground-truth signal

## Changes from Attempt 24

1. **New metrics patches** (3 new patches):
   - `metrics_actor.patch`: Gradient norm (from DeepSpeed engine), advantage
     stats (mean/std/max/min), token entropy distribution (mean/p10/p90)
   - `metrics_experience.patch`: Reward distribution stats (mean/std/min/max/p90),
     high-reward-but-wrong detection rate
   - `metrics_trainer.patch`: Per-step JSONL metrics to /mnt/scratch, sample
     output logging every save_steps

2. **Held-out evaluation** (`train_grpo.sh`):
   - Added `--eval_dataset` pointing to AIME eval set (18 problems)
   - `--eval_steps 10`: evaluate every 10 training steps
   - `--eval_temperature 0.0`: deterministic for EM measurement
   - `--eval_n_samples_per_prompt 8`: pass@8 computation

3. **Entropy logging** (`train_grpo.sh`):
   - Added `--entropy_loss_coef 0`: enables entropy computation without
     affecting the loss function (coef=0 means log-only)

4. **Disk space utilization**:
   - Checkpoints: `/mnt/data/rft_checkpoints/` (1TB NVMe)
   - Model output: `/mnt/data/rft_output/` (1TB NVMe)
   - Training metrics JSONL: `/mnt/scratch/rft_metrics/` (5.9TB RAID-0)
   - Sample outputs: `/mnt/scratch/rft_samples/` (5.9TB RAID-0)
   - `--max_ckpt_num 3` (up from 2, since disk is no longer a constraint)

## Metrics Being Logged

### Per-step (global step summary)
| Metric | Source | Purpose |
|--------|--------|---------|
| policy_loss | actor | Policy gradient loss (not monotonic) |
| grad_norm | actor | Detect gradient spikes |
| advantage_mean/std/max/min | actor | Advantage distribution health |
| entropy_mean/p10/p90 | actor | Detect collapse (entropy drops) |
| reward/reward_mean/std/min/max/p90 | experience | Reward distribution |
| correctness | experience | Train batch exact match |
| reasoning_quality | experience | Gemini judge score |
| high_reward_but_wrong | experience | Reward hacking detector |
| ppo_kl | actor | Approximate KL (from importance ratios) |
| ppo_clip_ratio | actor | Update aggressiveness |
| has_answer | experience | Parse success rate |
| response_length | actor | Length drift detection |
| truncated | actor | Verbosity drift |

### Per-eval (every 10 steps)
| Metric | Source | Purpose |
|--------|--------|---------|
| eval_pass1 | eval | Held-out exact match (ground truth) |
| eval_pass8 | eval | Held-out pass@8 |

### Persistent storage
| Path | Content |
|------|---------|
| /mnt/scratch/rft_metrics/training_metrics.jsonl | All per-step metrics |
| /mnt/scratch/rft_samples/samples_stepN.jsonl | Model outputs at checkpoints |

## Command
```bash
bash train_grpo.sh > train.log 2>&1
```

Key flags: `--offload`, `--adam_offload`, `--gradient_checkpointing_use_reentrant`,
`--vllm_sync_with_ray`, `--zero_stage 3`, `--vllm_gpu_memory_utilization 0.80`,
`--init_kl_coef 0`, `--entropy_loss_coef 0`, `--eval_steps 10`,
`--eval_dataset data/eval_prompts.jsonl`, `--disable_ds_ckpt`, `--save_hf_ckpt`

## Baseline (from Attempt 24)
- Teacher (Gemini 3 Pro) on AIME eval: 0% (18 problems)
- Teacher on MATH45 eval: 27.8% (18 problems)
- Base student: TBD
- Attempt 24 training batch correctness: 9.4–64.1% (noisy, 16 steps)

## Result
- Pending (waiting for attempt 24 to stop, then restart)
