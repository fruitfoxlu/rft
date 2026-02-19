# Attempt 22

## Hypothesis
- Non-reentrant gradient checkpointing (PyTorch default) validates tensor metadata during backward
  recomputation, but ZeRO-3 CPU param offload changes param shape between forward ([2880]) and
  backward ([0], partitioned/offloaded).
- Reentrant gradient checkpointing skips this strict validation.
- With `--gradient_checkpointing_use_reentrant`, backward pass should complete successfully.

## Changes from Attempt 21
1. Added `--gradient_checkpointing_use_reentrant` to `train_grpo.sh`

## Command
```bash
bash train_grpo.sh > train.log 2>&1
```

Key flags: `--offload`, `--adam_offload`, `--gradient_checkpointing_use_reentrant`, `--vllm_sync_with_ray`, `--zero_stage 3`, `--vllm_gpu_memory_utilization 0.80`, `--init_kl_coef 0`

## Result
- Training in progress...
