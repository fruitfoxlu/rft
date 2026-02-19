# Attempt 23

## Hypothesis
- Attempt 22 succeeded through episode 1 training (16/16 steps completed in 12:05)
- Weight sync from actor to vLLM engines failed after training:
  - `RuntimeError: Unable to meet other processes at the rendezvous store`
  - Root cause: Ray collective's NCCL communicator is created lazily on first `broadcast()`.
    The rendezvous requires all 5 ranks (1 actor + 4 vLLM workers) to call `broadcast()`
    within 180s, but vLLM v1's EngineCore subprocess indirection causes timing issues.
- Fix: Add a warm-up broadcast during `init_process_group` (when all ranks are synchronized)
  to eagerly create the NCCL communicator, avoiding the timing-sensitive lazy creation
  during the first actual weight update.
- Also: Gemini reward function now has model fallback chain:
  gemini-3-pro-preview → gemini-3.1-pro-preview → gemini-3-flash-preview

## Changes from Attempt 22
1. Added warm-up broadcast in `ppo_actor.py` after `init_collective_group()` (actor rank 0)
2. Added warm-up broadcast in `vllm_worker_wrap.py` after `init_collective_group()` (worker ranks)
3. Updated `reward_func.py` with Gemini model fallback chain

## Command
```bash
bash train_grpo.sh > train.log 2>&1
```

Key flags: `--offload`, `--adam_offload`, `--gradient_checkpointing_use_reentrant`, `--vllm_sync_with_ray`, `--zero_stage 3`, `--vllm_gpu_memory_utilization 0.80`, `--init_kl_coef 0`

## Result
- Training in progress...
