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
1. **Ray object store weight sync**: Replaced Ray collective broadcast (which uses NCCL
   rendezvous that fails between actor and vLLM EngineCore subprocess workers) with
   Ray object store transfer (ray.put/ray.get). This completely avoids NCCL for weight sync.
   - `ppo_actor.py`: Actor puts param data into Ray object store, sends ref to engines
   - `vllm_engine.py`: Added `update_weight_from_ref()` that passes weight_ref via collective_rpc
   - `vllm_worker_wrap.py`: Added `update_weight_from_ref()` that uses ray.get(weight_ref)
   - Removed collective group initialization (no longer needed with --vllm_sync_with_ray)
2. Updated `reward_func.py` with Gemini model fallback chain

## Failed sub-attempts
- Warm-up broadcast during init_process_group: Same rendezvous.meet() failure
  (NCCLUniqueIDStore named actor not findable across Ray namespaces)

## Command
```bash
bash train_grpo.sh > train.log 2>&1
```

Key flags: `--offload`, `--adam_offload`, `--gradient_checkpointing_use_reentrant`, `--vllm_sync_with_ray`, `--zero_stage 3`, `--vllm_gpu_memory_utilization 0.80`, `--init_kl_coef 0`

## Result
- Training in progress...
