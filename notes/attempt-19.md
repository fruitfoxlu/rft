# Attempt 19

## Hypothesis
- `--vllm_sync_with_ray` bypasses the direct `PyNcclCommunicator` that crashes with "NCCL error: unhandled cuda error" after `deepspeed.zero.Init(module=model)` partitions the 234GB dequantized model.
- Remove string values from `extra_logs` (reward_func returned `model_answer` as a string, causing `torch.tensor([str])` to fail).

## Changes from Attempt 18
1. Added `--vllm_sync_with_ray` to `train_grpo.sh`
2. Changed `NCCL_DEBUG=WARN` to `NCCL_DEBUG=INFO` + added `NCCL_CUMEM_ENABLE=0`
3. Fixed `reward_func.py`: replaced `"model_answer": model_answer` (string) with `"has_answer": float(model_answer != "")` (numeric)
4. Fixed `experience_maker.py`: filter `extra_logs` to only include `int`/`float` values
5. Added `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` before sync group creation in `ppo_actor.py`

## Command
```bash
bash train_grpo.sh > train.log 2>&1
```

Key flags: `--vllm_sync_with_ray`, `--zero_stage 3`, `--adam_offload`, `--init_kl_coef 0`, `--actor_num_gpus_per_node 4`, `--vllm_num_engines 2`, `--vllm_tensor_parallel_size 2`

## Result
- NCCL sync group issue RESOLVED (Ray collective group created successfully)
- `ValueError: too many dimensions 'str'` from extra_logs fixed
- Training in progress...
