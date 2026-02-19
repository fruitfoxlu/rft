# Attempt 21

## Hypothesis
- CPU parameter offloading (`--offload`) keeps ZeRO-3 params on CPU, loading one layer at a time to GPU during forward/backward pass.
- Without offload: ~66 GB/GPU for params, only ~13 GB headroom → OOM on 8.79 GiB activation alloc.
- With offload: Only ~6 GB/GPU per layer in flight, leaving ~73 GB headroom → plenty for activations.
- Trade-off: Slower training (CPU→GPU transfer per layer), but fits in memory.

## Changes from Attempt 20
1. Added `--offload` flag to `train_grpo.sh` (enables `offload_param.device = "cpu"` in DeepSpeed ZeRO-3 config)

## Command
```bash
bash train_grpo.sh > train.log 2>&1
```

Key flags: `--offload`, `--adam_offload`, `--vllm_sync_with_ray`, `--zero_stage 3`, `--vllm_gpu_memory_utilization 0.80`, `--init_kl_coef 0`, `--actor_num_gpus_per_node 4`, `--vllm_num_engines 2`, `--vllm_tensor_parallel_size 2`

## Result
- Training in progress...
