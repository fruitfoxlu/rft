# Attempt 20

## Hypothesis
- Reducing `--vllm_gpu_memory_utilization` from default 0.9 to 0.80 frees ~8 GB per vLLM GPU, potentially giving enough headroom for the MoE forward pass.

## Changes from Attempt 19
1. Added `--vllm_gpu_memory_utilization 0.80` to `train_grpo.sh`

## Command
```bash
bash train_grpo.sh > train.log 2>&1
```

## Result
- **FAILED**: Same OOM as attempt 19.
- Rollout generation succeeded (vLLM inference works fine).
- OOM hits during the **training forward pass** on actor GPUs (not vLLM GPUs).
- Each actor GPU: ~66 GB PyTorch allocated (ZeRO-3 partitioned model), only ~13 GB free.
- MoE BMM (`gate_up = torch.bmm(hidden_states, self.gate_up_proj)`) tries to allocate ~8.79 GiB for the all-gather + activations, but only ~5 GB free.
- Error: `torch.OutOfMemoryError: Tried to allocate 8.79 GiB. GPU 0 has 79.10 GiB total, 4.98 GiB free.`
- **Root cause**: ZeRO-3 partitions 234 GB across 4 GPUs = ~58.5 GB/GPU for params alone. With LoRA + buffers = ~66 GB, leaving insufficient room for forward pass activations.

## GPU Layout (during training)
- GPUs 0-3: vLLM engines (2 engines × 2 TP), ~68 GB each
- GPUs 4-7: Actor model (ZeRO-3), ~66 GB each → OOM during forward pass

## Fix for Next Attempt
Add `--offload` flag to enable `offload_param: {"device": "cpu"}` in DeepSpeed config. This keeps model parameters on CPU, loading one layer at a time to GPU during forward/backward pass (~6 GB per layer). Trades CPU↔GPU bandwidth for GPU memory headroom.
