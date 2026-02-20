# Attempt 24

## Hypothesis
- Attempt 23 resolved weight sync (Ray object store instead of NCCL)
- Training ran but filled disk (1TB) with checkpoints:
  - DeepSpeed ZeRO-3 intermediate checkpoints: 219 GB each (full model + optimizer states)
  - Final HuggingFace save: 203 GB (full merged model, not just LoRA adapters)
  - Root cause: `PeftModel.save_pretrained()` with ZeRO-3 writes full-model sharded
    `pytorch_model-*.bin` files, and OpenRLHF doesn't clean them up after saving
    the correct `adapter_model.bin`
- Fix: Disable DeepSpeed checkpoints, enable HuggingFace LoRA-only saves, clean up
  stale full-model shard files after adapter save

## Changes from Attempt 23
1. **Disk space fix** (`train_grpo.sh`):
   - Added `--disable_ds_ckpt` to skip 219 GB ZeRO-3 intermediate checkpoints
   - Added `--save_hf_ckpt` to enable HuggingFace format saves (LoRA-only)
   - Added `--max_ckpt_num 2` to keep at most 2 checkpoints
   - Added `--ckpt_path` explicit path
   - Result: checkpoint size drops from ~200 GB to ~2 GB per save

2. **Save model shard cleanup** (`patches/save_model.patch`):
   - After saving `adapter_model.bin` with correct LoRA weights, clean up
     `pytorch_model-*.bin` and `model-*.safetensors` shard files that
     `PeftModel.save_pretrained()` incorrectly writes with ZeRO-3
   - Also includes the CPU parameter offloading fix from attempt 21

3. **Patch format fix** (all `patches/*.patch` + `apply_patches.sh`):
   - Regenerated patches with git-style relative paths (`a/openrlhf/...`)
   - Changed `apply_patches.sh` from `-p0` to `-p1` for proper path stripping
   - Patches now apply cleanly without "dangerous file name" warnings

4. **Gemini model fallback order** (`reward_func.py`):
   - Updated model list: gemini-3.1-pro-preview -> gemini-3-pro-preview -> gemini-3-flash-preview

## Command
```bash
bash train_grpo.sh > train.log 2>&1
```

Key flags: `--offload`, `--adam_offload`, `--gradient_checkpointing_use_reentrant`,
`--vllm_sync_with_ray`, `--zero_stage 3`, `--vllm_gpu_memory_utilization 0.80`,
`--init_kl_coef 0`, `--disable_ds_ckpt`, `--save_hf_ckpt`, `--max_ckpt_num 2`

## Baseline
- Teacher (Gemini 3 Pro) on AIME eval: 0% (18 problems)
- Teacher on MATH45 eval: 27.8% (18 problems)
- Base student: TBD (will evaluate post-training)

## Result
- Training in progress...
