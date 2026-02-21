#!/usr/bin/env bash
# GRPO training for gpt-oss-20b — Attempt 27
#
# SINGLE TRAINING CHANGE from Attempt 26:
#   eps_clip: 0.2 → 0.1  (tighter PPO ratio clipping to reduce gradient spikes)
#
# INSTRUMENTATION CHANGES (do not affect training dynamics):
#   - eval_dataset: AIME (18) → OOD probe (202 MATH+Apex, disjoint from training)
#   - eval_n_samples_per_prompt: 8 → 1  (greedy_pass@1 instead of nondet_proxy_pass@8)
#   - eval_steps: 10 → 5  (more frequent eval for early stopping)
#   - save_steps: 10 → 5  (match eval frequency)
#   - max_ckpt_num: 3 → 8  (keep more checkpoints with more frequent saves)
#   - num_episodes: 20 → 5  (cap ~40 steps; attempt-26 peaked at step 30)
#   - Separate output dirs to preserve attempt-26 checkpoints
#   - SPIKE_LOG_PATH set for spike sample logging
#
# IDENTICAL to Attempt 26:
#   LR=5e-7, micro_train_batch_size=1, n_samples_per_prompt=8,
#   generate_max_len=4096, lora_rank=32, lora_alpha=64,
#   init_kl_coef=0.001, rollout_batch_size=16, train_batch_size=16,
#   zero_stage=3, max_len=5120, prompt_max_len=1024
#
# PATCHES TO APPLY BEFORE RUNNING (bash apply_patches.sh):
#   - ratio_logging_loss.patch: log-ratio clamping + ratio tail stats
#   - ppo_actor.patch: ratio stat logging + spike prompt hash dumping
#
# Side-by-side config comparison:
#   ┌──────────────────────────┬────────────┬────────────┐
#   │ Parameter                │ Attempt-26 │ Attempt-27 │
#   ├──────────────────────────┼────────────┼────────────┤
#   │ eps_clip                 │ 0.2        │ 0.1 ★      │
#   │ actor_learning_rate      │ 5e-7       │ 5e-7       │
#   │ micro_train_batch_size   │ 1          │ 1          │
#   │ n_samples_per_prompt     │ 8          │ 8          │
#   │ generate_max_len         │ 4096       │ 4096       │
#   │ rollout_batch_size       │ 16         │ 16         │
#   │ train_batch_size         │ 16         │ 16         │
#   │ lora_rank / alpha        │ 32 / 64    │ 32 / 64    │
#   │ init_kl_coef             │ 0.001      │ 0.001      │
#   │ max_norm (grad clip)     │ 1.0        │ 1.0        │
#   │ lr_warmup_ratio          │ 0.05       │ 0.05       │
#   ├──────────────────────────┼────────────┼────────────┤
#   │ eval_dataset             │ AIME (18)  │ OOD (202)  │
#   │ eval metric              │ ndet_p@8   │ greedy_p@1 │
#   │ eval_steps               │ 10         │ 5          │
#   │ save_steps               │ 10         │ 5          │
#   │ num_episodes             │ 20         │ 5          │
#   └──────────────────────────┴────────────┴────────────┘

set -euo pipefail

# Ensure Python output is unbuffered for real-time log monitoring
export PYTHONUNBUFFERED=1

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0

# Metrics logging (separate dirs from attempt-26)
export METRICS_LOG_DIR="/mnt/scratch/rft_metrics_20b_a27"
export SAMPLES_LOG_DIR="/mnt/scratch/rft_samples_20b_a27"
export SPIKE_LOG_PATH="/mnt/scratch/rft_metrics_20b_a27/spike_log.jsonl"
mkdir -p "$METRICS_LOG_DIR" "$SAMPLES_LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATA="${SCRIPT_DIR}/data/sft_rl_pool.jsonl"
EVAL_DATA="${SCRIPT_DIR}/data/probe_set_200_ood.jsonl"
REWARD_FUNC="${SCRIPT_DIR}/reward_func_em.py"
SAVE_PATH="/mnt/data/rft_output/gpt-oss-20b-grpo-a27"
CKPT_PATH="/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a27"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: $TRAIN_DATA not found."
    echo "Run: python prepare_sft_data.py"
    exit 1
fi

echo "=== GRPO Training — Attempt 27 (gpt-oss-20b) ==="
echo "  Model:       openai/gpt-oss-20b"
echo "  Train data:  $TRAIN_DATA ($(wc -l < "$TRAIN_DATA") prompts)"
echo "  Eval data:   $EVAL_DATA (OOD probe: 170 MATH + 32 Apex)"
echo "  Reward:      Pure exact-match (no Gemini judge)"
echo "  Save path:   $SAVE_PATH"
echo "  Checkpoints: $CKPT_PATH"
echo ""
echo "  ★ CHANGE: eps_clip 0.2 → 0.1"
echo "  ★ EVAL:   greedy_pass@1 on OOD probe every 5 steps"
echo ""

python -m openrlhf.cli.train_ppo_ray \
    --pretrain openai/gpt-oss-20b \
    --prompt_data "$TRAIN_DATA" \
    --input_key input \
    --label_key label \
    --apply_chat_template \
    --remote_rm_url "$REWARD_FUNC" \
    --advantage_estimator dr_grpo \
    --init_kl_coef 0.001 \
    --n_samples_per_prompt 8 \
    --max_epochs 1 \
    --num_episodes 5 \
    --rollout_batch_size 16 \
    --micro_rollout_batch_size 4 \
    --train_batch_size 16 \
    --micro_train_batch_size 1 \
    --max_len 5120 \
    --prompt_max_len 1024 \
    --generate_max_len 4096 \
    --eps_clip 0.1 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --colocate_actor_ref \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 4 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.85 \
    --zero_stage 3 \
    --gradient_checkpointing \
    --gradient_checkpointing_use_reentrant \
    --param_dtype bf16 \
    --actor_learning_rate 5e-7 \
    --lr_warmup_ratio 0.05 \
    --entropy_loss_coef 0 \
    --save_path "$SAVE_PATH" \
    --ckpt_path "$CKPT_PATH" \
    --disable_ds_ckpt \
    --save_hf_ckpt \
    --save_steps 5 \
    --max_ckpt_num 8 \
    --logging_steps 1 \
    --eval_dataset "$EVAL_DATA" \
    --eval_steps 5 \
    --eval_temperature 0.0 \
    --eval_n_samples_per_prompt 1 \
    --vllm_sync_with_ray \
    "$@"
