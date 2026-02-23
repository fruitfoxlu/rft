#!/usr/bin/env bash
# Profile run: micro_train_batch_size=2 (vs production micro_train_batch_size=1)
#
# PURPOSE: Check if micro_train_batch_size=2 fits in GPU memory.
#   - 5 global steps (40 gradient steps) — just enough to see peak memory + step time
#   - No checkpointing, no eval (save_steps/eval_steps set to 999999)
#   - Output goes to /mnt/scratch/rft_profile_micro2/
#
# ONLY differences from train_grpo_20b_a29b.sh:
#   1. micro_train_batch_size: 1 → 2
#   2. Training data: 80 prompts (5 global steps worth)
#   3. Output/checkpoint paths → profile dirs
#   4. save_steps/eval_steps → 999999 (skip)
#
# Step count:
#   gradient_steps = 80 * 8 / 16 * 1 = 40
#   grad_per_global = 8 * 16 / 16 = 8
#   global_steps = 40 / 8 = 5

set -euo pipefail

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0

# --- Profile-specific paths ---
ATTEMPT="profile_micro2"
export METRICS_LOG_DIR="/mnt/scratch/rft_profile_micro2"
export SAMPLES_LOG_DIR="/mnt/scratch/rft_profile_micro2/samples"
export SPIKE_LOG_PATH="/mnt/scratch/rft_profile_micro2/spike_log.jsonl"
mkdir -p "$METRICS_LOG_DIR" "$SAMPLES_LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATA="${SCRIPT_DIR}/data/profile_pool_80.jsonl"
EVAL_DATA="${SCRIPT_DIR}/data/probe_set_200_ood.jsonl"
REWARD_FUNC="${SCRIPT_DIR}/reward_func_em.py"
SAVE_PATH="/mnt/scratch/rft_profile_micro2/output"
CKPT_PATH="/mnt/scratch/rft_profile_micro2/checkpoints"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: $TRAIN_DATA not found."
    exit 1
fi

echo "=== GPU Memory Profile — micro_train_batch_size=2 ==="
echo "  Model:       openai/gpt-oss-20b"
echo "  Train data:  $TRAIN_DATA ($(wc -l < "$TRAIN_DATA") prompts = 5 global steps)"
echo "  Reward:      Pure exact-match"
echo "  Save path:   $SAVE_PATH"
echo ""
echo "  ★ micro_train_batch_size=2 (production uses 1)"
echo "  ★ 5 global steps (40 gradient steps), no checkpointing, no eval"
echo "  ★ Watch for OOM or DeepSpeed cache flush warnings"
echo ""

python -m openrlhf.cli.train_ppo_ray \
    --pretrain openai/gpt-oss-20b \
    --seed 42 \
    --prompt_data "$TRAIN_DATA" \
    --input_key input \
    --label_key label \
    --apply_chat_template \
    --remote_rm_url "$REWARD_FUNC" \
    --advantage_estimator dr_grpo \
    --init_kl_coef 0.001 \
    --n_samples_per_prompt 8 \
    --max_epochs 1 \
    --num_episodes 1 \
    --rollout_batch_size 16 \
    --micro_rollout_batch_size 4 \
    --train_batch_size 16 \
    --micro_train_batch_size 2 \
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
    --save_steps 999999 \
    --max_ckpt_num 1 \
    --logging_steps 1 \
    --eval_dataset "$EVAL_DATA" \
    --eval_steps 999999 \
    --eval_temperature 0.0 \
    --eval_n_samples_per_prompt 1 \
    --vllm_sync_with_ray \
    "$@"
