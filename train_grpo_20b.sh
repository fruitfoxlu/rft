#!/usr/bin/env bash
# GRPO training for gpt-oss-20b on NuminaMath with pure exact-match reward.
#
# Key differences from 120b:
#   - Model: gpt-oss-20b (21B total, 3.6B active MoE, ~11GB MXFP4, ~84GB bf16)
#   - Data: NuminaMath-1.5 RL pool (50k integer-only problems)
#   - Reward: Pure exact-match (no Gemini judge)
#   - GPU: 4 for actor (21GB/GPU, no offload needed), 4 for vLLM
#   - LoRA: r=32, alpha=64 (attention-only, MoE-compatible)
#
# GPU allocation (8 GPUs total):
#   - 4 GPUs for actor (LoRA training via DeepSpeed ZeRO-3, NO CPU offload)
#   - 4 GPUs for vLLM rollout (2 engines × TP=2, MXFP4 inference)
#
# Memory: 84GB bf16 / 4 GPUs = 21GB/GPU for params, leaving ~59GB for
# activations + optimizer states. Should fit without CPU offload.
#
# Usage:
#   bash train_grpo_20b.sh
#   bash train_grpo_20b.sh --num_episodes 100  # override any arg

set -euo pipefail

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0

# Metrics logging
export METRICS_LOG_DIR="/mnt/scratch/rft_metrics_20b"
export SAMPLES_LOG_DIR="/mnt/scratch/rft_samples_20b"
mkdir -p "$METRICS_LOG_DIR" "$SAMPLES_LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATA="${SCRIPT_DIR}/data/sft_rl_pool.jsonl"
EVAL_DATA="${SCRIPT_DIR}/data/aime_eval.jsonl"
REWARD_FUNC="${SCRIPT_DIR}/reward_func_em.py"
SAVE_PATH="/mnt/data/rft_output/gpt-oss-20b-grpo"
CKPT_PATH="/mnt/data/rft_checkpoints/gpt-oss-20b-grpo"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: $TRAIN_DATA not found."
    echo "Run: python prepare_sft_data.py"
    exit 1
fi

echo "=== GRPO Training (gpt-oss-20b) ==="
echo "  Model:       openai/gpt-oss-20b"
echo "  Train data:  $TRAIN_DATA ($(wc -l < "$TRAIN_DATA") prompts)"
echo "  Eval data:   $EVAL_DATA"
echo "  Reward:      Pure exact-match (no Gemini judge)"
echo "  Save path:   $SAVE_PATH"
echo "  Checkpoints: $CKPT_PATH"
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
    --num_episodes 20 \
    --rollout_batch_size 16 \
    --micro_rollout_batch_size 4 \
    --train_batch_size 16 \
    --micro_train_batch_size 1 \
    --max_len 5120 \
    --prompt_max_len 1024 \
    --generate_max_len 4096 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
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
    --save_steps 10 \
    --max_ckpt_num 3 \
    --logging_steps 1 \
    --eval_dataset "$EVAL_DATA" \
    --eval_steps 10 \
    --eval_temperature 0.0 \
    --eval_n_samples_per_prompt 8 \
    --vllm_sync_with_ray \
    "$@"
