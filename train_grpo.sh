#!/usr/bin/env bash
# GRPO training for gpt-oss-120b on math with Gemini judge reward.
#
# Prerequisites:
#   1. pip install google-genai kernels
#   2. python prepare_data.py
#   3. python eval_baseline.py --auto-symlink
#   4. data/train_prompts.jsonl exists (symlinked by eval_baseline.py)
#
# GPU allocation (8 GPUs total):
#   - 4 GPUs for actor (LoRA training via DeepSpeed ZeRO-3 + CPU param offload)
#   - 4 GPUs for vLLM rollout engines (2 engines x 2 TP, MXFP4 inference)
#
# Memory strategy: The dequantized model is ~240GB bf16. ZeRO-3 partitions
# across 4 actor GPUs (60GB each) but that leaves insufficient GPU headroom.
# Enabling CPU parameter offloading keeps params on CPU and brings one layer
# at a time to GPU during compute (~6GB per layer), solving GPU OOM.
#
# Notes:
#   - The model's MXFP4 weights are dequantized to bf16 for the actor (training).
#   - vLLM uses native MXFP4 inference kernels for fast rollouts.
#   - No reference model: saves GPU memory. GRPO dr_grpo doesn't strictly need KL.
#   - If KL is desired later, add: --use_kl_loss --init_kl_coef 0.01
#     (requires additional GPUs or CPU offload for reference model)
#
# Usage:
#   bash train_grpo.sh
#   bash train_grpo.sh --num_episodes 100  # override any arg

set -euo pipefail

# Use Vertex AI project auth (not API key) for Gemini reward function
unset GOOGLE_API_KEY GEMINI_API_KEY 2>/dev/null || true

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATA="${SCRIPT_DIR}/data/train_prompts.jsonl"
REWARD_FUNC="${SCRIPT_DIR}/reward_func.py"
SAVE_PATH="${SCRIPT_DIR}/output/gpt-oss-120b-grpo"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: $TRAIN_DATA not found."
    echo "Run: python prepare_data.py && python eval_baseline.py --auto-symlink"
    exit 1
fi

echo "=== GRPO Training ==="
echo "  Model:       openai/gpt-oss-120b"
echo "  Train data:  $TRAIN_DATA"
echo "  Reward func: $REWARD_FUNC"
echo "  Save path:   $SAVE_PATH"
echo "  Train size:  $(wc -l < "$TRAIN_DATA") prompts"
echo ""

python -m openrlhf.cli.train_ppo_ray \
    --pretrain openai/gpt-oss-120b \
    --prompt_data "$TRAIN_DATA" \
    --input_key input \
    --label_key label \
    --apply_chat_template \
    --remote_rm_url "$REWARD_FUNC" \
    --advantage_estimator dr_grpo \
    --init_kl_coef 0 \
    --n_samples_per_prompt 8 \
    --max_epochs 1 \
    --num_episodes 50 \
    --rollout_batch_size 8 \
    --micro_rollout_batch_size 2 \
    --train_batch_size 8 \
    --micro_train_batch_size 1 \
    --max_len 4096 \
    --prompt_max_len 1024 \
    --generate_max_len 3072 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --target_modules q_proj k_proj v_proj o_proj \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 4 \
    --vllm_num_engines 2 \
    --vllm_tensor_parallel_size 2 \
    --vllm_gpu_memory_utilization 0.80 \
    --zero_stage 3 \
    --offload \
    --adam_offload \
    --gradient_checkpointing \
    --gradient_checkpointing_use_reentrant \
    --gradient_checkpointing_use_reentrant \
    --param_dtype bf16 \
    --actor_learning_rate 1e-6 \
    --save_path "$SAVE_PATH" \
    --save_steps 10 \
    --logging_steps 1 \
    --vllm_sync_with_ray \
    "$@"
