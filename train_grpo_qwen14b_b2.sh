#!/usr/bin/env bash
# GRPO training for Qwen2.5-14B-Instruct — Track B2 (eps_clip=0.2)
#
# Track B2: Relaxed clipping experiment (§14, Track B)
# Changed from Q1: eps_clip 0.1 → 0.2
# Everything else identical to Q1.
#
# Hypothesis: D1 showed the policy IS moving but changes are small and
#   random. Larger clipping range allows bigger policy updates when
#   gradient signal exists, potentially escaping local basins.
#
# CHANGES FROM Q1 (train_grpo_qwen14b_q01.sh):
#   - eps_clip: 0.1 → 0.2  ★ (THE one changed variable)
#   - Output paths: *q01* → *b2*
#
# Risk: Training instability if updates too large. Monitor ratio_max
#   and KL carefully.
#
# UNCHANGED from Q1:
#   - pretrain: Qwen/Qwen2.5-14B-Instruct
#   - dr_grpo, init_kl_coef=0.001
#   - LoRA rank=32, alpha=64, targets q_proj k_proj v_proj o_proj
#   - n_samples_per_prompt=8
#   - rollout_batch_size=16, train_batch_size=16, micro_train_batch_size=2
#   - micro_rollout_batch_size=4
#   - LR=5e-7, warmup_ratio=0.05
#   - colocate_actor_ref
#   - 200 global steps, save/eval every 10
#   - generate_max_len=2048, max_len=3072
#   - 2x vLLM TP=2 engines
#
# Config delta from Q1:
#   ┌──────────────────────────┬───────────┬───────────┐
#   │ Parameter                │ Q1        │ B2        │
#   ├──────────────────────────┼───────────┼───────────┤
#   │ eps_clip                 │ 0.1       │ 0.2 ★     │
#   │ (all others)             │ same      │ same      │
#   └──────────────────────────┴───────────┴───────────┘

set -euo pipefail

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0

# --- Attempt-specific paths ---
ATTEMPT="b2"
export METRICS_LOG_DIR="/mnt/scratch/rft_metrics_qwen14b_${ATTEMPT}"
export SAMPLES_LOG_DIR="/mnt/scratch/rft_samples_qwen14b_${ATTEMPT}"
export SPIKE_LOG_PATH="/mnt/scratch/rft_metrics_qwen14b_${ATTEMPT}/spike_log.jsonl"
mkdir -p "$METRICS_LOG_DIR" "$SAMPLES_LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATA="${SCRIPT_DIR}/data/sft_rl_pool_3200_boxed.jsonl"
EVAL_DATA="${SCRIPT_DIR}/data/probe_set_200_ood_boxed.jsonl"
REWARD_FUNC="${SCRIPT_DIR}/reward_func_em.py"
SAVE_PATH="/mnt/data/rft_output/qwen14b-grpo-${ATTEMPT}"
CKPT_PATH="/mnt/data/rft_checkpoints/qwen14b-grpo-${ATTEMPT}"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: $TRAIN_DATA not found."
    echo "Run prepare_rl_data_qwen.py first."
    exit 1
fi

if [ ! -f "$EVAL_DATA" ]; then
    echo "Error: $EVAL_DATA not found."
    echo "Run prepare_rl_data_qwen.py first."
    exit 1
fi

# --- Pre-flight LR check ---
echo "=== Pre-flight LR schedule check ==="
POOL=$(wc -l < "$TRAIN_DATA") NS=8 TBS=16 RBS=16 EP=1 WARMUP=0.05 LR=5e-7 \
    bash "${SCRIPT_DIR}/preflight_lr.sh" || { echo "ABORT: LR schedule check failed."; exit 1; }
echo ""

echo "=== GRPO Training — Track B2: eps_clip=0.2 (Qwen2.5-14B-Instruct) ==="
echo "  Model:       Qwen/Qwen2.5-14B-Instruct"
echo "  Train data:  $TRAIN_DATA ($(wc -l < "$TRAIN_DATA") prompts)"
echo "  Eval data:   $EVAL_DATA (OOD probe: 202 problems, BOXED_SUFFIX)"
echo "  Reward:      Pure exact-match"
echo "  Save path:   $SAVE_PATH"
echo "  Checkpoints: $CKPT_PATH"
echo ""
echo "  ★ Track B2: eps_clip=0.2 (was 0.1 in Q1)"
echo "  ★ Qwen2.5-14B-Instruct (OOD baseline 67.0%, 33% error headroom)"
echo "  ★ 200 global steps, same speed as Q1"
echo "  ★ Eval: greedy_pass@1 on OOD probe every 10 global steps"
echo ""

python -m openrlhf.cli.train_ppo_ray \
    --pretrain Qwen/Qwen2.5-14B-Instruct \
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
    --max_len 3072 \
    --prompt_max_len 1024 \
    --generate_max_len 2048 \
    --eps_clip 0.2 \
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
    --save_steps 10 \
    --max_ckpt_num 20 \
    --logging_steps 1 \
    --eval_dataset "$EVAL_DATA" \
    --eval_steps 10 \
    --eval_temperature 0.0 \
    --eval_n_samples_per_prompt 1 \
    --vllm_sync_with_ray \
    "$@"
