#!/usr/bin/env bash
# GRPO training for gpt-oss-20b — Attempt 28A (eps_clip=0.2 baseline)
#
# A/B EXPERIMENT: eps_clip comparison with correct LR schedule.
#   28A: eps_clip=0.2 (OpenRLHF default)
#   28B: eps_clip=0.1 (tighter clipping)
#
# CHANGES FROM A26/A27:
#   - pool: 50K → 400 prompts (sft_rl_pool_400.jsonl, seed=42)
#   - num_episodes: 5 → 1 (yields 200 steps with 400 prompts)
#   - LR warmup reaches 100% by step 10 (vs never reaching meaningful LR in A26/A27)
#   - Seed fixed (--seed 42) for reproducibility
#   - Updated spike logging (trigger on per-micro-batch ratio/loss, not stale grad_norm)
#
# LR schedule (verified by preflight_lr.sh):
#   step 1:  10%, step 10: 100% (peak), step 50: 91%, step 100: 59%, step 200: 10% (min_lr)
#
# Pool subset provenance:
#   sft_rl_pool_400.jsonl: 400 prompts, seed=42, SHA256=1d09bb26d12774ea...
#   No overlap with OOD probe (202 problems) or AIME eval (18 problems)
#
# Config:
#   ┌──────────────────────────┬───────────┬───────────┐
#   │ Parameter                │ 28A       │ 28B       │
#   ├──────────────────────────┼───────────┼───────────┤
#   │ eps_clip                 │ 0.2 ★     │ 0.1 ★     │
#   ├──────────────────────────┼───────────┼───────────┤
#   │ prompt_data              │ pool_400  │ pool_400  │
#   │ num_episodes             │ 1         │ 1         │
#   │ seed                     │ 42        │ 42        │
#   │ actor_learning_rate      │ 5e-7      │ 5e-7      │
#   │ micro_train_batch_size   │ 1         │ 1         │
#   │ n_samples_per_prompt     │ 8         │ 8         │
#   │ generate_max_len         │ 4096      │ 4096      │
#   │ rollout_batch_size       │ 16        │ 16        │
#   │ train_batch_size         │ 16        │ 16        │
#   │ lora_rank / alpha        │ 32 / 64   │ 32 / 64   │
#   │ init_kl_coef             │ 0.001     │ 0.001     │
#   │ lr_warmup_ratio          │ 0.05      │ 0.05      │
#   │ eval_dataset             │ OOD (202) │ OOD (202) │
#   │ eval metric              │ greedy@1  │ greedy@1  │
#   │ eval_steps               │ 10        │ 10        │
#   │ save_steps               │ 10        │ 10        │
#   └──────────────────────────┴───────────┴───────────┘

set -euo pipefail

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0

# --- Attempt-specific paths ---
ATTEMPT="a28a"
export METRICS_LOG_DIR="/mnt/scratch/rft_metrics_20b_${ATTEMPT}"
export SAMPLES_LOG_DIR="/mnt/scratch/rft_samples_20b_${ATTEMPT}"
export SPIKE_LOG_PATH="/mnt/scratch/rft_metrics_20b_${ATTEMPT}/spike_log.jsonl"
mkdir -p "$METRICS_LOG_DIR" "$SAMPLES_LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATA="${SCRIPT_DIR}/data/sft_rl_pool_400.jsonl"
EVAL_DATA="${SCRIPT_DIR}/data/probe_set_200_ood.jsonl"
REWARD_FUNC="${SCRIPT_DIR}/reward_func_em.py"
SAVE_PATH="/mnt/data/rft_output/gpt-oss-20b-grpo-${ATTEMPT}"
CKPT_PATH="/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-${ATTEMPT}"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: $TRAIN_DATA not found."
    exit 1
fi

# --- Pre-flight LR check ---
echo "=== Pre-flight LR schedule check ==="
POOL=$(wc -l < "$TRAIN_DATA") NS=8 TBS=16 EP=1 WARMUP=0.05 LR=5e-7 \
    bash "${SCRIPT_DIR}/preflight_lr.sh" || { echo "ABORT: LR schedule check failed."; exit 1; }
echo ""

echo "=== GRPO Training — Attempt 28A (gpt-oss-20b, eps_clip=0.2) ==="
echo "  Model:       openai/gpt-oss-20b"
echo "  Train data:  $TRAIN_DATA ($(wc -l < "$TRAIN_DATA") prompts)"
echo "  Eval data:   $EVAL_DATA (OOD probe: 170 MATH + 32 Apex)"
echo "  Reward:      Pure exact-match"
echo "  Save path:   $SAVE_PATH"
echo "  Checkpoints: $CKPT_PATH"
echo ""
echo "  ★ eps_clip=0.2 (baseline)"
echo "  ★ 200 steps, LR peaks at step 10"
echo "  ★ Eval: greedy_pass@1 on OOD probe every 10 steps"
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
    --micro_train_batch_size 1 \
    --max_len 5120 \
    --prompt_max_len 1024 \
    --generate_max_len 4096 \
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
