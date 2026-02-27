#!/usr/bin/env bash
# GRPO training for Qwen2.5-14B-Instruct — Attempt Q1
#
# Adapted from train_grpo_20b_a30.sh (gpt-oss-20b) for Qwen2.5-14B.
# Model selection rationale (§11.28-11.29):
#   - OOD-1000 baseline: 67.0% (33% error zone = RL headroom)
#   - ID-200 baseline: 69.5%
#   - 99.2% boxed parsing, 0% parse failures
#   - max_tokens=2048 optimal (no accuracy loss vs 4096)
#
# CHANGES FROM train_grpo_20b_a30.sh:
#   - pretrain: openai/gpt-oss-20b -> Qwen/Qwen2.5-14B-Instruct
#   - generate_max_len: 4096 -> 2048 (§11.29 truncation study)
#   - max_len: 5120 -> 3072 (prompt 1024 + gen 2048)
#   - prompt_data: sft_rl_pool_3200.jsonl -> sft_rl_pool_3200_boxed.jsonl
#   - eval_dataset: probe_set_200_ood.jsonl -> probe_set_200_ood_boxed.jsonl
#   - vllm_num_engines: 2 -> 2 (unchanged; TP=1 crashed — see below)
#   - vllm_tensor_parallel_size: 2 -> 2 (unchanged; TP=1 fails with --vllm_sync_with_ray)
#   NOTE: TP=1 with 4 engines crashed: ray.util.collective.init_collective_group()
#         requires Ray actor context, but TP=1 uses uniproc_executor (plain subprocess).
#         Reverted to TP=2 × 2 engines (proven A30 config). Same 4 GPU allocation.
#   - Output paths: *20b*a30* -> *qwen14b*q01*
#
# UNCHANGED (proven config from A30):
#   - dr_grpo, eps_clip=0.1, init_kl_coef=0.001
#   - LoRA rank=32, alpha=64, targets q_proj k_proj v_proj o_proj
#   - rollout_batch_size=16, train_batch_size=16, micro_train_batch_size=2
#   - n_samples_per_prompt=8, LR=5e-7, warmup_ratio=0.05
#   - colocate_actor_ref, ref_num_nodes=1, ref_num_gpus_per_node=4
#   - 200 global steps, save/eval every 10
#
# Step count (verified by preflight_lr.sh):
#   gradient_steps = pool * n_samples / train_batch * episodes = 3200 * 8 / 16 * 1 = 1600
#   grad_per_global = n_samples * rollout_batch / train_batch = 8 * 16 / 16 = 8
#   global_steps = 1600 / 8 = 200
#
# LR schedule (verified by preflight_lr.sh):
#   warmup: 80 gradient steps (~10 global steps)
#   global step  10: 100% of target (peak)
#   global step  50:  91%
#   global step 100:  59%
#   global step 150:  25%
#   global step 200:  10% (min_lr)
#
# GPU allocation:
#   GPUs 0-3: 2x vLLM TP=2 engines (same as A30; TP=1 crashes with ray sync)
#   GPUs 4-7: ZeRO-3 actor + co-located reference model
#
# Data provenance:
#   sft_rl_pool_3200_boxed.jsonl:  SHA256=e4e4354facc1104c848a4eee1ff87fcfe3a3316f35db2e4ab6a3af52e8bf11b7
#   probe_set_200_ood_boxed.jsonl: SHA256=a45232fc213841b0f1f02f098d89272fb6933c3a724bd3e6247c75cf2739595a
#
# Pre-training guardrails completed:
#   G0: verify_prompt_parity.py — PASS/PENDING
#   G2: eval_model_sweep.py (boxed>=98%, fail=0%) — PASS/PENDING
#   G4: eval_passk_headroom.py (pass@8>=15%) — PASS/PENDING
#
# Config:
#   ┌──────────────────────────┬───────────┬───────────┐
#   │ Parameter                │ A30       │ Q1        │
#   ├──────────────────────────┼───────────┼───────────┤
#   │ pretrain                 │ gpt-20b   │ qwen-14b  │
#   │ generate_max_len         │ 4096      │ 2048 ★    │
#   │ max_len                  │ 5120      │ 3072 ★    │
#   │ prompt_data              │ pool_3200 │ pool_boxed│
#   │ eval_dataset             │ ood_200   │ ood_boxed │
#   │ vllm_num_engines         │ 2         │ 2         │
#   │ vllm_tensor_parallel     │ 2         │ 2         │
#   ├──────────────────────────┼───────────┼───────────┤
#   │ eps_clip                 │ 0.1       │ 0.1       │
#   │ init_kl_coef             │ 0.001     │ 0.001     │
#   │ n_samples_per_prompt     │ 8         │ 8         │
#   │ rollout_batch_size       │ 16        │ 16        │
#   │ train_batch_size         │ 16        │ 16        │
#   │ micro_train_batch_size   │ 2         │ 2         │
#   │ lora_rank / alpha        │ 32 / 64   │ 32 / 64   │
#   │ actor_learning_rate      │ 5e-7      │ 5e-7      │
#   │ lr_warmup_ratio          │ 0.05      │ 0.05      │
#   │ global_steps             │ 200       │ 200       │
#   │ save_steps / eval_steps  │ 10        │ 10        │
#   │ seed                     │ 42        │ 42        │
#   └──────────────────────────┴───────────┴───────────┘

set -euo pipefail

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0

# --- Attempt-specific paths ---
ATTEMPT="q01"
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

echo "=== GRPO Training — Attempt Q1 (Qwen2.5-14B-Instruct) ==="
echo "  Model:       Qwen/Qwen2.5-14B-Instruct"
echo "  Train data:  $TRAIN_DATA ($(wc -l < "$TRAIN_DATA") prompts)"
echo "  Eval data:   $EVAL_DATA (OOD probe: 202 problems, BOXED_SUFFIX)"
echo "  Reward:      Pure exact-match"
echo "  Save path:   $SAVE_PATH"
echo "  Checkpoints: $CKPT_PATH"
echo ""
echo "  ★ Qwen2.5-14B-Instruct (OOD baseline 67.0%, 33% error headroom)"
echo "  ★ generate_max_len=2048 (no accuracy loss vs 4096)"
echo "  ★ 2x vLLM engines TP=2 (TP=1 crashes with ray sync; same GPU alloc)"
echo "  ★ BOXED_SUFFIX appended to all prompts"
echo "  ★ 200 global steps (1600 gradient steps), LR peaks at global step ~10"
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
    --save_steps 10 \
    --max_ckpt_num 20 \
    --logging_steps 1 \
    --eval_dataset "$EVAL_DATA" \
    --eval_steps 10 \
    --eval_temperature 0.0 \
    --eval_n_samples_per_prompt 1 \
    --vllm_sync_with_ray \
    "$@"
