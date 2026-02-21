#!/usr/bin/env bash
# Dry-run for attempt-27: 1 episode to verify ratio tail logging + spike logging
set -euo pipefail

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0
export METRICS_LOG_DIR="/mnt/scratch/rft_metrics_20b_a27_dryrun"
export SAMPLES_LOG_DIR="/mnt/scratch/rft_samples_20b_a27_dryrun"
export SPIKE_LOG_PATH="/mnt/scratch/rft_metrics_20b_a27_dryrun/spike_log.jsonl"
mkdir -p "$METRICS_LOG_DIR" "$SAMPLES_LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python -m openrlhf.cli.train_ppo_ray \
    --pretrain openai/gpt-oss-20b \
    --prompt_data "${SCRIPT_DIR}/data/sft_rl_pool.jsonl" \
    --input_key input \
    --label_key label \
    --apply_chat_template \
    --remote_rm_url "${SCRIPT_DIR}/reward_func_em.py" \
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
    --save_path "/mnt/data/rft_output/gpt-oss-20b-grpo-a27-dryrun" \
    --ckpt_path "/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a27-dryrun" \
    --disable_ds_ckpt \
    --save_hf_ckpt \
    --save_steps 999 \
    --max_ckpt_num 1 \
    --logging_steps 1 \
    --eval_dataset "${SCRIPT_DIR}/data/probe_set_200_ood.jsonl" \
    --eval_steps 999 \
    --eval_temperature 0.0 \
    --eval_n_samples_per_prompt 1 \
    --vllm_sync_with_ray
