#!/usr/bin/env bash
# SFT training for Qwen2.5-14B-Instruct — Track A (RSFT)
#
# Fine-tunes on rejection-sampled correct solutions from the baseline model.
# Uses OpenRLHF's SFT trainer with DeepSpeed ZeRO-2 + LoRA.
#
# Same LoRA config as Q1 GRPO for direct comparison:
#   - rank=32, alpha=64, targets: q_proj k_proj v_proj o_proj
#
# Training data: /mnt/scratch/qwen14b_rsft/sft_train.jsonl
#   Format: {input: "problem + BOXED_SUFFIX", output: "correct solution", ...}
#
# Usage:
#   bash train_sft_qwen14b_rsft.sh
#
set -euo pipefail

# ── Paths ──────────────────────────────────────────────────────────────
MODEL="Qwen/Qwen2.5-14B-Instruct"
DATASET="/mnt/scratch/qwen14b_rsft/sft_train.jsonl"
SAVE_DIR="/mnt/data/rft_checkpoints/qwen14b-rsft-a01"
LOG_DIR="/mnt/scratch/qwen14b_rsft"

mkdir -p "$SAVE_DIR" "$LOG_DIR"

# ── Training ───────────────────────────────────────────────────────────
deepspeed --num_gpus 8 \
    --module openrlhf.cli.train_sft \
    --pretrain "$MODEL" \
    --dataset "$DATASET" \
    --input_key input \
    --output_key output \
    --apply_chat_template \
    --save_path "$SAVE_DIR" \
    --save_steps 200 \
    --logging_steps 10 \
    --max_len 3072 \
    --max_epochs 3 \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --learning_rate 2e-5 \
    --lr_scheduler cosine \
    --lr_warmup_ratio 0.05 \
    --max_norm 1.0 \
    --gradient_checkpointing \
    --zero_stage 2 \
    --param_dtype bf16 \
    --seed 42 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj \
    --lora_dropout 0.0 \
    --packing_samples \
    --save_hf_ckpt \
    --trust_remote_code \
    2>&1 | tee "$LOG_DIR/sft_train.log"

echo ""
echo "=== SFT Training Complete ==="
echo "Checkpoints: $SAVE_DIR"
echo "Log: $LOG_DIR/sft_train.log"
