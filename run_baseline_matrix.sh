#!/bin/bash
# Run baseline eval matrix: 2 models × 3 token limits
# Usage: bash run_baseline_matrix.sh <model_name> <port>

MODEL_NAME=$1  # e.g., "20b" or "120b"
PORT=$2        # e.g., 8000 or 8001
URL="http://localhost:${PORT}/v1"
OUT_DIR="/mnt/scratch/baseline_matrix"
mkdir -p "$OUT_DIR"

echo "=== Baseline matrix for ${MODEL_NAME} on port ${PORT} ==="

for MAX_TOKENS in 20000 65536 131072; do
    OUT="${OUT_DIR}/${MODEL_NAME}_${MAX_TOKENS}.jsonl"
    echo ""
    echo ">>> ${MODEL_NAME} max_tokens=${MAX_TOKENS}"
    echo ">>> Output: ${OUT}"

    python eval_gptoss_aime.py \
        --input data/aime_eval.jsonl \
        --out "$OUT" \
        --vllm-url "$URL" \
        --reasoning-effort high \
        --max-tokens "$MAX_TOKENS" \
        --temperature 1.0 \
        --retries 2

    echo ">>> Done: ${MODEL_NAME} max_tokens=${MAX_TOKENS}"
    echo ""
done

echo "=== All runs complete for ${MODEL_NAME} ==="
