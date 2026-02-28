#!/usr/bin/env bash
# GRPO training for Qwen2.5-14B-Instruct — Track C1 (Judge reward, EM=1 bonus)
#
# Track C1: LLM-as-judge quality differentiation among correct solutions.
# Changed from Q1: reward function adds judge bonus on EM=1 responses.
#
# Reward formula:
#   reward = EM × (1 + α × judge_score)
#   When EM=1: reward = 1 + α × judge_score ∈ [1, 1+α]
#   When EM=0: reward = 0 (unchanged)
#
# Judge: Qwen2.5-14B-Instruct (base model) served on GPU 2, port 8001
# α = 0.5 (configurable via JUDGE_ALPHA env var)
#
# GPU allocation:
#   GPUs 0-1: 1x vLLM TP=2 engine (rollouts)
#   GPU 2: 1x judge server TP=1 (Qwen2.5-14B-Instruct, persistent)
#   GPU 3: unused
#   GPUs 4-7: ZeRO-3 actor + co-located reference model
#
# Config delta from Q1:
#   ┌──────────────────────────┬───────────────────────────┬───────────────────────────┐
#   │ Parameter                │ Q1                        │ C1                        │
#   ├──────────────────────────┼───────────────────────────┼───────────────────────────┤
#   │ reward function          │ reward_func_em.py         │ reward_func_judge_c1.py ★ │
#   │ vllm_num_engines         │ 2                         │ 1 ★                       │
#   │ JUDGE_ALPHA              │ N/A                       │ 0.5 ★                     │
#   │ (all others)             │ same                      │ same                      │
#   └──────────────────────────┴───────────────────────────┴───────────────────────────┘

set -euo pipefail

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0

# ── Judge configuration ────────────────────────────────────────────────
export JUDGE_ALPHA="0.5"
export JUDGE_BASE_URL="http://localhost:8001/v1"
export JUDGE_MODEL="Qwen/Qwen2.5-14B-Instruct"
JUDGE_PORT=8001
JUDGE_GPU="2"

# --- Attempt-specific paths ---
ATTEMPT="c1"
export METRICS_LOG_DIR="/mnt/scratch/rft_metrics_qwen14b_${ATTEMPT}"
export SAMPLES_LOG_DIR="/mnt/scratch/rft_samples_qwen14b_${ATTEMPT}"
export SPIKE_LOG_PATH="/mnt/scratch/rft_metrics_qwen14b_${ATTEMPT}/spike_log.jsonl"
mkdir -p "$METRICS_LOG_DIR" "$SAMPLES_LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATA="${SCRIPT_DIR}/data/sft_rl_pool_3200_boxed.jsonl"
EVAL_DATA="${SCRIPT_DIR}/data/probe_set_200_ood_boxed.jsonl"
REWARD_FUNC="${SCRIPT_DIR}/reward_func_judge_c1.py"
SAVE_PATH="/mnt/data/rft_output/qwen14b-grpo-${ATTEMPT}"
CKPT_PATH="/mnt/data/rft_checkpoints/qwen14b-grpo-${ATTEMPT}"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: $TRAIN_DATA not found."
    exit 1
fi

if [ ! -f "$EVAL_DATA" ]; then
    echo "Error: $EVAL_DATA not found."
    exit 1
fi

# ── Start judge server ─────────────────────────────────────────────────
echo "=== Starting judge server ==="
echo "  Model: $JUDGE_MODEL"
echo "  GPU: $JUDGE_GPU"
echo "  Port: $JUDGE_PORT"

CUDA_VISIBLE_DEVICES="$JUDGE_GPU" python -m vllm.entrypoints.openai.api_server \
    --model "$JUDGE_MODEL" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --port "$JUDGE_PORT" \
    --trust-remote-code \
    > "$METRICS_LOG_DIR/judge_server.log" 2>&1 &
JUDGE_PID=$!
echo "  Judge PID: $JUDGE_PID"

echo "  Waiting for judge server..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${JUDGE_PORT}/v1/models" | grep -q "id"; then
        echo "  Judge server ready!"
        break
    fi
    if ! kill -0 "$JUDGE_PID" 2>/dev/null; then
        echo "ERROR: Judge server died. Check $METRICS_LOG_DIR/judge_server.log"
        exit 1
    fi
    sleep 5
done

echo "  Testing judge..."
JUDGE_TEST=$(curl -s "http://localhost:${JUDGE_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$JUDGE_MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Say 'ready'\"}], \"max_tokens\": 5, \"temperature\": 0}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
echo "  Judge test response: $JUDGE_TEST"

cleanup() {
    echo "Stopping judge server (PID $JUDGE_PID)..."
    kill "$JUDGE_PID" 2>/dev/null
    wait "$JUDGE_PID" 2>/dev/null
}
trap cleanup EXIT

# --- Pre-flight LR check ---
echo ""
echo "=== Pre-flight LR schedule check ==="
POOL=$(wc -l < "$TRAIN_DATA") NS=8 TBS=16 RBS=16 EP=1 WARMUP=0.05 LR=5e-7 \
    bash "${SCRIPT_DIR}/preflight_lr.sh" || { echo "ABORT: LR schedule check failed."; exit 1; }
echo ""

echo "=== GRPO Training — Track C1: Judge reward α=$JUDGE_ALPHA (Qwen2.5-14B-Instruct) ==="
echo "  Model:       Qwen/Qwen2.5-14B-Instruct"
echo "  Train data:  $TRAIN_DATA ($(wc -l < "$TRAIN_DATA") prompts)"
echo "  Eval data:   $EVAL_DATA (OOD probe: 202 problems, BOXED_SUFFIX)"
echo "  Reward:      EM×(1+α×judge), α=$JUDGE_ALPHA"
echo "  Judge:       $JUDGE_MODEL on GPU $JUDGE_GPU port $JUDGE_PORT"
echo "  Save path:   $SAVE_PATH"
echo "  Checkpoints: $CKPT_PATH"
echo ""
echo "  ★ Track C1: Judge bonus on correct solutions (quality differentiation)"
echo "  ★ EM=1 reward range: [1.0, 1.5], EM=0 reward: 0.0"
echo "  ★ 1 vLLM engine (freed GPU for judge), 200 global steps"
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
    --vllm_num_engines 1 \
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
