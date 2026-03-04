#!/usr/bin/env bash
# GRPO training for Qwen2.5-14B-Instruct — Track C2 (Judge reward, EM=0 bonus)
#
# Track C2: LLM-as-judge process reward (§14, Track C)
# Changed from Q1:
#   - Reward function: reward_func_em.py → reward_func_judge_c2.py
#
# Hypothesis: Binary EM reward fails at credit assignment because it scores
#   the entire response as 0 or 1. C2 gives partial credit to wrong answers
#   based on reasoning quality, providing gradient signal on all-wrong groups.
#
# Reward formula:
#   reward = EM + α × judge_score × (1 - EM)
#   When EM=1: reward = 1.0 (correctness always dominates)
#   When EM=0: reward = α × judge_score ∈ [0, α]
#   Critical invariant: α < 1.0 ensures max(wrong) < min(correct)
#
# Judge: Gemini 2.5 Pro via Google API (external, no GPU needed)
# α = 0.3 (configurable via JUDGE_ALPHA env var)
#
# GPU allocation (same as Q1/B1/B2 — no local judge needed):
#   GPUs 0-3: 2x vLLM TP=2 engines (rollouts)
#   GPUs 4-7: ZeRO-3 actor + co-located reference model
#
# Config delta from Q1:
#   ┌──────────────────────────┬───────────────────────────┬───────────────────────────┐
#   │ Parameter                │ Q1                        │ C2                        │
#   ├──────────────────────────┼───────────────────────────┼───────────────────────────┤
#   │ reward function          │ reward_func_em.py         │ reward_func_judge_c2.py ★ │
#   │ judge                    │ N/A                       │ Gemini 2.5 Pro (API) ★    │
#   │ JUDGE_ALPHA              │ N/A                       │ 0.3 ★                     │
#   │ (all others)             │ same                      │ same                      │
#   └──────────────────────────┴───────────────────────────┴───────────────────────────┘

set -euo pipefail

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_CUMEM_ENABLE=0

# ── Judge configuration (Gemini API — no local GPU needed) ────────────
export JUDGE_ALPHA="0.3"
# Uses Vertex AI with gcloud application-default credentials (project=wf30-poc)

# --- Attempt-specific paths ---
ATTEMPT="c2"
export METRICS_LOG_DIR="/mnt/scratch/rft_metrics_qwen14b_${ATTEMPT}"
export SAMPLES_LOG_DIR="/mnt/scratch/rft_samples_qwen14b_${ATTEMPT}"
export SPIKE_LOG_PATH="/mnt/scratch/rft_metrics_qwen14b_${ATTEMPT}/spike_log.jsonl"
mkdir -p "$METRICS_LOG_DIR" "$SAMPLES_LOG_DIR"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DATA="${SCRIPT_DIR}/data/sft_rl_pool_3200_boxed.jsonl"
EVAL_DATA="${SCRIPT_DIR}/data/probe_set_200_ood_boxed.jsonl"
REWARD_FUNC="${SCRIPT_DIR}/reward_func_judge_c2.py"
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

# ── Verify Gemini 3.1 Pro access (google-genai SDK) ─────────────────
echo "=== Testing Gemini 3.1 Pro (genai SDK, project=${GCP_PROJECT:-wf30-poc}) ==="
JUDGE_OK=0
for _try in 1 2 3; do
    if JUDGE_TEST=$(python3 -c "
from google import genai
client = genai.Client(vertexai=True, project='wf30-poc', location='global')
resp = client.models.generate_content(model='gemini-3.1-pro-preview', contents='Say ready')
print(resp.text.strip())
" 2>&1); then
        JUDGE_OK=1
        break
    fi
    echo "  Attempt $_try failed, retrying in 5s..."
    sleep 5
done
echo "  Gemini test: $JUDGE_TEST"
if [ "$JUDGE_OK" -ne 1 ]; then
    echo "ERROR: Gemini API preflight failed after 3 attempts."
    exit 1
fi
if [ -z "$JUDGE_TEST" ]; then
    echo "WARNING: Gemini API test succeeded but returned empty output."
fi

# --- Pre-flight LR check ---
echo ""
echo "=== Pre-flight LR schedule check ==="
POOL=$(wc -l < "$TRAIN_DATA") NS=8 TBS=16 RBS=16 EP=1 WARMUP=0.05 LR=5e-7 \
    bash "${SCRIPT_DIR}/preflight_lr.sh" || { echo "ABORT: LR schedule check failed."; exit 1; }
echo ""

echo "=== GRPO Training — Track C2: Judge reward α=$JUDGE_ALPHA (Gemini 3.1 Pro via genai SDK) ==="
echo "  Model:       Qwen/Qwen2.5-14B-Instruct"
echo "  Train data:  $TRAIN_DATA ($(wc -l < "$TRAIN_DATA") prompts)"
echo "  Eval data:   $EVAL_DATA (OOD probe: 202 problems, BOXED_SUFFIX)"
echo "  Reward:      EM + α×judge×(1-EM), α=$JUDGE_ALPHA"
echo "  Judge:       Gemini 3.1 Pro (genai SDK, project=wf30-poc)"
echo "  Save path:   $SAVE_PATH"
echo "  Checkpoints: $CKPT_PATH"
echo ""
echo "  ★ Track C2: Gemini-judged partial credit for EM=0 responses"
echo "  ★ Invariant: max(EM=0 reward) = $JUDGE_ALPHA < 1.0 = min(EM=1 reward)"
echo "  ★ 2 vLLM engines (no GPU needed for judge), 200 global steps"
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
