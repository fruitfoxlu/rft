# Plan: Adapting Marco-o1 MCTS Pipeline for Math Reasoning

## Motivation

After 5 null results with GRPO + EM reward on Qwen2.5-14B-Instruct (Q1, B1, B2, Track A RSFT, all Δ < +2pp on OOD-1000), we pivot to the Marco-o1 approach: **offline MCTS CoT data construction → SFT → DPO**.

Key insight from the Marco-o1 v2 paper: small/mid-size models (≤14B) exhibit "formalistic long-time thinking" when trained with standard RL or distillation — content repetition, over-reflection, no final answer. MCTS-constructed CoT data avoids this by building structured reasoning trees from scratch using the model's own capabilities, then extracting the best paths for supervised training.

### Why this should work where GRPO failed

| Problem with GRPO+EM | How MCTS+SFT+DPO addresses it |
|---|---|
| Binary reward (0/1) gives no gradient on all-wrong groups (~33% of problems) | MCTS explores multiple paths per problem; even hard problems yield partial successes |
| LoRA rank 32 may lack capacity for online RL exploration | SFT/DPO is much more stable with LoRA — proven at scale |
| 200 steps × 16 batch = 3200 unique prompts seen once | MCTS can generate 8-32 reasoning paths per problem offline, then SFT on all of them |
| No reward shaping (only final-answer correctness) | MCTS tree structure naturally decomposes reasoning into verifiable sub-steps |

---

## Architecture Overview

```
Phase 1: MCTS CoT Generation (offline, GPU-intensive)
    Serve Qwen2.5-14B via vLLM → MCTS tree search → correct/incorrect paths

Phase 2: Data Processing
    Extract (problem, best_CoT) pairs for SFT
    Extract (problem, good_CoT, bad_CoT) triples for DPO

Phase 3: SFT Training
    Fine-tune Qwen2.5-14B on MCTS-generated CoT data

Phase 4: DPO Training
    Preference optimization using correct vs incorrect MCTS paths

Phase 5: Evaluation
    OOD-1000 paired eval (same as GRPO pipeline)
```

---

## Phase 1: MCTS CoT Data Generation

### 1.1 Data Preparation

Convert our training pool to Marco-o1 input format.

**Source**: `data/sft_rl_pool_3200.jsonl` (3200 math problems, fields: `input`, `label`)

**Target format** (Marco-o1 expects):
```json
{
    "id": "math_0001",
    "problem": "Let x, y, and z be positive real numbers such that xyz = 27. Find the minimum value of...",
    "solution": "102"
}
```

**Script**: `prepare_mcts_data.py`
- Read `sft_rl_pool_3200.jsonl`
- Strip any BOXED_SUFFIX (MCTS has its own prompt template)
- Convert `input` → `problem`, `label` → `solution`, add sequential `id`
- Output: `data/mcts_input_3200.json` (JSON array, not JSONL)

### 1.2 Math Evaluator

The Marco-o1 repo's `math_evaluator` is a stub (`return True`). We need a real one.

**Create**: `mcts_math_evaluator.py`
- Reuse our proven `extract_model_answer()` + `check_correctness()` from `reward_func_em.py`
- Interface must match Marco-o1's evaluator API: `evaluator(answer_node, ground_truth) -> int`
- Extract answer from `answer_node.node_value` (which contains the `<answer>...</answer>` block)
- Parse `\boxed{...}` from the answer text
- Compare against ground truth using our normalize + exact-match logic
- Return 1 (correct) or 0 (incorrect)

### 1.3 MCTS Config for Math

**Create**: `mcts_configs/math_config.json`

```json
{
    "desc": "math_reasoning",
    "mode": "normal",
    "max_rollout_time": 16,
    "max_tokens": 512,
    "output_tree": true,
    "input_path": "./data/mcts_input_3200.json",
    "output_folder": "math_cot",
    "generate_func": "local",
    "search_reward_threshold": [2, 4],
    "evaluate_func": "math",
    "use_for_wrong_answer": {"double_check": {"reflection": 1}},
    "use_mini_step": false,
    "use_function_call": false,
    "use_step": false,
    "url1": "http://localhost:40000/generate",
    "url2": "http://localhost:40001/generate",
    "base_prompt": "<MATH_PROMPT>",
    "action_tree": { ... }
}
```

Key parameter decisions:

| Parameter | Value | Rationale |
|---|---|---|
| `max_rollout_time` | 16 | Paper used 32 for letter-counting; math is harder per rollout but 16 gives enough diversity. Can increase to 32 if success rate is low. |
| `max_tokens` | 512 | Per-node token limit. Math steps are usually shorter than 512 tokens. |
| `search_reward_threshold` | [2, 4] | Stop after finding 4 correct paths, or give up after 4× max_rollout if <2 correct. Balance between quality and compute. |
| `generate_func` | `local` | Uses HTTP API to local vLLM server (most efficient) |
| `use_for_wrong_answer` | `{"double_check": {"reflection": 1}}` | On wrong answers, backtrack to double_check node and add a reflection node. Forces the model to reconsider. |

### 1.4 Math-Specific Base Prompt

```
You are solving a mathematical problem. Show your work step by step.

First, break down the problem in <sub-task> to identify what needs to be solved.
In <thinking>, work through each sub-task carefully, showing all calculations.
In <double-check>, verify your calculations and logic for errors.
In <reflection>, if you found errors, correct them and rethink.
In <answer>, state your final answer clearly using \boxed{answer}.

Note: these tags cannot be nested but can be sequential. Keep actions within tags atomic.

Now the question is:
```

### 1.5 Action Tree for Math

```json
{
    "base": {
        "prefill_text": [],
        "description": "User Question",
        "show_in_history": true,
        "special_model": false,
        "next_step": {"sub_task": 2}
    },
    "sub_task": {
        "prefill_text": [
            "Let me break down this problem.\n- 1.",
            "I need to identify what's being asked.\n- 1.",
            "First, let me understand the problem.\n- 1."
        ],
        "description": "Break down the math problem",
        "show_in_history": true,
        "special_model": false,
        "next_step": {"thinking": 2}
    },
    "thinking": {
        "prefill_text": [],
        "description": "Work through calculations",
        "show_in_history": true,
        "special_model": false,
        "next_step": {
            "double_check": 2,
            "thinking": 2
        }
    },
    "reflection": {
        "prefill_text": [
            "Wait, I made an error. Let me reconsider.\n</reflection>",
            "Hold on, something is wrong with my approach.\n</reflection>"
        ],
        "description": "Reflect on errors",
        "show_in_history": true,
        "special_model": false,
        "next_step": {"thinking_from_scratch": 2}
    },
    "double_check": {
        "prefill_text": [
            "Let me verify my calculation",
            "Time to check my work",
            "Let me make sure this is correct"
        ],
        "description": "Verify calculations",
        "show_in_history": true,
        "special_model": false,
        "next_step": {"answer": 1}
    },
    "answer": {
        "prefill_text": ["The answer is:"],
        "description": "Final answer",
        "show_in_history": true,
        "special_model": false,
        "next_step": {}
    },
    "thinking_from_scratch": {
        "prefill_text": [
            "My previous approach was wrong. Let me start fresh.\n",
            "I need to rethink this entirely.\n"
        ],
        "description": "Rethink from scratch",
        "show_in_history": true,
        "special_model": false,
        "next_step": {"thinking": 2}
    }
}
```

Note: `special_model: false` for all nodes — we use only Qwen2.5-14B (no secondary model). The paper's dual-model approach used a separate model for double-checking to avoid confirmation bias, but we'll start with single-model and iterate.

### 1.6 vLLM Server Setup

```bash
# Single-model setup: Qwen2.5-14B on 2 GPUs (TP=2)
python src/v2/src/tree_search/utils/start_vllm_server.py \
    --path Qwen/Qwen2.5-14B-Instruct \
    --port 40000 \
    --usage 0.85
```

GPU allocation: GPUs 0-1 for vLLM (TP=2). Remaining GPUs idle during MCTS generation.

Alternative: run 2 independent vLLM servers (GPUs 0-1 and 2-3) to parallelize MCTS processing across problems. This would require a wrapper script to split input data and merge results.

### 1.7 Estimated Compute

- 3200 problems × 16 max rollouts × ~4 vLLM calls per rollout = ~205K inference calls
- Each call: ~512 tokens generated
- At ~1000 tok/s throughput (Qwen-14B, TP=2): ~105K seconds ≈ 29 hours for single server
- With 2 parallel servers: ~15 hours
- With 4 parallel servers (4×TP=1, if 14B fits single A100): ~8 hours

**Recommendation**: Start with 2 TP=2 servers on GPUs 0-3, expect ~15 hours.

---

## Phase 2: Data Processing

### 2.1 Extract SFT Data

**Script**: `extract_mcts_sft_data.py`

For each problem, extract the **best correct path** (highest visit count among correct terminal nodes):

```json
{
    "instruction": "Problem text...",
    "response": "<sub-task>\n...\n</sub-task>\n<thinking>\n...\n</thinking>\n<double-check>\n...\n</double-check>\n<answer>\nThe answer is: \\boxed{42}\n</answer>"
}
```

Processing steps:
1. Read MCTS tree output JSON files from `output/math_cot/tree_output/`
2. For each problem, find the best chain (highest-reward path from root to terminal)
3. Extract the concatenated `all_path_value` as the response
4. Pair with the original problem text as instruction
5. Filter: skip problems where MCTS found 0 correct paths (these are genuinely too hard)
6. Output: `data/mcts_sft_data.jsonl`

Expected yield: ~2000-2500 problems with at least 1 correct MCTS path (based on our pass@8 study showing ~15-20% of wrong-at-greedy problems become solvable with sampling).

### 2.2 Extract DPO Data

**Script**: `extract_mcts_dpo_data.py`

For each problem with both correct and incorrect paths, create preference pairs:

```json
{
    "instruction": "Problem text...",
    "chosen": "<sub-task>...<answer>\\boxed{42}</answer>",
    "rejected": "<sub-task>...<answer>\\boxed{37}</answer>"
}
```

Processing steps:
1. Read MCTS tree outputs
2. For each problem, collect all correct and incorrect terminal paths
3. Pair: chosen = best correct path, rejected = best incorrect path (highest visit count)
4. Length-aware selection: prefer pairs where chosen and rejected are similar length (paper finding: length-balanced pairs train better)
5. Output: `data/mcts_dpo_data.jsonl`

### 2.3 Quality Filters

Apply before training:
- **Min length**: Skip responses shorter than 100 characters (too terse to learn from)
- **Max length**: Truncate responses longer than 2048 tokens (matches our generate_max_len)
- **Tag integrity**: Verify all opened tags are closed (no malformed XML-like structure)
- **Answer presence**: Ensure `\boxed{...}` appears in the `<answer>` section

---

## Phase 3: SFT Training

### 3.1 Training Script

**Create**: `train_sft_mcts_qwen14b.sh`

Use standard HuggingFace Trainer (or OpenRLHF's SFT mode) to fine-tune Qwen2.5-14B-Instruct on the MCTS-generated CoT data.

```bash
python -m openrlhf.cli.train_sft \
    --pretrain Qwen/Qwen2.5-14B-Instruct \
    --dataset data/mcts_sft_data.jsonl \
    --input_key instruction \
    --output_key response \
    --apply_chat_template \
    --max_len 3072 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj \
    --learning_rate 2e-5 \
    --lr_scheduler cosine \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --save_steps 100 \
    --logging_steps 10 \
    --bf16 \
    --output_dir /mnt/data/rft_output/qwen14b-mcts-sft \
    --gradient_checkpointing
```

Key differences from GRPO training:
- **LR**: 2e-5 (standard SFT LR, 40× higher than GRPO's 5e-7 — SFT is supervised, doesn't need the conservative RL learning rate)
- **Epochs**: 3 (standard for SFT; GRPO was 1 epoch online)
- **No reward model**: Pure supervised learning on demonstration data
- **Same LoRA config**: rank 32, alpha 64, same target modules (for comparability)

### 3.2 Evaluation Checkpoint Selection

Same approach as GRPO pipeline:
- Save every 100 steps
- Eval on OOD-202 probe set during training
- Select best checkpoint by OOD-202 accuracy

---

## Phase 4: DPO Training

### 4.1 Training Script

**Create**: `train_dpo_mcts_qwen14b.sh`

Start from the best SFT checkpoint, then run DPO with MCTS preference pairs.

```bash
python -m openrlhf.cli.train_dpo \
    --pretrain /mnt/data/rft_output/qwen14b-mcts-sft/best_checkpoint \
    --dataset data/mcts_dpo_data.jsonl \
    --chosen_key chosen \
    --rejected_key rejected \
    --prompt_key instruction \
    --apply_chat_template \
    --max_len 3072 \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules q_proj k_proj v_proj o_proj \
    --learning_rate 5e-6 \
    --beta 0.1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --save_steps 50 \
    --logging_steps 10 \
    --bf16 \
    --output_dir /mnt/data/rft_output/qwen14b-mcts-dpo \
    --gradient_checkpointing
```

Key parameters:
- **beta**: 0.1 (standard DPO KL penalty; paper tested 0.05-0.2)
- **LR**: 5e-6 (lower than SFT since we're fine-tuning an already-finetuned model)
- **Epochs**: 1 (DPO typically needs fewer epochs to avoid overfitting preferences)

### 4.2 Paper-Recommended DPO Variants

The Marco-o1 paper proposes three CoT-aware DPO modifications. We implement the simplest first and add complexity if needed:

1. **Length-balanced pair selection** (Phase 2 already handles this)
2. **Conservative DPO (cDPO)**: Adds label smoothing to the DPO loss to handle potential noise in MCTS labels. Implementation: add `--label_smoothing 0.1` if supported by OpenRLHF.
3. **Masking-based DPO**: Masks the shared prefix between chosen and rejected responses so the model only learns from divergence points. This requires custom loss function code — defer to v2 if basic DPO works.

---

## Phase 5: Evaluation

### 5.1 Standard Eval Pipeline

Reuse existing eval infrastructure from GRPO pipeline:

1. Merge LoRA checkpoint into base model
2. Serve with vLLM (TP=2)
3. Eval on OOD-1000 (greedy, temp=0, max_tokens=2048)
4. Paired comparison vs baseline (67.1%)
5. McNemar test, bootstrap 95% CI

### 5.2 Checkpoints to Eval

- Best SFT checkpoint (by OOD-202 monitor)
- Best DPO checkpoint (by OOD-202 monitor)
- Final DPO checkpoint

### 5.3 Success Criteria

Same gates as GRPO pipeline:
- **Gate-1a**: Δ ≥ +2.0pp AND p < 0.10 → promising, run second seed
- **Gate-1b**: Δ ≥ +3.0pp AND p < 0.05 → claimable win

---

## Implementation Order

```
Step 1: prepare_mcts_data.py              (convert our data → MCTS input format)
Step 2: mcts_math_evaluator.py            (real math evaluator for MCTS)
Step 3: math_config.json + base_prompt     (MCTS config for math domain)
Step 4: run_mcts_generation.sh            (launch vLLM + run MCTS)
Step 5: extract_mcts_sft_data.py          (tree → SFT training pairs)
Step 6: extract_mcts_dpo_data.py          (tree → DPO preference pairs)
Step 7: train_sft_mcts_qwen14b.sh         (SFT training)
Step 8: train_dpo_mcts_qwen14b.sh         (DPO training)
Step 9: eval_mcts_checkpoints.sh          (OOD-1000 paired eval)
```

Steps 1-3 can be done immediately. Step 4 requires GPU (after C1/C2 pipeline finishes or by killing it).

---

## Files to Create

| # | File | Purpose |
|---|------|---------|
| 1 | `prepare_mcts_data.py` | Convert `sft_rl_pool_3200.jsonl` → MCTS input format |
| 2 | `mcts_math_evaluator.py` | Real math evaluator using our EM logic |
| 3 | `mcts_configs/math_config.json` | MCTS config for math domain |
| 4 | `run_mcts_generation.sh` | Launch vLLM servers + run MCTS tree search |
| 5 | `extract_mcts_sft_data.py` | Extract best correct paths → SFT data |
| 6 | `extract_mcts_dpo_data.py` | Extract correct/incorrect pairs → DPO data |
| 7 | `train_sft_mcts_qwen14b.sh` | SFT training script |
| 8 | `train_dpo_mcts_qwen14b.sh` | DPO training script |
| 9 | `eval_mcts_checkpoints.sh` | Eval pipeline (reuses existing infra) |

## Files to Modify (in Marco-o1 repo)

| File | Change |
|------|--------|
| `src/v2/src/tree_search/evaluator/evaluator.py` | Add `math_evaluator` implementation |
| `src/v2/src/tree_search/utils/start_vllm_server.py` | Adjust `tensor_parallel_size` for our GPU setup |

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| MCTS finds few correct paths for hard problems | Medium | Our pass@8 study showed ~15-20% of wrong problems become solvable. For the 67% that are already correct at greedy, MCTS should find paths easily. |
| MCTS generation takes too long (>24h) | Medium | Parallelize with multiple vLLM servers. Reduce `max_rollout_time` from 16 to 8. Process easier problems first. |
| SFT on structured CoT doesn't transfer to free-form eval | Low | Eval uses same BOXED_SUFFIX format. The `<thinking>`/`<answer>` tags structure the output but don't prevent the model from also producing clean `\boxed{}` answers. |
| DPO overfits to MCTS data distribution | Medium | Use conservative DPO (cDPO) with label smoothing. Eval on OOD set. Keep DPO to 1 epoch. |
| Model degrades on problems it already solves | Low | Monitor both "base correct → checkpoint correct" (should be >95%) and "base wrong → checkpoint correct" in paired eval. |

---

## Estimated Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Data prep (Steps 1-3) | 1 hour | Scripting only, no GPU |
| MCTS generation (Step 4) | 12-15 hours | 2 vLLM servers (TP=2), 3200 problems |
| Data extraction (Steps 5-6) | 1 hour | Post-processing |
| SFT training (Step 7) | 4-6 hours | 3 epochs, ~2500 examples, 8 GPUs |
| DPO training (Step 8) | 2-3 hours | 1 epoch, ~2000 pairs, 8 GPUs |
| Evaluation (Step 9) | 2-3 hours | 3 checkpoints × 1000 problems |
| **Total** | **~24-30 hours** | |

---

## Hardware Requirements

- **8× A100 80GB** (same as current setup)
- MCTS generation: 2-4 GPUs (vLLM TP=2)
- SFT/DPO training: 4-8 GPUs (same as GRPO)
- Eval: 2 GPUs (vLLM TP=2)

No external API needed (unlike C1/C2 tracks which required Gemini API).

---

## Comparison with Current GRPO Approach

| Dimension | GRPO + EM | MCTS + SFT + DPO |
|-----------|-----------|-------------------|
| Training signal | Binary (0/1 per response) | Rich (structured CoT paths) |
| Data efficiency | Each problem seen ~1× in 200 steps | Each problem generates 4-16 reasoning paths |
| Training stability | RL instability (ratio clipping, KL penalty) | SFT is stable; DPO is semi-supervised |
| Compute | ~17h training | ~15h MCTS + ~8h training = ~23h |
| Interpretability | Black-box policy gradient | Explicit reasoning trees, inspectable |
| Prior results | 5 null results (Δ < +2pp) | Untested on our setup |
