"""EM + LLM-as-judge reward for OpenRLHF GRPO training (Track C3).

Variant C3: Unified reward — bonus for correct, negative penalty for wrong.

reward formula:
  When EM=1: reward = 1.0 + α_pos × judge_score    ∈ [1.0, 1.5]
  When EM=0: reward = α_neg × (judge_score - 0.5)   ∈ [-0.30, +0.24]

This combines C1 (quality bonus on correct) and C2 (partial credit on wrong)
with the addition of NEGATIVE reward for bad wrong answers. The negative signal
gives the model a stronger push away from poor reasoning.

Invariant: max(EM=0 reward) = 0.3 < 1.0 = min(EM=1 reward).  ✓

Judge model: Gemini 3.1 Pro via Vertex AI (google-genai SDK, project=wf30-poc).

Loaded by OpenRLHF via --remote_rm_url /home/rlu/Code/rft/reward_func_judge_c3.py
Must export: reward_func(queries, prompts, labels) -> dict
"""

import logging
import os
import re
import time

from reward_func_em import (
    extract_model_answer, check_correctness,
    _strip_harmony_tokens,
)

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────
ALPHA_POS = float(os.environ.get("JUDGE_ALPHA_POS", "0.5"))   # bonus for correct
ALPHA_NEG = float(os.environ.get("JUDGE_ALPHA_NEG", "0.6"))   # scale for wrong
GCP_PROJECT = os.environ.get("GCP_PROJECT", "wf30-poc")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_FALLBACK = os.environ.get("GEMINI_FALLBACK", "gemini-3.1-flash-lite-preview")

# Lazy-init genai client
_genai_client = None


def _get_client():
    global _genai_client
    if _genai_client is None:
        from google import genai
        _genai_client = genai.Client(
            vertexai=True, project=GCP_PROJECT, location="global",
        )
    return _genai_client


# ── Judge prompts ─────────────────────────────────────────────────────
JUDGE_PROMPT_CORRECT = """You are evaluating a student's mathematical reasoning.

Problem: {problem}
Correct answer: {ground_truth}
Student's solution: {response}

The student's final answer is CORRECT. Score the reasoning quality from 0.0 to 1.0:
- 0.8-1.0: Clear, logical steps; correct approach throughout; well-organized
- 0.6-0.8: Correct approach, but some unnecessary steps, minor confusion, or unclear explanation
- 0.4-0.6: Got the right answer but reasoning has gaps, skipped steps, or partially wrong logic
- 0.2-0.4: Mostly wrong reasoning that happened to arrive at the correct answer (lucky)
- 0.0-0.2: No meaningful reasoning, or answer appears without justification

Output ONLY a single decimal number between 0.0 and 1.0, nothing else."""

JUDGE_PROMPT_WRONG = """You are evaluating a student's mathematical reasoning.

Problem: {problem}
Correct answer: {ground_truth}
Student's solution: {response}

The student's final answer is WRONG. Score the reasoning quality from 0.0 to 1.0:
- 0.8-1.0: Correct approach and method, only minor arithmetic/calculation error at the end
- 0.6-0.8: Right general approach, but errors in setup or intermediate steps
- 0.4-0.6: Partially correct approach, shows understanding of relevant concepts
- 0.2-0.4: Wrong approach but shows some mathematical understanding
- 0.0-0.2: No meaningful reasoning, completely wrong approach, or nonsensical

Output ONLY a single decimal number between 0.0 and 1.0, nothing else."""


def _parse_score(text: str) -> float | None:
    """Extract a score from judge response text. Returns None if unparseable."""
    if text is None:
        return None
    match = re.search(r"(\d+\.?\d*)", text.strip())
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return None


def _call_judge(problem: str, response: str, ground_truth: str,
                is_correct: bool) -> float:
    """Call Gemini 3.1 Pro to score reasoning quality, with flash-lite fallback."""
    from google.genai import types

    template = JUDGE_PROMPT_CORRECT if is_correct else JUDGE_PROMPT_WRONG
    prompt = template.format(
        problem=problem,
        response=response[:3000],
        ground_truth=ground_truth,
    )

    # Default: 0.5 for correct (neutral bonus), 0.0 for wrong (no partial credit)
    default = 0.5 if is_correct else 0.0
    config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=16384,
    )

    # Try primary model (gemini-3.1-pro-preview) with 3 retries
    for attempt in range(3):
        try:
            client = _get_client()
            resp = client.models.generate_content(
                model=GEMINI_MODEL, contents=prompt, config=config,
            )
            score = _parse_score(resp.text)
            if score is not None:
                return score
            logger.warning(f"Judge returned unparseable: {resp.text!r}")
            return default
        except Exception as e:
            logger.warning(f"Judge call attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            continue

    # Fallback to flash-lite model
    try:
        logger.warning(f"Primary judge failed, falling back to {GEMINI_FALLBACK}")
        client = _get_client()
        resp = client.models.generate_content(
            model=GEMINI_FALLBACK, contents=prompt, config=config,
        )
        score = _parse_score(resp.text)
        if score is not None:
            return score
        logger.warning(f"Fallback judge returned unparseable: {resp.text!r}")
    except Exception as e:
        logger.warning(f"Fallback judge failed: {e}")

    logger.warning(f"All judge attempts failed, defaulting to {default}")
    return default


# ── Main reward function ───────────────────────────────────────────────
def reward_func(queries: list[str], prompts: list[str], labels: list[str]) -> dict:
    """Compute reward for OpenRLHF (Track C3: unified judge reward).

    When EM=1: reward = 1.0 + α_pos × judge_score   (bonus for good reasoning)
    When EM=0: reward = α_neg × (judge_score - 0.5)  (negative for bad reasoning)
    """
    query = queries[0]
    prompt = prompts[0]
    label = labels[0]

    # Extract generated portion
    if query.startswith(prompt):
        generation = query[len(prompt):]
    else:
        generation = query

    # Extract model's final answer
    model_answer = extract_model_answer(generation)

    # Pure exact-match
    correctness = check_correctness(model_answer, label)

    # Extract problem text from chat template
    problem_text = prompt
    if "<|im_start|>" in problem_text:
        user_match = re.search(r"<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|$)",
                               problem_text, re.DOTALL)
        if user_match:
            problem_text = user_match.group(1).strip()

    # Judge scoring — always called (for both correct and wrong)
    is_correct = (correctness == 1.0)
    judge_score = _call_judge(problem_text, generation, label, is_correct)

    # C3 reward
    if is_correct:
        reward = 1.0 + ALPHA_POS * judge_score
    else:
        reward = ALPHA_NEG * (judge_score - 0.5)

    # Logging
    response_stripped = _strip_harmony_tokens(generation)
    boxed_matches = list(re.finditer(r"\\boxed\{", response_stripped))
    has_boxed = bool(boxed_matches)

    if has_boxed:
        parse_method = 2.0
    elif model_answer:
        parse_method = 1.0
    else:
        parse_method = 0.0

    if has_boxed and len(response_stripped) > 0:
        last_boxed_start = boxed_matches[-1].start()
        relative_pos = last_boxed_start / len(response_stripped)
        boxed_in_final = 1.0 if relative_pos >= 0.8 else 0.0
    else:
        boxed_in_final = 0.0

    stripped_tail = response_stripped.rstrip()
    is_very_long = len(response_stripped) > 3000
    ends_mid_sentence = (
        bool(stripped_tail)
        and stripped_tail[-1] not in ".!?)}\n"
        and not stripped_tail.endswith("$$")
    )
    truncated_response = 1.0 if (
        (not has_boxed and is_very_long)
        or ends_mid_sentence
    ) else 0.0

    extra_logs = {
        "correctness": float(correctness),
        "judge_score": float(judge_score),
        "judge_alpha_pos": float(ALPHA_POS),
        "judge_alpha_neg": float(ALPHA_NEG),
        "has_answer": 1.0 if model_answer else 0.0,
        "has_boxed": 1.0 if has_boxed else 0.0,
        "parse_method": parse_method,
        "boxed_in_final": boxed_in_final,
        "truncated_response": truncated_response,
    }

    return {
        "rewards": float(reward),
        "scores": float(correctness),  # scores always EM for comparability
        "extra_logs": extra_logs,
    }
