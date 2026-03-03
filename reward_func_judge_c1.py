"""EM + LLM-as-judge reward for OpenRLHF GRPO training (Track C1).

Variant C1: Judge bonus when EM=1 (quality differentiation among correct).

reward = EM × (1 + α × judge_score)
  When EM=1: reward = 1 + α × judge_score ∈ [1, 1+α]
  When EM=0: reward = 0 (unchanged)

This differentiates correct solutions by reasoning quality: clean logical
reasoning gets higher reward than getting lucky with wrong reasoning.

Judge model: Gemini 3.1 Pro via Vertex AI (google-genai SDK, project=wf30-poc).

Loaded by OpenRLHF via --remote_rm_url /home/rlu/Code/rft/reward_func_judge_c1.py
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
ALPHA = float(os.environ.get("JUDGE_ALPHA", "0.5"))
GCP_PROJECT = os.environ.get("GCP_PROJECT", "wf30-poc")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_FALLBACK = os.environ.get("GEMINI_FALLBACK", "gemini-3-flash-preview")

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


# ── Judge prompt ────────────────────────────────────────────────────────
JUDGE_PROMPT_TEMPLATE = """You are evaluating a student's mathematical reasoning.

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


def _parse_score(text: str) -> float | None:
    """Extract a score from judge response text. Returns None if unparseable."""
    if text is None:
        return None
    match = re.search(r"(\d+\.?\d*)", text.strip())
    if match:
        return max(0.0, min(1.0, float(match.group(1))))
    return None


def _call_judge(problem: str, response: str, ground_truth: str) -> float:
    """Call Gemini 3.1 Pro to score reasoning quality, with flash-lite fallback."""
    from google.genai import types

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        problem=problem,
        response=response[:3000],
        ground_truth=ground_truth,
    )

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
            return 0.5
        except Exception as e:
            logger.warning(f"Judge call attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            continue

    # Fallback to gemini-3-flash-preview with 3 retries
    logger.warning(f"Primary judge failed, falling back to {GEMINI_FALLBACK}")
    for attempt in range(3):
        try:
            client = _get_client()
            resp = client.models.generate_content(
                model=GEMINI_FALLBACK, contents=prompt, config=config,
            )
            score = _parse_score(resp.text)
            if score is not None:
                return score
            logger.warning(f"Fallback judge returned unparseable: {resp.text!r}")
            return 0.5
        except Exception as e:
            logger.warning(f"Fallback judge attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            continue

    logger.warning("All judge attempts failed, defaulting to 0.5")
    return 0.5


# ── Main reward function ───────────────────────────────────────────────
def reward_func(queries: list[str], prompts: list[str], labels: list[str]) -> dict:
    """Compute reward for OpenRLHF (Track C1: EM × (1 + α × judge) for correct).

    When EM=1: reward = 1 + α × judge_score  (bonus for good reasoning)
    When EM=0: reward = 0                     (unchanged from pure EM)
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

    # Pure exact-match reward
    correctness = check_correctness(model_answer, label)

    # Judge scoring for EM=1 responses only
    judge_score = 0.0
    if correctness == 1.0:
        # Extract the problem text
        problem_text = prompt
        if "<|im_start|>" in problem_text:
            user_match = re.search(r"<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|$)",
                                   problem_text, re.DOTALL)
            if user_match:
                problem_text = user_match.group(1).strip()

        judge_score = _call_judge(problem_text, generation, label)

    # C1 reward: EM × (1 + α × judge_score)
    reward = correctness * (1.0 + ALPHA * judge_score)

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
        "judge_alpha": float(ALPHA),
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
