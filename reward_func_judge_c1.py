"""EM + LLM-as-judge reward for OpenRLHF GRPO training (Track C1).

Variant C1: Judge bonus when EM=1 (quality differentiation among correct).

reward = EM × (1 + α × judge_score)
  When EM=1: reward = 1 + α × judge_score ∈ [1, 1+α]
  When EM=0: reward = 0 (unchanged)

This differentiates correct solutions by reasoning quality: clean logical
reasoning gets higher reward than getting lucky with wrong reasoning.

Judge model: Gemini 3.1 Pro via Google API (external, no GPU needed).

Loaded by OpenRLHF via --remote_rm_url /home/rlu/Code/rft/reward_func_judge_c1.py
Must export: reward_func(queries, prompts, labels) -> dict
"""

import logging
import os
import re
import time
import urllib.request
import json as json_mod

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────
ALPHA = float(os.environ.get("JUDGE_ALPHA", "0.5"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"


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


def _call_judge(problem: str, response: str, ground_truth: str) -> float:
    """Call Gemini API to score reasoning quality."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        problem=problem,
        response=response[:3000],  # Truncate very long responses
        ground_truth=ground_truth,
    )

    payload = json_mod.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 64,
        },
    }).encode("utf-8")

    for attempt in range(3):
        try:
            url = f"{GEMINI_URL}?key={GEMINI_API_KEY}"
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json_mod.loads(resp.read())

            parts = data["candidates"][0]["content"]["parts"]
            text = parts[0]["text"]
            match = re.search(r"(\d+\.?\d*)", text.strip())
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            logger.warning(f"Judge returned unparseable: {text!r}")
            return 0.5  # Default to middle for correct answers
        except Exception as e:
            logger.warning(f"Judge call attempt {attempt+1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            continue

    logger.warning("All judge attempts failed, defaulting to 0.5")
    return 0.5  # Default to middle for correct answers


# ── Shared utilities (from reward_func_em.py) ──────────────────────────
def normalize_answer(answer: str) -> str:
    s = answer.strip()
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    s = s.rstrip(".")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    for wrapper in ["\\mathrm{", "\\mathbf{"]:
        if s.startswith(wrapper) and s.endswith("}"):
            s = s[len(wrapper):-1].strip()
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*,\s*", ",", s)
    s = s.strip()
    if re.match(r"^-?[\d,]+$", s):
        s = s.replace(",", "")
    if re.match(r"^-?0+\d+$", s):
        s = str(int(s))
    return s


def _strip_harmony_tokens(text: str) -> str:
    if "<|channel|>" not in text and "<|start|>" not in text:
        return text
    final_match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
        text, re.DOTALL
    )
    if final_match:
        return final_match.group(1).strip()
    msg_matches = re.findall(
        r"<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|<\|call\|>|$)",
        text, re.DOTALL
    )
    if msg_matches:
        return msg_matches[-1].strip()
    cleaned = re.sub(r"<\|[^|]*\|>", "", text)
    return cleaned.strip()


def extract_model_answer(response: str) -> str:
    response = _strip_harmony_tokens(response)
    boxed_matches = list(re.finditer(r"\\boxed\{", response))
    if boxed_matches:
        last_match = boxed_matches[-1]
        idx = last_match.end()
        depth = 1
        end = idx
        while end < len(response) and depth > 0:
            if response[end] == "{":
                depth += 1
            elif response[end] == "}":
                depth -= 1
            end += 1
        return response[idx:end - 1].strip()
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", response)
    if numbers:
        return numbers[-1].replace(",", "")
    return ""


def check_correctness(model_answer: str, ground_truth: str) -> float:
    norm_model = normalize_answer(model_answer)
    norm_truth = normalize_answer(ground_truth)
    if norm_model == norm_truth:
        return 1.0
    try:
        if abs(float(norm_model) - float(norm_truth)) < 1e-6:
            return 1.0
    except (ValueError, OverflowError):
        pass
    return 0.0


# ── Main reward function ───────────────────────────────────────────────
def reward_func(queries: list[str], prompts: list[str], labels: list[str]) -> dict:
    """Compute reward for OpenRLHF (Track C1: EM × (1 + α × judge) for correct).

    Args:
        queries: Full model responses (prompt + generation), list of 1.
        prompts: Original prompts, list of 1.
        labels: Ground truth answers, list of 1.

    Returns:
        dict with "rewards" (float), "scores" (float), "extra_logs" (dict).
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

    # Judge scoring for EM=1 responses
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
