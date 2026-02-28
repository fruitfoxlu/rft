"""EM + LLM-as-judge reward for OpenRLHF GRPO training (Track C2).

Variant C2: Judge bonus when EM=0 (partial credit for wrong solutions).

reward = EM + α × judge_score × (1 - EM)
  When EM=1: reward = 1.0 (correctness always dominates)
  When EM=0: reward = α × judge_score ∈ [0, α]

Critical invariant: α < 1.0 ensures max(EM=0 reward) < min(EM=1 reward).

Judge model: Served on a separate vLLM instance (port 8001).
Must be started BEFORE training begins.

Loaded by OpenRLHF via --remote_rm_url /home/rlu/Code/rft/reward_func_judge_c2.py
Must export: reward_func(queries, prompts, labels) -> dict
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────────
ALPHA = float(os.environ.get("JUDGE_ALPHA", "0.3"))
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "http://localhost:8001/v1")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-14B-Instruct")

# Lazy-initialized judge client
_judge_client = None


def _get_judge_client():
    global _judge_client
    if _judge_client is None:
        from openai import OpenAI
        _judge_client = OpenAI(base_url=JUDGE_BASE_URL, api_key="unused")
    return _judge_client


# ── Judge prompt ────────────────────────────────────────────────────────
JUDGE_PROMPT_TEMPLATE = """You are evaluating a student's mathematical reasoning.

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


def _call_judge(problem: str, response: str, ground_truth: str) -> float:
    """Call the judge model to score reasoning quality."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        problem=problem,
        response=response,
        ground_truth=ground_truth,
    )

    try:
        client = _get_judge_client()
        resp = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0.0,
            n=1,
        )
        text = resp.choices[0].message.content or ""
        # Extract float from response
        match = re.search(r"(\d+\.?\d*)", text.strip())
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]
        logger.warning(f"Judge returned unparseable: {text!r}")
        return 0.0
    except Exception as e:
        logger.warning(f"Judge call failed: {e}")
        return 0.0


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
    """Compute reward for OpenRLHF (Track C2: EM + judge bonus on EM=0).

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

    # Judge scoring for EM=0 responses
    judge_score = 0.0
    if correctness == 0.0:
        # Extract the problem text (remove BOXED_SUFFIX for cleaner judge prompt)
        problem_text = prompt
        # Remove chat template tokens if present
        if "<|im_start|>" in problem_text:
            # Extract user message content
            user_match = re.search(r"<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|$)",
                                   problem_text, re.DOTALL)
            if user_match:
                problem_text = user_match.group(1).strip()

        judge_score = _call_judge(problem_text, generation, label)

    # C2 reward: EM + α × judge_score × (1 - EM)
    reward = correctness + ALPHA * judge_score * (1.0 - correctness)

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
