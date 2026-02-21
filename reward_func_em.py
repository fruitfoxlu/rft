"""Pure exact-match reward function for OpenRLHF GRPO training.

Loaded by OpenRLHF via --remote_rm_url /home/rlu/Code/rft/reward_func_em.py
Must export: reward_func(queries, prompts, labels) -> dict

Scoring:
  - reward = 1.0 if extracted answer matches ground truth, else 0.0
  - No LLM judge, no Gemini dependency
  - Deterministic, fast, zero API cost
"""

import logging
import re

logger = logging.getLogger(__name__)


def normalize_answer(answer: str) -> str:
    """Normalize a math answer for comparison."""
    s = answer.strip()
    # Remove enclosing $...$ or \(...\)
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    # Remove trailing period
    s = s.rstrip(".")
    # Remove common LaTeX wrappers
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    for wrapper in ["\\mathrm{", "\\mathbf{"]:
        if s.startswith(wrapper) and s.endswith("}"):
            s = s[len(wrapper):-1].strip()
    # Normalize fractions
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*,\s*", ",", s)
    s = s.strip()
    # Remove commas from numbers
    if re.match(r"^-?[\d,]+$", s):
        s = s.replace(",", "")
    # Remove leading zeros
    if re.match(r"^-?0+\d+$", s):
        s = str(int(s))
    return s


def _strip_harmony_tokens(text: str) -> str:
    """Strip Harmony format tokens (gpt-oss channel-based output)."""
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
    """Extract the final answer from a model's response.

    Tries \\boxed{...} first, then falls back to the last number.
    """
    response = _strip_harmony_tokens(response)

    # Try \boxed{...} (take the last one if multiple)
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

    # Fallback: last number-like token
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", response)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def check_correctness(model_answer: str, ground_truth: str) -> float:
    """Return 1.0 if answers match, 0.0 otherwise."""
    norm_model = normalize_answer(model_answer)
    norm_truth = normalize_answer(ground_truth)
    if norm_model == norm_truth:
        return 1.0
    # Try numeric comparison
    try:
        if abs(float(norm_model) - float(norm_truth)) < 1e-6:
            return 1.0
    except (ValueError, OverflowError):
        pass
    return 0.0


def reward_func(queries: list[str], prompts: list[str], labels: list[str]) -> dict:
    """Compute reward for OpenRLHF.

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

    # Determine parse method for logging
    response_stripped = _strip_harmony_tokens(generation)
    has_boxed = bool(re.search(r"\\boxed\{", response_stripped))

    reward = float(correctness)

    extra_logs = {
        "correctness": float(correctness),
        "has_answer": 1.0 if model_answer else 0.0,
        "has_boxed": 1.0 if has_boxed else 0.0,
    }

    return {
        "rewards": reward,
        "scores": float(correctness),
        "extra_logs": extra_logs,
    }
