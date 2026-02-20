"""Outcome-based reward function for OpenRLHF GRPO training.

Loaded by OpenRLHF via --remote_rm_url /home/rlu/Code/rft/reward_func.py
Must export: reward_func(queries, prompts, labels) -> dict

Scoring:
  - 0.6 * exact_match_correctness + 0.4 * gemini_reasoning_quality
  - Falls back to pure exact-match if Gemini API fails
"""

import logging
import os
import re
import time

# Force Vertex AI auth (not API key) to use paid project quota
os.environ.pop("GOOGLE_API_KEY", None)
os.environ.pop("GEMINI_API_KEY", None)

logger = logging.getLogger(__name__)

# Lazy-init Gemini client (initialized on first call)
_gemini_client = None
GEMINI_MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
]
GEMINI_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "wf30-poc")
GEMINI_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")

CORRECTNESS_WEIGHT = 0.6
REASONING_WEIGHT = 0.4
GEMINI_MAX_RETRIES = 3
GEMINI_RETRY_DELAY = 2.0


def _get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        _gemini_client = genai.Client(
            vertexai=True,
            project=GEMINI_PROJECT,
            location=GEMINI_LOCATION,
        )
    return _gemini_client


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
    # Remove common LaTeX wrappers (including \text{...} anywhere in the string)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    for wrapper in ["\\mathrm{", "\\mathbf{"]:
        if s.startswith(wrapper) and s.endswith("}"):
            s = s[len(wrapper) : -1].strip()
    # Normalize \dfrac -> \frac
    s = s.replace("\\dfrac", "\\frac")
    s = s.replace("\\tfrac", "\\frac")
    # Normalize whitespace: collapse multiple spaces, strip around operators
    s = re.sub(r"\s+", " ", s).strip()
    # Remove spaces around LaTeX operators for consistent matching
    s = re.sub(r"\s*\\cup\s*", r"\\cup ", s)
    s = re.sub(r"\s*\\cap\s*", r"\\cap ", s)
    s = re.sub(r"\s*,\s*", ",", s)
    s = s.strip()
    # Remove commas from numbers (e.g. "1,234" -> "1234")
    if re.match(r"^-?[\d,]+$", s):
        s = s.replace(",", "")
    # Remove leading zeros for integers
    if re.match(r"^-?0+\d+$", s):
        s = str(int(s))
    return s


def _strip_harmony_tokens(text: str) -> str:
    """Strip Harmony format tokens (gpt-oss uses channel-based output).

    Harmony tokens: <|start|>, <|end|>, <|message|>, <|channel|>, <|return|>, etc.
    We extract the final channel content if present, otherwise strip all tokens.
    """
    # Check if text contains Harmony tokens
    if "<|channel|>" not in text and "<|start|>" not in text:
        return text

    # Try to extract content from the 'final' channel
    final_match = re.search(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
        text, re.DOTALL
    )
    if final_match:
        return final_match.group(1).strip()

    # Try to extract content from any <|message|> ... <|end|> or <|return|>
    msg_matches = re.findall(
        r"<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|<\|call\|>|$)",
        text, re.DOTALL
    )
    if msg_matches:
        # Return the last message content (likely the final answer)
        return msg_matches[-1].strip()

    # Fallback: strip all Harmony tokens
    cleaned = re.sub(r"<\|[^|]*\|>", "", text)
    return cleaned.strip()


def extract_model_answer(response: str) -> str:
    """Extract the final answer from a model's response.

    Handles Harmony format (gpt-oss) and standard format.
    Tries \\boxed{...} first, then falls back to the last number.
    """
    # Strip Harmony format tokens if present
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
        return response[idx : end - 1].strip()

    # Fallback: last number-like token (integer or decimal, not trailing periods)
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
    # Try numeric comparison for floats
    try:
        if abs(float(norm_model) - float(norm_truth)) < 1e-6:
            return 1.0
    except (ValueError, OverflowError):
        pass
    return 0.0


def _get_safety_settings():
    """Return permissive safety settings -- math solutions should never be blocked."""
    from google.genai import types

    return [
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="OFF",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="OFF",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="OFF",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="OFF",
        ),
    ]


def _extract_score(text: str) -> float | None:
    """Extract a score from Gemini's response text, handling various formats."""
    text = text.strip()
    # Try direct float parse
    try:
        return max(0.0, min(1.0, float(text)))
    except ValueError:
        pass
    # Look for number patterns: "0.7", "1.0", "0", "1", "0.85"
    numbers = re.findall(r"(\d+\.?\d*)", text)
    if numbers:
        for n in numbers:
            try:
                val = float(n)
                if 0.0 <= val <= 1.0:
                    return val
            except ValueError:
                continue
    return None


GEMINI_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "Reasoning quality from 0.0 to 1.0"},
        "reason": {"type": "string", "description": "Brief explanation of the score"},
    },
    "required": ["score", "reason"],
}


def get_reasoning_quality(problem: str, solution: str, ground_truth: str) -> float:
    """Ask Gemini to rate the reasoning quality of a solution (0.0-1.0).

    Uses JSON schema output for reliable structured responses.
    """
    import random

    # Truncate solution to keep within context limits
    solution_truncated = solution[:6000] if len(solution) > 6000 else solution

    prompt = (
        f"PROBLEM: {problem}\n"
        f"SOLUTION: {solution_truncated}\n"
        f"CORRECT ANSWER: {ground_truth}\n"
        f"Rate the reasoning quality of this math solution."
    )

    client = _get_gemini_client()
    safety = _get_safety_settings()

    for model_id in GEMINI_MODELS:
        for attempt in range(GEMINI_MAX_RETRIES):
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config={
                        "max_output_tokens": 4096,
                        "temperature": 0.0,
                        "safety_settings": safety,
                        "response_mime_type": "application/json",
                        "response_json_schema": GEMINI_JSON_SCHEMA,
                    },
                )

                # Try parsed JSON first (most reliable)
                if response.parsed and isinstance(response.parsed, dict):
                    score = response.parsed.get("score")
                    if score is not None:
                        return max(0.0, min(1.0, float(score)))

                # Fallback: parse text as JSON
                resp_text = None
                try:
                    resp_text = response.text
                except Exception:
                    pass

                if resp_text:
                    import json
                    try:
                        data = json.loads(resp_text)
                        score = data.get("score")
                        if score is not None:
                            return max(0.0, min(1.0, float(score)))
                    except (json.JSONDecodeError, ValueError):
                        pass
                    # Last resort: extract number from text
                    score = _extract_score(resp_text)
                    if score is not None:
                        return score

                # Log failure
                finish = None
                if response.candidates:
                    finish = getattr(response.candidates[0], "finish_reason", None)
                logger.warning(
                    f"Gemini [{model_id}] no parseable score, finish_reason={finish} "
                    f"(attempt {attempt + 1}/{GEMINI_MAX_RETRIES})"
                )
                if attempt < GEMINI_MAX_RETRIES - 1:
                    time.sleep(GEMINI_RETRY_DELAY * (attempt + 1) + random.uniform(0, 1))
            except Exception as e:
                logger.warning(f"Gemini [{model_id}] API error (attempt {attempt + 1}/{GEMINI_MAX_RETRIES}): {e}")
                if attempt < GEMINI_MAX_RETRIES - 1:
                    time.sleep(GEMINI_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 2))
        logger.warning(f"Gemini [{model_id}] exhausted retries, falling back to next model")

    return -1.0  # Sentinel: all Gemini models failed


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

    # Extract the generated portion (query = prompt + generation)
    if query.startswith(prompt):
        generation = query[len(prompt) :]
    else:
        generation = query

    # Extract model's final answer
    model_answer = extract_model_answer(generation)

    # Hard signal: exact-match correctness
    correctness = check_correctness(model_answer, label)

    # Soft signal: Gemini reasoning quality
    reasoning_quality = get_reasoning_quality(prompt, generation, label)

    if reasoning_quality < 0:
        # Gemini failed -- fall back to pure exact-match
        reward = float(correctness)
        extra_logs = {
            "correctness": float(correctness),
            "reasoning_quality": -1.0,
            "gemini_fallback": 1.0,
            "has_answer": 1.0 if model_answer else 0.0,
        }
    else:
        reward = float(CORRECTNESS_WEIGHT * correctness + REASONING_WEIGHT * reasoning_quality)
        extra_logs = {
            "correctness": float(correctness),
            "reasoning_quality": float(reasoning_quality),
            "gemini_fallback": 0.0,
            "has_answer": 1.0 if model_answer else 0.0,
        }

    return {
        "rewards": float(reward),
        "scores": float(correctness),  # Track raw correctness separately
        "extra_logs": extra_logs,
    }
