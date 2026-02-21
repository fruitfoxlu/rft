#!/usr/bin/env python3
"""Evaluate gpt-oss models on MathArena Apex Shortlist 2025.

Supports integer and fraction answers. Skips parametric/symbolic answers.

Usage:
  python eval_apex.py \
    --vllm-url http://localhost:8000/v1 \
    --reasoning-effort medium \
    --max-tokens 20000
"""

import argparse
import json
import os
import re
import time
from fractions import Fraction
from typing import Any, Dict, List, Optional

from openai import OpenAI


def normalize_latex_answer(ans: str) -> Optional[str]:
    """Normalize a LaTeX answer to a comparable string.

    Returns None if the answer is parametric/symbolic and not evaluable.
    """
    s = ans.strip()

    # Skip parametric/symbolic answers
    if re.search(r"[nNk][\s\(\)\+\-\*]", s) and not s.replace(",", "").replace(" ", "").lstrip("-").isdigit():
        return None
    if re.search(r"\binom", s):
        return None

    # Handle 2^{k} - style
    pow_match = re.match(r"^(\d+)\^\{(\d+)\}\s*-\s*(\d+)$", s)
    if pow_match:
        base, exp, sub = int(pow_match.group(1)), int(pow_match.group(2)), int(pow_match.group(3))
        return str(base**exp - sub)

    pow_match2 = re.match(r"^(\d+)\^\{(\d+)\}$", s)
    if pow_match2:
        base, exp = int(pow_match2.group(1)), int(pow_match2.group(2))
        return str(base**exp)

    # Handle \frac{a}{b}
    frac_match = re.match(r"^\\frac\{(-?\d+)\}\{(-?\d+)\}$", s)
    if frac_match:
        num, den = int(frac_match.group(1)), int(frac_match.group(2))
        f = Fraction(num, den)
        return f"{f.numerator}/{f.denominator}"

    # Handle a/b
    slash_match = re.match(r"^(-?\d+)/(\d+)$", s)
    if slash_match:
        num, den = int(slash_match.group(1)), int(slash_match.group(2))
        f = Fraction(num, den)
        return f"{f.numerator}/{f.denominator}"

    # Handle expressions with \sqrt or \cdot — skip
    if "\\sqrt" in s or "\\cdot" in s:
        return None

    # Handle plain integers (possibly with commas)
    cleaned = s.replace(",", "").replace(" ", "")
    try:
        return str(int(cleaned))
    except ValueError:
        pass

    # Handle decimals
    try:
        return str(float(cleaned))
    except ValueError:
        pass

    return None


def extract_answer_from_response(text: str) -> Optional[str]:
    """Extract the answer from model response, trying boxed first."""
    # Try \boxed{...}
    boxed_matches = list(re.finditer(r"\\boxed\{", text))
    if boxed_matches:
        last_match = boxed_matches[-1]
        idx = last_match.end()
        depth = 1
        end = idx
        while end < len(text) and depth > 0:
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
            end += 1
        raw = text[idx:end - 1].strip()
        return raw

    # Fallback: last number
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def answers_match(pred_raw: str, gold_normalized: str) -> bool:
    """Check if predicted answer matches gold answer."""
    pred_norm = normalize_latex_answer(pred_raw)
    if pred_norm is None:
        # Try direct string comparison
        return pred_raw.strip() == gold_normalized

    return pred_norm == gold_normalized


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/apex_shortlist.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--reasoning-effort", default="", choices=["", "low", "medium", "high"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=20000)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    client = OpenAI(base_url=args.vllm_url, api_key="unused")

    if args.model:
        model = args.model
    else:
        models = client.models.list()
        model = models.data[0].id
    print(f"Model: {model}")
    print(f"Reasoning effort: {args.reasoning_effort or 'none'}")
    print(f"Max tokens: {args.max_tokens}")

    items = load_jsonl(args.input)
    reasoning_effort = args.reasoning_effort or None

    # Pre-process: identify evaluable problems
    evaluable = []
    skipped = []
    for item in items:
        gold_norm = normalize_latex_answer(item["label"])
        if gold_norm is None:
            skipped.append(item)
        else:
            evaluable.append((item, gold_norm))

    print(f"\nTotal problems: {len(items)}")
    print(f"Evaluable (integer/fraction): {len(evaluable)}")
    print(f"Skipped (symbolic/parametric): {len(skipped)}")
    for s in skipped:
        print(f"  Skipped: [{s.get('source', '')}] answer='{s['label']}'")

    # Load existing results
    done_keys = set()
    if os.path.exists(args.out):
        with open(args.out, "r") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line.strip())
                    done_keys.add(r["problem_idx"])

    total = 0
    correct = 0

    with open(args.out, "a") as fout:
        for item, gold_norm in evaluable:
            pidx = item.get("problem_idx", 0)
            if pidx in done_keys:
                # Count existing
                continue

            problem = item["input"]
            messages = [
                {
                    "role": "user",
                    "content": (
                        f"{problem}\n"
                        "Please reason step by step, and put your final answer within \\boxed{}."
                    ),
                }
            ]

            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            if reasoning_effort:
                kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

            pred_raw = ""
            comp_tokens = 0
            error = None

            for attempt in range(args.retries + 1):
                try:
                    response = client.chat.completions.create(**kwargs)
                    msg = response.choices[0].message
                    text = msg.content or ""
                    reasoning = (
                        getattr(msg, "reasoning_content", None)
                        or getattr(msg, "reasoning", None)
                        or ""
                    )
                    if not isinstance(reasoning, str):
                        reasoning = str(reasoning)
                    comp_tokens = response.usage.completion_tokens if response.usage else 0

                    # Extract from content first, then reasoning
                    pred_raw = extract_answer_from_response(text) or ""
                    if not pred_raw and reasoning:
                        pred_raw = extract_answer_from_response(reasoning) or ""

                    break
                except Exception as e:
                    error = str(e)
                    if attempt < args.retries:
                        time.sleep(2 ** attempt)

            is_correct = answers_match(pred_raw, gold_norm) if pred_raw else False
            total += 1
            correct += int(is_correct)

            result = {
                "problem_idx": pidx,
                "source": item.get("source", ""),
                "gold": item["label"],
                "gold_normalized": gold_norm,
                "pred_raw": pred_raw,
                "correct": is_correct,
                "completion_tokens": comp_tokens,
                "error": error,
            }
            fout.write(json.dumps(result) + "\n")
            fout.flush()

            status = "OK" if is_correct else "WRONG"
            print(f"[{pidx:2d}] pred='{pred_raw}' gold='{gold_norm}' {status} "
                  f"tokens={comp_tokens} source={item.get('source', '')}")

    # Also count any pre-existing results
    if done_keys:
        with open(args.out, "r") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line.strip())
                    total += 1
                    correct += int(r["correct"])

    acc = correct / total if total else 0.0
    print(f"\n=== Summary ===")
    print(f"Model: {model}")
    print(f"Reasoning effort: {args.reasoning_effort or 'none'}")
    print(f"Evaluable problems: {len(evaluable)}")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
