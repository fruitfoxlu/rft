#!/usr/bin/env python3
"""Quick format compliance check for gpt-oss-20b.

Tests whether the model reliably uses \\boxed{} and produces parseable answers.
Runs on a small sample (default 20 problems) and reports parse/format rates.

Usage:
  python check_format_compliance.py \
    --vllm-url http://localhost:8000/v1 \
    --reasoning-effort medium \
    --max-tokens 8192 \
    --sample-size 20
"""

import argparse
import json
import os
import re
import time
from typing import Optional

from openai import OpenAI


def has_boxed(text: str) -> bool:
    """Check if text contains \\boxed{...}."""
    return bool(re.search(r"\\boxed\{", text))


def extract_boxed(text: str) -> Optional[str]:
    """Extract content from last \\boxed{...}."""
    matches = list(re.finditer(r"\\boxed\{", text))
    if not matches:
        return None
    last = matches[-1]
    idx = last.end()
    depth = 1
    end = idx
    while end < len(text) and depth > 0:
        if text[end] == "{":
            depth += 1
        elif text[end] == "}":
            depth -= 1
        end += 1
    return text[idx:end - 1].strip()


def extract_last_number(text: str) -> Optional[str]:
    """Fallback: extract last number from text."""
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/math45_eval.jsonl",
                    help="JSONL file with {input, label} problems")
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1")
    ap.add_argument("--reasoning-effort", default="medium",
                    choices=["", "low", "medium", "high"])
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--sample-size", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="/mnt/scratch/format_compliance.jsonl")
    args = ap.parse_args()

    client = OpenAI(base_url=args.vllm_url, api_key="unused")
    models = client.models.list()
    model = models.data[0].id
    print(f"Model: {model}")

    # Load and sample problems
    with open(args.input) as f:
        all_problems = [json.loads(l) for l in f if l.strip()]

    import random
    random.seed(42)
    sample = random.sample(all_problems, min(args.sample_size, len(all_problems)))
    print(f"Testing {len(sample)} problems from {args.input}")
    print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    results = []
    for i, item in enumerate(sample):
        problem = item["input"]
        gold = item["label"]

        messages = [
            {
                "role": "user",
                "content": (
                    f"{problem}\n"
                    "Please reason step by step, and put your final answer within \\boxed{}."
                ),
            }
        ]

        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        if args.reasoning_effort:
            kwargs["extra_body"] = {"reasoning_effort": args.reasoning_effort}

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
            finish_reason = response.choices[0].finish_reason

            # Check format compliance
            content_has_boxed = has_boxed(text)
            reasoning_has_boxed = has_boxed(reasoning) if reasoning else False
            boxed_answer = extract_boxed(text)
            if boxed_answer is None and reasoning:
                boxed_answer = extract_boxed(reasoning)

            fallback_answer = extract_last_number(text)
            if fallback_answer is None and reasoning:
                fallback_answer = extract_last_number(reasoning)

            # Determine parse method
            if boxed_answer is not None:
                parse_method = "boxed"
                extracted = boxed_answer
            elif fallback_answer is not None:
                parse_method = "last_number"
                extracted = fallback_answer
            else:
                parse_method = "none"
                extracted = ""

            # Check if extracted answer is a clean integer
            is_integer = False
            try:
                int(extracted.replace(",", ""))
                is_integer = True
            except (ValueError, AttributeError):
                pass

            result = {
                "idx": i,
                "gold": gold,
                "extracted": extracted,
                "parse_method": parse_method,
                "content_has_boxed": content_has_boxed,
                "reasoning_has_boxed": reasoning_has_boxed,
                "is_integer": is_integer,
                "completion_tokens": comp_tokens,
                "finish_reason": finish_reason,
                "content_len": len(text),
                "reasoning_len": len(reasoning) if reasoning else 0,
                "content_preview": text[-300:] if text else "",
            }
            results.append(result)

            status_icon = "B" if content_has_boxed else ("b" if reasoning_has_boxed else "X")
            print(f"[{i+1:2d}/{len(sample)}] [{status_icon}] parse={parse_method:12s} "
                  f"extracted='{extracted[:30]}' gold='{gold[:30]}' "
                  f"tokens={comp_tokens} finish={finish_reason}")

        except Exception as e:
            print(f"[{i+1:2d}/{len(sample)}] ERROR: {e}")
            results.append({"idx": i, "gold": gold, "error": str(e)})

    # Save raw results
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nRaw results saved to {args.out}")

    # Summary
    valid = [r for r in results if "error" not in r]
    print(f"\n=== Format Compliance Summary ({len(valid)} valid responses) ===")

    n_content_boxed = sum(1 for r in valid if r["content_has_boxed"])
    n_reasoning_boxed = sum(1 for r in valid if r["reasoning_has_boxed"])
    n_any_boxed = sum(1 for r in valid if r["content_has_boxed"] or r["reasoning_has_boxed"])
    n_parsed_boxed = sum(1 for r in valid if r["parse_method"] == "boxed")
    n_parsed_fallback = sum(1 for r in valid if r["parse_method"] == "last_number")
    n_parsed_none = sum(1 for r in valid if r["parse_method"] == "none")
    n_integer = sum(1 for r in valid if r["is_integer"])
    n_truncated = sum(1 for r in valid if r["finish_reason"] == "length")

    print(f"  \\boxed{{}} in content:     {n_content_boxed}/{len(valid)} ({n_content_boxed/len(valid)*100:.0f}%)")
    print(f"  \\boxed{{}} in reasoning:   {n_reasoning_boxed}/{len(valid)} ({n_reasoning_boxed/len(valid)*100:.0f}%)")
    print(f"  \\boxed{{}} anywhere:       {n_any_boxed}/{len(valid)} ({n_any_boxed/len(valid)*100:.0f}%)")
    print(f"  Parsed via boxed:         {n_parsed_boxed}/{len(valid)} ({n_parsed_boxed/len(valid)*100:.0f}%)")
    print(f"  Parsed via last_number:   {n_parsed_fallback}/{len(valid)} ({n_parsed_fallback/len(valid)*100:.0f}%)")
    print(f"  Parse failed:             {n_parsed_none}/{len(valid)} ({n_parsed_none/len(valid)*100:.0f}%)")
    print(f"  Extracted is integer:      {n_integer}/{len(valid)} ({n_integer/len(valid)*100:.0f}%)")
    print(f"  Truncated (hit max_len):  {n_truncated}/{len(valid)} ({n_truncated/len(valid)*100:.0f}%)")

    # Decision
    parse_rate = (n_parsed_boxed + n_parsed_fallback) / len(valid) if valid else 0
    boxed_rate = n_any_boxed / len(valid) if valid else 0
    print(f"\n=== Decision ===")
    if boxed_rate >= 0.90:
        print(f"  \\boxed{{}} rate = {boxed_rate:.0%} >= 90% → SFT likely NOT needed for format")
    elif boxed_rate >= 0.70:
        print(f"  \\boxed{{}} rate = {boxed_rate:.0%} (70-90%) → Micro-SFT RECOMMENDED (2k examples)")
    else:
        print(f"  \\boxed{{}} rate = {boxed_rate:.0%} < 70% → Micro-SFT NEEDED (2k-5k examples)")

    if parse_rate < 0.95:
        print(f"  Parse rate = {parse_rate:.0%} < 95% → Answer extraction needs improvement")
    else:
        print(f"  Parse rate = {parse_rate:.0%} >= 95% → Answer extraction OK")


if __name__ == "__main__":
    main()
