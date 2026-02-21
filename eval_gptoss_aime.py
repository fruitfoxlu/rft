#!/usr/bin/env python3
"""Evaluate gpt-oss models on AIME 2025 via vLLM OpenAI-compatible API.

Supports reasoning_effort parameter for gpt-oss reasoning models.
Uses \boxed{} answer extraction matching the official gpt-oss eval.

Usage:
  python eval_gptoss_aime.py \
    --input data/aime_eval.jsonl \
    --out /mnt/scratch/gptoss20b_aime.jsonl \
    --vllm-url http://localhost:8000/v1 \
    --reasoning-effort high \
    --max-tokens 65536 \
    --temperature 1.0
"""

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from openai import OpenAI


@dataclass
class TrialResult:
    id: str
    trial: int
    gold: int
    pred: Optional[int]
    correct: bool
    response_len: int
    reasoning_len: int
    completion_tokens: int
    error: Optional[str] = None


def extract_boxed_text(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} or \\framebox{}, matching gpt-oss official eval."""
    pattern = r'boxed{(.*?)}|framebox{(.*?)}'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        for match in matches[::-1]:
            for group in match:
                if group != "":
                    return group.split(',')[-1].strip()
    # Fallback: get the last integer
    pattern = r'\d+'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    return None


def extract_answer_as_int(text: str) -> Optional[int]:
    """Extract and convert answer to integer."""
    raw = extract_boxed_text(text)
    if raw is None:
        return None
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def call_model(
    client: OpenAI,
    model: str,
    problem: str,
    reasoning_effort: Optional[str],
    temperature: float,
    max_tokens: int,
    retries: int,
) -> tuple:
    """Call the model and return (pred, raw_text, error)."""
    messages = [
        {
            "role": "user",
            "content": (
                f"{problem}\n"
                "Please reason step by step, and put your final answer within \\boxed{}."
            ),
        }
    ]

    last_err = None
    for attempt in range(retries + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if reasoning_effort:
                kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

            response = client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            text = msg.content or ""
            # vLLM may use 'reasoning_content' or 'reasoning' depending on version
            reasoning = (
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or ""
            )
            if not isinstance(reasoning, str):
                reasoning = str(reasoning)
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            finish_reason = response.choices[0].finish_reason

            # Try to extract answer from visible content first
            pred = extract_answer_as_int(text)
            # If no answer in content, try reasoning (model may put boxed in reasoning)
            if pred is None and reasoning:
                pred = extract_answer_as_int(reasoning)

            return pred, text, reasoning, completion_tokens, None
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(2 ** attempt)

    return None, "", "", 0, last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="AIME JSONL file (input/label format)")
    ap.add_argument("--out", required=True, help="Output JSONL results file")
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="", help="Model name (auto-detected from vLLM if empty)")
    ap.add_argument("--reasoning-effort", default="", choices=["", "low", "medium", "high"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    client = OpenAI(base_url=args.vllm_url, api_key="unused")

    # Auto-detect model
    if args.model:
        model = args.model
    else:
        models = client.models.list()
        model = models.data[0].id
    print(f"Model: {model}")
    print(f"Reasoning effort: {args.reasoning_effort or 'none'}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")

    items = load_jsonl(args.input)
    reasoning_effort = args.reasoning_effort or None

    total = 0
    correct = 0

    # Load existing results to support resume
    done_keys = set()
    if os.path.exists(args.out):
        with open(args.out, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                done_keys.add((r["id"], r["trial"]))
                total += 1
                correct += int(r["correct"])
        if done_keys:
            print(f"Resuming: {len(done_keys)} results already done ({correct}/{total} correct)")

    with open(args.out, "a") as fout:
        for idx, obj in enumerate(items):
            # Support both formats: {input, label} and {id, problem, answer}
            if "problem" in obj:
                pid = str(obj.get("id", f"problem-{idx+1}"))
                problem = obj["problem"]
                gold = int(obj["answer"])
            else:
                pid = f"problem-{idx+1}"
                problem = obj["input"]
                gold = int(obj["label"])

            for t in range(args.trials):
                if (pid, t) in done_keys:
                    continue

                pred, raw_text, reasoning, comp_tokens, err = call_model(
                    client, model, problem,
                    reasoning_effort, args.temperature,
                    args.max_tokens, args.retries,
                )

                is_correct = (pred == gold)
                tr = TrialResult(
                    id=pid, trial=t, gold=gold, pred=pred,
                    correct=is_correct,
                    response_len=len(raw_text),
                    reasoning_len=len(reasoning),
                    completion_tokens=comp_tokens,
                    error=err,
                )

                fout.write(json.dumps(asdict(tr)) + "\n")
                fout.flush()

                total += 1
                correct += int(is_correct)

                status = "OK" if is_correct else "WRONG"
                print(f"[{pid}][trial {t}] pred={pred} gold={gold} {status} "
                      f"resp={len(raw_text)} reason={len(reasoning)} tokens={comp_tokens}")

    acc = correct / total if total else 0.0
    print(f"\n=== Summary ===")
    print(f"model: {model}")
    print(f"reasoning_effort: {args.reasoning_effort or 'none'}")
    print(f"accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
