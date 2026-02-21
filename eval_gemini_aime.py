#!/usr/bin/env python3
"""
Evaluate Gemini 3 Pro on AIME 2025 using pass@1 (single attempt per problem),
optionally enabling Code Execution tool.

Input format: JSONL, one problem per line:
  {"id":"AIME2025-I-01","problem":"...","answer":104}

Output format: JSONL with per-trial records.

Notes:
- Uses Structured Outputs (JSON Schema) to force {"answer": int, "reason": str?}.
- AIME answers are integers 0..999 (leading zeros allowed in official format, but we compare as int).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError


class AimeResponse(BaseModel):
    answer: int = Field(..., ge=0, le=999, description="Final AIME answer as an integer 0..999.")
    reason: Optional[str] = Field(
        default=None,
        description="Optional short justification (keep it concise).",
    )


@dataclass
class TrialResult:
    id: str
    trial: int
    gold: int
    pred: Optional[int]
    correct: bool
    used_code_execution: bool
    raw_text: str
    error: Optional[str] = None


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no}: {e}") from e
            for k in ("id", "problem", "answer"):
                if k not in obj:
                    raise ValueError(f"Missing key '{k}' on line {line_no}")
            items.append(obj)
    return items


def build_client(args: argparse.Namespace) -> genai.Client:
    if args.vertex:
        if not args.project or not args.location:
            raise ValueError("--vertex requires --project and --location")
        return genai.Client(vertexai=True, project=args.project, location=args.location)

    api_key = args.api_key or os.getenv(args.api_key_env)
    if not api_key:
        raise ValueError(
            f"Missing API key. Provide --api_key or set env var {args.api_key_env}."
        )
    return genai.Client(api_key=api_key)


def make_prompt(problem_text: str) -> str:
    # Keep prompt minimal. Structured output enforces JSON shape.
    return (
        "Solve the following AIME 2025 problem.\n"
        "Return the final answer as an integer 0..999.\n"
        "If you provide 'reason', keep it very short.\n\n"
        f"Problem:\n{problem_text}\n"
    )


def detect_code_execution_used(resp: Any) -> bool:
    # When code execution is used, the response may include executable_code / code_execution_result parts.
    try:
        cand = resp.candidates[0]
        parts = cand.content.parts or []
        for p in parts:
            if getattr(p, "executable_code", None) is not None:
                return True
            if getattr(p, "code_execution_result", None) is not None:
                return True
    except Exception:
        return False
    return False


def call_model_once(
    client: genai.Client,
    model: str,
    problem_text: str,
    mode: str,
    temperature: float,
    max_output_tokens: int,
    thinking_level: Optional[str],
    retries: int,
) -> Tuple[Optional[int], bool, str, Optional[str]]:
    prompt = make_prompt(problem_text)

    cfg_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "max_output_tokens": max_output_tokens,
        # Force JSON output that matches schema:
        "response_mime_type": "application/json",
        "response_json_schema": AimeResponse.model_json_schema(),
    }

    if thinking_level:
        # Gemini 3 uses thinking_level via ThinkingConfig.
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

    if mode == "with_code_execution":
        cfg_kwargs["tools"] = [types.Tool(code_execution=types.ToolCodeExecution)]
    elif mode != "no_tools":
        raise ValueError(f"Unknown mode: {mode}")

    config = types.GenerateContentConfig(**cfg_kwargs)

    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )
            raw_text = resp.text or ""
            used_tool = detect_code_execution_used(resp)

            # Primary parse path: strict JSON schema.
            parsed = AimeResponse.model_validate_json(raw_text)
            return parsed.answer, used_tool, raw_text, None

        except (ValidationError, json.JSONDecodeError) as e:
            raw_text = ""
            try:
                raw_text = resp.text or ""  # type: ignore[name-defined]
            except Exception:
                pass
            last_err = f"Structured output parse failed: {e}"
            # No point retrying if the model consistently violates schema, but we still allow retries.
        except Exception as e:
            last_err = f"Request failed: {e}"

        if attempt < retries:
            sleep_s = min(2 ** attempt + random.random(), 10.0)
            time.sleep(sleep_s)

    return None, False, "", last_err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to AIME 2025 JSONL file.")
    ap.add_argument("--out", required=True, help="Write per-trial JSONL results here.")
    ap.add_argument("--model", default="gemini-3-pro-preview")
    ap.add_argument("--mode", choices=["no_tools", "with_code_execution"], default="no_tools")
    ap.add_argument("--trials", type=int, default=1, help="Repeat each problem N times (pass@1 averaged).")
    ap.add_argument("--max_examples", type=int, default=0, help="If >0, only run first N problems.")
    ap.add_argument("--seed", type=int, default=0)

    # Auth
    ap.add_argument("--vertex", action="store_true", help="Use Vertex AI auth.")
    ap.add_argument("--project", default="", help="GCP project (Vertex).")
    ap.add_argument("--location", default="us-central1", help="GCP location (Vertex).")
    ap.add_argument("--api_key", default="", help="Gemini Developer API key (non-Vertex).")
    ap.add_argument("--api_key_env", default="GEMINI_API_KEY", help="Env var for API key.")

    # Generation controls
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_output_tokens", type=int, default=4096)
    ap.add_argument("--thinking_level", default="", help="Gemini 3: low|high (default is high if unset).")

    ap.add_argument("--retries", type=int, default=2)
    args = ap.parse_args()

    if args.seed:
        random.seed(args.seed)

    client = build_client(args)
    items = load_jsonl(args.input)
    if args.max_examples and args.max_examples > 0:
        items = items[: args.max_examples]

    thinking_level = args.thinking_level.strip() or None

    results: List[TrialResult] = []
    total = 0
    correct = 0
    tool_used_count = 0

    with open(args.out, "w", encoding="utf-8") as fout:
        for obj in items:
            pid = str(obj["id"])
            problem = str(obj["problem"])
            gold = int(obj["answer"])

            for t in range(args.trials):
                pred, used_tool, raw_text, err = call_model_once(
                    client=client,
                    model=args.model,
                    problem_text=problem,
                    mode=args.mode,
                    temperature=args.temperature,
                    max_output_tokens=args.max_output_tokens,
                    thinking_level=thinking_level,
                    retries=args.retries,
                )

                is_correct = (pred == gold)
                tr = TrialResult(
                    id=pid,
                    trial=t,
                    gold=gold,
                    pred=pred,
                    correct=is_correct,
                    used_code_execution=used_tool,
                    raw_text=raw_text,
                    error=err,
                )
                results.append(tr)

                fout.write(json.dumps(asdict(tr), ensure_ascii=False) + "\n")
                fout.flush()

                total += 1
                correct += int(is_correct)
                tool_used_count += int(used_tool)

                # Minimal progress line
                status = "OK" if is_correct else "WRONG"
                print(f"[{pid}][trial {t}] pred={pred} gold={gold} {status} tool={used_tool}")

    acc = correct / total if total else 0.0
    tool_rate = tool_used_count / total if total else 0.0

    print("\n=== Summary ===")
    print(f"model: {args.model}")
    print(f"mode:  {args.mode}")
    print(f"n_problems: {len(items)}")
    print(f"trials/problem: {args.trials}")
    print(f"accuracy: {acc:.4f} ({correct}/{total})")
    if args.mode == "with_code_execution":
        print(f"code_execution_used_rate: {tool_rate:.4f}")


if __name__ == "__main__":
    main()
