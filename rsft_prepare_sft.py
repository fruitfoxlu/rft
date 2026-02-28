#!/usr/bin/env python3
"""RSFT Phase 2: Filter correct samples and prepare SFT training data.

Reads the raw samples from Phase 1, filters for EM=1, deduplicates,
and formats as chat-style SFT training data for Qwen2.5.

Usage:
    python rsft_prepare_sft.py [--k 16] [--max-per-problem 4]

Input:  /mnt/scratch/qwen14b_rsft/samples_k{k}.jsonl
Output: /mnt/scratch/qwen14b_rsft/sft_train.jsonl
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reward_func_em import normalize_answer

OUT_DIR = Path("/mnt/scratch/qwen14b_rsft")

BOXED_SUFFIX = (
    "\n\nPlease reason step by step but keep it concise, and put your final "
    "answer within \\boxed{...}. In the final line, output ONLY "
    "\\boxed{<integer>} and nothing else."
)


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="RSFT Data Preparation")
    parser.add_argument("--k", type=int, default=16, help="k used in sampling")
    parser.add_argument("--max-per-problem", type=int, default=4,
                        help="Max correct samples to keep per problem (diversity)")
    args = parser.parse_args()

    samples_path = OUT_DIR / f"samples_k{args.k}.jsonl"
    sft_path = OUT_DIR / "sft_train.jsonl"

    print(f"=== RSFT Data Preparation: {datetime.now(timezone.utc).isoformat()} ===")
    print(f"  Input: {samples_path}")

    samples = load_jsonl(samples_path)
    print(f"  Total samples: {len(samples)}")

    # Filter for correct samples
    correct_samples = [s for s in samples if s["correct"] == 1]
    print(f"  Correct samples: {len(correct_samples)} ({len(correct_samples)/len(samples)*100:.1f}%)")

    # Group by problem
    by_problem = defaultdict(list)
    for s in correct_samples:
        by_problem[s["problem_hash"]].append(s)

    print(f"  Problems with correct samples: {len(by_problem)}")

    # Deduplicate within each problem: keep unique (normalized_answer, output_hash) pairs
    # Then cap at max_per_problem for diversity
    sft_records = []
    for p_hash, problem_samples in sorted(by_problem.items()):
        seen = set()
        unique = []
        for s in problem_samples:
            # Deduplicate by normalized answer + first 200 chars of output
            norm_ans = normalize_answer(s.get("model_answer", ""))
            output_sig = hashlib.md5(s["output"][:200].encode()).hexdigest()[:8]
            dedup_key = (norm_ans, output_sig)
            if dedup_key not in seen:
                seen.add(dedup_key)
                unique.append(s)

        # Cap at max_per_problem, prefer shorter (more concise) solutions
        unique.sort(key=lambda s: len(s["output"]))
        selected = unique[:args.max_per_problem]

        for s in selected:
            # Format for OpenRLHF SFT (input/output keys + apply_chat_template)
            prompt_text = s["problem_input"] + BOXED_SUFFIX
            sft_records.append({
                "input": prompt_text,
                "output": s["output"],
                "problem_hash": s["problem_hash"],
                "truth": s["truth"],
                "model_answer": s["model_answer"],
                "parse_method": s["parse_method"],
            })

    print(f"  SFT training records: {len(sft_records)} "
          f"(max {args.max_per_problem} per problem)")

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(sft_path, "w") as f:
        for r in sft_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  Saved: {sft_path}")

    # Distribution stats
    problems_by_count = defaultdict(int)
    problem_counts = defaultdict(int)
    for r in sft_records:
        problem_counts[r["problem_hash"]] += 1
    for count in problem_counts.values():
        problems_by_count[count] += 1

    print(f"\n  Distribution (samples per problem):")
    for count in sorted(problems_by_count.keys()):
        print(f"    {count} samples: {problems_by_count[count]} problems")

    # Save summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": str(samples_path),
        "total_samples": len(samples),
        "correct_samples": len(correct_samples),
        "problems_with_correct": len(by_problem),
        "max_per_problem": args.max_per_problem,
        "sft_records": len(sft_records),
        "output": str(sft_path),
    }
    with open(OUT_DIR / "prepare_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary saved: {OUT_DIR / 'prepare_summary.json'}")


if __name__ == "__main__":
    main()
