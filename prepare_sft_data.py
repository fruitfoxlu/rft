#!/usr/bin/env python3
"""Prepare NuminaMath-1.5 SFT data for gpt-oss-20b training.

Filters for integer-only answers, excludes AMC/AIME (holdout),
formats as OpenRLHF-compatible JSONL with solution ending in \\boxed{answer}.

Outputs:
  data/sft_train.jsonl   — Training set
  data/sft_dev.jsonl     — Dev set (for eval during SFT)
  data/sft_rl_pool.jsonl — Separate pool for RL (no overlap with SFT train)

Usage:
  python prepare_sft_data.py [--sft-size 50000] [--rl-pool-size 50000] [--seed 42]
"""

import argparse
import hashlib
import json
import os
import re
import random
from collections import Counter

from datasets import load_dataset


def is_clean_integer(answer: str) -> bool:
    """Check if answer is a clean integer."""
    s = answer.strip()
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\textbf\{[^}]*\}", "", s)
    s = re.sub(r"[\$]", "", s)
    s = s.strip().replace(",", "")
    try:
        int(s)
        return True
    except ValueError:
        return False


def normalize_integer_answer(answer: str) -> str:
    """Normalize answer to a clean integer string."""
    s = answer.strip()
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\textbf\{[^}]*\}", "", s)
    s = re.sub(r"[\$]", "", s)
    s = s.strip().replace(",", "")
    return str(int(s))


def problem_hash(problem_text: str) -> str:
    """Hash problem text for deduplication."""
    # Normalize whitespace and case for hashing
    normalized = re.sub(r"\s+", " ", problem_text.strip().lower())
    return hashlib.md5(normalized.encode()).hexdigest()


def ensure_boxed_ending(solution: str, answer: str) -> str:
    """Ensure the solution ends with \\boxed{answer}.

    If the solution already ends with \\boxed{...}, keep it.
    Otherwise, append \\boxed{answer}.
    """
    # Check if solution already has \boxed at the end
    if re.search(r"\\boxed\{[^}]*\}\s*\.?\s*$", solution):
        return solution

    # Append boxed answer
    solution = solution.rstrip()
    if not solution.endswith("."):
        solution += "."
    solution += f"\n\nThe answer is $\\boxed{{{answer}}}$."
    return solution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-size", type=int, default=50000,
                        help="Number of SFT training examples (default: 50k)")
    parser.add_argument("--rl-pool-size", type=int, default=50000,
                        help="Number of RL pool examples (default: 50k)")
    parser.add_argument("--dev-size", type=int, default=500,
                        help="Number of dev examples (default: 500)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require-valid", action="store_true",
                        help="Also require problem_is_valid=Yes and solution_is_valid=Yes")
    parser.add_argument("--max-answer-abs", type=int, default=0,
                        help="If >0, filter out answers with |answer| > this value")
    args = parser.parse_args()

    print("=== Loading NuminaMath-1.5 ===")
    ds = load_dataset("AI-MO/NuminaMath-1.5", split="train")
    print(f"Total: {len(ds):,}")

    # Apply filters
    candidates = []
    seen_hashes = set()
    filter_stats = Counter()

    for row in ds:
        # Filter: question_type
        if row["question_type"] != "math-word-problem":
            filter_stats["reject_question_type"] += 1
            continue

        # Filter: integer answer
        ans = (row["answer"] or "").strip()
        if not ans or not is_clean_integer(ans):
            filter_stats["reject_non_integer"] += 1
            continue

        # Filter: exclude AMC/AIME
        src = row["source"].lower()
        if "amc" in src or "aime" in src:
            filter_stats["reject_amc_aime"] += 1
            continue

        # Filter: validity (optional)
        if args.require_valid:
            if row["problem_is_valid"] != "Yes" or row["solution_is_valid"] != "Yes":
                filter_stats["reject_invalid"] += 1
                continue

        # Filter: answer magnitude (optional)
        int_ans = int(normalize_integer_answer(ans))
        if args.max_answer_abs > 0 and abs(int_ans) > args.max_answer_abs:
            filter_stats["reject_magnitude"] += 1
            continue

        # Dedup by problem text hash
        h = problem_hash(row["problem"])
        if h in seen_hashes:
            filter_stats["reject_duplicate"] += 1
            continue
        seen_hashes.add(h)

        # Filter: very short problems (likely garbage)
        if len(row["problem"].strip()) < 20:
            filter_stats["reject_too_short"] += 1
            continue

        # Filter: empty solution
        if not row["solution"] or len(row["solution"].strip()) < 10:
            filter_stats["reject_empty_solution"] += 1
            continue

        candidates.append({
            "problem": row["problem"],
            "solution": row["solution"],
            "answer": normalize_integer_answer(ans),
            "source": row["source"],
            "synthetic": row["synthetic"],
        })
        filter_stats["passed"] += 1

    print(f"\nFilter stats:")
    for k, v in filter_stats.most_common():
        print(f"  {k:25s}: {v:>7,}")
    print(f"Candidates after filtering: {len(candidates):,}")

    # Shuffle
    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    # Split into SFT train, dev, and RL pool
    total_needed = args.sft_size + args.dev_size + args.rl_pool_size
    if total_needed > len(candidates):
        print(f"\nWarning: requested {total_needed:,} but only {len(candidates):,} available.")
        print(f"Scaling down proportionally.")
        ratio = len(candidates) / total_needed
        args.sft_size = int(args.sft_size * ratio)
        args.rl_pool_size = int(args.rl_pool_size * ratio)
        args.dev_size = min(args.dev_size, len(candidates) - args.sft_size - args.rl_pool_size)

    dev_set = candidates[:args.dev_size]
    sft_train = candidates[args.dev_size:args.dev_size + args.sft_size]
    rl_pool = candidates[args.dev_size + args.sft_size:args.dev_size + args.sft_size + args.rl_pool_size]

    print(f"\nSplit sizes:")
    print(f"  SFT train: {len(sft_train):,}")
    print(f"  Dev:        {len(dev_set):,}")
    print(f"  RL pool:    {len(rl_pool):,}")
    print(f"  Unused:     {len(candidates) - len(sft_train) - len(dev_set) - len(rl_pool):,}")

    # Source distribution
    for name, subset in [("SFT train", sft_train), ("RL pool", rl_pool)]:
        src_dist = Counter(s["source"] for s in subset)
        print(f"\n{name} source distribution:")
        for k, v in src_dist.most_common(10):
            print(f"  {k:25s}: {v:>7,} ({v/len(subset)*100:.1f}%)")

    # Format and save
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    def save_jsonl(items, filename, include_solution=True):
        path = os.path.join(data_dir, filename)
        with open(path, "w") as f:
            for item in items:
                if include_solution:
                    # SFT format: input (problem) + output (solution with boxed answer)
                    solution = ensure_boxed_ending(item["solution"], item["answer"])
                    record = {
                        "input": item["problem"],
                        "output": solution,
                        "label": item["answer"],
                    }
                else:
                    # RL format: just input + label (model generates solution)
                    record = {
                        "input": item["problem"],
                        "label": item["answer"],
                    }
                f.write(json.dumps(record) + "\n")
        print(f"Saved {len(items):,} examples to {path}")
        return path

    sft_train_path = save_jsonl(sft_train, "sft_train.jsonl", include_solution=True)
    dev_path = save_jsonl(dev_set, "sft_dev.jsonl", include_solution=True)
    rl_path = save_jsonl(rl_pool, "sft_rl_pool.jsonl", include_solution=False)

    # Verify no overlap
    sft_hashes = {problem_hash(s["problem"]) for s in sft_train}
    dev_hashes = {problem_hash(s["problem"]) for s in dev_set}
    rl_hashes = {problem_hash(s["problem"]) for s in rl_pool}
    assert len(sft_hashes & dev_hashes) == 0, "SFT/dev overlap!"
    assert len(sft_hashes & rl_hashes) == 0, "SFT/RL overlap!"
    assert len(dev_hashes & rl_hashes) == 0, "Dev/RL overlap!"
    print("\nNo overlap between splits (verified).")

    # Print a few examples
    print("\n=== Sample SFT examples ===")
    for item in sft_train[:3]:
        sol_formatted = ensure_boxed_ending(item["solution"], item["answer"])
        print(f"\n--- Source: {item['source']} | Answer: {item['answer']} ---")
        print(f"Problem: {item['problem'][:200]}...")
        print(f"Solution (last 200 chars): ...{sol_formatted[-200:]}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
