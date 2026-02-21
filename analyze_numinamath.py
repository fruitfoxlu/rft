#!/usr/bin/env python3
"""Analyze NuminaMath-1.5 for SFT data preparation.

Reports:
  1. Filter coverage: how many samples pass each filter stage
  2. Source distribution after filtering
  3. Token length distributions (prompt + solution) for max_seq_len selection
  4. Edge case inspection: samples near filter boundaries

Usage:
  python analyze_numinamath.py [--tokenizer openai/gpt-oss-20b]
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from datasets import load_dataset


def compute_percentiles(lengths: list[int], label: str):
    """Print percentile statistics for a list of lengths."""
    arr = np.array(lengths)
    print(f"\n  {label} token length distribution (n={len(arr)}):")
    for p in [50, 75, 90, 95, 99, 99.5]:
        print(f"    p{p:5.1f}: {int(np.percentile(arr, p)):>7,}")
    print(f"    max  : {int(arr.max()):>7,}")
    print(f"    mean : {arr.mean():>7,.0f}")

    # Report truncation rates at various max_seq_len
    print(f"\n  Truncation rates by max_seq_len:")
    for max_len in [4096, 8192, 16384, 32768]:
        truncated = (arr > max_len).sum()
        pct = truncated / len(arr) * 100
        print(f"    {max_len:>6}: {truncated:>6,} truncated ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="openai/gpt-oss-20b",
                        help="Tokenizer to use for length estimation")
    parser.add_argument("--sample-size", type=int, default=0,
                        help="If >0, sample this many for tokenization (faster)")
    args = parser.parse_args()

    print("=== Loading NuminaMath-1.5 ===")
    ds = load_dataset("AI-MO/NuminaMath-1.5", split="train")
    total = len(ds)
    print(f"Total examples: {total:,}")

    # --- Stage 1: Explore field distributions ---
    print("\n=== Field Distribution (full dataset) ===")

    question_types = Counter()
    problem_types = Counter()
    sources = Counter()
    valid_problem = Counter()
    valid_solution = Counter()
    synthetic_count = Counter()
    answer_types = Counter()

    for row in ds:
        question_types[row["question_type"]] += 1
        problem_types[row["problem_type"]] += 1
        sources[row["source"]] += 1
        valid_problem[row["problem_is_valid"]] += 1
        valid_solution[row["solution_is_valid"]] += 1
        synthetic_count[row["synthetic"]] += 1
        # Classify answer
        ans = (row["answer"] or "").strip()
        if not ans:
            answer_types["empty"] += 1
        elif ans.lower() in ("proof", "notfound", "n/a"):
            answer_types["non-numeric"] += 1
        else:
            # Check if it looks numeric (integer, decimal, fraction)
            answer_types["has_answer"] += 1

    print("\nquestion_type:")
    for k, v in question_types.most_common():
        print(f"  {k:30s}: {v:>7,} ({v/total*100:.1f}%)")

    print("\nproblem_type:")
    for k, v in problem_types.most_common(20):
        print(f"  {k:30s}: {v:>7,} ({v/total*100:.1f}%)")
    if len(problem_types) > 20:
        print(f"  ... and {len(problem_types) - 20} more types")

    print("\nsource (top 20):")
    for k, v in sources.most_common(20):
        print(f"  {k:30s}: {v:>7,} ({v/total*100:.1f}%)")
    if len(sources) > 20:
        print(f"  ... and {len(sources) - 20} more sources")

    print("\nproblem_is_valid:")
    for k, v in valid_problem.most_common():
        print(f"  {k:30s}: {v:>7,} ({v/total*100:.1f}%)")

    print("\nsolution_is_valid:")
    for k, v in valid_solution.most_common():
        print(f"  {k:30s}: {v:>7,} ({v/total*100:.1f}%)")

    print("\nsynthetic:")
    for k, v in synthetic_count.most_common():
        print(f"  {str(k):30s}: {v:>7,} ({v/total*100:.1f}%)")

    print("\nanswer classification:")
    for k, v in answer_types.most_common():
        print(f"  {k:30s}: {v:>7,} ({v/total*100:.1f}%)")

    # --- Stage 2: Apply filtering ---
    print("\n=== Applying Filters ===")

    # Check which sources match amc/aime
    amc_aime_sources = [s for s in sources if "amc" in s.lower() or "aime" in s.lower()]
    print(f"\nSources matching 'amc' or 'aime': {amc_aime_sources}")
    for s in amc_aime_sources:
        print(f"  {s}: {sources[s]:,}")

    filtered = []
    filter_stats = Counter()

    for row in ds:
        # Filter 1: question_type
        if row["question_type"] != "math-word-problem":
            filter_stats["rejected_question_type"] += 1
            continue

        # Filter 2: valid answer (not proof/notfound/empty)
        ans = (row["answer"] or "").strip()
        if not ans or ans.lower() in ("proof", "notfound", "n/a", "none"):
            filter_stats["rejected_answer"] += 1
            continue

        # Filter 3: exclude amc_aime sources
        src = row["source"].lower()
        if "amc" in src or "aime" in src:
            filter_stats["rejected_amc_aime"] += 1
            continue

        # Filter 4: valid problem and solution (optional, let's see impact)
        # Skip this for now — just count
        if row["problem_is_valid"] != "Yes":
            filter_stats["note_invalid_problem"] += 1
        if row["solution_is_valid"] != "Yes":
            filter_stats["note_invalid_solution"] += 1

        filtered.append(row)
        filter_stats["passed"] += 1

    print(f"\nFilter results:")
    print(f"  Total:                {total:>7,}")
    for k, v in filter_stats.most_common():
        print(f"  {k:24s}: {v:>7,} ({v/total*100:.1f}%)")
    print(f"  Pass rate:            {len(filtered)/total*100:.1f}%")

    # Post-filter source distribution
    post_sources = Counter(r["source"] for r in filtered)
    print(f"\nPost-filter source distribution (top 20, n={len(filtered):,}):")
    for k, v in post_sources.most_common(20):
        print(f"  {k:30s}: {v:>7,} ({v/len(filtered)*100:.1f}%)")

    # Additional filter: also require valid problem + solution
    strict_filtered = [r for r in filtered
                       if r["problem_is_valid"] == "Yes"
                       and r["solution_is_valid"] == "Yes"]
    print(f"\nStrict filter (also require valid problem + solution): {len(strict_filtered):,} "
          f"({len(strict_filtered)/total*100:.1f}%)")

    # --- Stage 3: Token length distribution ---
    print("\n=== Token Length Distribution ===")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        print(f"Using tokenizer: {args.tokenizer}")
    except Exception as e:
        print(f"Could not load tokenizer {args.tokenizer}: {e}")
        print("Falling back to character-based length estimate (÷4 for rough token count)")
        tokenizer = None

    # Sample if requested (tokenization can be slow for 500k+ samples)
    if args.sample_size > 0 and args.sample_size < len(filtered):
        import random
        random.seed(42)
        sample = random.sample(filtered, args.sample_size)
        print(f"Sampling {args.sample_size:,} examples for tokenization")
    else:
        sample = filtered
        print(f"Tokenizing all {len(filtered):,} filtered examples")

    prompt_lengths = []
    solution_lengths = []
    combined_lengths = []

    # Format as the SFT training would: user prompt + assistant solution
    for i, row in enumerate(sample):
        prompt = row["problem"]
        solution = row["solution"]

        if tokenizer is not None:
            p_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            s_len = len(tokenizer.encode(solution, add_special_tokens=False))
        else:
            p_len = len(prompt) // 4
            s_len = len(solution) // 4

        prompt_lengths.append(p_len)
        solution_lengths.append(s_len)
        combined_lengths.append(p_len + s_len)

        if (i + 1) % 50000 == 0:
            print(f"  Tokenized {i+1:,}/{len(sample):,}...")

    compute_percentiles(prompt_lengths, "Prompt")
    compute_percentiles(solution_lengths, "Solution")
    compute_percentiles(combined_lengths, "Combined (prompt + solution)")

    # --- Stage 4: Edge cases ---
    print("\n=== Edge Case Inspection ===")

    # Samples with very long solutions
    if combined_lengths:
        sorted_idx = np.argsort(combined_lengths)
        print("\nLongest 5 examples (combined tokens):")
        for i in sorted_idx[-5:]:
            row = sample[i]
            print(f"  {combined_lengths[i]:,} tokens | source={row['source']} | "
                  f"answer={row['answer'][:50]} | problem={row['problem'][:80]}...")

        print("\nShortest 5 examples (combined tokens):")
        for i in sorted_idx[:5]:
            row = sample[i]
            print(f"  {combined_lengths[i]:,} tokens | source={row['source']} | "
                  f"answer={row['answer'][:50]} | problem={row['problem'][:80]}...")

    # Samples with suspicious answers
    print("\nAnswer format inspection (random 20 from filtered):")
    import random
    random.seed(123)
    for row in random.sample(filtered, min(20, len(filtered))):
        ans = row["answer"].strip()
        print(f"  [{row['source'][:20]:20s}] answer='{ans[:60]}'")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
