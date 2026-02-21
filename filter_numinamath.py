#!/usr/bin/env python3
"""Filter NuminaMath-1.5 for SFT data: integer-only answers, exclude AMC/AIME.

Reports filter coverage and prepares filtered dataset.
"""

import json
import os
import re
from collections import Counter

from datasets import load_dataset


def is_clean_integer(answer: str) -> bool:
    """Check if answer is a clean integer (possibly negative, with commas)."""
    s = answer.strip()
    # Remove LaTeX wrappers
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\textbf\{[^}]*\}", "", s)
    s = re.sub(r"[\$]", "", s)
    s = s.strip()
    # Remove commas from numbers
    s = s.replace(",", "")
    try:
        int(s)
        return True
    except ValueError:
        return False


def is_clean_numeric(answer: str) -> bool:
    """Check if answer is a clean numeric (integer, decimal, or simple fraction)."""
    s = answer.strip()
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\textbf\{[^}]*\}", "", s)
    s = re.sub(r"[\$]", "", s)
    s = s.strip()
    s = s.replace(",", "")
    # Plain integer
    try:
        int(s)
        return True
    except ValueError:
        pass
    # Plain float
    try:
        float(s)
        return True
    except ValueError:
        pass
    # \frac{a}{b}
    if re.match(r"^\\(?:d?frac|tfrac)\{-?\d+\}\{-?\d+\}$", s):
        return True
    # a/b format
    if re.match(r"^-?\d+/\d+$", s):
        return True
    return False


def main():
    print("=== Loading NuminaMath-1.5 ===")
    ds = load_dataset("AI-MO/NuminaMath-1.5", split="train")
    total = len(ds)
    print(f"Total examples: {total:,}")

    integer_only = []
    numeric_only = []
    rejected_answers = Counter()

    for row in ds:
        if row["question_type"] != "math-word-problem":
            continue
        ans = (row["answer"] or "").strip()
        if not ans or ans.lower() in ("proof", "notfound", "n/a", "none"):
            continue
        src = row["source"].lower()
        if "amc" in src or "aime" in src:
            continue

        if is_clean_integer(ans):
            integer_only.append(row)
        elif is_clean_numeric(ans):
            numeric_only.append(row)
        else:
            rejected_answers[ans[:60]] += 1

    print(f"\n=== Filter Results ===")
    print(f"Integer-only answers: {len(integer_only):,}")
    print(f"Other numeric (decimal/fraction): {len(numeric_only):,}")
    print(f"Total numeric: {len(integer_only) + len(numeric_only):,}")
    print(f"Rejected (non-numeric): {sum(rejected_answers.values()):,}")

    # Source distribution for integer-only
    src_dist = Counter(r["source"] for r in integer_only)
    print(f"\nInteger-only source distribution:")
    for k, v in src_dist.most_common(15):
        print(f"  {k:25s}: {v:>7,} ({v/len(integer_only)*100:.1f}%)")

    # Top rejected answers
    print(f"\nTop 20 rejected answer patterns:")
    for ans, cnt in rejected_answers.most_common(20):
        print(f"  [{cnt:>4}] {ans}")

    # Also check: what do the numeric-only (non-integer) look like?
    print(f"\nSample non-integer numeric answers (first 20):")
    for row in numeric_only[:20]:
        print(f"  [{row['source'][:20]:20s}] answer='{row['answer'].strip()[:60]}'")

    # Report answer range for integers
    int_values = []
    for row in integer_only:
        s = row["answer"].strip()
        s = re.sub(r"\\text\{[^}]*\}", "", s)
        s = re.sub(r"[\$]", "", s).strip().replace(",", "")
        try:
            int_values.append(int(s))
        except ValueError:
            pass

    if int_values:
        import numpy as np
        arr = np.array(int_values)
        print(f"\nInteger answer value distribution:")
        print(f"  min: {arr.min()}, max: {arr.max()}")
        print(f"  p25: {int(np.percentile(arr, 25))}, median: {int(np.percentile(arr, 50))}, p75: {int(np.percentile(arr, 75))}")
        in_0_999 = ((arr >= 0) & (arr <= 999)).sum()
        print(f"  In [0, 999]: {in_0_999:,} ({in_0_999/len(arr)*100:.1f}%)")


if __name__ == "__main__":
    main()
