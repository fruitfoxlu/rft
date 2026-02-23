#!/usr/bin/env python3
"""Build OOD-1000 probe set for experiment matrix.

Combines:
- All 32 integer-answer Apex Shortlist problems
- All 32 integer-answer competition problems from existing OOD probe (non-math45)
- Remaining ~936 from MATH45 (integer-answer, Level 4-5)

Verified disjoint from:
- Training pool (sft_rl_pool.jsonl, 50k problems)
- ID probe (probe_set_200.jsonl, 200 problems)
- AIME eval (aime_eval.jsonl, 18 problems)

Output: data/probe_set_1000_ood.jsonl
"""
import json
import hashlib
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
SEED = 42
TARGET_SIZE = 1000


def normalize_text(text: str) -> str:
    """Normalize problem text for deduplication."""
    return " ".join(text.split()).strip().lower()


def content_hash(text: str) -> str:
    return hashlib.md5(normalize_text(text).encode()).hexdigest()


def load_jsonl(path):
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def is_integer_label(label) -> bool:
    """Check if label is a plain integer (possibly negative)."""
    return str(label).strip().lstrip("-").isdigit()


def main():
    # 1. Load exclusion sets (training pool, ID probe, AIME)
    print("Loading exclusion sets...")
    exclude_hashes = set()

    for name, fname in [
        ("training pool", "sft_rl_pool.jsonl"),
        ("ID probe", "probe_set_200.jsonl"),
        ("AIME eval", "aime_eval.jsonl"),
    ]:
        problems = load_jsonl(DATA_DIR / fname)
        for p in problems:
            exclude_hashes.add(content_hash(p["input"]))
        print(f"  {name}: {len(problems)} problems, {len(exclude_hashes)} total exclusions")

    # 2. Load existing OOD probe (all 202 problems are our starting set)
    print("\nLoading existing OOD probe (202 problems)...")
    existing_ood = load_jsonl(DATA_DIR / "probe_set_200_ood.jsonl")
    ood_hashes = {content_hash(p["input"]) for p in existing_ood}
    print(f"  {len(existing_ood)} problems loaded")

    # Verify existing OOD is disjoint from exclusion sets
    overlap = ood_hashes & exclude_hashes
    if overlap:
        print(f"  WARNING: {len(overlap)} existing OOD problems overlap with exclusion sets!")
    else:
        print("  Verified: existing OOD is disjoint from training pool/ID/AIME")

    # Start with all existing OOD problems
    result = list(existing_ood)
    seen_hashes = set(ood_hashes)

    # 3. Load Apex Shortlist (add any integer-answer problems not already in OOD)
    print("\nLoading Apex Shortlist...")
    apex = load_jsonl(DATA_DIR / "apex_shortlist.jsonl")
    apex_added = 0
    for p in apex:
        if not is_integer_label(p.get("label", "")):
            continue
        h = content_hash(p["input"])
        if h in seen_hashes or h in exclude_hashes:
            continue
        result.append({
            "input": p["input"],
            "label": str(p["label"]).strip(),
            "source": p.get("source", "apex"),
        })
        seen_hashes.add(h)
        apex_added += 1
    print(f"  Added {apex_added} new Apex problems (total now: {len(result)})")

    # 4. Load MATH45 (train + eval), filter to integer answers, exclude overlaps
    print("\nLoading MATH45 (train + eval)...")
    math45_candidates = []
    for fname in ["math45_train.jsonl", "math45_eval.jsonl"]:
        problems = load_jsonl(DATA_DIR / fname)
        for p in problems:
            if not is_integer_label(p.get("label", "")):
                continue
            h = content_hash(p["input"])
            if h in seen_hashes or h in exclude_hashes:
                continue
            math45_candidates.append({
                "input": p["input"],
                "label": str(p["label"]).strip(),
                "source": "math45",
                "_hash": h,
            })
    print(f"  {len(math45_candidates)} unique integer-answer MATH45 problems available")

    # 5. Sample from MATH45 to reach target
    needed = TARGET_SIZE - len(result)
    if needed > len(math45_candidates):
        print(f"  WARNING: Need {needed} but only {len(math45_candidates)} available. Using all.")
        needed = len(math45_candidates)

    rng = random.Random(SEED)
    rng.shuffle(math45_candidates)
    for p in math45_candidates[:needed]:
        seen_hashes.add(p["_hash"])
        result.append({
            "input": p["input"],
            "label": p["label"],
            "source": p["source"],
        })
    print(f"  Added {needed} MATH45 problems (total now: {len(result)})")

    # 6. Final verification
    print(f"\nFinal verification:")
    final_hashes = set()
    for p in result:
        h = content_hash(p["input"])
        if h in final_hashes:
            print(f"  DUPLICATE found!")
        final_hashes.add(h)

    overlap_train = final_hashes & exclude_hashes
    print(f"  Total problems: {len(result)}")
    print(f"  Unique hashes: {len(final_hashes)}")
    print(f"  Overlap with exclusion sets: {len(overlap_train)}")

    # Source breakdown
    source_counts = {}
    for p in result:
        src = p.get("source", "unknown")
        if src == "math45":
            source_counts["math45"] = source_counts.get("math45", 0) + 1
        else:
            source_counts["competition"] = source_counts.get("competition", 0) + 1
    print(f"  Sources: {source_counts}")

    # Statistical power
    import math
    n = len(result)
    p_est = 0.65
    se = math.sqrt(p_est * (1 - p_est) / n)
    print(f"\n  SE at p=0.65: {se*100:.2f}%")
    print(f"  95% CI: ±{1.96*se*100:.2f}pp")
    print(f"  Detectable effect (2σ): {2*1.96*se*100:.2f}pp")

    # 7. Save
    output_path = DATA_DIR / "probe_set_1000_ood.jsonl"
    with open(output_path, "w") as f:
        for p in result:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(result)} problems to {output_path}")


if __name__ == "__main__":
    main()
