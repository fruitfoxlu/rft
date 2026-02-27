#!/usr/bin/env python3
"""Prepare RL training data for Qwen2.5-14B: append BOXED_SUFFIX to prompts.

Reads source JSONL files and appends the BOXED_SUFFIX instruction to each
problem's 'input' field. Produces suffixed versions for both the training
pool and the OOD probe set.

Outputs:
  data/sft_rl_pool_3200_boxed.jsonl   (training pool, 3200 prompts)
  data/probe_set_200_ood_boxed.jsonl  (in-run OOD monitor, 202 problems)

Usage:
    python prepare_rl_data_qwen.py
"""

import hashlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

BOXED_SUFFIX = (
    "\n\nPlease reason step by step but keep it concise, and put your final "
    "answer within \\boxed{...}. In the final line, output ONLY "
    "\\boxed{<integer>} and nothing else."
)

# Source -> output mapping
FILE_PAIRS = [
    (DATA_DIR / "sft_rl_pool_3200.jsonl", DATA_DIR / "sft_rl_pool_3200_boxed.jsonl"),
    (DATA_DIR / "probe_set_200_ood.jsonl", DATA_DIR / "probe_set_200_ood_boxed.jsonl"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def process_file(src: Path, dst: Path) -> int:
    """Append BOXED_SUFFIX to each record's 'input' field. Returns count."""
    records = []
    with open(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record["input"] = record["input"] + BOXED_SUFFIX
            records.append(record)

    with open(dst, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(records)


def main():
    print("=== Preparing BOXED_SUFFIX data for Qwen2.5-14B RL training ===\n")
    print(f"BOXED_SUFFIX ({len(BOXED_SUFFIX)} chars):")
    print(f"  {BOXED_SUFFIX!r}\n")

    for src, dst in FILE_PAIRS:
        if not src.exists():
            print(f"ERROR: Source file not found: {src}", file=sys.stderr)
            sys.exit(1)

        count = process_file(src, dst)
        src_hash = sha256_file(src)
        dst_hash = sha256_file(dst)

        print(f"{src.name} -> {dst.name}")
        print(f"  Records: {count}")
        print(f"  Source SHA256: {src_hash}")
        print(f"  Output SHA256: {dst_hash}")
        print()

    print("Done. Embed output SHA256 hashes in training script header for provenance.")


if __name__ == "__main__":
    main()
