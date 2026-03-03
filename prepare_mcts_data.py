#!/usr/bin/env python3
"""Convert our JSONL training pool to Marco-o1 MCTS input format.

Input:  data/sft_rl_pool_3200.jsonl  (fields: input, label)
Output: data/mcts_input_3200.json    (JSON array of {id, problem, solution})

The MCTS engine expects a JSON array where each entry has:
  - id: unique identifier string
  - problem: the math problem text
  - solution: ground truth answer (string or number)
"""

import json
import hashlib

INPUT_PATH = "data/sft_rl_pool_3200.jsonl"
OUTPUT_PATH = "data/mcts_input_3200.json"

# BOXED_SUFFIX that may have been appended in _boxed variants — strip it
BOXED_SUFFIX = (
    "\n\nPlease reason step by step but keep it concise, and put your final "
    "answer within \\boxed{...}. In the final line, output ONLY \\boxed{<integer>} "
    "and nothing else."
)


def main():
    with open(INPUT_PATH) as f:
        lines = f.readlines()

    records = []
    for i, line in enumerate(lines):
        entry = json.loads(line)
        problem = entry["input"]
        label = entry["label"]

        # Strip BOXED_SUFFIX if present (MCTS uses its own prompt template)
        if problem.endswith(BOXED_SUFFIX):
            problem = problem[: -len(BOXED_SUFFIX)]

        records.append({
            "id": f"math_{i:04d}",
            "problem": problem,
            "solution": str(label),
        })

    with open(OUTPUT_PATH, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # Provenance
    sha = hashlib.sha256(open(OUTPUT_PATH, "rb").read()).hexdigest()
    print(f"Wrote {len(records)} problems to {OUTPUT_PATH}")
    print(f"SHA256: {sha}")


if __name__ == "__main__":
    main()
