"""Download math datasets, split train/eval, format for OpenRLHF.

Datasets:
  - AIME: MathArena/aime_2024_I, aime_2024_II, aime_2025_I, aime_2025_II, aime_2026
  - MATH Level 4-5: EleutherAI/hendrycks_math (all configs, filtered to Level 4+5)

Output:
  data/aime_train.jsonl, data/aime_eval.jsonl
  data/math45_train.jsonl, data/math45_eval.jsonl
  data/train_prompts.jsonl  (selected dataset, symlinked after eval_baseline.py)
  data/eval_prompts.jsonl
"""

import json
import os
import re
import random

from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

AIME_DATASETS = [
    "MathArena/aime_2024_I",
    "MathArena/aime_2024_II",
    "MathArena/aime_2025_I",
    "MathArena/aime_2025_II",
    "MathArena/aime_2026",
]

MATH_CONFIGS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]

EVAL_FRACTION = 0.2  # 20% held out for eval
SEED = 42


def extract_boxed_answer(solution: str) -> str:
    """Extract the answer from \\boxed{...} in a MATH solution string."""
    # Handle nested braces by counting depth
    idx = solution.rfind("\\boxed{")
    if idx == -1:
        return ""
    idx += len("\\boxed{")
    depth = 1
    end = idx
    while end < len(solution) and depth > 0:
        if solution[end] == "{":
            depth += 1
        elif solution[end] == "}":
            depth -= 1
        end += 1
    return solution[idx : end - 1].strip()


def load_aime() -> list[dict]:
    """Load all AIME datasets and format as {input, label}."""
    problems = []
    for ds_name in AIME_DATASETS:
        try:
            ds = load_dataset(ds_name, split="train", trust_remote_code=False)
        except Exception as e:
            print(f"  Warning: could not load {ds_name}: {e}")
            continue
        short_name = ds_name.split("/")[-1]
        for row in ds:
            problems.append(
                {
                    "input": row["problem"],
                    "label": str(row["answer"]),
                    "source": short_name,
                }
            )
        print(f"  Loaded {len(ds)} problems from {ds_name}")
    return problems


def load_math_level45() -> list[dict]:
    """Load MATH dataset (Level 4 and 5 only) and format as {input, label}."""
    problems = []
    for config in MATH_CONFIGS:
        for split in ["train", "test"]:
            try:
                ds = load_dataset(
                    "EleutherAI/hendrycks_math", config, split=split, trust_remote_code=False
                )
            except Exception as e:
                print(f"  Warning: could not load hendrycks_math/{config}/{split}: {e}")
                continue
            count = 0
            for row in ds:
                level_str = row.get("level", "")
                try:
                    level_num = int(level_str.split()[-1]) if level_str.startswith("Level") else 0
                except ValueError:
                    level_num = 0
                if level_num < 4:
                    continue
                answer = extract_boxed_answer(row["solution"])
                if not answer:
                    continue
                problems.append(
                    {
                        "input": row["problem"],
                        "label": answer,
                        "source": f"math_{config}_L{level_num}",
                    }
                )
                count += 1
            print(f"  Loaded {count} Level 4-5 problems from hendrycks_math/{config}/{split}")
    return problems


def split_and_save(problems: list[dict], name: str):
    """Shuffle, split into train/eval, and save as JSONL."""
    random.Random(SEED).shuffle(problems)
    n_eval = max(1, int(len(problems) * EVAL_FRACTION))
    eval_set = problems[:n_eval]
    train_set = problems[n_eval:]

    train_path = os.path.join(DATA_DIR, f"{name}_train.jsonl")
    eval_path = os.path.join(DATA_DIR, f"{name}_eval.jsonl")

    for path, data in [(train_path, train_set), (eval_path, eval_set)]:
        with open(path, "w") as f:
            for item in data:
                # Only write input and label (drop source metadata)
                f.write(json.dumps({"input": item["input"], "label": item["label"]}) + "\n")

    print(f"  {name}: {len(train_set)} train, {len(eval_set)} eval")
    print(f"    Saved to {train_path}")
    print(f"    Saved to {eval_path}")
    return train_path, eval_path


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=== Loading AIME datasets ===")
    aime_problems = load_aime()
    print(f"Total AIME problems: {len(aime_problems)}")

    print("\n=== Loading MATH Level 4-5 datasets ===")
    math_problems = load_math_level45()
    print(f"Total MATH Level 4-5 problems: {len(math_problems)}")

    print("\n=== Splitting and saving ===")
    split_and_save(aime_problems, "aime")
    split_and_save(math_problems, "math45")

    print("\n=== Done ===")
    print(
        "Run eval_baseline.py next to decide which dataset to use for training."
    )
    print(
        "After that, symlink or copy the chosen train/eval files to "
        "data/train_prompts.jsonl and data/eval_prompts.jsonl."
    )


if __name__ == "__main__":
    main()
