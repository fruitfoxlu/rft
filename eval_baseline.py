"""Pre-flight evaluation: Teacher (Gemini 3 Pro) + Base Student (gpt-oss-120b).

Evaluates both models on eval sets from each dataset, prints accuracy table,
and recommends which dataset to use for GRPO training.

Usage:
  python eval_baseline.py [--max-problems N] [--vllm-url URL] [--maj-k K]
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# Reuse answer extraction/checking from reward_func
from reward_func import check_correctness, extract_model_answer, normalize_answer

GEMINI_MODEL = "gemini-3-pro-preview"
STUDENT_MODEL = "openai/gpt-oss-120b"

MATH_SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{}."
)


def load_eval_problems(path: str) -> list[dict]:
    """Load eval JSONL file."""
    problems = []
    with open(path) as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


# ---------------------------------------------------------------------------
# Teacher: Gemini 3 Pro
# ---------------------------------------------------------------------------

def eval_teacher_single(client, problem: str, label: str) -> dict:
    """Evaluate a single problem with Gemini 3 Pro."""
    from google.genai import types

    prompt = f"{MATH_SYSTEM_PROMPT}\n\nProblem: {problem}"
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=3072,
                    temperature=0.0,
                ),
            )
            text = response.text
            model_answer = extract_model_answer(text)
            correct = check_correctness(model_answer, label)
            return {
                "model_answer": model_answer,
                "correct": correct,
                "response": text,
            }
        except Exception as e:
            print(f"  Gemini error (attempt {attempt + 1}): {e}")
            time.sleep(2 * (attempt + 1))
    return {"model_answer": "", "correct": 0.0, "response": ""}


def eval_teacher(problems: list[dict], max_problems: int) -> list[dict]:
    """Evaluate Teacher (Gemini 3 Pro) on a set of problems."""
    from google import genai

    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "wf30-poc"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    subset = problems[:max_problems]
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(eval_teacher_single, client, p["input"], p["label"]): i
            for i, p in enumerate(subset)
        }
        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results.append((idx, result))
            status = "CORRECT" if result["correct"] > 0 else "WRONG"
            print(f"  Teacher [{idx + 1}/{len(subset)}] {status} "
                  f"(answer: {result['model_answer']}, truth: {subset[idx]['label']})")

    results.sort(key=lambda x: x[0])
    return [r for _, r in results]


# ---------------------------------------------------------------------------
# Base Student: gpt-oss-120b via vLLM (OpenAI-compatible API)
# ---------------------------------------------------------------------------

def eval_student_single(
    client, model: str, problem: str, label: str, n: int = 1
) -> dict:
    """Evaluate a single problem with the student model via vLLM OpenAI API."""
    prompt = f"{MATH_SYSTEM_PROMPT}\n\nProblem: {problem}"
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3072,
                temperature=0.0 if n == 1 else 0.7,
                n=n,
            )
            answers = []
            for choice in response.choices:
                text = choice.message.content or ""
                model_answer = extract_model_answer(text)
                correct = check_correctness(model_answer, label)
                answers.append({
                    "model_answer": model_answer,
                    "correct": correct,
                    "response": text,
                })
            return {"answers": answers}
        except Exception as e:
            print(f"  Student error (attempt {attempt + 1}): {e}")
            time.sleep(2 * (attempt + 1))
    return {"answers": [{"model_answer": "", "correct": 0.0, "response": ""}] * n}


def eval_student(
    problems: list[dict],
    max_problems: int,
    vllm_url: str,
    maj_k: int = 1,
) -> list[dict]:
    """Evaluate Base Student (gpt-oss-120b) on a set of problems."""
    from openai import OpenAI

    client = OpenAI(base_url=vllm_url, api_key="unused")

    # Detect the model name served by vLLM
    models = client.models.list()
    model_name = models.data[0].id if models.data else STUDENT_MODEL
    print(f"  Using vLLM model: {model_name}")

    subset = problems[:max_problems]
    results = []

    for i, p in enumerate(subset):
        result = eval_student_single(client, model_name, p["input"], p["label"], n=maj_k)
        results.append(result)

        # pass@1 from first answer
        pass1 = result["answers"][0]["correct"]
        # maj@k from majority vote
        if maj_k > 1:
            answers_list = [a["model_answer"] for a in result["answers"]]
            from collections import Counter
            counts = Counter(normalize_answer(a) for a in answers_list)
            majority_answer = counts.most_common(1)[0][0]
            maj_correct = check_correctness(majority_answer, p["label"])
        else:
            maj_correct = pass1

        status = "CORRECT" if pass1 > 0 else "WRONG"
        print(f"  Student [{i + 1}/{len(subset)}] pass@1={status} "
              f"(answer: {result['answers'][0]['model_answer']}, truth: {p['label']})")
        result["pass1"] = pass1
        result["maj_correct"] = maj_correct

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_table(dataset_name: str, teacher_results, student_results, maj_k: int):
    """Print accuracy comparison table."""
    teacher_acc = sum(r["correct"] for r in teacher_results) / len(teacher_results) * 100
    student_pass1 = sum(r["pass1"] for r in student_results) / len(student_results) * 100
    student_maj = sum(r["maj_correct"] for r in student_results) / len(student_results) * 100

    print(f"\n{'=' * 60}")
    print(f"  Dataset: {dataset_name} ({len(teacher_results)} problems)")
    print(f"{'=' * 60}")
    print(f"  {'Model':<30} {'Accuracy':>10}")
    print(f"  {'-' * 40}")
    print(f"  {'Teacher (Gemini 3 Pro)':<30} {teacher_acc:>9.1f}%")
    print(f"  {'Base Student pass@1':<30} {student_pass1:>9.1f}%")
    if maj_k > 1:
        print(f"  {'Base Student maj@' + str(maj_k):<30} {student_maj:>9.1f}%")
    print(f"{'=' * 60}")

    return teacher_acc, student_pass1, student_maj


def recommend_dataset(results: dict[str, dict]) -> str:
    """Recommend dataset based on base student accuracy."""
    print(f"\n{'=' * 60}")
    print("  RECOMMENDATION")
    print(f"{'=' * 60}")

    best_dataset = None
    best_score = -1

    for name, stats in results.items():
        acc = stats["student_pass1"]
        if 20 <= acc <= 70:
            # In the sweet spot -- prefer the one closest to 40-50%
            score = 100 - abs(acc - 45)
            label = "IDEAL (20-70%)"
        elif acc > 70:
            score = 10
            label = "TOO EASY (>70%)"
        elif acc > 10:
            score = 50
            label = "HARD BUT FEASIBLE (10-20%)"
        else:
            score = 5
            label = "TOO HARD (<10%)"

        print(f"  {name}: student={acc:.1f}% -- {label}")
        if score > best_score:
            best_score = score
            best_dataset = name

    print(f"\n  --> Use: {best_dataset}")
    print(f"{'=' * 60}")
    return best_dataset


def setup_dataset_symlinks(dataset_name: str):
    """Create symlinks for the chosen dataset as train_prompts.jsonl / eval_prompts.jsonl."""
    train_src = os.path.join(DATA_DIR, f"{dataset_name}_train.jsonl")
    eval_src = os.path.join(DATA_DIR, f"{dataset_name}_eval.jsonl")
    train_dst = os.path.join(DATA_DIR, "train_prompts.jsonl")
    eval_dst = os.path.join(DATA_DIR, "eval_prompts.jsonl")

    for dst in [train_dst, eval_dst]:
        if os.path.exists(dst):
            os.remove(dst)

    os.symlink(os.path.basename(train_src), train_dst)
    os.symlink(os.path.basename(eval_src), eval_dst)
    print(f"\n  Symlinked: {train_dst} -> {os.path.basename(train_src)}")
    print(f"  Symlinked: {eval_dst} -> {os.path.basename(eval_src)}")


def main():
    parser = argparse.ArgumentParser(description="Pre-flight baseline evaluation")
    parser.add_argument(
        "--max-problems", type=int, default=30,
        help="Max problems to evaluate per dataset (default: 30)",
    )
    parser.add_argument(
        "--vllm-url", type=str, default="http://localhost:8000/v1",
        help="vLLM OpenAI-compatible API URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--maj-k", type=int, default=8,
        help="Number of samples for majority voting (default: 8)",
    )
    parser.add_argument(
        "--skip-teacher", action="store_true",
        help="Skip teacher evaluation (if already done)",
    )
    parser.add_argument(
        "--skip-student", action="store_true",
        help="Skip student evaluation (useful for testing teacher only)",
    )
    parser.add_argument(
        "--auto-symlink", action="store_true",
        help="Automatically symlink the recommended dataset for training",
    )
    args = parser.parse_args()

    eval_files = {
        "aime": os.path.join(DATA_DIR, "aime_eval.jsonl"),
        "math45": os.path.join(DATA_DIR, "math45_eval.jsonl"),
    }

    # Check that eval files exist
    for name, path in eval_files.items():
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run prepare_data.py first.")
            sys.exit(1)

    all_results = {}

    for dataset_name, eval_path in eval_files.items():
        print(f"\n{'#' * 60}")
        print(f"  Evaluating: {dataset_name}")
        print(f"{'#' * 60}")

        problems = load_eval_problems(eval_path)
        print(f"  Loaded {len(problems)} eval problems")

        # Teacher evaluation
        if not args.skip_teacher:
            print(f"\n  --- Teacher (Gemini 3 Pro) ---")
            teacher_results = eval_teacher(problems, args.max_problems)
        else:
            print(f"\n  --- Skipping teacher evaluation ---")
            teacher_results = [{"correct": 0.0}] * min(len(problems), args.max_problems)

        # Student evaluation
        if not args.skip_student:
            print(f"\n  --- Base Student (gpt-oss-120b) ---")
            student_results = eval_student(
                problems, args.max_problems, args.vllm_url, args.maj_k
            )
        else:
            print(f"\n  --- Skipping student evaluation ---")
            student_results = [{"pass1": 0.0, "maj_correct": 0.0}] * min(
                len(problems), args.max_problems
            )

        teacher_acc, student_pass1, student_maj = print_table(
            dataset_name, teacher_results, student_results, args.maj_k
        )
        all_results[dataset_name] = {
            "teacher_acc": teacher_acc,
            "student_pass1": student_pass1,
            "student_maj": student_maj,
        }

    # Recommend dataset
    recommended = recommend_dataset(all_results)

    if args.auto_symlink:
        setup_dataset_symlinks(recommended)
    else:
        print(f"\nTo set up training data, run:")
        print(f"  python eval_baseline.py --auto-symlink")
        print(f"Or manually:")
        print(f"  cd {DATA_DIR}")
        print(f"  ln -sf {recommended}_train.jsonl train_prompts.jsonl")
        print(f"  ln -sf {recommended}_eval.jsonl eval_prompts.jsonl")


if __name__ == "__main__":
    main()
