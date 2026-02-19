"""Post-training evaluation: compare Teacher vs Base Student vs RL Student.

Loads the RL-finetuned checkpoint (base + LoRA adapter) and evaluates on
the same eval set used in pre-flight evaluation.

Usage:
  python eval_rl.py --checkpoint /home/rlu/Code/rft/output/gpt-oss-120b-grpo/latest
  python eval_rl.py --vllm-url http://localhost:8000/v1  # if already serving
"""

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

from reward_func import check_correctness, extract_model_answer, normalize_answer

GEMINI_MODEL = "gemini-3-pro-preview"
STUDENT_MODEL = "openai/gpt-oss-120b"

MATH_SYSTEM_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{}."
)


def load_eval_problems(path: str) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            problems.append(json.loads(line))
    return problems


def eval_model_via_vllm(
    vllm_url: str,
    problems: list[dict],
    max_problems: int,
    n: int = 1,
    label: str = "Model",
) -> list[dict]:
    """Evaluate a model served by vLLM."""
    from openai import OpenAI

    client = OpenAI(base_url=vllm_url, api_key="unused")
    models = client.models.list()
    model_name = models.data[0].id if models.data else STUDENT_MODEL
    print(f"  [{label}] Using model: {model_name}")

    subset = problems[:max_problems]
    results = []

    for i, p in enumerate(subset):
        prompt = f"{MATH_SYSTEM_PROMPT}\n\nProblem: {p['input']}"
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=3072,
                    temperature=0.0 if n == 1 else 0.7,
                    n=n,
                )
                break
            except Exception as e:
                print(f"  Error (attempt {attempt + 1}): {e}")
                time.sleep(2 * (attempt + 1))
                if attempt == 2:
                    response = None

        answers = []
        if response:
            for choice in response.choices:
                text = choice.message.content or ""
                model_answer = extract_model_answer(text)
                correct = check_correctness(model_answer, p["label"])
                answers.append({"model_answer": model_answer, "correct": correct})
        else:
            answers = [{"model_answer": "", "correct": 0.0}] * n

        pass1 = answers[0]["correct"]
        if n > 1:
            answer_counts = Counter(normalize_answer(a["model_answer"]) for a in answers)
            majority = answer_counts.most_common(1)[0][0]
            maj_correct = check_correctness(majority, p["label"])
        else:
            maj_correct = pass1

        results.append({"pass1": pass1, "maj_correct": maj_correct, "answers": answers})

        status = "CORRECT" if pass1 > 0 else "WRONG"
        print(
            f"  [{label}] [{i + 1}/{len(subset)}] {status} "
            f"(answer: {answers[0]['model_answer']}, truth: {p['label']})"
        )

    return results


def eval_teacher(problems: list[dict], max_problems: int) -> list[dict]:
    """Evaluate Teacher (Gemini 3 Pro)."""
    from google import genai
    from google.genai import types

    client = genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT", "wf30-poc"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
    )

    subset = problems[:max_problems]
    results = []

    for i, p in enumerate(subset):
        prompt = f"{MATH_SYSTEM_PROMPT}\n\nProblem: {p['input']}"
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
                correct = check_correctness(model_answer, p["label"])
                break
            except Exception as e:
                print(f"  Teacher error (attempt {attempt + 1}): {e}")
                time.sleep(2 * (attempt + 1))
                model_answer, correct = "", 0.0

        results.append({"model_answer": model_answer, "correct": correct})
        status = "CORRECT" if correct > 0 else "WRONG"
        print(
            f"  [Teacher] [{i + 1}/{len(subset)}] {status} "
            f"(answer: {model_answer}, truth: {p['label']})"
        )

    return results


def print_comparison(
    dataset_name: str,
    n_problems: int,
    teacher_results: list[dict],
    base_results: list[dict],
    rl_results: list[dict],
    maj_k: int,
):
    """Print comparison table."""
    teacher_acc = sum(r["correct"] for r in teacher_results) / len(teacher_results) * 100
    base_pass1 = sum(r["pass1"] for r in base_results) / len(base_results) * 100
    base_maj = sum(r["maj_correct"] for r in base_results) / len(base_results) * 100
    rl_pass1 = sum(r["pass1"] for r in rl_results) / len(rl_results) * 100
    rl_maj = sum(r["maj_correct"] for r in rl_results) / len(rl_results) * 100

    delta_pass1 = rl_pass1 - base_pass1
    delta_maj = rl_maj - base_maj

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {dataset_name} ({n_problems} problems)")
    print(f"{'=' * 70}")
    print(f"  {'Model':<35} {'pass@1':>10} {'maj@' + str(maj_k):>10}")
    print(f"  {'-' * 55}")
    print(f"  {'Teacher (Gemini 3 Pro)':<35} {teacher_acc:>9.1f}% {'--':>10}")
    print(f"  {'Base Student (gpt-oss-120b)':<35} {base_pass1:>9.1f}% {base_maj:>9.1f}%")
    print(f"  {'RL Student (+ GRPO)':<35} {rl_pass1:>9.1f}% {rl_maj:>9.1f}%")
    print(f"  {'-' * 55}")
    sign_p = "+" if delta_pass1 >= 0 else ""
    sign_m = "+" if delta_maj >= 0 else ""
    print(f"  {'Delta (RL - Base)':<35} {sign_p}{delta_pass1:>8.1f}% {sign_m}{delta_maj:>8.1f}%")
    print(f"{'=' * 70}")

    if delta_pass1 > 0:
        print("  GRPO training improved pass@1 accuracy.")
    elif delta_pass1 == 0:
        print("  No change in pass@1 accuracy. Consider tuning hyperparameters.")
    else:
        print("  WARNING: pass@1 accuracy decreased. Check for reward hacking or overfit.")

    return {
        "teacher_acc": teacher_acc,
        "base_pass1": base_pass1,
        "base_maj": base_maj,
        "rl_pass1": rl_pass1,
        "rl_maj": rl_maj,
        "delta_pass1": delta_pass1,
        "delta_maj": delta_maj,
    }


def main():
    parser = argparse.ArgumentParser(description="Post-training RL evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/rlu/Code/rft/output/gpt-oss-120b-grpo",
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--base-vllm-url",
        type=str,
        default=None,
        help="vLLM URL for base student (if already serving). If not set, skips base eval.",
    )
    parser.add_argument(
        "--rl-vllm-url",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM URL for RL student (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--max-problems", type=int, default=30,
        help="Max problems to evaluate (default: 30)",
    )
    parser.add_argument(
        "--maj-k", type=int, default=8,
        help="Samples for majority voting (default: 8)",
    )
    parser.add_argument(
        "--skip-teacher", action="store_true",
        help="Skip teacher evaluation",
    )
    parser.add_argument(
        "--eval-file", type=str, default=None,
        help="Eval JSONL file (default: data/eval_prompts.jsonl)",
    )
    parser.add_argument(
        "--output-json", type=str, default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    eval_path = args.eval_file or os.path.join(DATA_DIR, "eval_prompts.jsonl")
    if not os.path.exists(eval_path):
        print(f"Error: {eval_path} not found. Run prepare_data.py and eval_baseline.py first.")
        sys.exit(1)

    problems = load_eval_problems(eval_path)
    print(f"Loaded {len(problems)} eval problems from {eval_path}")

    # --- Teacher ---
    if not args.skip_teacher:
        print(f"\n--- Teacher (Gemini 3 Pro) ---")
        teacher_results = eval_teacher(problems, args.max_problems)
    else:
        n = min(len(problems), args.max_problems)
        teacher_results = [{"correct": 0.0}] * n

    # --- Base Student ---
    if args.base_vllm_url:
        print(f"\n--- Base Student ---")
        base_results = eval_model_via_vllm(
            args.base_vllm_url, problems, args.max_problems,
            n=args.maj_k, label="Base Student",
        )
    else:
        print(f"\n--- Skipping base student (no --base-vllm-url) ---")
        print(f"  Tip: Serve base model with vLLM and pass --base-vllm-url")
        n = min(len(problems), args.max_problems)
        base_results = [{"pass1": 0.0, "maj_correct": 0.0}] * n

    # --- RL Student ---
    print(f"\n--- RL Student (checkpoint: {args.checkpoint}) ---")
    print(f"  Ensure the RL checkpoint is being served at {args.rl_vllm_url}")
    print(f"  Example: vllm serve {STUDENT_MODEL} --enable-lora --lora-modules "
          f"rl-student={args.checkpoint}")
    rl_results = eval_model_via_vllm(
        args.rl_vllm_url, problems, args.max_problems,
        n=args.maj_k, label="RL Student",
    )

    # --- Comparison ---
    dataset_name = os.path.basename(eval_path).replace("_eval.jsonl", "").replace("eval_prompts", "selected")
    stats = print_comparison(
        dataset_name,
        min(len(problems), args.max_problems),
        teacher_results,
        base_results,
        rl_results,
        args.maj_k,
    )

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
