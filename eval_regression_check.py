#!/usr/bin/env python3
"""Pipeline regression check: compare in-training eval with post-hoc eval.

Validates that the LoRA accumulation fix produces matching results between
the in-training eval path (vLLM worker apply_lora_delta) and the post-hoc
eval path (PEFT merge_and_unload + vLLM serve).

Usage:
    # After training frees GPUs:
    python eval_regression_check.py \
        --merged-model /mnt/scratch/merged_models/a30_step10 \
        --in-training-score 0.6337 \
        --gpus 0,1

    # Or specify custom eval dataset:
    python eval_regression_check.py \
        --merged-model /mnt/scratch/merged_models/a30_step10 \
        --eval-data data/probe_set_200_ood.jsonl \
        --in-training-score 0.6337
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reward_func_em import extract_model_answer, check_correctness


def main():
    parser = argparse.ArgumentParser(description="Pipeline regression check")
    parser.add_argument("--merged-model", required=True,
                        help="Path to merged model directory")
    parser.add_argument("--eval-data", default=str(SCRIPT_DIR / "data" / "probe_set_200_ood.jsonl"),
                        help="Evaluation dataset (JSONL with input/label)")
    parser.add_argument("--in-training-score", type=float, required=True,
                        help="In-training eval score to compare against (e.g., 0.6337)")
    parser.add_argument("--gpus", default="0,1",
                        help="GPU indices for vLLM server")
    parser.add_argument("--tp", type=int, default=2,
                        help="Tensor parallel size")
    parser.add_argument("--port", type=int, default=8100,
                        help="vLLM server port (avoid conflict with training)")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    # Load problems
    problems = []
    with open(args.eval_data) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    print(f"Loaded {len(problems)} problems from {args.eval_data}")

    # Start vLLM server
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.merged_model,
        "--tensor-parallel-size", str(args.tp),
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "5120",
        "--port", str(args.port),
        "--trust-remote-code",
    ]
    print(f"Starting vLLM server on GPUs {args.gpus}...")
    log_file = open("/mnt/scratch/regression_check_vllm.log", "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)

    # Wait for server
    base_url = f"http://localhost:{args.port}/v1"
    from openai import OpenAI
    start = time.time()
    ready = False
    while time.time() - start < 300:
        try:
            client = OpenAI(base_url=base_url, api_key="unused")
            models = client.models.list()
            if models.data:
                print(f"Server ready: {models.data[0].id} ({time.time()-start:.0f}s)")
                ready = True
                break
        except Exception:
            pass
        time.sleep(5)

    if not ready:
        print("ERROR: vLLM server did not start")
        proc.terminate()
        proc.wait()
        log_file.close()
        sys.exit(1)

    # Evaluate
    print(f"\nEvaluating {len(problems)} problems (greedy, max_tokens={args.max_tokens})...")
    client = OpenAI(base_url=base_url, api_key="unused")
    correct_count = 0
    results = []

    for i, p in enumerate(problems):
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=args.merged_model,
                    messages=[{"role": "user", "content": p["input"]}],
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    n=1,
                )
                text = response.choices[0].message.content or ""
                break
            except Exception as e:
                print(f"  Error (attempt {attempt + 1}): {e}")
                time.sleep(2 * (attempt + 1))
                text = ""

        model_answer = extract_model_answer(text)
        correct = check_correctness(model_answer, str(p["label"]))
        correct_count += correct
        results.append({"correct": correct, "model_answer": model_answer, "truth": str(p["label"])})

        if (i + 1) % 50 == 0 or i == len(problems) - 1:
            print(f"  {i+1}/{len(problems)}: {correct_count}/{i+1} = {correct_count/(i+1)*100:.2f}%")

    # Shutdown server
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    log_file.close()

    # Compare
    posthoc_score = correct_count / len(problems)
    in_training = args.in_training_score
    diff_problems = abs(posthoc_score - in_training) * len(problems)

    print(f"\n{'='*60}")
    print(f"REGRESSION CHECK RESULT")
    print(f"{'='*60}")
    print(f"  In-training eval:  {in_training*100:.2f}% ({int(in_training*len(problems))}/{len(problems)})")
    print(f"  Post-hoc eval:     {posthoc_score*100:.2f}% ({correct_count}/{len(problems)})")
    print(f"  Difference:        {abs(posthoc_score-in_training)*100:.2f}pp ({diff_problems:.1f} problems)")

    if diff_problems <= 2.0:
        print(f"\n  PASS: Scores match within 2 problems ({diff_problems:.1f})")
        print(f"  The LoRA accumulation fix is validated.")
        print(f"  In-training eval curves are trustworthy going forward.")
    elif diff_problems <= 5.0:
        print(f"\n  MARGINAL: Scores differ by {diff_problems:.1f} problems")
        print(f"  This is within sampling noise for N={len(problems)} but warrants attention.")
    else:
        print(f"\n  FAIL: Scores differ by {diff_problems:.1f} problems")
        print(f"  This suggests a remaining pipeline inconsistency.")

    # Save results
    output = {
        "in_training_score": in_training,
        "posthoc_score": posthoc_score,
        "n_problems": len(problems),
        "diff_problems": diff_problems,
        "correct_count": correct_count,
        "pass": diff_problems <= 2.0,
    }
    output_path = "/mnt/scratch/regression_check_result.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
