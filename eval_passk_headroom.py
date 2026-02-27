#!/usr/bin/env python3
"""G4 Guardrail: Pass@k headroom check for Qwen2.5-14B GRPO training.

Determines whether GRPO has usable signal by checking how many baseline-wrong
problems become solvable with multiple samples.

Steps:
  1. Load OOD-1000 baseline results (greedy@1)
  2. Collect problems where baseline is wrong (~330 expected)
  3. Serve qwen2.5-14b with vLLM, generate k=8 samples per wrong problem
  4. Report pass@8: fraction where at least 1/8 samples is correct

Go/no-go guidance:
  - pass@8 >= 15%: GRPO has usable signal -> proceed
  - pass@8 < 5%:   most errors are hard-wrong -> reconsider

Usage:
    python eval_passk_headroom.py [--k 8] [--temp 0.6] [--max_tokens 2048]
                                  [--baseline /mnt/scratch/model_sweep/qwen2.5-14b_ood1000.jsonl]
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

DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = Path("/mnt/scratch/qwen14b_headroom")

MODEL_HF_ID = "Qwen/Qwen2.5-14B-Instruct"
BASELINE_PATH = Path("/mnt/scratch/model_sweep/qwen2.5-14b_ood1000.jsonl")
OOD_DATA_PATH = DATA_DIR / "probe_set_1000_ood.jsonl"

BOXED_SUFFIX = (
    "\n\nPlease reason step by step but keep it concise, and put your final "
    "answer within \\boxed{...}. In the final line, output ONLY "
    "\\boxed{<integer>} and nothing else."
)

PORT = 8000


def _ensure_hf_token():
    if os.environ.get("HF_TOKEN"):
        return
    bashrc = Path.home() / ".bashrc"
    if bashrc.exists():
        for line in bashrc.read_text().splitlines():
            line = line.strip()
            if line.startswith("export HF_TOKEN="):
                token = line.split("=", 1)[1].strip().strip("'\"")
                os.environ["HF_TOKEN"] = token
                print(f"  Loaded HF_TOKEN from ~/.bashrc", flush=True)
                return


_ensure_hf_token()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def start_vllm_server(port: int = PORT) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_HF_ID,
        "--tensor-parallel-size", "2",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "8192",
        "--port", str(port),
        "--trust-remote-code",
    ]
    print(f"  Starting vLLM: {' '.join(cmd)}", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = open(OUTPUT_DIR / "vllm_server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    proc._log_file = log_file
    return proc


def wait_for_server(base_url: str, timeout: int = 600) -> bool:
    from openai import OpenAI
    start = time.time()
    while time.time() - start < timeout:
        try:
            client = OpenAI(base_url=base_url, api_key="unused")
            models = client.models.list()
            if models.data:
                print(f"  Server ready: {models.data[0].id}", flush=True)
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def stop_server(proc: subprocess.Popen):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if hasattr(proc, "_log_file"):
        proc._log_file.close()


def main():
    parser = argparse.ArgumentParser(description="G4: Pass@k headroom check")
    parser.add_argument("--k", type=int, default=8, help="Number of samples per problem")
    parser.add_argument("--temp", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument("--baseline", type=str, default=str(BASELINE_PATH),
                        help="Path to baseline OOD-1000 results JSONL")
    args = parser.parse_args()

    print(f"=== G4: Pass@{args.k} Headroom Check ===\n")
    print(f"Model: {MODEL_HF_ID}")
    print(f"k={args.k}, temp={args.temp}, max_tokens={args.max_tokens}")
    print(f"Baseline: {args.baseline}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Load baseline results and identify wrong problems
    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        print(f"ERROR: Baseline results not found: {baseline_path}")
        print("Run eval_model_sweep.py first to generate OOD-1000 baseline.")
        sys.exit(1)

    baseline_results = load_jsonl(baseline_path)
    wrong_hashes = {r["problem_hash"] for r in baseline_results if r["correct"] == 0}
    print(f"Baseline: {len(baseline_results)} total, "
          f"{len(baseline_results) - len(wrong_hashes)} correct, "
          f"{len(wrong_hashes)} wrong\n")

    # Step 2: Load OOD-1000 source problems, filter to wrong ones
    if not OOD_DATA_PATH.exists():
        print(f"ERROR: OOD data not found: {OOD_DATA_PATH}")
        sys.exit(1)

    all_problems = load_jsonl(OOD_DATA_PATH)

    # Build hash -> problem mapping
    import hashlib
    def problem_hash(p):
        text = " ".join(p["input"].split()).strip().lower()
        return hashlib.md5(text.encode()).hexdigest()

    wrong_problems = []
    for p in all_problems:
        h = problem_hash(p)
        if h in wrong_hashes:
            p["_hash"] = h
            wrong_problems.append(p)

    print(f"Wrong problems matched: {len(wrong_problems)} "
          f"(expected ~{len(wrong_hashes)})\n")

    if len(wrong_problems) == 0:
        print("No wrong problems found — baseline is 100% correct. Nothing to check.")
        sys.exit(0)

    # Step 3: Start vLLM server and generate k samples per wrong problem
    base_url = f"http://localhost:{PORT}/v1"
    proc = start_vllm_server(port=PORT)

    if not wait_for_server(base_url, timeout=600):
        print("ERROR: vLLM server did not start", flush=True)
        log_path = OUTPUT_DIR / "vllm_server.log"
        if log_path.exists():
            print("--- Last 30 lines of vLLM log ---", flush=True)
            lines = log_path.read_text().splitlines()
            for line in lines[-30:]:
                print(f"  {line}", flush=True)
        stop_server(proc)
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="unused")

    results = []
    pass_count = 0
    t0 = time.time()

    for i, p in enumerate(wrong_problems):
        prompt_text = p["input"] + BOXED_SUFFIX
        truth = str(p["label"])

        # Generate k samples
        samples_correct = []
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL_HF_ID,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=args.max_tokens,
                    temperature=args.temp,
                    n=args.k,
                )
                for choice in response.choices:
                    text = choice.message.content or ""
                    model_answer = extract_model_answer(text)
                    correct = int(check_correctness(model_answer, truth))
                    samples_correct.append(correct)
                break
            except Exception as e:
                print(f"  Error (attempt {attempt+1}): {e}", flush=True)
                time.sleep(2 * (attempt + 1))
                samples_correct = [0] * args.k

        any_correct = int(any(c == 1 for c in samples_correct))
        pass_count += any_correct

        results.append({
            "problem_hash": p["_hash"],
            "truth": truth,
            "n_samples": len(samples_correct),
            "n_correct": sum(samples_correct),
            "any_correct": any_correct,
        })

        if (i + 1) % 25 == 0 or i == len(wrong_problems) - 1:
            elapsed = time.time() - t0
            passk = pass_count / (i + 1) * 100
            print(f"  [{i+1}/{len(wrong_problems)}] "
                  f"pass@{args.k} so far: {pass_count}/{i+1} = {passk:.1f}% "
                  f"[{elapsed:.0f}s]", flush=True)

    stop_server(proc)

    # Step 4: Report results
    total_wrong = len(wrong_problems)
    passk_rate = pass_count / total_wrong if total_wrong > 0 else 0.0
    total_samples_correct = sum(r["n_correct"] for r in results)
    total_samples = sum(r["n_samples"] for r in results)

    print(f"\n{'='*60}")
    print(f"G4 RESULTS: Pass@{args.k} Headroom Check")
    print(f"{'='*60}")
    print(f"  Model: {MODEL_HF_ID}")
    print(f"  Wrong problems (greedy@1): {total_wrong}")
    print(f"  Sampling: k={args.k}, temp={args.temp}, max_tokens={args.max_tokens}")
    print(f"")
    print(f"  Pass@{args.k}: {pass_count}/{total_wrong} = {passk_rate*100:.1f}%")
    print(f"  Per-sample accuracy: {total_samples_correct}/{total_samples} "
          f"= {total_samples_correct/total_samples*100:.1f}%")

    # Distribution of n_correct per problem
    from collections import Counter
    dist = Counter(r["n_correct"] for r in results)
    print(f"\n  Distribution of correct samples per wrong problem:")
    for nc in sorted(dist.keys()):
        bar = "#" * dist[nc]
        print(f"    {nc}/{args.k} correct: {dist[nc]:>4} problems  {bar}")

    # Go/no-go guidance
    print(f"\n  Go/no-go guidance:")
    if passk_rate >= 0.15:
        print(f"  >>> PROCEED: pass@{args.k} = {passk_rate*100:.1f}% >= 15% "
              f"-> GRPO has usable signal")
    elif passk_rate >= 0.05:
        print(f"  >>> MARGINAL: pass@{args.k} = {passk_rate*100:.1f}% "
              f"(between 5-15%) -> proceed with caution")
    else:
        print(f"  >>> RECONSIDER: pass@{args.k} = {passk_rate*100:.1f}% < 5% "
              f"-> most errors are hard-wrong")

    # Save results
    out_path = OUTPUT_DIR / f"passk_headroom_k{args.k}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n  Per-problem results saved to {out_path}")

    summary = {
        "model": MODEL_HF_ID,
        "k": args.k,
        "temp": args.temp,
        "max_tokens": args.max_tokens,
        "n_wrong": total_wrong,
        "n_pass": pass_count,
        "passk_rate": passk_rate,
        "per_sample_accuracy": total_samples_correct / total_samples if total_samples > 0 else 0,
    }
    summary_path = OUTPUT_DIR / f"passk_headroom_k{args.k}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
