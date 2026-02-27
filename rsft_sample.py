#!/usr/bin/env python3
"""RSFT Phase 1: Sample k solutions per problem from baseline, grade with EM.

Generates k=16 solutions per problem from the baseline model on the
3200-problem RL training pool. Each solution is graded with binary EM.
Outputs all samples with metadata for downstream filtering.

Usage:
    python rsft_sample.py [--k 16] [--pool-size 3200] [--tp 4]

Output: /mnt/scratch/qwen14b_rsft/samples_k{k}.jsonl
Schema per record:
  {problem_hash, problem_input, truth, sample_idx, output, model_answer,
   correct, parse_method, finish_reason, source}
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reward_func_em import extract_model_answer, check_correctness

# ── Config ─────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
RL_POOL_3200 = SCRIPT_DIR / "data" / "sft_rl_pool_3200.jsonl"
RL_POOL_FULL = SCRIPT_DIR / "data" / "sft_rl_pool.jsonl"
OUT_DIR = Path("/mnt/scratch/qwen14b_rsft")

BOXED_SUFFIX = (
    "\n\nPlease reason step by step but keep it concise, and put your final "
    "answer within \\boxed{...}. In the final line, output ONLY "
    "\\boxed{<integer>} and nothing else."
)

PORT = 8000
MAX_TOKENS = 2048
TEMPERATURE = 0.6


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
                return

_ensure_hf_token()


def problem_hash(problem: dict) -> str:
    text = " ".join(problem["input"].split()).strip().lower()
    return hashlib.md5(text.encode()).hexdigest()


def detect_parse_method(output: str) -> str:
    output = output or ""
    if re.search(r"\\boxed\{", output):
        return "boxed"
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", output)
    if numbers:
        return "last_number"
    return "none"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def start_vllm_server(model_path: str, tp: int, port: int = PORT) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", "0.90",
        "--max-model-len", "4096",
        "--port", str(port),
        "--trust-remote-code",
        "--max-num-seqs", "64",
    ]
    print(f"  Starting vLLM: {' '.join(cmd)}", flush=True)
    log_file = open(OUT_DIR / "vllm_sample.log", "w")
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


def sample_problems(base_url: str, model_name: str, problems: list[dict],
                    k: int, out_path: Path):
    """Sample k solutions per problem, grade with EM, write results."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="unused")

    total_correct = 0
    total_samples = 0
    problems_with_any_correct = 0

    # Open file for streaming writes (resume-friendly)
    existing_hashes = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                r = json.loads(line.strip())
                existing_hashes.add(r["problem_hash"])
        print(f"  Resuming: {len(existing_hashes)} problems already sampled", flush=True)

    f_out = open(out_path, "a")

    for i, p in enumerate(problems):
        p_hash = problem_hash(p)
        if p_hash in existing_hashes:
            continue

        prompt_text = p["input"] + BOXED_SUFFIX
        truth = str(p["label"])

        # Request k samples in one call
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    n=k,
                )
                break
            except Exception as e:
                print(f"  Error (attempt {attempt + 1}): {e}", flush=True)
                time.sleep(2 * (attempt + 1))
                if attempt == 2:
                    response = None

        if response is None:
            continue

        any_correct = False
        for j, choice in enumerate(response.choices):
            text = choice.message.content or ""
            finish = choice.finish_reason or "unknown"
            model_answer = extract_model_answer(text)
            correct = int(check_correctness(model_answer, truth))

            total_samples += 1
            total_correct += correct
            if correct:
                any_correct = True

            record = {
                "problem_hash": p_hash,
                "problem_input": p["input"],
                "truth": truth,
                "sample_idx": j,
                "output": text,
                "model_answer": model_answer,
                "correct": correct,
                "parse_method": detect_parse_method(text),
                "finish_reason": finish,
                "source": p.get("source", ""),
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

        if any_correct:
            problems_with_any_correct += 1

        done = i + 1 - len(existing_hashes)
        total_problems = len(problems) - len(existing_hashes)
        if done % 100 == 0 or done == total_problems:
            acc = total_correct / total_samples if total_samples > 0 else 0
            print(f"  [{done}/{total_problems}] "
                  f"samples={total_samples}, correct={total_correct} ({acc*100:.1f}%), "
                  f"problems_with_any_correct={problems_with_any_correct}",
                  flush=True)

    f_out.close()


def main():
    parser = argparse.ArgumentParser(description="RSFT Sampling")
    parser.add_argument("--k", type=int, default=16, help="Samples per problem")
    parser.add_argument("--pool-size", type=int, default=3200,
                        help="Use 3200 (training pool) or 50000 (full pool)")
    parser.add_argument("--tp", type=int, default=4,
                        help="Tensor parallel size for vLLM")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load problems
    pool_path = RL_POOL_3200 if args.pool_size <= 3200 else RL_POOL_FULL
    problems = load_jsonl(pool_path)[:args.pool_size]
    print(f"=== RSFT Sampling: {datetime.now(timezone.utc).isoformat()} ===")
    print(f"  Model: {BASE_MODEL}")
    print(f"  Problems: {len(problems)} (from {pool_path})")
    print(f"  k={args.k}, temperature={TEMPERATURE}")
    print(f"  Output: {OUT_DIR}\n")

    out_path = OUT_DIR / f"samples_k{args.k}.jsonl"
    base_url = f"http://localhost:{PORT}/v1"

    # Start server
    proc = start_vllm_server(BASE_MODEL, tp=args.tp, port=PORT)
    if not wait_for_server(base_url, timeout=600):
        print("ERROR: vLLM server failed to start", flush=True)
        stop_server(proc)
        sys.exit(1)

    t0 = time.time()
    sample_problems(base_url, BASE_MODEL, problems, args.k, out_path)
    elapsed = time.time() - t0

    stop_server(proc)

    # Summary
    samples = load_jsonl(out_path)
    n_correct = sum(r["correct"] for r in samples)
    unique_hashes = set(r["problem_hash"] for r in samples)
    hashes_with_correct = set(r["problem_hash"] for r in samples if r["correct"])

    print(f"\n=== Sampling Complete [{elapsed:.0f}s] ===")
    print(f"  Total samples: {len(samples)}")
    print(f"  Correct samples: {n_correct} ({n_correct/len(samples)*100:.1f}%)")
    print(f"  Problems sampled: {len(unique_hashes)}")
    print(f"  Problems with ≥1 correct: {len(hashes_with_correct)} "
          f"({len(hashes_with_correct)/len(unique_hashes)*100:.1f}%)")
    print(f"  Output: {out_path}")

    # Save summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": BASE_MODEL,
        "pool": str(pool_path),
        "pool_size": len(problems),
        "k": args.k,
        "temperature": TEMPERATURE,
        "total_samples": len(samples),
        "correct_samples": n_correct,
        "correct_rate": n_correct / len(samples),
        "problems_sampled": len(unique_hashes),
        "problems_with_correct": len(hashes_with_correct),
        "elapsed_seconds": elapsed,
    }
    with open(OUT_DIR / "sampling_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
