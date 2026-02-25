#!/usr/bin/env python3
"""Truncation sensitivity check: qwen2.5-14b on 200 OOD problems,
max_tokens=4096 vs 2048.

If accuracy is basically unchanged but trunc% drops materially,
standardize max_tokens=2048 going forward (saves compute for RL).

Usage:
    python eval_trunc_sensitivity.py
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reward_func_em import extract_model_answer, check_correctness


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
    print("WARNING: HF_TOKEN not found", flush=True)


_ensure_hf_token()

DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = Path("/mnt/scratch/model_sweep/trunc_sensitivity")
MODEL_HF_ID = "Qwen/Qwen2.5-14B-Instruct"
TP = 2
MAX_MODEL_LEN = 8192
PORT = 8000
N_PROBLEMS = 200

BOXED_SUFFIX = (
    "\n\nPlease reason step by step but keep it concise, and put your final "
    "answer within \\boxed{...}. In the final line, output ONLY "
    "\\boxed{<integer>} and nothing else."
)


def load_problems(path: str, limit: int) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
                if len(problems) >= limit:
                    break
    return problems


def problem_hash(problem: dict) -> str:
    text = " ".join(problem["input"].split()).strip().lower()
    return hashlib.md5(text.encode()).hexdigest()


def detect_parse_method(response: str, model_answer: str) -> str:
    if re.search(r"\\boxed\{", response):
        return "boxed"
    if model_answer:
        return "last_number"
    return "fail"


def start_vllm_server() -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_HF_ID,
        "--tensor-parallel-size", str(TP),
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", str(MAX_MODEL_LEN),
        "--port", str(PORT),
        "--trust-remote-code",
    ]
    print(f"  Starting vLLM: {' '.join(cmd)}", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = open(OUTPUT_DIR / "vllm_server.log", "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    proc._log_file = log_file
    return proc


def wait_for_server(timeout: int = 600) -> bool:
    from openai import OpenAI
    base_url = f"http://localhost:{PORT}/v1"
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


def eval_with_max_tokens(problems: list[dict], max_tokens: int,
                         label: str) -> list[dict]:
    from openai import OpenAI
    client = OpenAI(base_url=f"http://localhost:{PORT}/v1", api_key="unused")
    results = []
    correct_count = 0

    for i, p in enumerate(problems):
        p_hash = problem_hash(p)
        prompt_text = p["input"] + BOXED_SUFFIX
        text = ""
        finish_reason = "unknown"

        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL_HF_ID,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    n=1,
                )
                text = response.choices[0].message.content or ""
                finish_reason = response.choices[0].finish_reason or "unknown"
                break
            except Exception as e:
                print(f"  [{label}] Error (attempt {attempt + 1}): {e}",
                      flush=True)
                time.sleep(2 * (attempt + 1))

        model_answer = extract_model_answer(text)
        correct = int(check_correctness(model_answer, str(p["label"])))
        correct_count += correct
        parse_method = detect_parse_method(text, model_answer)

        results.append({
            "problem_hash": p_hash,
            "correct": correct,
            "model_answer": model_answer,
            "truth": str(p["label"]),
            "parse_method": parse_method,
            "output": text,
            "finish_reason": finish_reason,
            "source": p.get("source", ""),
            "max_tokens": max_tokens,
        })

        if (i + 1) % 50 == 0 or i == len(problems) - 1:
            trunc = sum(1 for r in results if r["finish_reason"] == "length")
            print(f"  [{label}] {i+1}/{len(problems)}: "
                  f"{correct_count}/{i+1} = {correct_count/(i+1)*100:.1f}% "
                  f"(trunc={trunc}/{i+1}={trunc/(i+1)*100:.0f}%)", flush=True)

    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load first 200 OOD problems
    problems = load_problems(str(DATA_DIR / "probe_set_1000_ood.jsonl"),
                             limit=N_PROBLEMS)
    print(f"  Loaded {len(problems)} OOD problems for truncation check",
          flush=True)

    # Start server once, run both settings
    proc = start_vllm_server()
    if not wait_for_server():
        print("ERROR: vLLM server did not start", flush=True)
        stop_server(proc)
        sys.exit(1)

    all_results = {}
    for max_tokens in [4096, 2048]:
        label = f"qwen2.5-14b/mt={max_tokens}"
        print(f"\n{'='*60}", flush=True)
        print(f"EVALUATING: {label}", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        results = eval_with_max_tokens(problems, max_tokens, label)
        elapsed = time.time() - t0

        n = len(results)
        correct = sum(r["correct"] for r in results)
        boxed = sum(1 for r in results if r["parse_method"] == "boxed")
        trunc = sum(1 for r in results if r["finish_reason"] == "length")

        all_results[max_tokens] = {
            "results": results,
            "accuracy": correct / n,
            "correct": correct,
            "n": n,
            "boxed_pct": boxed / n * 100,
            "trunc_pct": trunc / n * 100,
            "elapsed_s": round(elapsed, 1),
        }

        # Save per-problem JSONL
        out_path = OUTPUT_DIR / f"qwen2.5-14b_ood200_mt{max_tokens}.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Saved to {out_path}", flush=True)

    stop_server(proc)

    # Paired comparison
    r4096 = {r["problem_hash"]: r for r in all_results[4096]["results"]}
    r2048 = {r["problem_hash"]: r for r in all_results[2048]["results"]}

    common = sorted(set(r4096.keys()) & set(r2048.keys()))
    n_paired = len(common)

    both_correct = sum(1 for h in common
                       if r4096[h]["correct"] and r2048[h]["correct"])
    only_4096 = sum(1 for h in common
                    if r4096[h]["correct"] and not r2048[h]["correct"])
    only_2048 = sum(1 for h in common
                    if not r4096[h]["correct"] and r2048[h]["correct"])
    both_wrong = sum(1 for h in common
                     if not r4096[h]["correct"] and not r2048[h]["correct"])

    s4 = all_results[4096]
    s2 = all_results[2048]

    print(f"\n{'='*70}", flush=True)
    print(f"TRUNCATION SENSITIVITY: qwen2.5-14b, OOD-200", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  {'Setting':<20} {'Accuracy':>10} {'Boxed%':>8} {'Trunc%':>8}",
          flush=True)
    print(f"  {'-'*48}", flush=True)
    print(f"  {'max_tokens=4096':<20} {s4['accuracy']*100:>9.1f}% "
          f"{s4['boxed_pct']:>7.1f} {s4['trunc_pct']:>7.1f}", flush=True)
    print(f"  {'max_tokens=2048':<20} {s2['accuracy']*100:>9.1f}% "
          f"{s2['boxed_pct']:>7.1f} {s2['trunc_pct']:>7.1f}", flush=True)
    print(f"\n  Δ accuracy: {(s2['accuracy'] - s4['accuracy'])*100:+.1f}pp",
          flush=True)
    print(f"  Δ trunc%:   {s2['trunc_pct'] - s4['trunc_pct']:+.1f}pp",
          flush=True)
    print(f"\n  Paired (N={n_paired}):", flush=True)
    print(f"    Both correct: {both_correct}", flush=True)
    print(f"    Only 4096:    {only_4096} (lost by cutting to 2048)", flush=True)
    print(f"    Only 2048:    {only_2048} (gained by cutting to 2048)", flush=True)
    print(f"    Both wrong:   {both_wrong}", flush=True)

    # Decision
    acc_delta = abs(s2["accuracy"] - s4["accuracy"]) * 100
    trunc_drop = s4["trunc_pct"] - s2["trunc_pct"]

    print(f"\n  DECISION:", flush=True)
    if acc_delta <= 2.0 and trunc_drop < -5.0:
        # trunc_drop is negative when 2048 has MORE truncation, which doesn't make sense
        # Actually: if trunc% goes UP with shorter max_tokens, that's expected
        # We want: trunc% with 4096 to be high, trunc% with 2048 to be... also high but we don't care
        # The question is whether accuracy drops
        pass

    if acc_delta <= 2.0:
        print(f"  ✓ Accuracy delta is small ({acc_delta:.1f}pp). "
              f"max_tokens=2048 is viable.", flush=True)
        if s4["trunc_pct"] > 20:
            print(f"  ✓ 4096 already has {s4['trunc_pct']:.0f}% truncation — "
                  f"model rarely uses the extra budget productively.", flush=True)
    else:
        print(f"  ✗ Accuracy drops by {acc_delta:.1f}pp with max_tokens=2048. "
              f"Keep max_tokens=4096.", flush=True)

    # Save summary
    summary = {
        "max_tokens_4096": {k: v for k, v in s4.items() if k != "results"},
        "max_tokens_2048": {k: v for k, v in s2.items() if k != "results"},
        "paired": {
            "n": n_paired,
            "both_correct": both_correct,
            "only_4096": only_4096,
            "only_2048": only_2048,
            "both_wrong": both_wrong,
            "acc_delta_pp": round((s2["accuracy"] - s4["accuracy"]) * 100, 2),
        },
    }
    summary_path = OUTPUT_DIR / "trunc_sensitivity_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
