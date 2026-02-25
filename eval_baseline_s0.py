#!/usr/bin/env python3
"""Evaluate Baseline-S0 (base model, no LoRA) and compute paired comparison
with Baseline-SOTA (A30 step 130) on OOD-1000 / ID-200 / AIME-18.

Answers the question: "Does RL help at all?"

Outputs:
  - Per-problem JSONL for both models (for reproducibility and post-hoc analysis)
  - Paired comparison: Δ, discordant counts (b, c), McNemar exact p-value
  - Bootstrap 95% CI for Δ

Usage:
    python eval_baseline_s0.py
"""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from scipy import stats as scipy_stats
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reward_func_em import extract_model_answer, check_correctness

DATA_DIR = SCRIPT_DIR / "data"
BASE_MODEL = "openai/gpt-oss-20b"
SOTA_MODEL = "/mnt/scratch/merged_models/a30_step130"
OUTPUT_DIR = Path("/mnt/scratch/baseline_s0_eval")

EVAL_SETS = {
    "ood1000": str(DATA_DIR / "probe_set_1000_ood.jsonl"),
    "id200": str(DATA_DIR / "probe_set_200.jsonl"),
    "aime": str(DATA_DIR / "aime_eval.jsonl"),
}


def load_problems(path: str) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def problem_hash(problem: dict) -> str:
    text = " ".join(problem["input"].split()).strip().lower()
    return hashlib.md5(text.encode()).hexdigest()


def start_vllm_server(model_path: str, tp: int = 2, port: int = 8000) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "5120",
        "--port", str(port),
        "--trust-remote-code",
    ]
    print(f"  Starting vLLM: {' '.join(cmd)}", flush=True)
    log_file = open(os.path.join(OUTPUT_DIR, "vllm_server.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    proc._log_file = log_file
    return proc


def wait_for_server(base_url: str, timeout: int = 300) -> bool:
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
    if hasattr(proc, '_log_file'):
        proc._log_file.close()


def eval_model(base_url: str, model_name: str, problems: list[dict],
               max_tokens: int, label: str) -> list[dict]:
    """Evaluate and return per-problem results."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="unused")
    results = []
    correct_count = 0

    for i, p in enumerate(problems):
        p_hash = problem_hash(p)
        text = ""
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": p["input"]}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                    n=1,
                )
                text = response.choices[0].message.content or ""
                break
            except Exception as e:
                print(f"  [{label}] Error (attempt {attempt + 1}): {e}", flush=True)
                time.sleep(2 * (attempt + 1))
                text = ""

        model_answer = extract_model_answer(text)
        correct = int(check_correctness(model_answer, str(p["label"])))
        correct_count += correct

        results.append({
            "problem_hash": p_hash,
            "correct": correct,
            "model_answer": model_answer,
            "truth": str(p["label"]),
            "output": text,
        })

        if (i + 1) % 50 == 0 or i == len(problems) - 1:
            print(f"  [{label}] {i+1}/{len(problems)}: "
                  f"{correct_count}/{i+1} = {correct_count/(i+1)*100:.1f}%",
                  flush=True)

    return results


def save_results(results: list[dict], path: str):
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(results)} results to {path}", flush=True)


def compute_paired_stats(results_base: list[dict], results_new: list[dict],
                         label: str) -> dict:
    """Compute paired comparison: Δ, discordants, McNemar, bootstrap CI."""
    # Match by problem_hash
    base_by_hash = {r["problem_hash"]: r["correct"] for r in results_base}
    new_by_hash = {r["problem_hash"]: r["correct"] for r in results_new}

    common_hashes = sorted(set(base_by_hash.keys()) & set(new_by_hash.keys()))
    n = len(common_hashes)

    if n == 0:
        print(f"  ERROR: No common problems for paired comparison!", flush=True)
        return {}

    y_base = np.array([base_by_hash[h] for h in common_hashes])
    y_new = np.array([new_by_hash[h] for h in common_hashes])

    acc_base = y_base.mean()
    acc_new = y_new.mean()
    delta = acc_new - acc_base

    # Discordant counts
    b = int(((y_base == 1) & (y_new == 0)).sum())  # base correct, new wrong
    c = int(((y_base == 0) & (y_new == 1)).sum())  # base wrong, new correct

    # McNemar exact test (two-sided)
    if b + c > 0:
        # Use binomial test: under H0, b ~ Binomial(b+c, 0.5)
        mcnemar_p = scipy_stats.binomtest(b, b + c, 0.5).pvalue
    else:
        mcnemar_p = 1.0

    # Bootstrap 95% CI for Δ
    rng = np.random.RandomState(42)
    n_boot = 10000
    boot_deltas = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_delta = y_new[idx].mean() - y_base[idx].mean()
        boot_deltas.append(boot_delta)
    boot_deltas = np.array(boot_deltas)
    ci_low = float(np.percentile(boot_deltas, 2.5))
    ci_high = float(np.percentile(boot_deltas, 97.5))

    print(f"\n{'='*70}", flush=True)
    print(f"PAIRED COMPARISON: {label}", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"  N (paired problems): {n}", flush=True)
    print(f"  Baseline-S0 accuracy: {acc_base*100:.2f}% ({int(y_base.sum())}/{n})", flush=True)
    print(f"  Baseline-SOTA accuracy: {acc_new*100:.2f}% ({int(y_new.sum())}/{n})", flush=True)
    print(f"  Δ (SOTA - S0): {delta*100:+.2f}pp", flush=True)
    print(f"  Discordants: b={b} (S0 correct, SOTA wrong), c={c} (S0 wrong, SOTA correct)", flush=True)
    print(f"  Net improvement: c - b = {c - b} problems", flush=True)
    print(f"  McNemar exact p-value: {mcnemar_p:.6f}", flush=True)
    print(f"  Bootstrap 95% CI for Δ: [{ci_low*100:+.2f}pp, {ci_high*100:+.2f}pp]", flush=True)

    # Gate checks
    print(f"\n  SOP Gate checks:", flush=True)
    if delta * 100 >= 3.0 and mcnemar_p < 0.05:
        print(f"  ✓ Gate-1b PASS: Δ={delta*100:+.2f}pp ≥ +3.0pp AND p={mcnemar_p:.4f} < 0.05", flush=True)
    elif delta * 100 >= 2.0 and (mcnemar_p < 0.10 or ci_low * 100 > -0.5):
        print(f"  ~ Gate-1a PASS (worth confirming): Δ={delta*100:+.2f}pp ≥ +2.0pp", flush=True)
    else:
        print(f"  ✗ Neither gate passes: Δ={delta*100:+.2f}pp", flush=True)

    return {
        "n_paired": n,
        "acc_base": float(acc_base),
        "acc_new": float(acc_new),
        "delta_pp": float(delta * 100),
        "b_discordant": b,
        "c_discordant": c,
        "mcnemar_p": float(mcnemar_p),
        "bootstrap_ci_low_pp": float(ci_low * 100),
        "bootstrap_ci_high_pp": float(ci_high * 100),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_url = "http://localhost:8000/v1"
    max_tokens = 4096
    port = 8000

    # Load all eval sets
    all_problems = {}
    for name, path in EVAL_SETS.items():
        all_problems[name] = load_problems(path)
        print(f"  Eval set: {name} = {len(all_problems[name])} problems", flush=True)

    all_results = {}

    # ---- Evaluate Baseline-S0 (base model, no LoRA) ----
    print(f"\n{'='*60}", flush=True)
    print(f"EVALUATING: Baseline-S0 (base model, no LoRA)", flush=True)
    print(f"  Model: {BASE_MODEL}", flush=True)
    print(f"{'='*60}", flush=True)

    proc = start_vllm_server(BASE_MODEL, tp=2, port=port)
    if not wait_for_server(base_url, timeout=300):
        print("ERROR: vLLM server did not start for base model", flush=True)
        stop_server(proc)
        sys.exit(1)

    all_results["baseline_s0"] = {}
    for eval_name, problems in all_problems.items():
        t0 = time.time()
        results = eval_model(base_url, BASE_MODEL, problems, max_tokens,
                             f"s0/{eval_name}")
        elapsed = time.time() - t0
        acc = sum(r["correct"] for r in results) / len(results)
        all_results["baseline_s0"][eval_name] = {
            "accuracy": acc,
            "correct": sum(r["correct"] for r in results),
            "n_problems": len(results),
            "elapsed_s": round(elapsed, 1),
        }
        save_results(results, str(OUTPUT_DIR / f"s0_{eval_name}.jsonl"))
        print(f"  -> {eval_name}: {acc*100:.2f}% ({sum(r['correct'] for r in results)}/{len(results)}) [{elapsed:.1f}s]",
              flush=True)

    stop_server(proc)
    time.sleep(5)

    # ---- Evaluate Baseline-SOTA (A30 step 130) ----
    print(f"\n{'='*60}", flush=True)
    print(f"EVALUATING: Baseline-SOTA (A30 step 130)", flush=True)
    print(f"  Model: {SOTA_MODEL}", flush=True)
    print(f"{'='*60}", flush=True)

    proc = start_vllm_server(SOTA_MODEL, tp=2, port=port)
    if not wait_for_server(base_url, timeout=300):
        print("ERROR: vLLM server did not start for SOTA model", flush=True)
        stop_server(proc)
        sys.exit(1)

    all_results["baseline_sota"] = {}
    for eval_name, problems in all_problems.items():
        t0 = time.time()
        results = eval_model(base_url, SOTA_MODEL, problems, max_tokens,
                             f"sota/{eval_name}")
        elapsed = time.time() - t0
        acc = sum(r["correct"] for r in results) / len(results)
        all_results["baseline_sota"][eval_name] = {
            "accuracy": acc,
            "correct": sum(r["correct"] for r in results),
            "n_problems": len(results),
            "elapsed_s": round(elapsed, 1),
        }
        save_results(results, str(OUTPUT_DIR / f"sota_{eval_name}.jsonl"))
        print(f"  -> {eval_name}: {acc*100:.2f}% ({sum(r['correct'] for r in results)}/{len(results)}) [{elapsed:.1f}s]",
              flush=True)

    stop_server(proc)

    # ---- Summary table ----
    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY TABLE", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Model':<20} {'OOD-1000':>16} {'ID-200':>14} {'AIME-18':>14}", flush=True)
    print("-" * 68, flush=True)
    for model_name in ["baseline_s0", "baseline_sota"]:
        row = f"{model_name:<20}"
        for eval_name in ["ood1000", "id200", "aime"]:
            r = all_results[model_name][eval_name]
            acc = r["accuracy"] * 100
            correct = r["correct"]
            n = r["n_problems"]
            row += f" {acc:>6.2f}% ({correct:>3}/{n})"
        print(row, flush=True)

    # ---- Paired comparison on OOD-1000 ----
    s0_ood = load_problems(str(OUTPUT_DIR / "s0_ood1000.jsonl"))
    sota_ood = load_problems(str(OUTPUT_DIR / "sota_ood1000.jsonl"))
    paired_stats_ood = compute_paired_stats(s0_ood, sota_ood, "OOD-1000")

    # Save paired records per SOP
    paired_records_path = str(OUTPUT_DIR / "paired_records_s0_sota_ood1000.jsonl")
    s0_by_hash = {r["problem_hash"]: r for r in s0_ood}
    sota_by_hash = {r["problem_hash"]: r for r in sota_ood}
    with open(paired_records_path, "w") as f:
        for h in sorted(set(s0_by_hash.keys()) & set(sota_by_hash.keys())):
            record = {
                "problem_hash": h,
                "y_baseline": s0_by_hash[h]["correct"],
                "y_new": sota_by_hash[h]["correct"],
                "output_baseline": s0_by_hash[h]["output"],
                "output_new": sota_by_hash[h]["output"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"\n  Paired records saved to {paired_records_path}", flush=True)

    # ---- Paired comparison on ID-200 ----
    s0_id = load_problems(str(OUTPUT_DIR / "s0_id200.jsonl"))
    sota_id = load_problems(str(OUTPUT_DIR / "sota_id200.jsonl"))
    paired_stats_id = compute_paired_stats(s0_id, sota_id, "ID-200")

    # ---- Save all results ----
    summary = {
        "aggregates": all_results,
        "paired_ood1000": paired_stats_ood,
        "paired_id200": paired_stats_id,
    }
    summary_path = str(OUTPUT_DIR / "baseline_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to {summary_path}", flush=True)


if __name__ == "__main__":
    main()
