#!/usr/bin/env python3
"""Q1 Diagnostics: Classify why GRPO shows null OOD effect on qwen2.5-14b.

D1: Did the policy actually move? (paired movement + KL/proxy stats)
D2: Does it help ID but not OOD? (held-out in-distribution eval)
D3: Pass@k improved but greedy didn't? (reliability vs capability)

Usage:
    python run_q1_diagnostics.py [--skip-d1] [--skip-d2] [--skip-d3]
                                 [--d2-size 1000] [--d3-size 200]

All artifacts written to /mnt/scratch/qwen14b_q01_eval/diag/
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reward_func_em import extract_model_answer, check_correctness, normalize_answer

# ── Paths ──────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
STEP_MERGED = {
    "step50": "/mnt/scratch/merged_models_qwen14b_q01/step50",
    "step100": "/mnt/scratch/merged_models_qwen14b_q01/step100",
    "step200": "/mnt/scratch/merged_models_qwen14b_q01/step200",
}
STEP100_MERGED = STEP_MERGED["step100"]

BASELINE_OOD1000 = Path("/mnt/scratch/model_sweep/qwen2.5-14b_ood1000.jsonl")
STEP_OOD1000 = {
    "step50": Path("/mnt/scratch/qwen14b_q01_eval/step50_ood1000.jsonl"),
    "step100": Path("/mnt/scratch/qwen14b_q01_eval/step100_ood1000.jsonl"),
    "step200": Path("/mnt/scratch/qwen14b_q01_eval/step200_ood1000.jsonl"),
}
TRAINING_METRICS = Path("/mnt/scratch/rft_metrics_qwen14b_q01/training_metrics.jsonl")
HEADROOM_PASSK = Path("/mnt/scratch/qwen14b_headroom/passk_headroom_k8.jsonl")

OOD_DATA = SCRIPT_DIR / "data" / "probe_set_1000_ood.jsonl"
RL_POOL_FULL = SCRIPT_DIR / "data" / "sft_rl_pool.jsonl"
RL_POOL_3200 = SCRIPT_DIR / "data" / "sft_rl_pool_3200.jsonl"
OOD_200 = SCRIPT_DIR / "data" / "probe_set_200_ood.jsonl"

DIAG_DIR = Path("/mnt/scratch/qwen14b_q01_eval/diag")

BOXED_SUFFIX = (
    "\n\nPlease reason step by step but keep it concise, and put your final "
    "answer within \\boxed{...}. In the final line, output ONLY "
    "\\boxed{<integer>} and nothing else."
)

PORT = 8000
MAX_TOKENS = 2048


# ── Utilities ──────────────────────────────────────────────────────────

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


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records: list[dict], path: Path):
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records)} records to {path}", flush=True)


def save_json(obj: dict, path: Path):
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"  Saved {path}", flush=True)


def problem_hash(problem: dict) -> str:
    text = " ".join(problem["input"].split()).strip().lower()
    return hashlib.md5(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_parse_method(output: str) -> str:
    """Detect which extraction method extract_model_answer would use."""
    output = output or ""
    if re.search(r"\\boxed\{", output):
        return "boxed"
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", output)
    if numbers:
        return "last_number"
    return "none"


def compute_paired_stats(y_base: np.ndarray, y_new: np.ndarray,
                         label: str) -> dict:
    """Compute Δ, discordants, McNemar, bootstrap CI."""
    n = len(y_base)
    acc_base = y_base.mean()
    acc_new = y_new.mean()
    delta = acc_new - acc_base

    b = int(((y_base == 1) & (y_new == 0)).sum())
    c = int(((y_base == 0) & (y_new == 1)).sum())

    if b + c > 0:
        mcnemar_p = scipy_stats.binomtest(b, b + c, 0.5).pvalue
    else:
        mcnemar_p = 1.0

    rng = np.random.RandomState(42)
    boot_deltas = []
    for _ in range(10000):
        idx = rng.choice(n, size=n, replace=True)
        boot_deltas.append(y_new[idx].mean() - y_base[idx].mean())
    boot_deltas = np.array(boot_deltas)
    ci_low = float(np.percentile(boot_deltas, 2.5))
    ci_high = float(np.percentile(boot_deltas, 97.5))

    return {
        "label": label,
        "n_paired": n,
        "acc_base": float(acc_base),
        "acc_new": float(acc_new),
        "delta_pp": float(delta * 100),
        "b_discordant": b,
        "c_discordant": c,
        "b_plus_c": b + c,
        "b_plus_c_pct": float((b + c) / n * 100),
        "mcnemar_p": float(mcnemar_p),
        "ci_low_pp": float(ci_low * 100),
        "ci_high_pp": float(ci_high * 100),
    }


def start_vllm_server(model_path: str, port: int = PORT) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--tensor-parallel-size", "2",
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "4096",
        "--port", str(port),
        "--trust-remote-code",
    ]
    print(f"  Starting vLLM: {' '.join(cmd)}", flush=True)
    log_file = open(DIAG_DIR / "vllm_server.log", "w")
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


def eval_greedy(base_url: str, model_name: str, problems: list[dict],
                label: str) -> list[dict]:
    """Greedy eval (temp=0, n=1) on a problem list. Returns per-problem results."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="unused")
    results = []
    correct_count = 0

    for i, p in enumerate(problems):
        p_hash = problem_hash(p)
        prompt_text = p["input"] + BOXED_SUFFIX
        truth = str(p["label"])

        text = ""
        finish = "error"
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=MAX_TOKENS,
                    temperature=0.0,
                    n=1,
                )
                text = response.choices[0].message.content or ""
                finish = response.choices[0].finish_reason or "unknown"
                break
            except Exception as e:
                print(f"  [{label}] Error (attempt {attempt + 1}): {e}", flush=True)
                time.sleep(2 * (attempt + 1))

        model_answer = extract_model_answer(text)
        correct = int(check_correctness(model_answer, truth))
        correct_count += correct

        results.append({
            "problem_hash": p_hash,
            "correct": correct,
            "truth": truth,
            "model_answer": model_answer,
            "output": text,
            "parse_method": detect_parse_method(text),
            "finish_reason": finish,
            "source": p.get("source", ""),
        })

        if (i + 1) % 100 == 0 or i == len(problems) - 1:
            print(f"  [{label}] {i+1}/{len(problems)}: "
                  f"{correct_count}/{i+1} = {correct_count/(i+1)*100:.1f}%",
                  flush=True)

    return results


def eval_passk(base_url: str, model_name: str, problems: list[dict],
               k: int, label: str) -> list[dict]:
    """Sampling eval (temp=0.6, n=k) for pass@k. Returns per-problem results."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="unused")
    results = []
    pass_count = 0

    for i, p in enumerate(problems):
        p_hash = problem_hash(p)
        prompt_text = p["input"] + BOXED_SUFFIX
        truth = str(p["label"])

        samples_correct = []
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=MAX_TOKENS,
                    temperature=0.6,
                    n=k,
                )
                for choice in response.choices:
                    text = choice.message.content or ""
                    ma = extract_model_answer(text)
                    samples_correct.append(int(check_correctness(ma, truth)))
                break
            except Exception as e:
                print(f"  [{label}] Error (attempt {attempt + 1}): {e}", flush=True)
                time.sleep(2 * (attempt + 1))
                samples_correct = [0] * k

        any_correct = int(any(c == 1 for c in samples_correct))
        pass_count += any_correct

        results.append({
            "problem_hash": p_hash,
            "truth": truth,
            "n_samples": len(samples_correct),
            "n_correct": sum(samples_correct),
            "any_correct": any_correct,
        })

        if (i + 1) % 50 == 0 or i == len(problems) - 1:
            print(f"  [{label}] {i+1}/{len(problems)}: "
                  f"pass@{k}={pass_count}/{i+1} = {pass_count/(i+1)*100:.1f}%",
                  flush=True)

    return results


# ══════════════════════════════════════════════════════════════════════
# Phase 0: Re-evaluate checkpoints missing full schema fields
# ══════════════════════════════════════════════════════════════════════

def _needs_reeval(path: Path) -> bool:
    """Check if a JSONL file lacks finish_reason field (→ needs re-run)."""
    if not path.exists():
        return True
    with open(path) as f:
        first_line = f.readline().strip()
        if not first_line:
            return True
        record = json.loads(first_line)
        return "finish_reason" not in record


def reeval_full_schema():
    """Re-evaluate any checkpoint OOD-1000 files that lack full schema.

    Full schema: {problem_hash, correct, truth, model_answer, output,
                  parse_method, finish_reason, source}
    """
    to_reeval = []
    for name, path in STEP_OOD1000.items():
        if _needs_reeval(path):
            to_reeval.append(name)
            print(f"  {name}: {path} — NEEDS RE-EVAL (missing finish_reason)")
        else:
            print(f"  {name}: {path} — OK (has full schema)")

    if not to_reeval:
        print("  All checkpoint eval files have full schema. Skipping Phase 0.\n")
        return

    # Load OOD problems (with source field)
    problems = load_jsonl(OOD_DATA)
    print(f"\n  OOD-1000: {len(problems)} problems")
    print(f"  Re-evaluating: {to_reeval}\n")

    base_url = f"http://localhost:{PORT}/v1"

    for name in to_reeval:
        model_path = STEP_MERGED[name]
        out_path = STEP_OOD1000[name]

        print(f"\n  {'='*60}")
        print(f"  RE-EVAL: {name} ({model_path})")
        print(f"  {'='*60}\n")

        proc = start_vllm_server(model_path, port=PORT)
        if not wait_for_server(base_url, timeout=600):
            print(f"  ERROR: vLLM server failed to start for {name}", flush=True)
            stop_server(proc)
            continue

        t0 = time.time()
        results = eval_greedy(base_url, model_path, problems, f"reeval/{name}")
        elapsed = time.time() - t0

        acc = sum(r["correct"] for r in results) / len(results)
        save_jsonl(results, out_path)
        print(f"\n  {name}: {acc*100:.2f}% ({sum(r['correct'] for r in results)}/{len(results)}) [{elapsed:.0f}s]")

        stop_server(proc)
        time.sleep(5)

    print(f"\n  Phase 0 complete: re-evaluated {len(to_reeval)} checkpoints with full schema.\n")


# ══════════════════════════════════════════════════════════════════════
# D1: Did the policy actually move?
# ══════════════════════════════════════════════════════════════════════

def run_d1():
    print(f"\n{'='*70}")
    print(f"D1: Did the policy actually move? (OOD-1000 paired movement)")
    print(f"{'='*70}\n")

    # Load results (Phase 0 ensures step100 has full schema)
    baseline = load_jsonl(BASELINE_OOD1000)
    step100 = load_jsonl(STEP_OOD1000["step100"])

    base_by_hash = {r["problem_hash"]: r for r in baseline}
    new_by_hash = {r["problem_hash"]: r for r in step100}
    common = sorted(set(base_by_hash.keys()) & set(new_by_hash.keys()))
    print(f"  Baseline: {len(baseline)} problems, Step100: {len(step100)}, Common: {len(common)}")

    # ── Part 1: Paired records ──
    paired_records = []
    answer_changed = 0
    for h in common:
        b = base_by_hash[h]
        s = new_by_hash[h]

        ans_base = normalize_answer(b.get("model_answer", ""))
        ans_step = normalize_answer(s.get("model_answer", ""))
        if ans_base != ans_step:
            answer_changed += 1

        paired_records.append({
            "problem_hash": h,
            "correct_baseline": b["correct"],
            "correct_step100": s["correct"],
            "answer_baseline": b.get("model_answer", ""),
            "answer_step100": s.get("model_answer", ""),
            "parse_method_baseline": b["parse_method"],
            "parse_method_step100": s["parse_method"],
            "finish_reason_baseline": b["finish_reason"],
            "finish_reason_step100": s["finish_reason"],
        })

    save_jsonl(paired_records, DIAG_DIR / "paired_records_ood1000_baseline_vs_step100.jsonl")

    # Compute paired stats
    y_base = np.array([base_by_hash[h]["correct"] for h in common])
    y_new = np.array([new_by_hash[h]["correct"] for h in common])
    paired = compute_paired_stats(y_base, y_new, "D1: OOD-1000 baseline vs step100")

    # Answer change analysis
    paired["answer_changed"] = answer_changed
    paired["answer_changed_pct"] = round(answer_changed / len(common) * 100, 2)

    # Discordant breakdown
    both_correct = int(((y_base == 1) & (y_new == 1)).sum())
    both_wrong = int(((y_base == 0) & (y_new == 0)).sum())
    paired["both_correct"] = both_correct
    paired["both_wrong"] = both_wrong

    print(f"\n  Paired results:")
    print(f"    Both correct: {both_correct}")
    print(f"    Both wrong:   {both_wrong}")
    print(f"    b (base✓ RL✗): {paired['b_discordant']}")
    print(f"    c (base✗ RL✓): {paired['c_discordant']}")
    print(f"    b+c total:    {paired['b_plus_c']} ({paired['b_plus_c_pct']:.1f}%)")
    print(f"    Δ: {paired['delta_pp']:+.2f}pp, McNemar p={paired['mcnemar_p']:.4f}")
    print(f"    95% CI: [{paired['ci_low_pp']:+.2f}, {paired['ci_high_pp']:+.2f}]pp")
    print(f"    Answer changed: {answer_changed}/{len(common)} ({answer_changed/len(common)*100:.1f}%)")

    # ── Part 2: Policy shift metrics from training ──
    metrics_records = load_jsonl(TRAINING_METRICS)
    shift_fields = ["kl", "ppo_kl", "log_ratio_max", "log_ratio_min",
                    "log_ratio_abs_p99", "ratio_max"]

    shift_metrics = {}
    for field in shift_fields:
        values = [m[field] for m in metrics_records if field in m]
        if values:
            shift_metrics[field] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p10": float(np.percentile(values, 10)),
                "p90": float(np.percentile(values, 90)),
                "final_10_mean": float(np.mean(values[-10:])),
            }

    # Also compute reward trajectory
    rewards = [m["reward"] for m in metrics_records if "reward" in m]
    shift_metrics["reward_trajectory"] = {
        "first_10_mean": float(np.mean(rewards[:10])),
        "last_10_mean": float(np.mean(rewards[-10:])),
        "overall_mean": float(np.mean(rewards)),
    }

    save_json(shift_metrics, DIAG_DIR / "d1_policy_shift_metrics.json")

    print(f"\n  Policy shift metrics (last 10 steps):")
    for field in shift_fields:
        if field in shift_metrics:
            print(f"    {field}: mean={shift_metrics[field]['final_10_mean']:.6f}")

    # ── D1 Summary ──
    d1_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "diagnostic": "D1",
        "description": "Did the policy actually move?",
        "baseline_model": BASE_MODEL,
        "rl_checkpoint": STEP100_MERGED,
        "dataset": str(OOD_DATA),
        "dataset_sha256": sha256_file(OOD_DATA),
        "decoding": {"temperature": 0.0, "n": 1, "max_tokens": MAX_TOKENS},
        "paired_stats": paired,
        "policy_shift": shift_metrics,
        "interpretation": (
            "DIRECTIONLESS" if paired["b_plus_c"] >= 30
            else "TOO_SMALL"
        ),
        "interpretation_detail": (
            f"b+c={paired['b_plus_c']} ({paired['b_plus_c_pct']:.1f}% of {len(common)}): "
            + ("policy moved substantially but changes cancel out (Δ≈0). "
               "Suggests objective/data mismatch, not LR/rank issue."
               if paired["b_plus_c"] >= 30
               else "policy barely moved. Updates too small — check LR, LoRA rank/targets.")
        ),
    }
    save_json(d1_summary, DIAG_DIR / "d1_movement_summary.json")
    return d1_summary


# ══════════════════════════════════════════════════════════════════════
# D2: Does it help ID but not OOD?
# ══════════════════════════════════════════════════════════════════════

def prepare_id_heldout(n: int = 1000) -> Path:
    """Create ID-heldout set from RL pool, disjoint from training subset."""
    out_path = SCRIPT_DIR / "data" / f"id_heldout_{n}_boxed.jsonl"
    sha_path = DIAG_DIR / f"id_heldout_{n}_sha256.txt"

    if out_path.exists():
        print(f"  ID-heldout already exists: {out_path} ({sum(1 for _ in open(out_path))} lines)")
        return out_path

    print(f"  Creating ID-heldout-{n} from {RL_POOL_FULL}...", flush=True)

    # Load full RL pool and training subset
    pool_full = load_jsonl(RL_POOL_FULL)
    pool_3200 = load_jsonl(RL_POOL_3200)

    # Compute hashes of training subset
    train_hashes = set()
    for p in pool_3200:
        train_hashes.add(problem_hash(p))

    # Also exclude OOD eval hashes
    ood_hashes = set()
    for path in [OOD_DATA, OOD_200]:
        if path.exists():
            for p in load_jsonl(path):
                ood_hashes.add(problem_hash(p))

    # Filter: keep problems NOT in training and NOT in OOD
    candidates = []
    for p in pool_full:
        h = problem_hash(p)
        if h not in train_hashes and h not in ood_hashes:
            candidates.append(p)

    print(f"  Full pool: {len(pool_full)}, Training: {len(train_hashes)}, "
          f"OOD excluded: {len(ood_hashes)}, Candidates: {len(candidates)}")

    if len(candidates) < n:
        print(f"  WARNING: Only {len(candidates)} candidates, requested {n}. Using all.")
        n = len(candidates)

    # Deterministic selection: sort by hash, take first n
    candidates.sort(key=lambda p: problem_hash(p))
    selected = candidates[:n]

    # Append BOXED_SUFFIX
    records = []
    for p in selected:
        records.append({
            "input": p["input"] + BOXED_SUFFIX,
            "label": p["label"],
        })

    save_jsonl(records, out_path)

    # Save SHA256
    sha = sha256_file(out_path)
    with open(sha_path, "w") as f:
        f.write(f"{sha}  {out_path.name}\n")
    print(f"  SHA256: {sha}")

    return out_path


def run_d2(n: int = 1000):
    print(f"\n{'='*70}")
    print(f"D2: Does it help ID but not OOD? (ID-heldout eval)")
    print(f"{'='*70}\n")

    # Prepare data
    id_heldout_path = prepare_id_heldout(n)
    id_problems_raw = load_jsonl(id_heldout_path)

    # These already have BOXED_SUFFIX in "input", so we need to NOT append it again.
    # Create wrapper that eval_greedy expects (with raw input for hashing)
    # Actually, the problems already have BOXED_SUFFIX in input. Our eval function
    # appends BOXED_SUFFIX again. We need to use the raw input for these.
    # Let's load the raw problems from the full pool instead.
    pool_full = load_jsonl(RL_POOL_FULL)
    pool_by_hash = {}
    for p in pool_full:
        pool_by_hash[problem_hash(p)] = p

    # Build problem list matching the heldout set (using raw inputs)
    id_problems = []
    for p_boxed in id_problems_raw:
        # The boxed version has BOXED_SUFFIX appended. Strip it to get the raw input.
        raw_input = p_boxed["input"]
        if raw_input.endswith(BOXED_SUFFIX):
            raw_input = raw_input[:-len(BOXED_SUFFIX)]
        id_problems.append({"input": raw_input, "label": p_boxed["label"]})

    print(f"  ID-heldout: {len(id_problems)} problems")

    base_url = f"http://localhost:{PORT}/v1"
    all_results = {}

    # ── Evaluate baseline ──
    print(f"\n  --- Evaluating baseline on ID-heldout ---")
    proc = start_vllm_server(BASE_MODEL, port=PORT)
    if not wait_for_server(base_url, timeout=600):
        print("  ERROR: vLLM server failed to start for baseline", flush=True)
        stop_server(proc)
        return None
    results_base = eval_greedy(base_url, BASE_MODEL, id_problems, "d2/baseline")
    all_results["baseline"] = results_base
    save_jsonl(results_base, DIAG_DIR / "d2_baseline_idheldout.jsonl")
    stop_server(proc)
    time.sleep(5)

    # ── Evaluate step100 ──
    print(f"\n  --- Evaluating step100 on ID-heldout ---")
    proc = start_vllm_server(STEP100_MERGED, port=PORT)
    if not wait_for_server(base_url, timeout=600):
        print("  ERROR: vLLM server failed to start for step100", flush=True)
        stop_server(proc)
        return None
    results_step = eval_greedy(base_url, STEP100_MERGED, id_problems, "d2/step100")
    all_results["step100"] = results_step
    save_jsonl(results_step, DIAG_DIR / "d2_step100_idheldout.jsonl")
    stop_server(proc)

    # ── Paired comparison ──
    base_by_hash = {r["problem_hash"]: r for r in results_base}
    step_by_hash = {r["problem_hash"]: r for r in results_step}
    common = sorted(set(base_by_hash.keys()) & set(step_by_hash.keys()))

    y_base = np.array([base_by_hash[h]["correct"] for h in common])
    y_new = np.array([step_by_hash[h]["correct"] for h in common])
    paired = compute_paired_stats(y_base, y_new, "D2: ID-heldout baseline vs step100")

    print(f"\n  ID-heldout paired results:")
    print(f"    Baseline: {paired['acc_base']*100:.2f}%")
    print(f"    Step100:  {paired['acc_new']*100:.2f}%")
    print(f"    Δ: {paired['delta_pp']:+.2f}pp, McNemar p={paired['mcnemar_p']:.4f}")
    print(f"    95% CI: [{paired['ci_low_pp']:+.2f}, {paired['ci_high_pp']:+.2f}]pp")
    print(f"    b+c: {paired['b_plus_c']} ({paired['b_plus_c_pct']:.1f}%)")

    # Load OOD comparison for context
    d1_path = DIAG_DIR / "d1_movement_summary.json"
    ood_delta = None
    if d1_path.exists():
        d1 = json.loads(d1_path.read_text())
        ood_delta = d1["paired_stats"]["delta_pp"]

    # Interpretation
    id_delta = paired["delta_pp"]
    if id_delta >= 2.0 and (ood_delta is None or ood_delta < 1.0):
        interp = "ID_OVERFITTING"
        detail = (f"ID-heldout Δ={id_delta:+.1f}pp vs OOD Δ={ood_delta:+.1f}pp: "
                  "RL helps in-distribution but not OOD. Distribution mismatch — "
                  "change training data distribution, not hyperparams.")
    elif id_delta < 1.0:
        interp = "ALSO_NULL"
        detail = (f"ID-heldout Δ={id_delta:+.1f}pp: also null. "
                  "RL dynamics / credit assignment / update magnitude issue.")
    else:
        interp = "MARGINAL"
        detail = f"ID-heldout Δ={id_delta:+.1f}pp: marginal improvement."

    d2_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "diagnostic": "D2",
        "description": "Does it help ID but not OOD?",
        "baseline_model": BASE_MODEL,
        "rl_checkpoint": STEP100_MERGED,
        "dataset": str(id_heldout_path),
        "dataset_sha256": sha256_file(id_heldout_path),
        "dataset_size": len(id_problems),
        "decoding": {"temperature": 0.0, "n": 1, "max_tokens": MAX_TOKENS},
        "paired_stats": paired,
        "ood_delta_pp_for_comparison": ood_delta,
        "interpretation": interp,
        "interpretation_detail": detail,
    }
    save_json(d2_summary, DIAG_DIR / "d2_idheldout_summary.json")
    return d2_summary


# ══════════════════════════════════════════════════════════════════════
# D3: Pass@k improved but greedy didn't?
# ══════════════════════════════════════════════════════════════════════

def prepare_wrong200(n: int = 200) -> tuple[Path, list[dict]]:
    """Select first n wrong problems from baseline OOD-1000 by sorted hash."""
    out_path = SCRIPT_DIR / "data" / f"ood1000_wrong{n}_boxed.jsonl"
    sha_path = DIAG_DIR / f"ood1000_wrong{n}_sha256.txt"

    # Load baseline results to find wrong problems
    baseline = load_jsonl(BASELINE_OOD1000)
    wrong_hashes = sorted([r["problem_hash"] for r in baseline if r["correct"] == 0])
    selected_hashes = set(wrong_hashes[:n])

    print(f"  Baseline wrong: {len(wrong_hashes)}, Selected: {len(selected_hashes)}")

    # Load OOD problems and filter
    ood_problems = load_jsonl(OOD_DATA)
    selected = []
    for p in ood_problems:
        h = problem_hash(p)
        if h in selected_hashes:
            selected.append(p)

    # Sort by hash for determinism
    selected.sort(key=lambda p: problem_hash(p))

    if not out_path.exists():
        records = []
        for p in selected:
            records.append({
                "input": p["input"] + BOXED_SUFFIX,
                "label": p["label"],
            })
        save_jsonl(records, out_path)
        sha = sha256_file(out_path)
        with open(sha_path, "w") as f:
            f.write(f"{sha}  {out_path.name}\n")
    else:
        print(f"  Already exists: {out_path}")

    return out_path, selected


def run_d3(n: int = 200):
    print(f"\n{'='*70}")
    print(f"D3: Pass@k improved but greedy didn't? (reliability vs capability)")
    print(f"{'='*70}\n")

    # Prepare wrong-N subset
    wrong_path, wrong_problems = prepare_wrong200(n)

    # ── Baseline pass@8: reuse from headroom eval ──
    print(f"\n  Loading baseline pass@8 from headroom eval...")
    headroom = load_jsonl(HEADROOM_PASSK)
    headroom_by_hash = {r["problem_hash"]: r for r in headroom}

    # Match with our wrong200 selection
    wrong_hashes = [problem_hash(p) for p in wrong_problems]
    baseline_pass8 = {}
    missing = 0
    for h in wrong_hashes:
        if h in headroom_by_hash:
            baseline_pass8[h] = headroom_by_hash[h]
        else:
            missing += 1

    print(f"  Headroom coverage: {len(baseline_pass8)}/{len(wrong_hashes)} "
          f"({missing} missing)")

    # ── Step100 pass@8: need to evaluate ──
    print(f"\n  --- Evaluating step100 pass@8 on wrong{n} ---")
    base_url = f"http://localhost:{PORT}/v1"
    proc = start_vllm_server(STEP100_MERGED, port=PORT)
    if not wait_for_server(base_url, timeout=600):
        print("  ERROR: vLLM server failed to start", flush=True)
        stop_server(proc)
        return None

    step100_results = eval_passk(base_url, STEP100_MERGED, wrong_problems, 8, "d3/step100")
    save_jsonl(step100_results, DIAG_DIR / "d3_step100_pass8_wrong200.jsonl")
    stop_server(proc)

    step100_by_hash = {r["problem_hash"]: r for r in step100_results}

    # If any baseline pass@8 results were missing from headroom, we need to
    # run baseline too. For now, run baseline pass@8 only on missing problems.
    if missing > 0:
        missing_problems = [p for p in wrong_problems
                           if problem_hash(p) not in baseline_pass8]
        print(f"\n  --- Evaluating baseline pass@8 on {len(missing_problems)} missing problems ---")
        proc = start_vllm_server(BASE_MODEL, port=PORT)
        if wait_for_server(base_url, timeout=600):
            extra_results = eval_passk(base_url, BASE_MODEL, missing_problems, 8,
                                       "d3/baseline-extra")
            for r in extra_results:
                baseline_pass8[r["problem_hash"]] = r
            save_jsonl(extra_results, DIAG_DIR / "d3_baseline_pass8_extra.jsonl")
        stop_server(proc)

    # ── Compute paired stats ──
    paired_records = []
    common_hashes = sorted(set(baseline_pass8.keys()) & set(step100_by_hash.keys()))

    for h in common_hashes:
        b = baseline_pass8[h]
        s = step100_by_hash[h]
        paired_records.append({
            "problem_hash": h,
            "pass8_baseline": b["any_correct"],
            "pass8_step100": s["any_correct"],
            "num_correct_baseline": b["n_correct"],
            "num_correct_step100": s["n_correct"],
        })

    save_jsonl(paired_records, DIAG_DIR / "d3_paired_pass8_wrong200.jsonl")

    # Overall pass@8 stats
    y_base = np.array([baseline_pass8[h]["any_correct"] for h in common_hashes])
    y_new = np.array([step100_by_hash[h]["any_correct"] for h in common_hashes])

    base_pass8_rate = float(y_base.mean())
    step_pass8_rate = float(y_new.mean())
    delta_pass8 = (step_pass8_rate - base_pass8_rate) * 100

    # Per-sample accuracy
    base_per_sample = sum(baseline_pass8[h]["n_correct"] for h in common_hashes) / \
                      sum(baseline_pass8[h]["n_samples"] for h in common_hashes)
    step_per_sample = sum(step100_by_hash[h]["n_correct"] for h in common_hashes) / \
                      sum(step100_by_hash[h]["n_samples"] for h in common_hashes)

    paired = compute_paired_stats(y_base, y_new, "D3: pass@8 baseline vs step100 (wrong200)")

    print(f"\n  Pass@8 results (wrong{n} subset):")
    print(f"    Baseline pass@8: {base_pass8_rate*100:.1f}% ({int(y_base.sum())}/{len(common_hashes)})")
    print(f"    Step100  pass@8: {step_pass8_rate*100:.1f}% ({int(y_new.sum())}/{len(common_hashes)})")
    print(f"    Δ pass@8: {delta_pass8:+.1f}pp")
    print(f"    Per-sample: baseline={base_per_sample*100:.1f}%, step100={step_per_sample*100:.1f}%")
    print(f"    McNemar p={paired['mcnemar_p']:.4f}")

    # Interpretation
    if delta_pass8 >= 5.0:
        interp = "PASSK_UP_GREEDY_FLAT"
        detail = (f"pass@8 Δ={delta_pass8:+.1f}pp: RL improves stochastic success "
                  "but not greedy mode. Consider RSFT/DPO to distill correct samples.")
    elif delta_pass8 <= -5.0:
        interp = "PASSK_DOWN"
        detail = (f"pass@8 Δ={delta_pass8:+.1f}pp: RL reduced sampling diversity. "
                  "Possible mode collapse.")
    else:
        interp = "ALSO_FLAT"
        detail = (f"pass@8 Δ={delta_pass8:+.1f}pp: also flat. "
                  "Model capability unchanged — focus on data/capacity/reward signal.")

    d3_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "diagnostic": "D3",
        "description": "Pass@k improved but greedy didn't?",
        "baseline_model": BASE_MODEL,
        "rl_checkpoint": STEP100_MERGED,
        "dataset": str(wrong_path),
        "dataset_sha256": sha256_file(wrong_path),
        "dataset_size": len(common_hashes),
        "decoding_greedy": {"temperature": 0.0, "n": 1, "max_tokens": MAX_TOKENS},
        "decoding_sampling": {"temperature": 0.6, "n": 8, "max_tokens": MAX_TOKENS},
        "baseline_pass8_rate": base_pass8_rate,
        "step100_pass8_rate": step_pass8_rate,
        "delta_pass8_pp": delta_pass8,
        "baseline_per_sample_acc": float(base_per_sample),
        "step100_per_sample_acc": float(step_per_sample),
        "paired_stats": paired,
        "interpretation": interp,
        "interpretation_detail": detail,
    }
    save_json(d3_summary, DIAG_DIR / "d3_pass8_summary.json")
    return d3_summary


# ══════════════════════════════════════════════════════════════════════
# README
# ══════════════════════════════════════════════════════════════════════

def write_readme(d1, d2, d3):
    lines = [
        "# Q1 Diagnostics: Why GRPO Shows Null OOD Effect",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Model: {BASE_MODEL}",
        f"RL checkpoint: step100 (best by G6)",
        "",
        "## Commands",
        "",
        "```bash",
        "python run_q1_diagnostics.py          # all three diagnostics",
        "python run_q1_diagnostics.py --skip-d2 --skip-d3  # D1 only (no GPU)",
        "python run_q1_diagnostics.py --skip-d1             # D2+D3 only (GPU)",
        "```",
        "",
        "## D1: Did the policy actually move?",
        "",
        "Quantifies how much the RL checkpoint changes answers vs baseline on OOD-1000.",
        "",
        "**PASS/FAIL interpretation:**",
        "- If b+c < 3% of N AND policy shift metrics are tiny: updates too small (LR/rank/targets issue).",
        "- If b+c >= ~10% but Δ≈0: policy moves but directionless (objective/data mismatch).",
        "",
    ]

    if d1:
        ps = d1["paired_stats"]
        lines += [
            f"**Result: {d1['interpretation']}**",
            f"- b+c = {ps['b_plus_c']} ({ps['b_plus_c_pct']:.1f}% of {ps['n_paired']})",
            f"- Δ = {ps['delta_pp']:+.2f}pp, McNemar p = {ps['mcnemar_p']:.4f}",
            f"- Answers changed: {ps['answer_changed']}/{ps['n_paired']} ({ps['answer_changed_pct']:.1f}%)",
            f"- {d1['interpretation_detail']}",
            "",
        ]

    lines += [
        "## D2: Does it help ID but not OOD?",
        "",
        "Evaluates baseline vs step100 on a held-out in-distribution set (disjoint from training pool).",
        "",
        "**PASS/FAIL interpretation:**",
        "- If ID-heldout shows clear positive Δ while OOD is null: distribution mismatch (change data, not hyperparams).",
        "- If ID-heldout is also null: RL dynamics / credit assignment / update magnitude issue.",
        "",
    ]

    if d2:
        ps = d2["paired_stats"]
        lines += [
            f"**Result: {d2['interpretation']}**",
            f"- ID-heldout: baseline {ps['acc_base']*100:.1f}%, step100 {ps['acc_new']*100:.1f}%",
            f"- Δ = {ps['delta_pp']:+.2f}pp, McNemar p = {ps['mcnemar_p']:.4f}",
            f"- OOD Δ = {d2.get('ood_delta_pp_for_comparison', 'N/A')}pp (for comparison)",
            f"- {d2['interpretation_detail']}",
            "",
        ]

    lines += [
        "## D3: Pass@k improved but greedy didn't?",
        "",
        "Tests whether RL increases stochastic success (pass@8) without improving greedy@1.",
        "Evaluated on 200 problems where baseline greedy is wrong.",
        "",
        "**PASS/FAIL interpretation:**",
        "- If pass@8 improves meaningfully but greedy@1 flat: need RSFT/DPO to distill correct samples.",
        "- If pass@8 also flat: model capability unchanged; focus on data/capacity/reward signal.",
        "",
    ]

    if d3:
        lines += [
            f"**Result: {d3['interpretation']}**",
            f"- Baseline pass@8: {d3['baseline_pass8_rate']*100:.1f}%",
            f"- Step100  pass@8: {d3['step100_pass8_rate']*100:.1f}%",
            f"- Δ pass@8 = {d3['delta_pass8_pp']:+.1f}pp",
            f"- {d3['interpretation_detail']}",
            "",
        ]

    lines += [
        "## Artifacts",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `d1_movement_summary.json` | D1 summary with paired stats + interpretation |",
        "| `d1_policy_shift_metrics.json` | KL, ratio, reward trajectory from training |",
        "| `paired_records_ood1000_baseline_vs_step100.jsonl` | Per-problem paired OOD-1000 |",
        "| `d2_idheldout_summary.json` | D2 summary with ID-heldout paired comparison |",
        "| `d2_baseline_idheldout.jsonl` | Per-problem baseline results on ID-heldout |",
        "| `d2_step100_idheldout.jsonl` | Per-problem step100 results on ID-heldout |",
        "| `d3_pass8_summary.json` | D3 summary with pass@8 comparison |",
        "| `d3_paired_pass8_wrong200.jsonl` | Per-problem paired pass@8 records |",
        "| `d3_step100_pass8_wrong200.jsonl` | Step100 pass@8 raw results |",
        "",
    ]

    readme_path = DIAG_DIR / "README.md"
    os.makedirs(DIAG_DIR, exist_ok=True)
    with open(readme_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Wrote {readme_path}")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Q1 Diagnostics")
    parser.add_argument("--skip-reeval", action="store_true",
                        help="Skip Phase 0 (re-eval checkpoints for full schema)")
    parser.add_argument("--skip-d1", action="store_true")
    parser.add_argument("--skip-d2", action="store_true")
    parser.add_argument("--skip-d3", action="store_true")
    parser.add_argument("--d2-size", type=int, default=1000)
    parser.add_argument("--d3-size", type=int, default=200)
    args = parser.parse_args()

    os.makedirs(DIAG_DIR, exist_ok=True)

    print(f"=== Q1 Diagnostics: {datetime.now(timezone.utc).isoformat()} ===\n")
    print(f"  Baseline: {BASE_MODEL}")
    print(f"  RL checkpoint: {STEP100_MERGED}")
    print(f"  Output: {DIAG_DIR}\n")

    # Phase 0: Re-evaluate checkpoint files missing full schema
    if not args.skip_reeval:
        print(f"{'='*70}")
        print(f"Phase 0: Verify / re-evaluate checkpoint OOD-1000 files")
        print(f"{'='*70}\n")
        reeval_full_schema()

    d1_result = None
    d2_result = None
    d3_result = None

    if not args.skip_d1:
        d1_result = run_d1()

    if not args.skip_d2:
        d2_result = run_d2(args.d2_size)

    if not args.skip_d3:
        d3_result = run_d3(args.d3_size)

    write_readme(d1_result, d2_result, d3_result)

    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC SUMMARY")
    print(f"{'='*70}")
    if d1_result:
        print(f"  D1: {d1_result['interpretation']} — {d1_result['interpretation_detail'][:80]}")
    if d2_result:
        print(f"  D2: {d2_result['interpretation']} — {d2_result['interpretation_detail'][:80]}")
    if d3_result:
        print(f"  D3: {d3_result['interpretation']} — {d3_result['interpretation_detail'][:80]}")
    print(f"\nAll artifacts in {DIAG_DIR}")


if __name__ == "__main__":
    main()
