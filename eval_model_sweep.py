#!/usr/bin/env python3
"""First-wave model sweep: evaluate 4 models on OOD-1000 / ID-200 / AIME-18.

Goal: find a model with baseline accuracy ~60-70% on OOD-1000 so that RL gains
are detectable (gpt-oss-20b already scores ~73.8%, leaving minimal headroom).

Stage 1 (default): OOD-1000 only for all 4 models.
Stage 2: ID-200 + AIME-18 for selected models only.

Usage:
    # Smoke test (20 problems per eval set per model):
    python eval_model_sweep.py --limit 20

    # Full Stage 1 sweep (OOD-1000 only):
    python eval_model_sweep.py --stage 1

    # Stage 2 for selected models:
    python eval_model_sweep.py --stage 2 --models llama3.1-8b qwen2.5-14b
"""

import argparse
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
    """Load HF_TOKEN from ~/.bashrc if not already in the environment."""
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
    print("WARNING: HF_TOKEN not found in environment or ~/.bashrc", flush=True)


_ensure_hf_token()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = Path("/mnt/scratch/model_sweep")

BOXED_SUFFIX = (
    "\n\nPlease reason step by step but keep it concise, and put your final "
    "answer within \\boxed{...}. In the final line, output ONLY "
    "\\boxed{<integer>} and nothing else."
)

MODELS = {
    "gpt-oss-20b": {
        "hf_id": "openai/gpt-oss-20b",
        "tp": 2,
        "max_model_len": 8192,
    },
    "llama3.1-8b": {
        "hf_id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "tp": 1,
        "max_model_len": 8192,
    },
    "qwen2.5-14b": {
        "hf_id": "Qwen/Qwen2.5-14B-Instruct",
        "tp": 2,
        "max_model_len": 8192,
    },
    "qwen2.5-32b": {
        "hf_id": "Qwen/Qwen2.5-32B-Instruct",
        "tp": 2,
        "max_model_len": 8192,
    },
}

EVAL_SETS = {
    "ood1000": str(DATA_DIR / "probe_set_1000_ood.jsonl"),
    "id200": str(DATA_DIR / "probe_set_200.jsonl"),
    "aime": str(DATA_DIR / "aime_eval.jsonl"),
}

MAX_TOKENS = 4096
PORT = 8000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_problems(path: str, limit: int | None = None) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
                if limit and len(problems) >= limit:
                    break
    return problems


def problem_hash(problem: dict) -> str:
    text = " ".join(problem["input"].split()).strip().lower()
    return hashlib.md5(text.encode()).hexdigest()


def detect_parse_method(response: str, model_answer: str) -> str:
    """Return 'boxed', 'last_number', or 'fail'."""
    boxed_matches = list(re.finditer(r"\\boxed\{", response))
    if boxed_matches:
        return "boxed"
    if model_answer:
        return "last_number"
    return "fail"


def is_truncated(response: str, finish_reason: str) -> bool:
    if finish_reason == "length":
        return True
    stripped = response.rstrip()
    is_long = len(response) > 3000
    ends_mid = (
        bool(stripped)
        and stripped[-1] not in ".!?)}\n"
        and not stripped.endswith("$$")
    )
    no_boxed = not re.search(r"\\boxed\{", response)
    return (no_boxed and is_long) or ends_mid


# ---------------------------------------------------------------------------
# vLLM server management
# ---------------------------------------------------------------------------


def start_vllm_server(model_cfg: dict, port: int = PORT) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_cfg["hf_id"],
        "--tensor-parallel-size", str(model_cfg["tp"]),
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", str(model_cfg["max_model_len"]),
        "--port", str(port),
        "--trust-remote-code",
    ]
    print(f"  Starting vLLM: {' '.join(cmd)}", flush=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_path = OUTPUT_DIR / "vllm_server.log"
    log_file = open(log_path, "w")
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def eval_model(base_url: str, model_hf_id: str, problems: list[dict],
               label: str) -> list[dict]:
    """Run greedy eval with BOXED_SUFFIX prompt, return per-problem results."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="unused")
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
                    model=model_hf_id,
                    messages=[{"role": "user", "content": prompt_text}],
                    max_tokens=MAX_TOKENS,
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
                text = ""
                finish_reason = "error"

        model_answer = extract_model_answer(text)
        correct = int(check_correctness(model_answer, str(p["label"])))
        correct_count += correct
        parse_method = detect_parse_method(text, model_answer)
        truncated = is_truncated(text, finish_reason)

        results.append({
            "problem_hash": p_hash,
            "correct": correct,
            "model_answer": model_answer,
            "truth": str(p["label"]),
            "parse_method": parse_method,
            "output": text,
            "finish_reason": finish_reason,
            "source": p.get("source", ""),
        })

        if (i + 1) % 50 == 0 or i == len(problems) - 1:
            print(f"  [{label}] {i+1}/{len(problems)}: "
                  f"{correct_count}/{i+1} = "
                  f"{correct_count/(i+1)*100:.1f}%", flush=True)

    return results


def save_results(results: list[dict], path: str):
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved {len(results)} results to {path}", flush=True)


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------


def compute_stats(results: list[dict], eval_name: str) -> dict:
    """Compute accuracy, bucket breakdown, parse stats, truncation rate."""
    n = len(results)
    if n == 0:
        return {}

    correct = sum(r["correct"] for r in results)
    accuracy = correct / n

    # Source buckets (only meaningful for OOD-1000)
    math_results = [r for r in results if r.get("source", "").startswith("math")]
    comp_results = [r for r in results if r.get("source", "") and not r["source"].startswith("math")]

    math_acc = (sum(r["correct"] for r in math_results) / len(math_results)
                if math_results else None)
    comp_acc = (sum(r["correct"] for r in comp_results) / len(comp_results)
                if comp_results else None)

    # Parse breakdown
    boxed_count = sum(1 for r in results if r["parse_method"] == "boxed")
    ln_count = sum(1 for r in results if r["parse_method"] == "last_number")
    fail_count = sum(1 for r in results if r["parse_method"] == "fail")

    # Truncation
    trunc_count = sum(1 for r in results if is_truncated(r["output"], r["finish_reason"]))

    return {
        "n": n,
        "correct": correct,
        "accuracy": accuracy,
        "math_n": len(math_results),
        "math_acc": math_acc,
        "comp_n": len(comp_results),
        "comp_acc": comp_acc,
        "boxed_pct": boxed_count / n * 100,
        "ln_pct": ln_count / n * 100,
        "fail_pct": fail_count / n * 100,
        "trunc_pct": trunc_count / n * 100,
    }


def print_summary_table(all_stats: dict[str, dict[str, dict]]):
    """Print formatted summary table across all models and eval sets."""
    print(f"\n{'='*100}", flush=True)
    print("SUMMARY TABLE", flush=True)
    print(f"{'='*100}", flush=True)

    # Header
    header = f"{'Model':<16}"
    eval_names = []
    for model_name in all_stats:
        eval_names = list(all_stats[model_name].keys())
        break
    for en in eval_names:
        header += f" | {'Acc':>7} {'MATH':>7} {'Comp':>7} {'Box%':>5} {'LN%':>5} {'Fail%':>5} {'Trunc%':>6}"
    # Print eval set labels
    label_row = f"{'':16}"
    for en in eval_names:
        label_row += f" | {'--- ' + en + ' ---':^50}"
    print(label_row, flush=True)
    print(header, flush=True)
    print("-" * (16 + len(eval_names) * 52), flush=True)

    for model_name in all_stats:
        row = f"{model_name:<16}"
        for en in eval_names:
            s = all_stats[model_name].get(en)
            if s is None:
                row += f" | {'--':>7} {'--':>7} {'--':>7} {'--':>5} {'--':>5} {'--':>5} {'--':>6}"
                continue
            acc_str = f"{s['accuracy']*100:.1f}%"
            math_str = f"{s['math_acc']*100:.1f}%" if s["math_acc"] is not None else "--"
            comp_str = f"{s['comp_acc']*100:.1f}%" if s["comp_acc"] is not None else "--"
            row += (f" | {acc_str:>7} {math_str:>7} {comp_str:>7}"
                    f" {s['boxed_pct']:>5.1f} {s['ln_pct']:>5.1f}"
                    f" {s['fail_pct']:>5.1f} {s['trunc_pct']:>6.1f}")
        print(row, flush=True)

    print(f"{'='*100}\n", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="First-wave model sweep")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Stage 1: OOD-1000 only. Stage 2: ID-200 + AIME-18.")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to evaluate (default: all for stage 1)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit problems per eval set (smoke test)")
    return parser.parse_args()


def main():
    args = parse_args()
    limit = args.limit or (int(os.environ["EVAL_LIMIT"])
                           if "EVAL_LIMIT" in os.environ else None)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_url = f"http://localhost:{PORT}/v1"

    # Determine which models to run
    if args.models:
        model_names = args.models
        for m in model_names:
            if m not in MODELS:
                print(f"ERROR: Unknown model '{m}'. "
                      f"Available: {list(MODELS.keys())}", flush=True)
                sys.exit(1)
    else:
        model_names = list(MODELS.keys())

    # Determine which eval sets to run
    if args.stage == 1:
        eval_set_names = ["ood1000"]
    else:
        eval_set_names = ["id200", "aime"]

    # For stage 1, gpt-oss-20b only runs OOD-1000
    # For stage 2, gpt-oss-20b is excluded (not a candidate)
    if args.stage == 2 and "gpt-oss-20b" in model_names:
        print("Note: Excluding gpt-oss-20b from Stage 2 (baseline only).",
              flush=True)
        model_names = [m for m in model_names if m != "gpt-oss-20b"]

    # Load problems
    all_problems = {}
    for en in eval_set_names:
        all_problems[en] = load_problems(EVAL_SETS[en], limit=limit)
        print(f"  Eval set: {en} = {len(all_problems[en])} problems"
              f"{f' (limited to {limit})' if limit else ''}", flush=True)

    # Collect all stats for summary table
    all_stats: dict[str, dict[str, dict]] = {}

    for model_name in model_names:
        model_cfg = MODELS[model_name]
        print(f"\n{'='*60}", flush=True)
        print(f"EVALUATING: {model_name}", flush=True)
        print(f"  HF ID: {model_cfg['hf_id']}", flush=True)
        print(f"  TP: {model_cfg['tp']}, max_model_len: {model_cfg['max_model_len']}", flush=True)
        print(f"{'='*60}", flush=True)

        proc = start_vllm_server(model_cfg, port=PORT)
        if not wait_for_server(base_url, timeout=600):
            print(f"ERROR: vLLM server did not start for {model_name}",
                  flush=True)
            # Print last 50 lines of log for debugging
            log_path = OUTPUT_DIR / "vllm_server.log"
            if log_path.exists():
                print("--- Last 50 lines of vLLM log ---", flush=True)
                lines = log_path.read_text().splitlines()
                for line in lines[-50:]:
                    print(f"  {line}", flush=True)
                print("--- End of log ---", flush=True)
            stop_server(proc)
            sys.exit(1)

        all_stats[model_name] = {}
        for en in eval_set_names:
            problems = all_problems[en]
            label = f"{model_name}/{en}"
            t0 = time.time()
            results = eval_model(base_url, model_cfg["hf_id"], problems, label)
            elapsed = time.time() - t0

            stats = compute_stats(results, en)
            all_stats[model_name][en] = stats

            # Save per-problem JSONL
            out_path = str(OUTPUT_DIR / f"{model_name}_{en}.jsonl")
            save_results(results, out_path)

            acc = stats["accuracy"] * 100
            print(f"  -> {en}: {acc:.2f}% "
                  f"({stats['correct']}/{stats['n']}) "
                  f"[boxed={stats['boxed_pct']:.0f}% "
                  f"ln={stats['ln_pct']:.0f}% "
                  f"fail={stats['fail_pct']:.0f}% "
                  f"trunc={stats['trunc_pct']:.0f}%] "
                  f"[{elapsed:.1f}s]", flush=True)

        stop_server(proc)
        print(f"  Server stopped for {model_name}. Waiting 10s...", flush=True)
        time.sleep(10)

    # Print summary table
    print_summary_table(all_stats)

    # Save summary JSON
    summary_path = str(OUTPUT_DIR / f"sweep_stage{args.stage}_summary.json")
    # Convert for JSON serialization
    json_stats = {}
    for mn, evals in all_stats.items():
        json_stats[mn] = {}
        for en, s in evals.items():
            json_stats[mn][en] = {k: (float(v) if v is not None else None)
                                   for k, v in s.items()}
    with open(summary_path, "w") as f:
        json.dump(json_stats, f, indent=2)
    print(f"Summary saved to {summary_path}", flush=True)

    # Selection guidance (Stage 1 only)
    if args.stage == 1 and "ood1000" in eval_set_names:
        print(f"\n{'='*60}", flush=True)
        print("SELECTION GUIDANCE", flush=True)
        print(f"{'='*60}", flush=True)
        print("Target: OOD-1000 accuracy in ~60-70% range", flush=True)
        print("Prefer: higher boxed%, lower fail%, smaller model\n", flush=True)

        candidates = []
        for mn in all_stats:
            s = all_stats[mn].get("ood1000")
            if s:
                candidates.append((mn, s))

        candidates.sort(key=lambda x: (
            # Prefer accuracy in 60-70% (penalize distance from 65%)
            abs(x[1]["accuracy"] * 100 - 65),
            # Then higher boxed%
            -x[1]["boxed_pct"],
            # Then lower fail%
            x[1]["fail_pct"],
        ))

        for rank, (mn, s) in enumerate(candidates, 1):
            acc = s["accuracy"] * 100
            in_range = "<<< IN RANGE" if 60 <= acc <= 70 else ""
            print(f"  {rank}. {mn}: {acc:.1f}% "
                  f"(boxed={s['boxed_pct']:.0f}%, "
                  f"fail={s['fail_pct']:.0f}%) {in_range}", flush=True)

        print(f"\nNext: run Stage 2 for top candidates:", flush=True)
        top = [mn for mn, s in candidates[:2] if mn != "gpt-oss-20b"]
        if top:
            print(f"  python eval_model_sweep.py --stage 2 --models "
                  f"{' '.join(top)}", flush=True)


if __name__ == "__main__":
    main()
