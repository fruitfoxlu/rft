#!/usr/bin/env python3
"""Evaluate a GRPO LoRA checkpoint on OOD-1000.

Merges LoRA adapter into base model, serves with vLLM, evaluates greedy
on OOD-1000, and runs paired McNemar test against baseline.

Usage:
    python eval_grpo_checkpoint.py --lora /mnt/data/rft_checkpoints/qwen14b-grpo-b1/global_step100_hf \
        --label b1-step100 --tp 2

    # Skip baseline if already evaluated:
    python eval_grpo_checkpoint.py --lora ... --label ... --skip-baseline
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

import numpy as np
from scipy import stats as scipy_stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reward_func_em import extract_model_answer, check_correctness, normalize_answer

# ── Config ─────────────────────────────────────────────────────────────
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
OOD_DATA = SCRIPT_DIR / "data" / "probe_set_1000_ood.jsonl"
OUT_DIR = Path("/mnt/scratch/qwen14b_eval")

BOXED_SUFFIX = (
    "\n\nPlease reason step by step but keep it concise, and put your final "
    "answer within \\boxed{...}. In the final line, output ONLY "
    "\\boxed{<integer>} and nothing else."
)

PORT = 8000
MAX_TOKENS = 2048


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


def merge_lora(base_model: str, lora_path: str, output_path: str):
    """Merge LoRA adapter into base model."""
    print(f"  Merging LoRA: {lora_path} → {output_path}", flush=True)
    if os.path.exists(output_path) and os.listdir(output_path):
        print(f"  Already merged: {output_path}", flush=True)
        return

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print(f"  Loading base model: {base_model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True
    )
    print(f"  Loading LoRA adapter: {lora_path}", flush=True)
    model = PeftModel.from_pretrained(model, lora_path)
    print(f"  Merging...", flush=True)
    model = model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)
    print(f"  Saved merged model: {output_path}", flush=True)


def start_vllm_server(model_path: str, tp: int, port: int = PORT) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--tensor-parallel-size", str(tp),
        "--gpu-memory-utilization", "0.85",
        "--max-model-len", "4096",
        "--port", str(port),
        "--trust-remote-code",
    ]
    print(f"  Starting vLLM: {' '.join(cmd)}", flush=True)
    log_file = open(OUT_DIR / "vllm_eval.log", "w")
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


def compute_paired_stats(y_base, y_new, label):
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

    return {
        "label": label,
        "n_paired": n,
        "acc_base": float(acc_base),
        "acc_new": float(acc_new),
        "delta_pp": float(delta * 100),
        "b_discordant": b,
        "c_discordant": c,
        "b_plus_c": b + c,
        "mcnemar_p": float(mcnemar_p),
        "ci_low_pp": float(np.percentile(boot_deltas, 2.5) * 100),
        "ci_high_pp": float(np.percentile(boot_deltas, 97.5) * 100),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate GRPO checkpoint")
    parser.add_argument("--lora", required=True, help="Path to LoRA checkpoint (HF format)")
    parser.add_argument("--label", required=True, help="Label for this experiment (e.g. b1-step100)")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip baseline eval (reuse existing)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    problems = load_jsonl(OOD_DATA)

    merged_path = OUT_DIR / f"merged_{args.label}"

    print(f"=== GRPO Checkpoint Evaluation: {datetime.now(timezone.utc).isoformat()} ===")
    print(f"  LoRA: {args.lora}")
    print(f"  Label: {args.label}")
    print(f"  Merged: {merged_path}")
    print(f"  OOD-1000: {len(problems)} problems\n")

    # Phase 1: Merge LoRA
    if not args.skip_merge:
        merge_lora(BASE_MODEL, args.lora, str(merged_path))

    base_url = f"http://localhost:{PORT}/v1"

    # Phase 2: Evaluate baseline (shared across all experiments)
    baseline_path = OUT_DIR / "baseline_ood1000.jsonl"
    if not args.skip_baseline or not baseline_path.exists():
        print(f"\n--- Evaluating baseline on OOD-1000 ---")
        proc = start_vllm_server(BASE_MODEL, tp=args.tp)
        if not wait_for_server(base_url):
            stop_server(proc)
            sys.exit(1)
        baseline_results = eval_greedy(base_url, BASE_MODEL, problems, "baseline")
        with open(baseline_path, "w") as f:
            for r in baseline_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stop_server(proc)
        time.sleep(5)
    else:
        print(f"  Loading existing baseline: {baseline_path}")
        baseline_results = load_jsonl(baseline_path)

    # Phase 3: Evaluate checkpoint
    ckpt_path = OUT_DIR / f"{args.label}_ood1000.jsonl"
    print(f"\n--- Evaluating {args.label} on OOD-1000 ---")
    proc = start_vllm_server(str(merged_path), tp=args.tp)
    if not wait_for_server(base_url):
        stop_server(proc)
        sys.exit(1)
    ckpt_results = eval_greedy(base_url, str(merged_path), problems, args.label)
    with open(ckpt_path, "w") as f:
        for r in ckpt_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stop_server(proc)

    # Phase 4: Paired comparison
    base_by_hash = {r["problem_hash"]: r for r in baseline_results}
    ckpt_by_hash = {r["problem_hash"]: r for r in ckpt_results}
    common = sorted(set(base_by_hash.keys()) & set(ckpt_by_hash.keys()))

    y_base = np.array([base_by_hash[h]["correct"] for h in common])
    y_ckpt = np.array([ckpt_by_hash[h]["correct"] for h in common])

    stats = compute_paired_stats(y_base, y_ckpt, f"{args.label} vs Baseline on OOD-1000")

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS: {args.label}")
    print(f"{'='*60}")
    print(f"  Baseline:   {stats['acc_base']*100:.1f}% ({int(y_base.sum())}/{len(common)})")
    print(f"  {args.label}: {stats['acc_new']*100:.1f}% ({int(y_ckpt.sum())}/{len(common)})")
    print(f"  Δ: {stats['delta_pp']:+.2f}pp")
    print(f"  McNemar p: {stats['mcnemar_p']:.4f}")
    print(f"  95% CI: [{stats['ci_low_pp']:+.2f}, {stats['ci_high_pp']:+.2f}]pp")
    print(f"  b (base✓ ckpt✗): {stats['b_discordant']}")
    print(f"  c (base✗ ckpt✓): {stats['c_discordant']}")
    print(f"  b+c: {stats['b_plus_c']}")

    # Gate checks
    gate_1a = stats['delta_pp'] >= 2.0 and stats['mcnemar_p'] < 0.10
    gate_1b = stats['delta_pp'] >= 3.0 and stats['mcnemar_p'] < 0.05
    print(f"\n  Gate-1a (Δ≥2pp, p<0.10): {'PASS' if gate_1a else 'FAIL'}")
    print(f"  Gate-1b (Δ≥3pp, p<0.05): {'PASS' if gate_1b else 'FAIL'}")

    # Save summary
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": args.label,
        "base_model": BASE_MODEL,
        "lora_path": args.lora,
        "merged_path": str(merged_path),
        "eval_dataset": str(OOD_DATA),
        "paired_stats": stats,
        "gate_1a": gate_1a,
        "gate_1b": gate_1b,
    }
    summary_path = OUT_DIR / f"{args.label}_eval_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
