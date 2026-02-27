#!/usr/bin/env python3
"""G6 Post-Run Evaluation for Qwen2.5-14B GRPO Q1.

Evaluates K=3 checkpoints on OOD-1000 with paired comparison vs baseline.

Checkpoints:
  - step 200 (final)
  - step 100 (mid)
  - step 50  (best-by-monitor, trailing 3-eval avg = 58.9%)

For each checkpoint:
  1. Merge LoRA into base model (PEFT)
  2. Serve with vLLM
  3. Eval on OOD-1000 (greedy, temp=0, max_tokens=2048, BOXED_SUFFIX)

Primary test (best checkpoint):
  - Paired records vs baseline qwen2.5-14b
  - Δ, discordants (b,c), McNemar exact p-value, bootstrap 95% CI

Decision gates:
  - Gate-1b: Δ ≥ +3.0pp AND p < 0.05 → claimable win
  - Gate-1a: Δ ≥ +2.0pp AND (p < 0.10 OR CI_low > −0.5pp) → run second seed
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reward_func_em import extract_model_answer, check_correctness

# --- Config ---
BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
CKPT_DIR = Path("/mnt/data/rft_checkpoints/qwen14b-grpo-q01")
MERGED_DIR = Path("/mnt/scratch/merged_models_qwen14b_q01")
OUTPUT_DIR = Path("/mnt/scratch/qwen14b_q01_eval")
BASELINE_PATH = Path("/mnt/scratch/model_sweep/qwen2.5-14b_ood1000.jsonl")
OOD_DATA = SCRIPT_DIR / "data" / "probe_set_1000_ood.jsonl"

BOXED_SUFFIX = (
    "\n\nPlease reason step by step but keep it concise, and put your final "
    "answer within \\boxed{...}. In the final line, output ONLY "
    "\\boxed{<integer>} and nothing else."
)

CHECKPOINTS = {
    "step50": "global_step50_hf",
    "step100": "global_step100_hf",
    "step200": "global_step200_hf",
}

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


def problem_hash(problem: dict) -> str:
    text = " ".join(problem["input"].split()).strip().lower()
    return hashlib.md5(text.encode()).hexdigest()


def merge_lora(adapter_path: str, output_path: str):
    """Merge LoRA adapter into base model using PEFT."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"  Merging LoRA: {adapter_path} -> {output_path}", flush=True)
    if Path(output_path).exists() and (Path(output_path) / "config.json").exists():
        print(f"  Already merged, skipping.", flush=True)
        return

    os.makedirs(output_path, exist_ok=True)
    t0 = time.time()

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Load and merge LoRA
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    # Save
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    elapsed = time.time() - t0
    print(f"  Merged in {elapsed:.0f}s", flush=True)


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


def eval_model(base_url: str, model_name: str, problems: list[dict],
               label: str) -> list[dict]:
    """Evaluate on OOD-1000 with BOXED_SUFFIX."""
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key="unused")
    results = []
    correct_count = 0

    for i, p in enumerate(problems):
        p_hash = problem_hash(p)
        prompt_text = p["input"] + BOXED_SUFFIX
        truth = str(p["label"])

        text = ""
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
                break
            except Exception as e:
                print(f"  [{label}] Error (attempt {attempt + 1}): {e}", flush=True)
                time.sleep(2 * (attempt + 1))
                text = ""

        model_answer = extract_model_answer(text)
        correct = int(check_correctness(model_answer, truth))
        correct_count += correct

        results.append({
            "problem_hash": p_hash,
            "correct": correct,
            "model_answer": model_answer,
            "truth": truth,
            "output": text,
        })

        if (i + 1) % 100 == 0 or i == len(problems) - 1:
            print(f"  [{label}] {i+1}/{len(problems)}: "
                  f"{correct_count}/{i+1} = {correct_count/(i+1)*100:.1f}%",
                  flush=True)

    return results


def compute_paired_stats(results_base: list[dict], results_new: list[dict],
                         label: str) -> dict:
    """Compute paired comparison: Δ, discordants, McNemar, bootstrap CI."""
    base_by_hash = {r["problem_hash"]: r["correct"] for r in results_base}
    new_by_hash = {r["problem_hash"]: r["correct"] for r in results_new}
    common_hashes = sorted(set(base_by_hash.keys()) & set(new_by_hash.keys()))
    n = len(common_hashes)

    if n == 0:
        print(f"  ERROR: No common problems!", flush=True)
        return {}

    y_base = np.array([base_by_hash[h] for h in common_hashes])
    y_new = np.array([new_by_hash[h] for h in common_hashes])

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
    print(f"  Baseline accuracy: {acc_base*100:.2f}% ({int(y_base.sum())}/{n})", flush=True)
    print(f"  RL model accuracy: {acc_new*100:.2f}% ({int(y_new.sum())}/{n})", flush=True)
    print(f"  Δ (RL - baseline): {delta*100:+.2f}pp", flush=True)
    print(f"  Discordants: b={b} (base correct, RL wrong), c={c} (base wrong, RL correct)", flush=True)
    print(f"  Net: c - b = {c - b} problems", flush=True)
    print(f"  McNemar exact p-value: {mcnemar_p:.6f}", flush=True)
    print(f"  Bootstrap 95% CI: [{ci_low*100:+.2f}pp, {ci_high*100:+.2f}pp]", flush=True)

    print(f"\n  Decision gates:", flush=True)
    if delta * 100 >= 3.0 and mcnemar_p < 0.05:
        print(f"  >>> Gate-1b PASS: Δ={delta*100:+.2f}pp >= +3.0pp AND p={mcnemar_p:.4f} < 0.05 → claimable win", flush=True)
    elif delta * 100 >= 2.0 and (mcnemar_p < 0.10 or ci_low * 100 > -0.5):
        print(f"  >>> Gate-1a PASS: Δ={delta*100:+.2f}pp >= +2.0pp → run second seed", flush=True)
    else:
        print(f"  >>> Neither gate passes: Δ={delta*100:+.2f}pp", flush=True)

    return {
        "n_paired": n,
        "acc_base": float(acc_base),
        "acc_new": float(acc_new),
        "delta_pp": float(delta * 100),
        "b_discordant": b,
        "c_discordant": c,
        "mcnemar_p": float(mcnemar_p),
        "ci_low_pp": float(ci_low * 100),
        "ci_high_pp": float(ci_high * 100),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MERGED_DIR, exist_ok=True)

    print(f"=== G6 Post-Run Evaluation: Qwen2.5-14B Q1 ===\n")
    print(f"Base model: {BASE_MODEL}")
    print(f"Checkpoints: {list(CHECKPOINTS.keys())}")
    print(f"Baseline: {BASELINE_PATH}")
    print(f"OOD data: {OOD_DATA}\n")

    # Verify baseline exists
    if not BASELINE_PATH.exists():
        print(f"ERROR: Baseline not found: {BASELINE_PATH}")
        sys.exit(1)

    # Load OOD problems
    problems = load_jsonl(OOD_DATA)
    print(f"OOD-1000: {len(problems)} problems\n")

    # Load baseline results
    baseline_results = load_jsonl(BASELINE_PATH)
    baseline_acc = sum(r["correct"] for r in baseline_results) / len(baseline_results)
    print(f"Baseline accuracy: {baseline_acc*100:.2f}% ({sum(r['correct'] for r in baseline_results)}/{len(baseline_results)})\n")

    # Step 1: Merge all LoRA checkpoints
    print(f"{'='*60}")
    print(f"STEP 1: Merge LoRA adapters")
    print(f"{'='*60}\n")

    merged_paths = {}
    for name, ckpt_dir in CHECKPOINTS.items():
        adapter_path = str(CKPT_DIR / ckpt_dir)
        output_path = str(MERGED_DIR / name)
        merge_lora(adapter_path, output_path)
        merged_paths[name] = output_path
        print()

    # Step 2: Evaluate each checkpoint
    all_results = {}
    base_url = f"http://localhost:{PORT}/v1"

    for name, model_path in merged_paths.items():
        print(f"\n{'='*60}")
        print(f"EVALUATING: {name} ({model_path})")
        print(f"{'='*60}\n")

        proc = start_vllm_server(model_path, port=PORT)
        if not wait_for_server(base_url, timeout=600):
            print(f"ERROR: vLLM server did not start for {name}", flush=True)
            log_path = OUTPUT_DIR / "vllm_server.log"
            if log_path.exists():
                print("--- Last 20 lines of vLLM log ---", flush=True)
                for l in log_path.read_text().splitlines()[-20:]:
                    print(f"  {l}", flush=True)
            stop_server(proc)
            continue

        t0 = time.time()
        results = eval_model(base_url, model_path, problems, name)
        elapsed = time.time() - t0
        acc = sum(r["correct"] for r in results) / len(results)

        all_results[name] = results
        out_path = OUTPUT_DIR / f"{name}_ood1000.jsonl"
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"\n  {name}: {acc*100:.2f}% ({sum(r['correct'] for r in results)}/{len(results)}) [{elapsed:.0f}s]")
        print(f"  Saved to {out_path}")

        stop_server(proc)
        time.sleep(5)

    # Step 3: Summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY TABLE: OOD-1000 Accuracy")
    print(f"{'='*60}")
    print(f"  {'Model':<20} {'Accuracy':>10} {'Correct':>10}")
    print(f"  {'-'*42}")
    print(f"  {'baseline':.<20} {baseline_acc*100:>9.2f}% {sum(r['correct'] for r in baseline_results):>6}/{len(baseline_results)}")
    for name in ["step50", "step100", "step200"]:
        if name in all_results:
            results = all_results[name]
            acc = sum(r["correct"] for r in results) / len(results)
            delta = (acc - baseline_acc) * 100
            print(f"  {name:<20} {acc*100:>9.2f}% {sum(r['correct'] for r in results):>6}/{len(results)}  (Δ={delta:+.2f}pp)")

    # Step 4: Paired comparison for each checkpoint vs baseline
    paired_stats = {}
    for name in ["step50", "step100", "step200"]:
        if name in all_results:
            stats = compute_paired_stats(baseline_results, all_results[name],
                                         f"{name} vs baseline (OOD-1000)")
            paired_stats[name] = stats

            # Save paired records
            paired_path = OUTPUT_DIR / f"paired_{name}_vs_baseline_ood1000.jsonl"
            base_by_hash = {r["problem_hash"]: r for r in baseline_results}
            new_by_hash = {r["problem_hash"]: r for r in all_results[name]}
            with open(paired_path, "w") as f:
                for h in sorted(set(base_by_hash.keys()) & set(new_by_hash.keys())):
                    record = {
                        "problem_hash": h,
                        "y_baseline": base_by_hash[h]["correct"],
                        "y_new": new_by_hash[h]["correct"],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save summary
    summary = {
        "experiment": "qwen14b_q01",
        "base_model": BASE_MODEL,
        "baseline_acc": float(baseline_acc),
        "checkpoint_results": {},
        "paired_stats": paired_stats,
    }
    for name in ["step50", "step100", "step200"]:
        if name in all_results:
            results = all_results[name]
            acc = sum(r["correct"] for r in results) / len(results)
            summary["checkpoint_results"][name] = {
                "accuracy": float(acc),
                "correct": sum(r["correct"] for r in results),
                "n_problems": len(results),
            }

    summary_path = OUTPUT_DIR / "g6_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFull results saved to {summary_path}")


if __name__ == "__main__":
    main()
