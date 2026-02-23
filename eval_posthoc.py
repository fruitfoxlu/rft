#!/usr/bin/env python3
"""Post-hoc evaluation of RL checkpoints on configurable eval sets.

Starts a vLLM server for each model, evaluates all problems via OpenAI API,
prints a summary table. Supports both already-merged models and raw LoRA
adapters (auto-detected by presence of adapter_config.json).

Usage:
    # Eval already-merged models on OOD-1000 + ID-200 + AIME-18
    python eval_posthoc.py \
      --model a30_step130=/mnt/scratch/merged_models/a30_step130 \
      --model a30_step200=/mnt/scratch/merged_models/a30_step200 \
      --eval ood1000=data/probe_set_1000_ood.jsonl \
      --eval id200=data/probe_set_200.jsonl \
      --eval aime=data/aime_eval.jsonl

    # Defaults: 3 merged models × (OOD-1000 + ID-200 + AIME-18)
    python eval_posthoc.py

    # Include base model
    python eval_posthoc.py --include-base
"""

import argparse
import gc
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from reward_func_em import extract_model_answer, check_correctness

DATA_DIR = SCRIPT_DIR / "data"
BASE_MODEL = "openai/gpt-oss-20b"

DEFAULT_MODELS = {
    "a29b_step200": "/mnt/scratch/merged_models/a29b_step200",
    "a30_step130": "/mnt/scratch/merged_models/a30_step130",
    "a30_step200": "/mnt/scratch/merged_models/a30_step200",
}

DEFAULT_EVALS = {
    "ood1000": str(DATA_DIR / "probe_set_1000_ood.jsonl"),
    "id200": str(DATA_DIR / "probe_set_200.jsonl"),
    "aime": str(DATA_DIR / "aime_eval.jsonl"),
}

# Adapter-to-merged model mapping for auto-merge
ADAPTER_CHECKPOINTS = {
    "a29b_step200": "/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a29b/global_step200_hf",
    "a30_step130": "/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a30/global_step130_hf",
    "a30_step200": "/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a30/global_step200_hf",
}


def load_problems(path: str) -> list[dict]:
    problems = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))
    return problems


def merge_checkpoint_subprocess(base_model: str, adapter_path: str, output_dir: str):
    """Merge LoRA adapter in a separate process (CPU-only, no CUDA pollution)."""
    merge_script = f"""
import json, os, gc
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

print("  Loading base model on CPU...", flush=True)
tokenizer = AutoTokenizer.from_pretrained("{base_model}", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "{base_model}", torch_dtype=torch.bfloat16, device_map="cpu",
    trust_remote_code=True,
)
print("  Loading LoRA adapter...", flush=True)
model = PeftModel.from_pretrained(model, "{adapter_path}", device_map="cpu")
model = model.merge_and_unload()

print("  Saving merged model to {output_dir}...", flush=True)
os.makedirs("{output_dir}", exist_ok=True)
model.save_pretrained("{output_dir}")
tokenizer.save_pretrained("{output_dir}")

del model
gc.collect()
print("  Merge complete.", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", merge_script],
        capture_output=False, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Merge failed for {adapter_path}")


def start_vllm_server(model_path: str, tp: int = 2, port: int = 8000,
                      log_dir: str = "/mnt/scratch") -> subprocess.Popen:
    """Start a vLLM server."""
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
    log_file = open(os.path.join(log_dir, "vllm_server.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    proc._log_file = log_file
    return proc


def wait_for_server(base_url: str, timeout: int = 300) -> bool:
    """Wait for vLLM server to be ready."""
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
    """Stop a vLLM server subprocess."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if hasattr(proc, '_log_file'):
        proc._log_file.close()


def eval_via_api(base_url: str, model_name: str, problems: list[dict],
                 max_tokens: int, label: str) -> tuple[float, list[dict]]:
    """Evaluate using OpenAI-compatible API (greedy, temp=0, n=1)."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="unused")
    results = []
    correct_count = 0

    for i, p in enumerate(problems):
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
        correct = check_correctness(model_answer, str(p["label"]))
        correct_count += correct

        results.append({
            "correct": correct,
            "model_answer": model_answer,
            "truth": str(p["label"]),
        })

        if (i + 1) % 50 == 0 or i == len(problems) - 1:
            print(f"  [{label}] {i+1}/{len(problems)}: "
                  f"{correct_count:.0f}/{i+1} = {correct_count/(i+1)*100:.1f}%",
                  flush=True)

    accuracy = correct_count / len(problems) if problems else 0
    return accuracy, results


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc RL checkpoint evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", action="append", metavar="NAME=PATH",
                        help="Model to evaluate: name=model_dir (repeatable). "
                             "If model_dir contains adapter_config.json, auto-merges first.")
    parser.add_argument("--eval", action="append", metavar="NAME=PATH",
                        help="Eval set: name=path (repeatable). "
                             "Default: ood1000 + id200 + aime.")
    parser.add_argument("--include-base", action="store_true",
                        help="Also eval base model (no LoRA)")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--output", type=str,
                        default="/mnt/scratch/posthoc_eval_results.json")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--merged-dir", type=str, default="/mnt/scratch/merged_models",
                        help="Directory for auto-merged models")
    args = parser.parse_args()

    # Parse --model flags
    models = {}
    if args.model:
        for spec in args.model:
            if "=" not in spec:
                raise ValueError(f"Invalid --model spec: {spec}. Use name=path format.")
            name, path = spec.split("=", 1)
            models[name] = path
    else:
        models = dict(DEFAULT_MODELS)

    # Parse --eval flags
    eval_sets = {}
    if args.eval:
        for spec in args.eval:
            if "=" not in spec:
                raise ValueError(f"Invalid --eval spec: {spec}. Use name=path format.")
            name, path = spec.split("=", 1)
            eval_sets[name] = path
    else:
        eval_sets = dict(DEFAULT_EVALS)

    base_url = f"http://localhost:{args.port}/v1"

    # Validate models
    for name, path in models.items():
        if not os.path.exists(path):
            # Check if it's a raw adapter that needs merging
            if name in ADAPTER_CHECKPOINTS and os.path.exists(ADAPTER_CHECKPOINTS[name]):
                print(f"  {name}: model dir not found, will auto-merge from adapter", flush=True)
            else:
                raise FileNotFoundError(f"Model {name} not found: {path}")

    # Validate eval sets
    for name, path in eval_sets.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Eval set {name} not found: {path}")

    # Load all problem sets once
    all_problems = {}
    for eval_name, eval_path in eval_sets.items():
        all_problems[eval_name] = load_problems(eval_path)
        n = len(all_problems[eval_name])
        print(f"  Eval set: {eval_name} = {n} problems ({Path(eval_path).name})", flush=True)

    all_results = {}
    os.makedirs(args.merged_dir, exist_ok=True)

    # --- Step 1: Auto-merge any adapters that need it ---
    for model_name, model_path in models.items():
        if os.path.exists(os.path.join(model_path, "config.json")):
            print(f"  {model_name}: ready (merged model found)", flush=True)
            continue
        if os.path.exists(os.path.join(model_path, "adapter_config.json")):
            # Raw adapter — merge it
            merged_path = os.path.join(args.merged_dir, model_name)
            if os.path.exists(os.path.join(merged_path, "config.json")):
                print(f"  {model_name}: already merged (cached at {merged_path})", flush=True)
                models[model_name] = merged_path
                continue
            print(f"  {model_name}: merging adapter...", flush=True)
            t0 = time.time()
            merge_checkpoint_subprocess(BASE_MODEL, model_path, merged_path)
            print(f"  {model_name}: merged in {time.time() - t0:.1f}s", flush=True)
            models[model_name] = merged_path
        elif model_name in ADAPTER_CHECKPOINTS and not os.path.exists(model_path):
            # Model dir missing but we know the adapter path
            adapter_path = ADAPTER_CHECKPOINTS[model_name]
            merged_path = os.path.join(args.merged_dir, model_name)
            if os.path.exists(os.path.join(merged_path, "config.json")):
                print(f"  {model_name}: already merged (cached at {merged_path})", flush=True)
                models[model_name] = merged_path
                continue
            print(f"  {model_name}: merging from {adapter_path}...", flush=True)
            t0 = time.time()
            merge_checkpoint_subprocess(BASE_MODEL, adapter_path, merged_path)
            print(f"  {model_name}: merged in {time.time() - t0:.1f}s", flush=True)
            models[model_name] = merged_path

    # --- Step 2: Base model eval ---
    if args.include_base:
        print(f"\n{'='*60}", flush=True)
        print(f"EVALUATING: BASE MODEL (no LoRA)", flush=True)
        print(f"{'='*60}", flush=True)

        proc = start_vllm_server(BASE_MODEL, tp=args.tp, port=args.port)
        if not wait_for_server(base_url, timeout=300):
            print("ERROR: vLLM server did not start for base model", flush=True)
            stop_server(proc)
            sys.exit(1)

        all_results["base"] = {}
        for eval_name, problems in all_problems.items():
            t0 = time.time()
            acc, details = eval_via_api(
                base_url, BASE_MODEL, problems,
                max_tokens=args.max_tokens, label=f"base/{eval_name}",
            )
            elapsed = time.time() - t0
            all_results["base"][eval_name] = {
                "accuracy": acc, "n_problems": len(problems),
                "correct": sum(r["correct"] for r in details),
                "elapsed_s": round(elapsed, 1),
            }
            print(f"  -> {eval_name}: {acc*100:.2f}% ({sum(r['correct'] for r in details)}/{len(problems)}) [{elapsed:.1f}s]", flush=True)

        stop_server(proc)
        time.sleep(5)

    # --- Step 3: Model evals ---
    for model_name, model_path in models.items():
        print(f"\n{'='*60}", flush=True)
        print(f"EVALUATING: {model_name}", flush=True)
        print(f"  Model path: {model_path}", flush=True)
        print(f"{'='*60}", flush=True)

        proc = start_vllm_server(model_path, tp=args.tp, port=args.port)
        if not wait_for_server(base_url, timeout=300):
            print(f"ERROR: vLLM server did not start for {model_name}", flush=True)
            stop_server(proc)
            continue

        all_results[model_name] = {}
        for eval_name, problems in all_problems.items():
            t0 = time.time()
            acc, details = eval_via_api(
                base_url, model_path, problems,
                max_tokens=args.max_tokens, label=f"{model_name}/{eval_name}",
            )
            elapsed = time.time() - t0
            n_correct = sum(r["correct"] for r in details)
            all_results[model_name][eval_name] = {
                "accuracy": acc, "n_problems": len(problems),
                "correct": n_correct,
                "elapsed_s": round(elapsed, 1),
            }
            print(f"  -> {eval_name}: {acc*100:.2f}% ({n_correct}/{len(problems)}) [{elapsed:.1f}s]", flush=True)

        stop_server(proc)
        time.sleep(5)

    # --- Summary ---
    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY TABLE", flush=True)
    print(f"{'='*80}", flush=True)

    # Build header from eval set names and sizes
    header = f"{'Model':<20}"
    eval_names = list(eval_sets.keys())
    for eval_name in eval_names:
        n = len(all_problems[eval_name])
        col = f"{eval_name}({n})"
        header += f" {col:>16}"
    print(header, flush=True)
    print("-" * (20 + 17 * len(eval_names)), flush=True)

    for model_name in all_results:
        row = f"{model_name:<20}"
        for eval_name in eval_names:
            if eval_name in all_results[model_name]:
                r = all_results[model_name][eval_name]
                acc = r["accuracy"] * 100
                correct = r["correct"]
                n = r["n_problems"]
                row += f" {acc:>6.2f}% ({correct:>3}/{n})"
            else:
                row += f" {'--':>16}"
        print(row, flush=True)

    # Statistical notes
    p_est = 0.65
    print(f"\nStatistical notes (SE at p≈{p_est}):", flush=True)
    for eval_name in eval_names:
        n = len(all_problems[eval_name])
        se = math.sqrt(p_est * (1 - p_est) / n) * 100
        print(f"  {eval_name} (N={n}): SE={se:.1f}%, 95% CI ≈ ±{1.96*se:.1f}pp", flush=True)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
