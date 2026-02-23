#!/usr/bin/env python3
"""Post-hoc evaluation of RL checkpoints on ID probe, OOD probe, and AIME.

Strategy:
1. Merge LoRA adapters into base model (CPU-only subprocess to avoid CUDA pollution)
2. Start a vLLM server with each merged model (TP=2, MXFP4)
3. Evaluate all problems via OpenAI-compatible API
4. Stop server, repeat for next checkpoint

Usage:
    python eval_posthoc.py
    python eval_posthoc.py --include-base
"""

import argparse
import gc
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
BASE_MODEL = "openai/gpt-oss-20b"

EVAL_SETS = {
    "id_probe": DATA_DIR / "probe_set_200.jsonl",
    "ood_probe": DATA_DIR / "probe_set_200_ood.jsonl",
    "aime": DATA_DIR / "aime_eval.jsonl",
}

DEFAULT_CHECKPOINTS = {
    "a29a_step200": "/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a29a/global_step200_hf",
    "a29b_step80": "/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a29b/global_step80_hf",
    "a29b_step200": "/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a29b/global_step200_hf",
}


def load_problems(path: Path) -> list[dict]:
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
# Note: quantization_config is preserved — MLP expert weights remain MXFP4

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
    print(f"  Starting vLLM: {' '.join(cmd)}")
    # Redirect to file to avoid pipe buffer deadlock
    log_file = open(os.path.join(log_dir, "vllm_server.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    proc._log_file = log_file  # keep reference to close later
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
                print(f"  Server ready: {models.data[0].id}")
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
    """Evaluate using OpenAI-compatible API."""
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
                print(f"  [{label}] Error (attempt {attempt + 1}): {e}")
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
                  f"{correct_count:.0f}/{i+1} = {correct_count/(i+1)*100:.1f}%")

    accuracy = correct_count / len(problems) if problems else 0
    return accuracy, results


def main():
    parser = argparse.ArgumentParser(description="Post-hoc RL checkpoint evaluation")
    parser.add_argument("--include-base", action="store_true",
                        help="Also eval base model (no LoRA)")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--output", type=str,
                        default="/mnt/scratch/posthoc_eval_results.json")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--merged-dir", type=str, default="/mnt/scratch/merged_models",
                        help="Directory for merged models")
    args = parser.parse_args()

    checkpoints = DEFAULT_CHECKPOINTS
    base_url = f"http://localhost:{args.port}/v1"

    # Validate
    for name, path in checkpoints.items():
        assert os.path.exists(path), f"Checkpoint {name} not found: {path}"
    for name, path in EVAL_SETS.items():
        assert path.exists(), f"Eval set {name} not found: {path}"
        print(f"  Eval set: {name} = {len(load_problems(path))} problems")

    # Load all problem sets once
    all_problems = {}
    for eval_name, eval_path in EVAL_SETS.items():
        all_problems[eval_name] = load_problems(eval_path)

    all_results = {}
    os.makedirs(args.merged_dir, exist_ok=True)

    # --- Step 1: Merge all checkpoints (CPU-only subprocesses) ---
    print(f"\n{'='*60}")
    print("STEP 1: Merging LoRA checkpoints")
    print(f"{'='*60}")
    for ckpt_name, ckpt_path in checkpoints.items():
        merged_path = os.path.join(args.merged_dir, ckpt_name)
        if os.path.exists(os.path.join(merged_path, "config.json")):
            print(f"  {ckpt_name}: already merged (cached)")
            continue
        print(f"  Merging {ckpt_name}...")
        t0 = time.time()
        merge_checkpoint_subprocess(BASE_MODEL, ckpt_path, merged_path)
        print(f"  {ckpt_name}: merged in {time.time() - t0:.1f}s")

    # --- Step 2: Base model eval ---
    if args.include_base:
        print(f"\n{'='*60}")
        print(f"EVALUATING: BASE MODEL (no LoRA)")
        print(f"{'='*60}")

        proc = start_vllm_server(BASE_MODEL, tp=args.tp, port=args.port)
        if not wait_for_server(base_url, timeout=300):
            print("ERROR: vLLM server did not start for base model")
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
            print(f"  -> {eval_name}: {acc*100:.2f}% [{elapsed:.1f}s]")

        stop_server(proc)
        time.sleep(5)

    # --- Step 3: Checkpoint evals ---
    for ckpt_name, ckpt_path in checkpoints.items():
        merged_path = os.path.join(args.merged_dir, ckpt_name)
        print(f"\n{'='*60}")
        print(f"EVALUATING: {ckpt_name}")
        print(f"  Merged model: {merged_path}")
        print(f"{'='*60}")

        proc = start_vllm_server(merged_path, tp=args.tp, port=args.port)
        if not wait_for_server(base_url, timeout=300):
            print(f"ERROR: vLLM server did not start for {ckpt_name}")
            stop_server(proc)
            continue

        all_results[ckpt_name] = {}
        for eval_name, problems in all_problems.items():
            t0 = time.time()
            acc, details = eval_via_api(
                base_url, merged_path, problems,
                max_tokens=args.max_tokens, label=f"{ckpt_name}/{eval_name}",
            )
            elapsed = time.time() - t0
            all_results[ckpt_name][eval_name] = {
                "accuracy": acc, "n_problems": len(problems),
                "correct": sum(r["correct"] for r in details),
                "elapsed_s": round(elapsed, 1),
            }
            print(f"  -> {eval_name}: {acc*100:.2f}% [{elapsed:.1f}s]")

        stop_server(proc)
        time.sleep(5)

    # --- Summary ---
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'ID(200)':>10} {'OOD(202)':>10} {'AIME(18)':>10}")
    print("-" * 55)
    for model_name in all_results:
        row = f"{model_name:<20}"
        for eval_name in ["id_probe", "ood_probe", "aime"]:
            if eval_name in all_results[model_name]:
                acc = all_results[model_name][eval_name]["accuracy"] * 100
                row += f" {acc:>9.2f}%"
            else:
                row += f" {'--':>10}"
        print(row)

    print(f"\nNote: SE ≈ sqrt(p(1-p)/N) at p~0.65:")
    print(f"  ID (N=200):  SE=3.4%, 95% CI ≈ ±6.7pp")
    print(f"  OOD (N=202): SE=3.4%, 95% CI ≈ ±6.6pp")
    print(f"  AIME (N=18): SE=11.2%, 95% CI ≈ ±22pp (sanity only)")

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
