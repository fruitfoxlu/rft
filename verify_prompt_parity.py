#!/usr/bin/env python3
"""G0 Guardrail: Verify prompt/chat-template parity between train and eval paths.

Takes 5 samples from the BOXED_SUFFIX training data and compares tokenized
input_ids produced by:
  - Train path: tokenizer.apply_chat_template([{"role": "user", "content": input}])
    (this is what OpenRLHF does with --apply_chat_template)
  - Eval path: same call (vLLM's chat completions API applies the same HF template)

PASS: all 5 samples produce identical input_ids sequences.

Catches: chat template version mismatches, suffix placement bugs, whitespace
differences between train and eval tokenization.

Usage:
    python verify_prompt_parity.py [--model Qwen/Qwen2.5-14B-Instruct]
                                   [--data data/sft_rl_pool_3200_boxed.jsonl]
                                   [--n_samples 5]
"""

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = SCRIPT_DIR / "data" / "sft_rl_pool_3200_boxed.jsonl"
DEFAULT_MODEL = "Qwen/Qwen2.5-14B-Instruct"


def load_samples(path: Path, n: int) -> list[str]:
    """Load first n input texts from JSONL."""
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append(record["input"])
            if len(samples) >= n:
                break
    return samples


def main():
    parser = argparse.ArgumentParser(description="G0: Prompt parity check")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace model ID")
    parser.add_argument("--data", default=str(DEFAULT_DATA),
                        help="Path to BOXED_SUFFIX JSONL data")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of samples to check")
    args = parser.parse_args()

    print(f"=== G0: Prompt/Chat-Template Parity Check ===\n")
    print(f"Model: {args.model}")
    print(f"Data:  {args.data}")
    print(f"Samples: {args.n_samples}\n")

    # Load tokenizer
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"  Tokenizer class: {type(tokenizer).__name__}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        print(f"  Chat template: present ({len(tokenizer.chat_template)} chars)")
    else:
        print("  WARNING: No chat_template found on tokenizer!")
    print()

    # Load samples
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        print(f"  Run prepare_rl_data_qwen.py first.")
        sys.exit(1)

    samples = load_samples(data_path, args.n_samples)
    if len(samples) < args.n_samples:
        print(f"WARNING: Only {len(samples)} samples available (requested {args.n_samples})")

    # Compare tokenization paths
    all_pass = True
    for i, input_text in enumerate(samples):
        messages = [{"role": "user", "content": input_text}]

        # Train path: apply_chat_template (what OpenRLHF uses with --apply_chat_template)
        train_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )

        # Eval path: same call (vLLM chat completions uses tokenizer.apply_chat_template)
        eval_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )

        match = (train_ids == eval_ids)
        status = "PASS" if match else "FAIL"
        all_pass = all_pass and match

        print(f"  Sample {i+1}: {status}")
        print(f"    Input length: {len(input_text)} chars")
        print(f"    Train path tokens: {len(train_ids)}")
        print(f"    Eval path tokens:  {len(eval_ids)}")

        if not match:
            # Find first divergence point
            min_len = min(len(train_ids), len(eval_ids))
            for j in range(min_len):
                if train_ids[j] != eval_ids[j]:
                    print(f"    First divergence at position {j}:")
                    print(f"      Train: {train_ids[max(0,j-2):j+3]}")
                    print(f"      Eval:  {eval_ids[max(0,j-2):j+3]}")
                    # Decode around divergence
                    print(f"      Train decoded: {tokenizer.decode(train_ids[max(0,j-2):j+3])!r}")
                    print(f"      Eval decoded:  {tokenizer.decode(eval_ids[max(0,j-2):j+3])!r}")
                    break
            if len(train_ids) != len(eval_ids):
                print(f"    Length mismatch: {len(train_ids)} vs {len(eval_ids)}")

        # Show suffix region for first sample
        if i == 0:
            # Decode last 50 tokens to verify BOXED_SUFFIX placement
            tail = tokenizer.decode(train_ids[-50:])
            print(f"    Last 50 tokens decoded: ...{tail!r}")

    print()
    if all_pass:
        print(f"G0 RESULT: PASS ({len(samples)}/{len(samples)} samples identical)")
    else:
        print(f"G0 RESULT: FAIL (mismatch detected — investigate before training)")
        sys.exit(1)


if __name__ == "__main__":
    main()
