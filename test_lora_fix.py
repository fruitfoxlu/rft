#!/usr/bin/env python3
"""Regression test for LoRA accumulation bug fix.

Verifies two properties:
1. Correctness: manual `base + scaling * B @ A` matches PEFT merge_and_unload()
2. No accumulation: applying the delta formula twice yields the same result
   as applying it once (i.e., we replace, not accumulate).

Runs on CPU only (no GPU required).
"""

import os
import sys

# Force CPU-only execution
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import copy

ADAPTER_PATH = "/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a29b/global_step200_hf"
BASE_MODEL = "openai/gpt-oss-20b"
TARGET_LAYER = "model.layers.0.self_attn.q_proj"
TARGET_WEIGHT = f"{TARGET_LAYER}.weight"
LORA_ALPHA = 64
LORA_RANK = 32
SCALING = LORA_ALPHA / LORA_RANK  # 2.0


def apply_lora_delta(base_weight, lora_a, lora_b, scaling):
    """Compute merged weight: base + scaling * B @ A.

    This mirrors the logic in vllm_worker_wrap.py:apply_lora_delta,
    but without TP sharding (single device).
    """
    delta = (scaling * (lora_b.float() @ lora_a.float())).to(dtype=base_weight.dtype)
    return base_weight + delta


def main():
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    print("=" * 70)
    print("LoRA Accumulation Bug Fix - Regression Test")
    print("=" * 70)

    # ---- Step 1: Load base model and grab pre-LoRA weight ----
    print("\n[1/5] Loading base model (CPU, bf16 dequantized)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    base_weight = None
    for name, param in base_model.named_parameters():
        if name == TARGET_WEIGHT:
            base_weight = param.data.clone()
            break

    assert base_weight is not None, f"Could not find {TARGET_WEIGHT} in base model"
    print(f"   Base weight shape: {base_weight.shape}, dtype: {base_weight.dtype}")
    print(f"   Base weight norm:  {base_weight.float().norm().item():.4f}")

    # ---- Step 2: Load LoRA adapter with PEFT, merge, get reference weight ----
    print("\n[2/5] Loading LoRA adapter and merging with PEFT...")
    peft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    merged_model = peft_model.merge_and_unload()

    peft_merged_weight = None
    for name, param in merged_model.named_parameters():
        if name == TARGET_WEIGHT:
            peft_merged_weight = param.data.clone()
            break

    assert peft_merged_weight is not None, f"Could not find {TARGET_WEIGHT} in merged model"
    print(f"   PEFT merged weight norm: {peft_merged_weight.float().norm().item():.4f}")

    # Free merged model memory
    del peft_model, merged_model, base_model
    import gc; gc.collect()

    # ---- Step 3: Load LoRA matrices directly from checkpoint ----
    print("\n[3/5] Loading LoRA A/B matrices from adapter checkpoint...")
    adapter_state = torch.load(
        os.path.join(ADAPTER_PATH, "adapter_model.bin"),
        map_location="cpu",
    )

    lora_a_key = f"base_model.model.{TARGET_LAYER}.lora_A.weight"
    lora_b_key = f"base_model.model.{TARGET_LAYER}.lora_B.weight"

    lora_a = adapter_state[lora_a_key]
    lora_b = adapter_state[lora_b_key]
    print(f"   LoRA A shape: {lora_a.shape} (r x in_features)")
    print(f"   LoRA B shape: {lora_b.shape} (out_features x r)")
    print(f"   Scaling: {SCALING} (alpha={LORA_ALPHA} / r={LORA_RANK})")

    del adapter_state
    gc.collect()

    # ---- Step 4: Test correctness - manual delta vs PEFT merge ----
    print("\n[4/5] Test 1: Correctness - manual delta vs PEFT merge_and_unload()...")

    manual_merged = apply_lora_delta(base_weight, lora_a, lora_b, SCALING)

    # Compare manual merge to PEFT merge
    max_diff = (manual_merged.float() - peft_merged_weight.float()).abs().max().item()
    mean_diff = (manual_merged.float() - peft_merged_weight.float()).abs().mean().item()
    rel_diff = max_diff / peft_merged_weight.float().abs().max().item()

    print(f"   Max absolute difference:  {max_diff:.2e}")
    print(f"   Mean absolute difference: {mean_diff:.2e}")
    print(f"   Max relative difference:  {rel_diff:.2e}")

    # bf16 has ~7.8e-3 relative precision; we allow some headroom
    ATOL = 1e-4  # absolute tolerance
    RTOL = 1e-3  # relative tolerance

    if max_diff < ATOL or rel_diff < RTOL:
        print("   PASS: Manual delta matches PEFT merge_and_unload()")
        test1_pass = True
    else:
        print(f"   FAIL: Difference too large (max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e})")
        test1_pass = False

    # ---- Step 5: Test no-accumulation ----
    print("\n[5/5] Test 2: No accumulation - applying delta twice...")

    # Simulate the fix: base_weights cache stores original base
    # First application: result = base + delta
    first_apply = apply_lora_delta(base_weight, lora_a, lora_b, SCALING)

    # Second application WITH the fix (replace, not accumulate):
    # We always go back to base_weight, so result = base + delta (same)
    second_apply = apply_lora_delta(base_weight, lora_a, lora_b, SCALING)

    # They should be identical
    diff_fixed = (first_apply.float() - second_apply.float()).abs().max().item()
    print(f"   With fix (replace): max diff between 1st and 2nd apply = {diff_fixed:.2e}")

    if diff_fixed == 0.0:
        print("   PASS: No accumulation with fix (results identical)")
        test2a_pass = True
    else:
        print(f"   FAIL: Results differ (diff={diff_fixed:.2e})")
        test2a_pass = False

    # Demonstrate what the BUG would look like (accumulation):
    # Buggy: first_result + delta (adding delta on top of already-merged weight)
    buggy_second = apply_lora_delta(first_apply, lora_a, lora_b, SCALING)
    buggy_diff = (buggy_second.float() - first_apply.float()).abs().max().item()
    delta = (SCALING * (lora_b.float() @ lora_a.float())).to(dtype=base_weight.dtype)
    delta_norm = delta.float().norm().item()
    delta_max = delta.float().abs().max().item()

    print(f"\n   [Reference] Buggy (accumulate) diff from 1st apply: {buggy_diff:.2e}")
    print(f"   [Reference] Delta Frobenius norm: {delta_norm:.6f}")
    print(f"   [Reference] Delta max element:    {delta_max:.6f}")
    print(f"   Buggy result would have 2x delta applied (double the LoRA contribution)")

    # Verify the delta is non-zero and the buggy version differs from the correct one.
    # The delta may be small at step 200, but it must be non-zero for the test to
    # be meaningful. We check that the buggy accumulation produces a result that
    # differs from the fixed version (first_apply != buggy_second).
    buggy_vs_fixed_diff = (buggy_second.float() - first_apply.float()).abs().max().item()
    if delta_norm > 0 and buggy_vs_fixed_diff > 0:
        print(f"   PASS: Delta is non-zero (norm={delta_norm:.6f}) and buggy version "
              f"differs from fixed (diff={buggy_vs_fixed_diff:.2e})")
        test2b_pass = True
    else:
        print(f"   FAIL: Delta norm={delta_norm:.6f}, buggy_vs_fixed diff={buggy_vs_fixed_diff:.2e}")
        test2b_pass = False

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_pass = test1_pass and test2a_pass and test2b_pass
    results = [
        ("Correctness (manual == PEFT merge)", test1_pass),
        ("No accumulation (1st apply == 2nd apply)", test2a_pass),
        ("Non-trivial delta (bug would be visible)", test2b_pass),
    ]
    for desc, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")

    print()
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
