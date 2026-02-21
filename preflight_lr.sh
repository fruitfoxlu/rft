#!/bin/bash
# preflight_lr.sh — Pre-flight LR schedule check for OpenRLHF training runs.
#
# Models the actual scheduler: linear warmup + cosine_with_min_lr (min_lr = 0.1 * target_lr).
# Fail-fast heuristic: abort if LR@step20 < 30% of target (near-zero LR run).
#
# Usage:
#   POOL=400 NS=8 TBS=16 EP=1 WARMUP=0.05 LR=5e-7 bash preflight_lr.sh
set -euo pipefail

export POOL="${POOL:-50000}"        # prompts in pool
export NS="${NS:-8}"                # n_samples_per_prompt
export TBS="${TBS:-16}"             # train_batch_size
export EP="${EP:-1}"                # num_episodes
export WARMUP="${WARMUP:-0.05}"     # lr_warmup_ratio
export LR="${LR:-5e-7}"             # target actor LR
export MINLR="${MINLR:-0.1}"        # min_lr ratio (OpenRLHF cosine_with_min_lr uses 0.1 * target_lr)

python3 - <<'PY'
import os, math

POOL=int(os.environ["POOL"])
NS=int(os.environ["NS"])
TBS=int(os.environ["TBS"])
EP=int(os.environ["EP"])
WARMUP=float(os.environ["WARMUP"])
LR=float(os.environ["LR"])
MINLR=float(os.environ["MINLR"])

max_steps = math.ceil((POOL * NS) / TBS) * EP
warmup_steps = max(1, math.ceil(max_steps * WARMUP))
min_lr = LR * MINLR
decay_steps = max(1, max_steps - warmup_steps)

def lr_at(step: int) -> float:
    # step is 1-indexed
    if step <= warmup_steps:
        return LR * (step / warmup_steps)
    # cosine decay from LR to min_lr
    t = (step - warmup_steps) / decay_steps  # in (0, 1]
    t = max(0.0, min(1.0, t))
    return min_lr + 0.5 * (LR - min_lr) * (1.0 + math.cos(math.pi * t))

print("=== PRE-FLIGHT LR SCHEDULE CHECK (warmup + cosine_with_min_lr) ===")
print(f"pool_size={POOL}, n_samples={NS}, train_batch_size={TBS}, num_episodes={EP}")
print(f"max_steps={max_steps}")
print(f"warmup_ratio={WARMUP:.4f} => warmup_steps={warmup_steps}")
print(f"target_lr={LR:.3e}, min_lr_ratio={MINLR:.3f} => min_lr={min_lr:.3e}")

# Key steps to inspect
check_steps = [1, 5, 10, 20, 50, 100, 150, 200, max_steps]
seen = set()
print("\nLR snapshot:")
for s in check_steps:
    if 1 <= s <= max_steps and s not in seen:
        seen.add(s)
        lr = lr_at(s)
        print(f"  LR@step{s:>5} = {lr:.3e} ({lr/LR*100:.1f}% of target)")

# Fail-fast heuristic for short-horizon experiments:
# by step 20 we want LR to be meaningfully high (>=30% target)
if max_steps >= 20:
    lr20 = lr_at(20)
    if lr20 < 0.3 * LR:
        print(f"\nWARNING: LR@step20 = {lr20:.3e} ({lr20/LR*100:.1f}% of target) < 30% threshold.")
        print("This run may be too 'near-zero LR' to be informative.")
        print("Fix: reduce max_steps (smaller pool/fewer episodes) or reduce warmup_ratio.")
        exit(1)
    else:
        print(f"\nOK: LR reaches >=30% of target by step 20 ({lr20/LR*100:.1f}%).")
else:
    print(f"\nNOTE: max_steps={max_steps} < 20; adjust checks accordingly.")
PY
