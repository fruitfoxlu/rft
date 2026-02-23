#!/bin/bash
# preflight_lr.sh — Pre-flight LR schedule check for OpenRLHF training runs.
#
# Models the actual scheduler: linear warmup + cosine_with_min_lr (min_lr = 0.1 * target_lr).
# Fail-fast heuristic: abort if LR@step20 < 30% of target (near-zero LR run).
#
# IMPORTANT: max_steps counts GRADIENT steps, not global steps.
# Each global step = n_samples * rollout_batch_size / train_batch_size gradient steps.
# This script now displays both to prevent miscalculation.
#
# Usage:
#   POOL=400 NS=8 TBS=16 EP=1 WARMUP=0.05 LR=5e-7 RBS=16 bash preflight_lr.sh
set -euo pipefail

export POOL="${POOL:-50000}"        # prompts in pool
export NS="${NS:-8}"                # n_samples_per_prompt
export TBS="${TBS:-16}"             # train_batch_size
export EP="${EP:-1}"                # num_episodes
export WARMUP="${WARMUP:-0.05}"     # lr_warmup_ratio
export LR="${LR:-5e-7}"             # target actor LR
export MINLR="${MINLR:-0.1}"        # min_lr ratio (OpenRLHF cosine_with_min_lr uses 0.1 * target_lr)
export RBS="${RBS:-16}"              # rollout_batch_size (for global step calculation)

python3 - <<'PY'
import os, math

POOL=int(os.environ["POOL"])
NS=int(os.environ["NS"])
TBS=int(os.environ["TBS"])
EP=int(os.environ["EP"])
WARMUP=float(os.environ["WARMUP"])
LR=float(os.environ["LR"])
MINLR=float(os.environ["MINLR"])
RBS=int(os.environ["RBS"])

max_steps = math.ceil((POOL * NS) / TBS) * EP  # gradient steps
grad_per_global = (NS * RBS) // TBS             # gradient steps per global step
global_steps = POOL // RBS * EP                  # global steps (rollout batches)
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
print(f"pool_size={POOL}, n_samples={NS}, train_batch_size={TBS}, rollout_batch_size={RBS}, num_episodes={EP}")
print(f"max_steps={max_steps} (GRADIENT steps, i.e., optimizer.step() calls)")
print(f"grad_steps_per_global_step={grad_per_global} (n_samples * rollout_batch / train_batch)")
print(f"global_steps={global_steps} (rollout batches = what you see in the training log)")
print(f"warmup_ratio={WARMUP:.4f} => warmup_steps={warmup_steps} gradient steps (~{warmup_steps/grad_per_global:.1f} global steps)")
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

# Fail-fast heuristics:
# 1. Warmup should not consume more than 20% of total gradient steps
# 2. LR should reach >=90% of target shortly after warmup ends
# 3. For very short runs (<20 global steps), warn
ok = True

warmup_frac = warmup_steps / max_steps
if warmup_frac > 0.20:
    print(f"\nWARNING: warmup consumes {warmup_frac*100:.1f}% of total gradient steps (>{20}%).")
    print("The model spends too long at low LR. Reduce warmup_ratio or increase pool/episodes.")
    ok = False

post_warmup_step = min(warmup_steps + 2, max_steps)
lr_post = lr_at(post_warmup_step)
if lr_post < 0.85 * LR:
    print(f"\nWARNING: LR@step{post_warmup_step} = {lr_post:.3e} ({lr_post/LR*100:.1f}%) — expected >=85% post-warmup.")
    ok = False

if global_steps < 20:
    print(f"\nWARNING: only {global_steps} global steps — may be too short for meaningful training/eval.")
    ok = False

if not ok:
    exit(1)
else:
    print(f"\nOK: warmup={warmup_frac*100:.1f}% of run, LR@post-warmup={lr_post/LR*100:.1f}%, {global_steps} global steps.")
PY
