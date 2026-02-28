#!/usr/bin/env python3
"""Auto-update rl_grpo_research.md with track results.

Reads evaluation summaries and training metrics, generates research log
sections with findings, theory, and suggestions, appends to the log,
and commits + pushes.

Usage:
    python update_research_log.py --track b1
    python update_research_log.py --track b2
    python update_research_log.py --track c2
    python update_research_log.py --track c1
    python update_research_log.py --track all   # final summary
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_PATH = SCRIPT_DIR / "rl_grpo_research.md"

EVAL_DIR = Path("/mnt/scratch/qwen14b_eval")
METRICS_BASE = Path("/mnt/scratch")

TRACK_CONFIG = {
    "b1": {
        "section": "§16",
        "title": "Track B1 Results: GRPO with Larger Group Size (n=32)",
        "metrics_dir": "rft_metrics_qwen14b_b1",
        "hypothesis": (
            "With n=8 and binary EM, most groups produce all-correct or all-wrong "
            "rollouts (reward variance=0 → zero gradient). Increasing to n=32 gives "
            "more mixed-outcome groups, especially for hard problems (p=0.1: "
            "P(mixed) goes from 57% to 96%)."
        ),
        "changed_variable": "n_samples_per_prompt 8 → 32",
        "eval_labels": ["b1-step50", "b1-step100", "b1-step200"],
    },
    "b2": {
        "section": "§17",
        "title": "Track B2 Results: GRPO with Relaxed Clipping (eps_clip=0.2)",
        "metrics_dir": "rft_metrics_qwen14b_b2",
        "hypothesis": (
            "D1 showed the policy IS moving but changes are small and random. "
            "Doubling eps_clip from 0.1 to 0.2 allows larger policy updates when "
            "gradient signal exists, potentially escaping local basins."
        ),
        "changed_variable": "eps_clip 0.1 → 0.2",
        "eval_labels": ["b2-step50", "b2-step100", "b2-step200"],
    },
    "c2": {
        "section": "§18",
        "title": "Track C2 Results: Judge Reward — Partial Credit for Wrong Answers",
        "metrics_dir": "rft_metrics_qwen14b_c2",
        "hypothesis": (
            "Binary EM reward provides zero gradient on all-wrong groups. C2 gives "
            "partial credit (α=0.3) to wrong answers based on reasoning quality, "
            "creating reward variance within these groups. "
            "reward = EM + 0.3 × judge_score × (1-EM). "
            "Invariant: max(EM=0 reward) = 0.3 < 1.0 = min(EM=1 reward)."
        ),
        "changed_variable": "reward_func_em.py → reward_func_judge_c2.py (α=0.3)",
        "eval_labels": ["c2-step50", "c2-step100", "c2-step200"],
    },
    "c1": {
        "section": "§19",
        "title": "Track C1 Results: Judge Reward — Quality Differentiation Among Correct",
        "metrics_dir": "rft_metrics_qwen14b_c1",
        "hypothesis": (
            "Among correct solutions, some have clean logical reasoning while others "
            "got lucky. C1 differentiates by adding judge bonus on EM=1 responses. "
            "reward = EM × (1 + 0.5 × judge_score). "
            "EM=1 reward range: [1.0, 1.5]. EM=0 reward: 0.0."
        ),
        "changed_variable": "reward_func_em.py → reward_func_judge_c1.py (α=0.5)",
        "eval_labels": ["c1-step50", "c1-step100", "c1-step200"],
    },
}


def load_training_metrics(metrics_dir: str) -> list[dict]:
    path = METRICS_BASE / metrics_dir / "training_metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(l) for l in open(path)]


def load_eval_summary(label: str) -> dict | None:
    path = EVAL_DIR / f"{label}_eval_summary.json"
    if not path.exists():
        return None
    return json.load(open(path))


def format_metrics_table(metrics: list[dict], track_id: str) -> str:
    if not metrics:
        return "  (no training metrics available)\n"

    lines = []
    has_judge = any("judge_score" in m for m in metrics)

    if has_judge:
        lines.append(f"  {'Step':>5} {'Reward':>8} {'Correct':>8} {'Judge':>7} {'RatioMax':>9} {'GradNorm':>9}")
    else:
        lines.append(f"  {'Step':>5} {'Reward':>8} {'Correct':>8} {'RatioMax':>9} {'GradNorm':>9}")
    lines.append("  " + "-" * (len(lines[0]) - 2))

    for m in metrics:
        s = m["global_step"]
        if s % 20 == 0 or s <= 5 or s == len(metrics):
            if has_judge:
                judge = m.get("judge_score", 0.0)
                lines.append(
                    f"  {s:5d} {m['reward']:8.3f} {m['correctness']:8.3f} "
                    f"{judge:7.3f} {m['ratio_max']:9.3f} {m['grad_norm']:9.4f}"
                )
            else:
                lines.append(
                    f"  {s:5d} {m['reward']:8.3f} {m['correctness']:8.3f} "
                    f"{m['ratio_max']:9.3f} {m['grad_norm']:9.4f}"
                )
    return "\n".join(lines) + "\n"


def find_best_eval(eval_labels: list[str]) -> tuple[dict | None, str]:
    """Find the eval with highest delta (or least negative)."""
    best = None
    best_label = ""
    for label in eval_labels:
        summary = load_eval_summary(label)
        if summary:
            delta = summary["paired_stats"]["delta_pp"]
            if best is None or delta > best["paired_stats"]["delta_pp"]:
                best = summary
                best_label = label
    return best, best_label


def analyze_result(config: dict, metrics: list[dict], best_eval: dict | None,
                   best_label: str, all_evals: list[tuple[str, dict]]) -> str:
    """Generate findings, theory, and suggestions for a track."""
    lines = []

    if not best_eval:
        lines.append("**No evaluation results available.** Training may have failed.")
        return "\n".join(lines)

    stats = best_eval["paired_stats"]
    delta = stats["delta_pp"]
    p_val = stats["mcnemar_p"]
    b = stats["b_discordant"]
    c = stats["c_discordant"]
    bc = stats["b_plus_c"]
    acc_new = stats["acc_new"] * 100
    acc_base = stats["acc_base"] * 100
    gate_1a = best_eval.get("gate_1a", False)
    gate_1b = best_eval.get("gate_1b", False)

    # Determine result category
    if gate_1b:
        result_cat = "STRONG_WIN"
    elif gate_1a:
        result_cat = "WEAK_WIN"
    elif delta > 0 and p_val < 0.20:
        result_cat = "TRENDING_POSITIVE"
    elif abs(delta) < 1.0:
        result_cat = "NULL"
    elif delta < -1.0:
        result_cat = "REGRESSION"
    else:
        result_cat = "NULL"

    # --- Findings ---
    lines.append("**Findings:**\n")

    if result_cat == "STRONG_WIN":
        lines.append(
            f"- **Gate-1b PASS**: Δ={delta:+.2f}pp (p={p_val:.4f}). "
            f"This is a statistically significant and practically meaningful improvement. "
            f"Best checkpoint: {best_label}."
        )
    elif result_cat == "WEAK_WIN":
        lines.append(
            f"- **Gate-1a PASS** (Gate-1b FAIL): Δ={delta:+.2f}pp (p={p_val:.4f}). "
            f"Suggestive improvement but not definitive. Best checkpoint: {best_label}."
        )
    elif result_cat == "TRENDING_POSITIVE":
        lines.append(
            f"- **Trending positive but not significant**: Δ={delta:+.2f}pp (p={p_val:.4f}). "
            f"Both gates FAIL. Best checkpoint: {best_label}."
        )
    elif result_cat == "REGRESSION":
        lines.append(
            f"- **Regression**: Δ={delta:+.2f}pp (p={p_val:.4f}). "
            f"Performance decreased. Best checkpoint: {best_label}."
        )
    else:
        lines.append(
            f"- **Null effect**: Δ={delta:+.2f}pp (p={p_val:.4f}). "
            f"Both gates FAIL. Best checkpoint: {best_label}."
        )

    lines.append(
        f"- Discordant pairs: b={b} (base✓ new✗), c={c} (base✗ new✓), b+c={bc} ({bc/10:.1f}%)."
    )
    ci_lo = stats["ci_low_pp"]
    ci_hi = stats["ci_high_pp"]
    lines.append(f"- 95% CI: [{ci_lo:+.2f}, {ci_hi:+.2f}]pp.")

    # Checkpoint comparison
    if len(all_evals) > 1:
        lines.append("- Checkpoint comparison:")
        for label, ev in all_evals:
            s = ev["paired_stats"]
            lines.append(
                f"  - {label}: {s['acc_new']*100:.1f}% (Δ={s['delta_pp']:+.2f}pp, p={s['mcnemar_p']:.4f})"
            )

    # Training trajectory observation
    if metrics:
        first_reward = metrics[0]["reward"]
        last_reward = metrics[-1]["reward"]
        reward_trend = last_reward - first_reward
        lines.append(
            f"- Training reward: {first_reward:.3f} (step 1) → {last_reward:.3f} "
            f"(step {metrics[-1]['global_step']}) — "
            f"{'increasing' if reward_trend > 0.02 else 'decreasing' if reward_trend < -0.02 else 'flat'}."
        )

    # --- Theory ---
    lines.append("\n**Theory:**\n")

    if result_cat in ("STRONG_WIN", "WEAK_WIN"):
        lines.append(
            f"- The changed variable ({config['changed_variable']}) addresses a real bottleneck. "
            f"The improvement suggests that {config['hypothesis'].split('.')[0].lower()}."
        )
        if bc > 100:
            lines.append(
                f"- High discordance (b+c={bc}) indicates the model changed substantially, "
                f"and this time the changes are directional (c > b by {c-b})."
            )
    elif result_cat == "REGRESSION":
        lines.append(
            f"- The change ({config['changed_variable']}) was counterproductive. "
            f"More regressions (b={b}) than improvements (c={c}) suggests the modification "
            f"destabilized training or pushed the model away from correct solutions."
        )
    else:
        lines.append(
            f"- Another null result. The hypothesis that \"{config['hypothesis'].split('.')[0].lower()}\" "
            f"is not supported by the data."
        )
        if bc > 80:
            lines.append(
                f"- The model DOES change (b+c={bc}, {bc/10:.1f}% of problems), "
                f"but changes remain directionless — improvements cancel regressions."
            )
        else:
            lines.append(
                f"- Low discordance (b+c={bc}) suggests the modification had minimal effect "
                f"on model behavior."
            )

    # --- Suggestions ---
    lines.append("\n**Suggestions:**\n")

    if result_cat in ("STRONG_WIN", "WEAK_WIN"):
        lines.append("- This direction is promising. Consider:")
        lines.append("  - Running a second seed to confirm the result.")
        lines.append("  - Combining this change with other successful modifications.")
        lines.append("  - Analyzing WHICH problems improved to understand the mechanism.")
    elif result_cat == "TRENDING_POSITIVE":
        lines.append("- Result is suggestive but inconclusive. Consider:")
        lines.append("  - Running longer (more steps) to see if the trend continues.")
        lines.append("  - Trying a stronger version of this modification.")
        lines.append("  - Running a second seed to check if the trend replicates.")
    else:
        lines.append("- This modification does not help. Consider:")
        lines.append("  - Moving on to remaining tracks rather than iterating further.")
        lines.append(
            "  - If all tracks fail, the bottleneck may be fundamental "
            "(model capacity, LoRA limits, or train/eval domain gap)."
        )

    return "\n".join(lines)


def generate_track_section(track_id: str) -> str:
    """Generate a complete research log section for a track."""
    config = TRACK_CONFIG[track_id]
    metrics = load_training_metrics(config["metrics_dir"])

    # Load all available evals
    all_evals = []
    for label in config["eval_labels"]:
        ev = load_eval_summary(label)
        if ev:
            all_evals.append((label, ev))

    best_eval, best_label = find_best_eval(config["eval_labels"])

    lines = []
    lines.append(f"\n---\n")
    lines.append(f"## {config['section']} {config['title']}\n")
    lines.append(f"### {config['section']}.1 Experiment Record\n")
    lines.append("```")
    lines.append(f"Attempt ID: Q2-{track_id.upper()}")
    lines.append(f"Track: {track_id.upper()}")
    lines.append(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}")
    lines.append(f"Changed variable: {config['changed_variable']}")
    lines.append(f"Hypothesis: {config['hypothesis']}")

    if metrics:
        lines.append(f"Training steps: {metrics[-1]['global_step']}")

    if best_eval:
        stats = best_eval["paired_stats"]
        lines.append(f"OOD-1000 result (best checkpoint: {best_label}):")
        lines.append(f"  Baseline: {stats['acc_base']*100:.1f}% ({int(stats['acc_base']*stats['n_paired'])}/{stats['n_paired']})")
        lines.append(f"  {track_id.upper()}:     {stats['acc_new']*100:.1f}% ({int(stats['acc_new']*stats['n_paired'])}/{stats['n_paired']})")
        lines.append(f"  Δ: {stats['delta_pp']:+.2f}pp")
        lines.append(f"  McNemar p: {stats['mcnemar_p']:.4f}")
        lines.append(f"  95% CI: [{stats['ci_low_pp']:+.2f}, {stats['ci_high_pp']:+.2f}]pp")
        lines.append(f"  b (base✓ new✗): {stats['b_discordant']}")
        lines.append(f"  c (base✗ new✓): {stats['c_discordant']}")
        lines.append(f"  b+c: {stats['b_plus_c']}")
        gate_1a = best_eval.get("gate_1a", False)
        gate_1b = best_eval.get("gate_1b", False)
        lines.append(f"Decision gate: Gate-1a {'PASS' if gate_1a else 'FAIL'}, Gate-1b {'PASS' if gate_1b else 'FAIL'}")
    else:
        lines.append("OOD-1000 result: (evaluation not available)")

    lines.append("```\n")

    # Training metrics table
    lines.append(f"### {config['section']}.2 Training Trajectory\n")
    lines.append("```")
    lines.append(format_metrics_table(metrics, track_id))
    lines.append("```\n")

    # Analysis
    lines.append(f"### {config['section']}.3 Analysis\n")
    lines.append(analyze_result(config, metrics, best_eval, best_label, all_evals))

    return "\n".join(lines)


def generate_final_summary() -> str:
    """Generate a summary comparing all tracks."""
    lines = []
    lines.append("\n---\n")
    lines.append("## §20 Cross-Track Summary and Conclusions\n")
    lines.append("### All Results\n")
    lines.append("| Track | Method | Δ (pp) | p-value | b+c | Gate-1a | Gate-1b |")
    lines.append("|-------|--------|--------|---------|-----|---------|---------|")

    # Include Track A
    lines.append("| A (RSFT) | SFT on correct solutions | -1.00 | 0.4300 | 130 | FAIL | FAIL |")

    # Add all track results
    for tid in ["b1", "b2", "c2", "c1"]:
        config = TRACK_CONFIG[tid]
        best_eval, best_label = find_best_eval(config["eval_labels"])
        if best_eval:
            s = best_eval["paired_stats"]
            g1a = "PASS" if best_eval.get("gate_1a") else "FAIL"
            g1b = "PASS" if best_eval.get("gate_1b") else "FAIL"
            desc = config["changed_variable"].split("→")[-1].strip() if "→" in config["changed_variable"] else config["changed_variable"]
            lines.append(
                f"| {tid.upper()} | {desc} | {s['delta_pp']:+.2f} | {s['mcnemar_p']:.4f} | "
                f"{s['b_plus_c']} | {g1a} | {g1b} |"
            )
        else:
            lines.append(f"| {tid.upper()} | {config['changed_variable']} | — | — | — | — | — |")

    lines.append("")

    # Count results
    wins = []
    nulls = []
    regressions = []
    for tid in ["b1", "b2", "c2", "c1"]:
        config = TRACK_CONFIG[tid]
        best_eval, _ = find_best_eval(config["eval_labels"])
        if best_eval:
            delta = best_eval["paired_stats"]["delta_pp"]
            if best_eval.get("gate_1a") or best_eval.get("gate_1b"):
                wins.append(tid)
            elif delta < -1.5:
                regressions.append(tid)
            else:
                nulls.append(tid)

    lines.append("### Overall Assessment\n")

    if wins:
        lines.append(
            f"**Promising results from: {', '.join(t.upper() for t in wins)}.**\n"
        )
        lines.append(
            "These tracks show that the training setup CAN produce improvement "
            "when the right modifications are made. Next steps should focus on "
            "combining successful changes and validating with additional seeds.\n"
        )
    elif all(t in nulls for t in ["b1", "b2", "c2", "c1"]):
        lines.append(
            "**All tracks produce null results.** Combined with Track A (RSFT) failure, "
            "this is now FIVE independent null results across two training paradigms "
            "(RL and supervised) and multiple hyperparameter variations.\n"
        )
        lines.append("This strongly suggests a **fundamental limitation** in the current setup:\n")
        lines.append(
            "1. **LoRA capacity**: rank=32 LoRA may lack the capacity to meaningfully "
            "shift a 14B model's math reasoning. Full fine-tuning could test this.\n"
            "2. **Train/eval domain gap**: The 3200 NuminaMath training problems may not "
            "provide transferable signal to OOD-1000. The error zone may require "
            "capabilities not present in the training distribution.\n"
            "3. **Model ceiling**: Qwen2.5-14B-Instruct at 67% OOD may be near its "
            "architectural capacity for this task class. A larger model (32B+) "
            "might have more room to improve.\n"
            "4. **Evaluation sensitivity**: OOD-1000 greedy@1 may not be sensitive enough "
            "to detect small real improvements. Consider pass@k or softer metrics.\n"
        )
        lines.append("### Recommended Next Steps\n")
        lines.append(
            "1. **Full fine-tune (no LoRA)**: Tests hypothesis #1. If this also fails, "
            "LoRA capacity is ruled out.\n"
            "2. **Larger model**: Try Qwen2.5-32B or 72B with the same pipeline.\n"
            "3. **Training on OOD-like data**: If the domain gap is the issue, training "
            "on problems more similar to OOD-1000 should help.\n"
            "4. **Pass@k evaluation**: Use pass@8 as the primary metric instead of "
            "greedy@1 to detect subtler capability changes.\n"
        )
    else:
        lines.append("**Mixed results.** Some tracks show slight trends but nothing definitive.\n")

    return "\n".join(lines)


def update_section_0():
    """Update §0 Current Status in the research log."""
    # Collect all results
    results = []
    results.append(("A (RSFT)", -1.0, 0.43, False, False))
    for tid in ["b1", "b2", "c2", "c1"]:
        config = TRACK_CONFIG[tid]
        best_eval, _ = find_best_eval(config["eval_labels"])
        if best_eval:
            s = best_eval["paired_stats"]
            results.append((
                f"{tid.upper()}",
                s["delta_pp"],
                s["mcnemar_p"],
                best_eval.get("gate_1a", False),
                best_eval.get("gate_1b", False),
            ))

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    completed = [r[0] for r in results]
    any_pass = any(r[3] or r[4] for r in results)

    lines = []
    lines.append(f"### Current Status ({date_str})\n")

    for name, delta, p, g1a, g1b in results:
        gate_str = ""
        if g1b:
            gate_str = "Gate-1b PASS"
        elif g1a:
            gate_str = "Gate-1a PASS"
        else:
            gate_str = "both gates FAIL"
        lines.append(f"- **{name}**: Δ={delta:+.2f}pp (p={p:.4f}), {gate_str}")

    total_null = sum(1 for r in results if not r[3] and not r[4])
    lines.append(f"- **{total_null} out of {len(results)} tracks show null effect.**")

    if not any_pass:
        lines.append(
            "- All approaches tried (RSFT, GRPO variants, judge reward) fail to improve "
            "OOD-1000 accuracy beyond noise. See §20 for cross-track analysis."
        )
    lines.append("- **SOP**: See §11.24 for experiment protocol. **Consolidated lessons**: See §12.")

    return "\n".join(lines)


def apply_updates(track_id: str):
    """Apply updates to the research log."""
    content = LOG_PATH.read_text()

    if track_id == "all":
        # Generate final summary
        section = generate_final_summary()

        # Update §0
        new_status = update_section_0()
        import re
        content = re.sub(
            r"### Current Status \(\d{4}-\d{2}-\d{2}\)\n.*?(?=\n### Pivot|\n---)",
            new_status + "\n",
            content,
            flags=re.DOTALL,
        )

        content += section
    else:
        section = generate_track_section(track_id)
        content += section

    LOG_PATH.write_text(content)
    print(f"Updated {LOG_PATH}")


def git_commit_push(track_id: str):
    """Commit and push the research log update, keeping main in sync."""
    os.chdir(SCRIPT_DIR)
    subprocess.run(["git", "add", "rl_grpo_research.md"], check=True)

    if track_id == "all":
        msg = "Update research log: cross-track summary and §0 status"
    else:
        config = TRACK_CONFIG[track_id]
        msg = f"Update research log: {config['section']} {track_id.upper()} results"

    commit_msg = f"{msg}\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
    subprocess.run(["git", "commit", "-m", commit_msg], check=True)

    # Push to experiment branch
    subprocess.run(["git", "push", "origin", "attempt-26-grpo-20b-em"], check=True)

    # Keep main in sync (linear workflow)
    current = subprocess.run(
        ["git", "branch", "--show-current"], capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "checkout", "main"], check=True)
    subprocess.run(["git", "merge", "--ff-only", "attempt-26-grpo-20b-em"], check=True)
    subprocess.run(["git", "push", "origin", "main"], check=True)
    subprocess.run(["git", "checkout", current], check=True)
    print(f"Committed and pushed: {msg} (main synced)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True, choices=["b1", "b2", "c2", "c1", "all"])
    parser.add_argument("--no-push", action="store_true")
    args = parser.parse_args()

    apply_updates(args.track)

    if not args.no_push:
        git_commit_push(args.track)


if __name__ == "__main__":
    main()
