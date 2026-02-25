# Spare Machine (h100-8-5) Operations Guide

## Quick Reference

| Item | Primary (h100-8-4) | Spare (h100-8-5) |
|------|---------------------|-------------------|
| Internal IP | `10.202.0.2` | `10.202.0.3` |
| SSH | direct (local) | `ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3` |
| Repo | `~/Code/rft` | `~/Code/rft` |
| Conda env | `rft` | `rft` |
| Root disk | `/dev/nvme0n1p1` -> `/` (1TB) | same |
| Data disk | `/dev/nvme0n2` -> `/mnt/data` (1TB) | same |
| Scratch | `/dev/md0` -> `/mnt/scratch` (RAID-0, 5.9TB) | same |
| HF cache | `/mnt/scratch/hf` | same |
| Merged models | `/mnt/scratch/merged_models/` | same |
| Eval logs | — | `/mnt/scratch/eval_spare/logs/` |

---

## 1. SSH Access

Both machines are in the same VPC (`us-east5-a`), so use direct internal IP:

```bash
# Shorthand (add to ~/.bashrc on primary)
alias spare='ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3'
alias spare-t='ssh -t -i ~/.ssh/google_compute_engine rlu@10.202.0.3'

# Usage
spare 'hostname'
spare 'nvidia-smi'
spare-t 'tmux attach -t eval-sweep'
```

From an external machine via IAP:
```bash
gcloud compute ssh --zone us-east5-a h100-8-5 --tunnel-through-iap --project wf30-poc
```

---

## 2. Model Sync SOP — Cache-First (Approach A)

### Rationale

- Base models (gpt-oss-20b, Llama-8B, Qwen-14B/32B) download once from HF
  and cache permanently on `/mnt/scratch/hf/`. No 40GB copies between machines.
- RL checkpoints are LoRA adapters (~50-200MB). Copy only adapters, then merge
  locally on spare. Merged output cached under `/mnt/scratch/merged_models/`.

### What Gets Synced

| Artifact | Size | How Synced | Location on Spare |
|----------|------|------------|-------------------|
| HF base models | 10-60GB each | Auto-download on first use | `/mnt/scratch/hf/` |
| LoRA adapters | 50-200MB | `rsync` from primary | `/mnt/data/rft_checkpoints/` |
| Merged models | 20-40GB | Built locally via `eval_posthoc.py` | `/mnt/scratch/merged_models/` |
| Code | <1MB | `git bundle` (data/ is gitignored) | `~/Code/rft/` |
| Eval datasets | ~25MB | `rsync` (gitignored, not in repo) | `~/Code/rft/data/` |

### Sync Commands

```bash
SSH="ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3"
SCP="scp -i ~/.ssh/google_compute_engine"
RSYNC="rsync -avz -e 'ssh -i ~/.ssh/google_compute_engine'"

# --- Step 1: Sync code (git bundle for unpushed commits) ---
cd ~/Code/rft
git bundle create /tmp/rft-sync.bundle origin/attempt-26-grpo-20b-em..HEAD
$SCP /tmp/rft-sync.bundle rlu@10.202.0.3:/tmp/rft-sync.bundle
$SSH 'cd ~/Code/rft && git fetch /tmp/rft-sync.bundle HEAD:refs/heads/bundle-tmp && git merge bundle-tmp --ff-only && git branch -d bundle-tmp && rm /tmp/rft-sync.bundle'

# --- Step 2: Sync data files (gitignored) ---
rsync -avz -e "ssh -i ~/.ssh/google_compute_engine" ~/Code/rft/data/ rlu@10.202.0.3:~/Code/rft/data/

# --- Step 3: Copy LoRA adapter (only when needed for merged eval) ---
rsync -avz -e "ssh -i ~/.ssh/google_compute_engine" \
  /mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a30/global_step130_hf/ \
  rlu@10.202.0.3:/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a30/global_step130_hf/

# --- Step 4: Merge on spare (auto-merge via eval_posthoc.py) ---
$SSH 'cd ~/Code/rft && conda activate rft && python eval_posthoc.py \
  --model a30_step130=/mnt/data/rft_checkpoints/gpt-oss-20b-grpo-a30/global_step130_hf \
  --merged-dir /mnt/scratch/merged_models'
```

### Verification

```bash
SSH="ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3"

# Git SHA match
LOCAL_SHA=$(cd ~/Code/rft && git rev-parse HEAD)
SPARE_SHA=$($SSH 'cd ~/Code/rft && git rev-parse HEAD')
[ "$LOCAL_SHA" = "$SPARE_SHA" ] && echo "OK: SHA match" || echo "MISMATCH!"

# Dataset SHA256 match
diff <(cd ~/Code/rft && sha256sum data/probe_set_1000_ood.jsonl data/probe_set_200.jsonl data/aime_eval.jsonl) \
     <($SSH 'cd ~/Code/rft && sha256sum data/probe_set_1000_ood.jsonl data/probe_set_200.jsonl data/aime_eval.jsonl') \
  && echo "OK: data match" || echo "MISMATCH!"

# Package versions
$SSH 'conda run -n rft python -c "import vllm,torch,peft; print(f\"vllm={vllm.__version__} torch={torch.__version__} peft={peft.__version__}\")"'
```

---

## 3. Task Dispatch

### Standard Pattern

```bash
SSH="ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3"
$SSH 'tmux new-session -d -s <SESSION> "cd ~/Code/rft && conda activate rft && <COMMAND> 2>&1 | tee /mnt/scratch/eval_spare/logs/<JOB>.log"'
```

### Monitoring

```bash
SSH="ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3"

$SSH 'tmux list-sessions'                                              # list jobs
ssh -t -i ~/.ssh/google_compute_engine rlu@10.202.0.3 'tmux attach -t <name>'  # attach (Ctrl-B D)
$SSH 'tail -50 /mnt/scratch/eval_spare/logs/<JOB>.log'                # tail log
$SSH 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'   # GPU status
```

---

## 4. Ready-to-Use Dispatch Examples

```bash
SSH="ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3"
```

### Example 1: Model Sweep Stage 1 (OOD-1000, all 4 models)

```bash
$SSH 'tmux new-session -d -s eval-sweep "\
  cd ~/Code/rft && conda activate rft && \
  python eval_model_sweep.py --stage 1 \
  2>&1 | tee /mnt/scratch/eval_spare/logs/sweep_stage1.log"'

# Monitor:
$SSH 'tail -20 /mnt/scratch/eval_spare/logs/sweep_stage1.log'
```

### Example 2: Post-hoc Eval (multiple checkpoints)

```bash
$SSH 'tmux new-session -d -s eval-posthoc "\
  cd ~/Code/rft && conda activate rft && \
  python eval_posthoc.py \
    --model a30_step130=/mnt/scratch/merged_models/a30_step130 \
    --eval aime=data/aime_eval.jsonl \
    --eval apex=data/apex_shortlist.jsonl \
    --max-tokens 4096 \
    --output /mnt/scratch/eval_spare/posthoc_results.json \
  2>&1 | tee /mnt/scratch/eval_spare/logs/posthoc.log"'

# Monitor:
$SSH 'tail -20 /mnt/scratch/eval_spare/logs/posthoc.log'
```

### Example 3: Paired OOD-1000 Eval (Base vs RL)

```bash
# Pre-check: ensure merged model exists on spare
$SSH 'ls /mnt/scratch/merged_models/a30_step130/config.json'

$SSH 'tmux new-session -d -s eval-ood1000 "\
  cd ~/Code/rft && conda activate rft && \
  python eval_baseline_s0.py \
  2>&1 | tee /mnt/scratch/eval_spare/logs/paired_ood1000.log"'

# Monitor:
$SSH 'tail -30 /mnt/scratch/eval_spare/logs/paired_ood1000.log'
```

---

## 5. Patch Sync (after reinstalling openrlhf on spare)

```bash
# On primary: tarball patched files
SITE_PKG=$(conda run -n rft python -c 'import openrlhf; import os; print(os.path.dirname(os.path.dirname(openrlhf.__file__)))')
cd $SITE_PKG && tar czf /tmp/patched_openrlhf.tar.gz \
  openrlhf/models/actor.py openrlhf/models/loss.py \
  openrlhf/trainer/ray/ppo_actor.py openrlhf/trainer/ray/vllm_engine.py \
  openrlhf/trainer/ray/vllm_worker_wrap.py openrlhf/trainer/ppo_utils/experience_maker.py \
  openrlhf/trainer/ppo_trainer.py openrlhf/cli/train_ppo_ray.py \
  openrlhf/utils/deepspeed/deepspeed.py

# Copy and extract on spare
scp -i ~/.ssh/google_compute_engine /tmp/patched_openrlhf.tar.gz rlu@10.202.0.3:/tmp/
ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3 bash -c '
  SITE_PKG=$(conda run -n rft python -c "import openrlhf; import os; print(os.path.dirname(os.path.dirname(openrlhf.__file__)))")
  cd $SITE_PKG && tar xzf /tmp/patched_openrlhf.tar.gz && rm /tmp/patched_openrlhf.tar.gz
'
```

---

## 6. Quick-Reference Cheatsheet

```bash
SSH="ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3"

# === SYNC ===
# Code (git bundle)
cd ~/Code/rft && git bundle create /tmp/rft-sync.bundle origin/attempt-26-grpo-20b-em..HEAD
scp -i ~/.ssh/google_compute_engine /tmp/rft-sync.bundle rlu@10.202.0.3:/tmp/
$SSH 'cd ~/Code/rft && git fetch /tmp/rft-sync.bundle HEAD:refs/heads/tmp && git merge tmp --ff-only && git branch -d tmp'

# Data (gitignored)
rsync -avz -e "ssh -i ~/.ssh/google_compute_engine" ~/Code/rft/data/ rlu@10.202.0.3:~/Code/rft/data/

# LoRA adapter
rsync -avz -e "ssh -i ~/.ssh/google_compute_engine" /mnt/data/rft_checkpoints/<path>/ rlu@10.202.0.3:/mnt/data/rft_checkpoints/<path>/

# === DISPATCH ===
$SSH 'tmux new-session -d -s <name> "cd ~/Code/rft && conda activate rft && <cmd> 2>&1 | tee /mnt/scratch/eval_spare/logs/<name>.log"'

# === MONITOR ===
$SSH 'tmux list-sessions'                                        # list jobs
ssh -t -i ~/.ssh/google_compute_engine rlu@10.202.0.3 'tmux attach -t <name>'  # attach
$SSH 'tail -30 /mnt/scratch/eval_spare/logs/<name>.log'          # tail log
$SSH 'nvidia-smi'                                                # GPU status

# === VERIFY ===
$SSH 'cd ~/Code/rft && git log --oneline -1'                     # check SHA
$SSH 'conda run -n rft pip show vllm torch peft 2>/dev/null | grep -E "Name|Version"'  # packages
$SSH 'df -h /mnt/data /mnt/scratch'                              # disks
```
