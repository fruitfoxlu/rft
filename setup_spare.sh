#!/usr/bin/env bash
# =============================================================================
# setup_spare.sh — Bootstrap h100-8-5 as eval/diagnostics spare
# Run ON THE SPARE machine (h100-8-5).
#
# This script is idempotent: safe to re-run (skips already-done steps).
#
# Prerequisites: run from a machine with conda at /opt/conda/bin
#
# Usage (from primary h100-8-4):
#   SSH="ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3"
#   scp -i ~/.ssh/google_compute_engine ~/Code/rft/setup_spare.sh rlu@10.202.0.3:~/
#   $SSH 'bash ~/setup_spare.sh 2>&1 | tee ~/setup_spare.log'
# =============================================================================
set -euo pipefail
export PATH="/opt/conda/bin:$PATH"

log()  { echo "[SETUP] $*"; }
warn() { echo "[WARN]  $*"; }
err()  { echo "[ERROR] $*"; }

# ---------------------------------------------------------------------------
# 1. DISK SETUP
# ---------------------------------------------------------------------------
echo ""
echo "========== 1. Disk Setup =========="

# --- 1a. /mnt/data on nvme0n2 (1TB) ---
if mountpoint -q /mnt/data 2>/dev/null; then
    log "/mnt/data already mounted."
else
    log "Setting up /mnt/data on /dev/nvme0n2..."
    if ! sudo blkid /dev/nvme0n2 | grep -q 'TYPE='; then
        sudo mkfs.ext4 -F -L data /dev/nvme0n2
    fi
    sudo mkdir -p /mnt/data
    sudo mount /dev/nvme0n2 /mnt/data
    sudo chown "$(whoami):$(whoami)" /mnt/data
fi

# --- 1b. /mnt/scratch as RAID-0 from 16×375G NVMe drives ---
if mountpoint -q /mnt/scratch 2>/dev/null; then
    log "/mnt/scratch already mounted."
else
    log "Setting up /mnt/scratch (RAID-0 from 16×375G NVMe)..."
    # Install mdadm if needed
    which mdadm >/dev/null 2>&1 || {
        sudo sed -i "/bullseye-backports/d" /etc/apt/sources.list 2>/dev/null || true
        sudo apt-get update -qq && sudo apt-get install -y -qq mdadm
    }
    RAID_DEVS=()
    for dev in /dev/nvme{1..16}n1; do
        [ -b "$dev" ] && RAID_DEVS+=("$dev")
    done
    log "Found ${#RAID_DEVS[@]} drives for RAID-0"
    if [ ! -b /dev/md0 ]; then
        sudo mdadm --create /dev/md0 --level=0 --raid-devices=${#RAID_DEVS[@]} "${RAID_DEVS[@]}" --force --run
    fi
    if ! sudo blkid /dev/md0 | grep -q 'TYPE='; then
        sudo mkfs.ext4 -F -L scratch /dev/md0
    fi
    sudo mkdir -p /mnt/scratch
    sudo mount /dev/md0 /mnt/scratch
    sudo chown "$(whoami):$(whoami)" /mnt/scratch
fi

# Persist mounts
if ! grep -q '/mnt/data' /etc/fstab; then
    DATA_UUID=$(sudo blkid -s UUID -o value /dev/nvme0n2)
    echo "UUID=${DATA_UUID}  /mnt/data  ext4  defaults,nofail  0 2" | sudo tee -a /etc/fstab
fi
if ! grep -q '/mnt/scratch' /etc/fstab; then
    SCRATCH_UUID=$(sudo blkid -s UUID -o value /dev/md0)
    echo "UUID=${SCRATCH_UUID}  /mnt/scratch  ext4  defaults,nofail  0 2" | sudo tee -a /etc/fstab
fi
sudo mdadm --detail --scan | sudo tee -a /etc/mdadm/mdadm.conf > /dev/null 2>&1 || true
sudo update-initramfs -u 2>/dev/null || true

log "Disk layout:"
df -h /mnt/data /mnt/scratch

# ---------------------------------------------------------------------------
# 2. CACHE DIRECTORIES & ENV VARS
# ---------------------------------------------------------------------------
echo ""
echo "========== 2. Cache directories & env vars =========="

mkdir -p /mnt/scratch/hf/transformers /mnt/scratch/hf/datasets
mkdir -p /mnt/scratch/vllm_cache /mnt/scratch/merged_models
mkdir -p /mnt/scratch/eval_spare/logs /mnt/scratch/model_sweep

MARKER="# === RFT SPARE ENV ==="
for rc_file in ~/.bashrc ~/.profile; do
    if ! grep -q "$MARKER" "$rc_file" 2>/dev/null; then
        cat >> "$rc_file" << 'ENVEOF'

# === RFT SPARE ENV ===
export HF_HOME=/mnt/scratch/hf
export TRANSFORMERS_CACHE=/mnt/scratch/hf/transformers
export HF_DATASETS_CACHE=/mnt/scratch/hf/datasets
export VLLM_CACHE_DIR=/mnt/scratch/vllm_cache
ENVEOF
        log "Env vars added to $rc_file"
    fi
done

export HF_HOME=/mnt/scratch/hf
export TRANSFORMERS_CACHE=/mnt/scratch/hf/transformers
export HF_DATASETS_CACHE=/mnt/scratch/hf/datasets
export VLLM_CACHE_DIR=/mnt/scratch/vllm_cache

# ---------------------------------------------------------------------------
# 3. REPO
# ---------------------------------------------------------------------------
echo ""
echo "========== 3. Repo =========="

REPO_DIR="$HOME/Code/rft"
REPO_URL="https://github.com/fruitfoxlu/rft.git"
TARGET_BRANCH="${TARGET_BRANCH:-attempt-26-grpo-20b-em}"

if [ -d "$REPO_DIR/.git" ]; then
    log "Repo exists, fetching..."
    cd "$REPO_DIR" && git fetch --all 2>&1 | tail -3
else
    log "Cloning repo..."
    mkdir -p "$(dirname "$REPO_DIR")"
    git clone "$REPO_URL" "$REPO_DIR" 2>&1 | tail -3
    cd "$REPO_DIR"
fi

git checkout "$TARGET_BRANCH" 2>&1 || git checkout -b "$TARGET_BRANCH" "origin/$TARGET_BRANCH" 2>&1
git pull origin "$TARGET_BRANCH" 2>&1 | tail -3 || true
log "HEAD: $(git log --oneline -1)"

# ---------------------------------------------------------------------------
# 4. CONDA ENV & PACKAGES
# ---------------------------------------------------------------------------
echo ""
echo "========== 4. Conda env & packages =========="

if conda env list | grep -q "rft"; then
    log "rft env exists"
else
    log "Creating rft env..."
    conda create -n rft python=3.10 -y 2>&1 | tail -3
fi

log "Python: $(conda run -n rft python -V 2>&1)"

log "Installing torch..."
conda run -n rft pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3

log "Installing flash-attn..."
conda run -n rft pip install flash-attn==2.8.3 --no-build-isolation 2>&1 | tail -3

log "Installing vllm..."
conda run -n rft pip install vllm==0.15.1 2>&1 | tail -3

log "Installing openrlhf + deps..."
conda run -n rft pip install openrlhf==0.9.3 scipy kernels 2>&1 | tail -3

# ---------------------------------------------------------------------------
# 5. PATCHES (applied by copying from primary — see spare_ops.md Section 5)
# ---------------------------------------------------------------------------
echo ""
echo "========== 5. Patches =========="
cd "$REPO_DIR"
conda run -n rft bash apply_patches.sh 2>&1
log "Note: some patches may fail due to path formats. Use patched file copy from primary."

# ---------------------------------------------------------------------------
# 6. VERIFICATION
# ---------------------------------------------------------------------------
echo ""
echo "========== 6. Verification =========="

conda run -n rft python -c "
import vllm, torch, peft, transformers, deepspeed, scipy
from openai import OpenAI
print(f'vllm={vllm.__version__} torch={torch.__version__} peft={peft.__version__}')
print(f'transformers={transformers.__version__} deepspeed={deepspeed.__version__} scipy={scipy.__version__}')
print(f'CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}')
" 2>/dev/null

cd "$REPO_DIR"
conda run -n rft python -c "
import sys; sys.path.insert(0, '.')
from reward_func_em import extract_model_answer, check_correctness
assert check_correctness('42', '42') == 1
print('reward_func_em: OK')
"

conda run -n rft pip freeze > /mnt/scratch/eval_spare/pip_freeze.txt
log "pip freeze saved ($(wc -l < /mnt/scratch/eval_spare/pip_freeze.txt) packages)"

echo ""
log "Dataset checksums:"
sha256sum data/probe_set_1000_ood.jsonl data/probe_set_200.jsonl data/aime_eval.jsonl 2>/dev/null || warn "Some data files missing (sync from primary)"

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -8

echo ""
echo "========== SETUP COMPLETE =========="
