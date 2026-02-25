#!/usr/bin/env bash
# =============================================================================
# verify_spare.sh — Run from PRIMARY (h100-8-4) to verify spare is at parity
# Usage: bash verify_spare.sh
# =============================================================================
set -euo pipefail

SSH="ssh -i ~/.ssh/google_compute_engine rlu@10.202.0.3"

pass() { echo "  [PASS] $*"; }
fail() { echo "  [FAIL] $*"; FAILURES=$((FAILURES + 1)); }

FAILURES=0
echo "========== Spare Machine Parity Check =========="

# 1. Git SHA
echo ""
echo "--- Git SHA ---"
LOCAL_SHA=$(cd ~/Code/rft && git rev-parse HEAD)
SPARE_SHA=$($SSH 'cd ~/Code/rft && git rev-parse HEAD' 2>/dev/null)
if [ "$LOCAL_SHA" = "$SPARE_SHA" ]; then
    pass "Git SHA match: ${LOCAL_SHA:0:8}"
else
    fail "Git SHA mismatch: primary=${LOCAL_SHA:0:8} spare=${SPARE_SHA:0:8}"
fi

# 2. Python version
echo ""
echo "--- Python version ---"
LOCAL_PY=$(conda run -n rft python -V 2>&1)
SPARE_PY=$($SSH 'export PATH="/opt/conda/bin:$PATH" && conda run -n rft python -V' 2>/dev/null)
if [ "$LOCAL_PY" = "$SPARE_PY" ]; then
    pass "Python: $LOCAL_PY"
else
    fail "Python: primary='$LOCAL_PY' spare='$SPARE_PY'"
fi

# 3. Key packages
echo ""
echo "--- Key packages ---"
LOCAL_PKGS=$(conda run -n rft python -c "import vllm,torch,peft; print(f'{vllm.__version__} {torch.__version__} {peft.__version__}')" 2>/dev/null)
SPARE_PKGS=$($SSH 'export PATH="/opt/conda/bin:$PATH" && conda run -n rft python -c "import vllm,torch,peft; print(f\"{vllm.__version__} {torch.__version__} {peft.__version__}\")"' 2>/dev/null)
if [ "$LOCAL_PKGS" = "$SPARE_PKGS" ]; then
    pass "Packages: $LOCAL_PKGS"
else
    fail "Packages: primary='$LOCAL_PKGS' spare='$SPARE_PKGS'"
fi

# 4. Dataset checksums
echo ""
echo "--- Datasets ---"
for f in data/probe_set_1000_ood.jsonl data/probe_set_200.jsonl data/aime_eval.jsonl; do
    LOCAL_SUM=$(cd ~/Code/rft && sha256sum "$f" 2>/dev/null | awk '{print $1}')
    SPARE_SUM=$($SSH "cd ~/Code/rft && sha256sum $f" 2>/dev/null | awk '{print $1}')
    if [ "$LOCAL_SUM" = "$SPARE_SUM" ]; then
        pass "$f"
    else
        fail "$f checksum mismatch"
    fi
done

# 5. Disk mounts
echo ""
echo "--- Disks ---"
if $SSH 'mountpoint -q /mnt/data' 2>/dev/null; then
    pass "/mnt/data mounted"
else
    fail "/mnt/data not mounted"
fi
if $SSH 'mountpoint -q /mnt/scratch' 2>/dev/null; then
    pass "/mnt/scratch mounted"
else
    fail "/mnt/scratch not mounted"
fi

# 6. GPUs
echo ""
echo "--- GPUs ---"
SPARE_GPUS=$($SSH 'nvidia-smi --query-gpu=count --format=csv,noheader | head -1' 2>/dev/null)
if [ "${SPARE_GPUS:-0}" -ge 8 ]; then
    pass "$SPARE_GPUS GPUs detected"
else
    fail "Expected 8 GPUs, got ${SPARE_GPUS:-0}"
fi

# 7. Env vars
echo ""
echo "--- Env vars ---"
SPARE_HF=$($SSH 'bash -lc "echo \$HF_HOME"' 2>/dev/null)
if [ "$SPARE_HF" = "/mnt/scratch/hf" ]; then
    pass "HF_HOME=/mnt/scratch/hf"
else
    fail "HF_HOME='$SPARE_HF'"
fi

# 8. All patches
echo ""
echo "--- Patches ---"
PATCH_CHECK=$($SSH 'export PATH="/opt/conda/bin:$PATH" && SITE_PKG=$(conda run -n rft python -c "import openrlhf,os; print(os.path.dirname(os.path.dirname(openrlhf.__file__)))") && grep -c "apply_lora_delta" $SITE_PKG/openrlhf/trainer/ray/vllm_worker_wrap.py' 2>/dev/null)
if [ "${PATCH_CHECK:-0}" -gt 0 ]; then
    pass "Patches applied (vllm_worker verified)"
else
    fail "Patches may not be applied"
fi

# Summary
echo ""
echo "========================================="
if [ $FAILURES -eq 0 ]; then
    echo "  ALL CHECKS PASSED"
else
    echo "  ${FAILURES} CHECK(S) FAILED"
fi
echo "========================================="
