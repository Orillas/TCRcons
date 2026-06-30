#!/usr/bin/env bash
# install-deeptcr-repro.sh — Reproducible DeepTCR installation
#
# Installs DeepTCR from GitHub with pinned dependency versions matching
# the verified working .venv (TF 2.15.1, Keras 2.15.0, etc.).
#
# Why a separate script?
#   DeepTCR's setup.py declares tensorflow==2.12.0 via install_requires,
#   which conflicts with the newer versions we want. Pip extras cannot
#   override transitive deps, so we install DeepTCR with --no-deps and
#   pin everything ourselves.
#
# Usage:
#   bash scripts/install-deeptcr-repro.sh            # default: pip
#   UV=1 bash scripts/install-deeptcr-repro.sh        # use uv instead of pip
#   DRY_RUN=1 bash scripts/install-deeptcr-repro.sh   # print commands only
#
# Requires:
#   - git (to clone DeepTCR)
#   - pip or uv
#   - CUDA-capable GPU recommended (CPU works but is slow)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PINNED_REQ="$REPO_DIR/requirements/deeptcr-pinned.txt"

# DeepTCR commit and URL
DEEPTCR_URL="https://github.com/sidhomj/DeepTCR.git"
DEEPTCR_COMMIT="3930ca05a987c7cc621b4f2ecfd740e2d62799d8"

# Choose package manager
if [[ "${UV:-}" == "1" ]]; then
    PIP="uv pip"
else
    PIP="pip"
fi

echo "========================================"
echo " Reproducible DeepTCR Install"
echo "========================================"
echo "  Package manager : $PIP"
echo "  DeepTCR version : 2.1.29 ($DEEPTCR_COMMIT)"
echo "  Pinned deps     : $PINNED_REQ"
echo "========================================"

# ---- Step 1: Install DeepTCR without its own deps ----
echo ""
echo "[1/2] Installing DeepTCR (--no-deps)..."

if [[ -n "${DRY_RUN:-}" ]]; then
    echo "  $PIP install --no-deps \"DeepTCR @ git+$DEEPTCR_URL@$DEEPTCR_COMMIT\""
else
    $PIP install --no-deps "DeepTCR @ git+$DEEPTCR_URL@$DEEPTCR_COMMIT"
    echo "  ✓ DeepTCR installed"
fi

# ---- Step 2: Install pinned dependencies ----
echo ""
echo "[2/2] Installing pinned dependencies from requirements/deeptcr-pinned.txt..."

if [ ! -f "$PINNED_REQ" ]; then
    echo "  ERROR: $PINNED_REQ not found. Run this script from the repo root."
    exit 1
fi

if [[ -n "${DRY_RUN:-}" ]]; then
    echo "  $PIP install -r \"$PINNED_REQ\""
else
    $PIP install -r "$PINNED_REQ"
    echo "  ✓ Pinned dependencies installed"
fi

# ---- Verify ----
echo ""
echo "========================================"
echo " Verification"
echo "========================================"
if [[ -z "${DRY_RUN:-}" ]]; then
    python3 -c "
import DeepTCR
print(f'  DeepTCR     : {DeepTCR.__path__[0]}')
try:
    import tensorflow as tf
    print(f'  TensorFlow  : {tf.__version__}')
except: pass
try:
    import keras
    print(f'  Keras       : {keras.__version__}')
except: pass
try:
    import numpy as np
    print(f'  NumPy       : {np.__version__}')
except: pass
try:
    import scipy
    print(f'  SciPy       : {scipy.__version__}')
except: pass
try:
    import sklearn
    print(f'  scikit-learn: {sklearn.__version__}')
except: pass
"
fi
echo ""
echo "✅ Reproducible DeepTCR install complete."
echo "   To test: python -c \"from DeepTCR.DeepTCR import DeepTCR_U; print('OK')\""
