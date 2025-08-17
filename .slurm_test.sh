#!/usr/bin/env bash
set -euxo pipefail

echo "[test] Hostname: $(hostname)"
date
echo "[test] nvidia-smi:" || true
nvidia-smi || true

# Activate environment (mamba first, then conda fallback)
if [ -f /home/sleemann/miniconda3/etc/profile.d/mamba.sh ]; then
    source /home/sleemann/miniconda3/etc/profile.d/mamba.sh || true
    mamba activate energy-net-zoo || true
fi
if ! command -v python >/dev/null 2>&1; then
    if [ -f /home/sleemann/miniconda3/etc/profile.d/conda.sh ]; then
        source /home/sleemann/miniconda3/etc/profile.d/conda.sh || true
        conda activate energy-net-zoo || true
    fi
fi

which python || true
python --version || true
python - <<'PY'
import torch, sys
print('torch_version', getattr(torch, '__version__', 'unknown'))
print('cuda_available', torch.cuda.is_available())
print('cuda_device_count', torch.cuda.device_count())
sys.exit(0)
PY

sleep 2
echo "[test] done"


