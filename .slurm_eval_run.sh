#!/usr/bin/env bash
set -euo pipefail
source /home/sleemann/miniconda3/etc/profile.d/mamba.sh
mamba activate energy-net-zoo
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/sleemann/energy-net-zoo-robust-safe-iso:$PYTHONPATH
stdbuf -oL -eL python -u safeiso/eval/evaluate_omnisafe.py --suite safeiso/eval/suites/baseline_suite.yaml --baseline-only --device cuda --loader evaluator --strict --verbose
