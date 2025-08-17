#!/usr/bin/env bash
set -u

PY=/home/sleemann/.local/share/mamba/envs/energy-net-zoo/bin/python
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/sleemann/energy-net-zoo-robust-safe-iso:${PYTHONPATH:-}

exec "$PY" -u safeiso/eval/evaluate_omnisafe.py \
  --suite safeiso/eval/suites/baseline_suite.yaml \
  --baseline-only \
  --device cuda \
  --loader auto \
  --strict \
  --verbose


