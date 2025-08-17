#!/bin/bash
set -euo pipefail

# Minimal smoke tests for all algorithms in this repo
# Usage:
#   bash /home/sleemann/energy-net-zoo-robust-safe-iso/scripts/smoke_all.sh
# Optional env overrides:
#   MODE=cmdp|mdp STEPS=96 HORIZON=48 EVAL_EPISODES=1 DEVICE=cpu COST_LIMIT=0.10 SEED=0 PCS_SPEC="static:0.0" PRESET=default EVAL_MODE=random|policy|none

ROOT="/home/sleemann/energy-net-zoo-robust-safe-iso"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT"
CONDA_ENV="${CONDA_ENV:-energy-net-zoo}"

# Build a python command as an array to support conda run with args
PYTHON_CMD=(python3)
if [[ -x "/home/sleemann/miniconda3/bin/conda" ]]; then
  PYTHON_CMD=("/home/sleemann/miniconda3/bin/conda" run -n "${CONDA_ENV}" python)
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
fi

STEPS="${STEPS:-96}"
HORIZON="${HORIZON:-48}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
DEVICE="${DEVICE:-cpu}"
COST_LIMIT="${COST_LIMIT:-0.10}"
PCS_SPEC="${PCS_SPEC:-static:0.0}"
PRESET="${PRESET:-default}"
SEED="${SEED:-0}"
MODE="${MODE:-cmdp}"
EVAL_MODE="${EVAL_MODE:-random}"

if [[ "$MODE" == "cmdp" ]]; then
  MODE_FLAG="--cmdp"
  TAG="cmdp"
else
  MODE_FLAG="--no-cmdp"
  TAG="mdp"
fi

ALGOS=(PPOLag CUP CPO FOCOPS SautePPO)

for ALGO in "${ALGOS[@]}"; do
  SAVE_DIR="$ROOT/runs/smoke_${ALGO}_${TAG}_${EVAL_MODE}"
  echo "== $ALGO ($TAG) eval_mode=$EVAL_MODE =="
  "${PYTHON_CMD[@]}" -m safeiso.train.train_omnisafe \
    --algo "$ALGO" \
    --steps "$STEPS" \
    --seed "$SEED" \
    --device "$DEVICE" \
    $MODE_FLAG \
    --cost_limit "$COST_LIMIT" \
    --pcs "$PCS_SPEC" \
    --preset "$PRESET" \
    --max_episode_steps "$HORIZON" \
    --eval_episodes "$EVAL_EPISODES" \
    --eval_mode "$EVAL_MODE" \
    --save_dir "$SAVE_DIR"
  echo
done

echo "All smoke tests completed (MODE=$TAG, EVAL_MODE=$EVAL_MODE)." 