#!/bin/bash
#SBATCH --job-name=safeiso_pcs_smoke_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

ROOT="${ROOT:-runs/pcs_smoke_fixcheck}"
SUITE="${SUITE:-safeiso/eval/suites/pcs_mix_smoke.yaml}"
EPISODES="${EPISODES:-8}"
HORIZON="${HORIZON:-48}"
COST_LIMIT="${COST_LIMIT:-0.10}"

source "$HOME/miniconda3/etc/profile.d/conda.sh" || true
conda activate energy-net-zoo
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/energy-net-zoo-robust-safe-iso:$PYTHONPATH"

echo "Starting evaluation with debug enabled..."

python -u -m safeiso.eval.evaluate_omnisafe \
  --suite "$SUITE" \
  --episodes "$EPISODES" --horizon "$HORIZON" \
  --seed 0 --device cpu --cost_limit "$COST_LIMIT" \
  --dirs "$ROOT" --debug

echo "Evaluation completed"
