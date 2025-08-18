#!/bin/bash
#SBATCH --job-name=safeiso_pcs_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -e

CONDA_ENV="${CONDA_ENV:-energy-net-zoo}"
SAVE_ROOT="${SAVE_ROOT:-runs/pcs_smoke_1seed}"

echo "== PCS Grid Evaluation =="
echo "Date: $(date)"

# Activate env
source "$HOME/miniconda3/etc/profile.d/conda.sh" || true
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/energy-net-zoo-robust-safe-iso:$PYTHONPATH"

# Check completion
echo "=== Verifying training completion ==="
EXPECTED=30
FOUND=$(find "$SAVE_ROOT" -name "evalpack" -type d 2>/dev/null | wc -l)
echo "Found $FOUND EvalPacks (Expected: $EXPECTED)"

if [ "$FOUND" -lt "$EXPECTED" ]; then
  echo "⚠️  Training incomplete, evaluating available runs only"
fi

# Evaluate all available runs
echo "=== Starting evaluation ==="
python -u -m safeiso.eval.evaluate_omnisafe \
  --suite safeiso/eval/suites/pcs_mix_smoke.yaml \
  --episodes 4 \
  --horizon 48 \
  --seed 0 \
  --device cpu \
  --cost_limit 0.10 \
  --dirs "$SAVE_ROOT"

# Quick results summary
echo "=== Evaluation Results Summary ==="
python - <<'PY'
import glob, pandas as pd
paths = sorted(glob.glob("runs/pcs_smoke_1seed/**/eval/stress_eval.csv", recursive=True))
print(f"Evaluated CSVs: {len(paths)}")
if paths:
    for p in paths[:6]:  # Show first 6
        df = pd.read_csv(p)
        algo_mode_pcs = p.split('/')[-3]
        result = df[['return_mean', 'avg_step_cost', 'pass_cost_limit']].iloc[0]
        print(f"{algo_mode_pcs}: return={result['return_mean']:.0f}, cost={result['avg_step_cost']:.3f}, pass={result['pass_cost_limit']}")
PY

echo "=== Evaluation completed at $(date) ==="
