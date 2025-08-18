#!/bin/bash
#SBATCH --job-name=safeiso_pcs_grid
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-29%6

set -e

CONDA_ENV="${CONDA_ENV:-energy-net-zoo}"
SAVE_ROOT="${SAVE_ROOT:-runs/pcs_smoke_1seed}"
STEPS="${STEPS:-9600}"
COST_LIMIT="${COST_LIMIT:-0.10}"
PRESET="${PRESET:-default}"
HORIZON="${HORIZON:-48}"
DEVICE="${DEVICE:-cuda}"

# 5 algos × 2 modes × 3 PCS = 30 jobs
ALGOS=(PPOLag PPOLag PPOLag CUP CUP CUP CPO CPO CPO FOCOPS FOCOPS FOCOPS SautePPO SautePPO SautePPO PPOLag PPOLag PPOLag CUP CUP CUP CPO CPO CPO FOCOPS FOCOPS FOCOPS SautePPO SautePPO SautePPO)
MODES=(cmdp cmdp cmdp cmdp cmdp cmdp cmdp cmdp cmdp cmdp cmdp cmdp cmdp cmdp cmdp mdp mdp mdp mdp mdp mdp mdp mdp mdp mdp mdp mdp mdp mdp mdp)
PCS_SPECS=("static:0.0" "static:3.0" "responsive:spread_prop?gain=0.6&clip=5" "static:0.0" "static:3.0" "responsive:spread_prop?gain=0.6&clip=5" "static:0.0" "static:3.0" "responsive:spread_prop?gain=0.6&clip=5" "static:0.0" "static:3.0" "responsive:spread_prop?gain=0.6&clip=5" "static:0.0" "static:3.0" "responsive:spread_prop?gain=0.6&clip=5" "static:0.0" "static:3.0" "responsive:spread_prop?gain=0.6&clip=5" "static:0.0" "static:3.0" "responsive:spread_prop?gain=0.6&clip=5" "static:0.0" "static:3.0" "responsive:spread_prop?gain=0.6&clip=5" "static:0.0" "static:3.0" "responsive:spread_prop?gain=0.6&clip=5" "static:0.0" "static:3.0" "responsive:spread_prop?gain=0.6&clip=5")

IDX=${SLURM_ARRAY_TASK_ID:-0}
ALGO="${ALGOS[$IDX]}"
MODE="${MODES[$IDX]}"
PCS="${PCS_SPECS[$IDX]}"

if [[ "$MODE" == "cmdp" ]]; then MODE_FLAG="--cmdp"; else MODE_FLAG="--no-cmdp"; fi
TAG=$(echo "$PCS" | sed 's|/|-|g' | sed 's|:|_|g' | sed 's|?||g' | sed 's|&|_|g' )
RUN_DIR="${SAVE_ROOT}/${ALGO}_${MODE}_${TAG}/seed0"

echo "== PCS Grid: $ALGO $MODE PCS=$PCS -> $RUN_DIR =="

# Activate env
source "$HOME/miniconda3/etc/profile.d/conda.sh" || true
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/energy-net-zoo-robust-safe-iso:$PYTHONPATH"

# Register envs
python - <<'PY'
import safeiso.utils.register_envs  # noqa
print("SafeISO envs registered.")
PY

# Train
stdbuf -oL -eL python -u -m safeiso.train.train_omnisafe \
  --algo "${ALGO}" \
  --steps "${STEPS}" \
  --seed 0 \
  --device "${DEVICE}" \
  ${MODE_FLAG} \
  --cost_limit "${COST_LIMIT}" \
  --pcs "${PCS}" \
  --preset "${PRESET}" \
  --max_episode_steps "${HORIZON}" \
  --eval_every "${STEPS}" \
  --eval_episodes 0 \
  --save_dir "${RUN_DIR}"

