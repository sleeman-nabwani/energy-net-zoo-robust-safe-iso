#!/bin/bash
#SBATCH --job-name=safeiso_saute_only
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5%3

set -e

CONDA_ENV="${CONDA_ENV:-energy-net-zoo}"
SAVE_ROOT="${SAVE_ROOT:-runs/baseline_static_saute_rerun}"
STEPS="${STEPS:-48000}"
COST_LIMIT="${COST_LIMIT:-0.10}"
PRESET="${PRESET:-default}"
HORIZON="${HORIZON:-48}"
DEVICE="${DEVICE:-cuda}"

ALGOS=(SautePPO SautePPO SautePPO SautePPO SautePPO SautePPO)
MODES=(cmdp cmdp cmdp mdp mdp mdp)
SEEDS=(0 1 2 0 1 2)

IDX=${SLURM_ARRAY_TASK_ID:-0}
ALGO="${ALGOS[$IDX]}"; MODE="${MODES[$IDX]}"; SEED="${SEEDS[$IDX]}"

if [[ "$MODE" == "cmdp" ]]; then MODE_FLAG="--cmdp"; MODE_TAG="cmdp"; else MODE_FLAG="--no-cmdp"; MODE_TAG="mdp"; fi
PCS="static:0.0"; PCS_TAG="static-0.0"
RUN_DIR="${SAVE_ROOT}/${ALGO}_${MODE_TAG}_${PRESET}_${PCS_TAG}/seed${SEED}"

echo "== SautePPO rerun =="; echo "ALGO=${ALGO} MODE=${MODE_TAG} SEED=${SEED} SAVE=${RUN_DIR}"

# Activate env
source "$HOME/miniconda3/etc/profile.d/conda.sh" || true
conda activate "$CONDA_ENV"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/energy-net-zoo-robust-safe-iso:$PYTHONPATH"

# Register envs once
python - <<'PY'
import safeiso.utils.register_envs  # noqa
print("SafeISO envs registered.")
PY

# Train (trainer exports EvalPack at end)
stdbuf -oL -eL python -u -m safeiso.train.train_omnisafe \
  --algo "${ALGO}" \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --device "${DEVICE}" \
  ${MODE_FLAG} \
  --cost_limit "${COST_LIMIT}" \
  --pcs "${PCS}" \
  --preset "${PRESET}" \
  --max_episode_steps "${HORIZON}" \
  --eval_every "${STEPS}" \
  --eval_episodes 0 \
  --save_dir "${RUN_DIR}"

