#!/bin/bash
#SBATCH --job-name=safeiso_baseline_evalpack
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-29%4

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-energy-net-zoo}"
SAVE_ROOT="${SAVE_ROOT:-runs/baseline_static}"
STEPS="${STEPS:-48000}"
COST_LIMIT="${COST_LIMIT:-0.10}"
PRESET="${PRESET:-default}"
HORIZON="${HORIZON:-48}"
DEVICE="${DEVICE:-cuda}"
EVAL_MODE="${EVAL_MODE:-none}"

ALGOS=(PPOLag CUP CPO FOCOPS SautePPO)
MODES=(cmdp mdp)
SEEDS=(0 1 2)

N_A=${#ALGOS[@]}; N_M=${#MODES[@]}; N_S=${#SEEDS[@]}
TID=${SLURM_ARRAY_TASK_ID:-0}
AIDX=$(( TID % N_A ))
MIDX=$(( (TID / N_A) % N_M ))
SIDX=$(( (TID / (N_A * N_M)) % N_S ))
ALGO="${ALGOS[$AIDX]}"; MODE="${MODES[$MIDX]}"; SEED="${SEEDS[$SIDX]}"

if [[ "$MODE" == "cmdp" ]]; then MODE_FLAG="--cmdp"; MODE_TAG="cmdp"; ENV_ID="SafeISO-CMDP-omni-v0"; else MODE_FLAG="--no-cmdp"; MODE_TAG="mdp"; ENV_ID="SafeISO-ISOOnly-omni-v0"; fi

PCS="static:0.0"; PCS_TAG="static-0.0"
RUN_DIR="${SAVE_ROOT}/${ALGO}_${MODE_TAG}_${PRESET}_${PCS_TAG}/seed${SEED}"

echo "== Train baseline w/ EvalPack =="
echo "ALGO=${ALGO} MODE=${MODE_TAG} SEED=${SEED} PCS=${PCS}"
echo "SAVE_DIR=${RUN_DIR}"

mkdir -p "$(dirname "$RUN_DIR")" logs

CONDABASE="$HOME/miniconda3"
if [ -f "$CONDABASE/etc/profile.d/mamba.sh" ]; then
  source "$CONDABASE/etc/profile.d/mamba.sh"
  mamba activate "$CONDA_ENV" || true
fi
if ! command -v python >/dev/null 2>&1; then
  source "$CONDABASE/etc/profile.d/conda.sh" || true
  conda activate "$CONDA_ENV" || true
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/energy-net-zoo-robust-safe-iso:$PYTHONPATH"

python - <<'PY'
import safeiso.utils.register_envs  # noqa: F401
print("SafeISO envs registered.")
PY

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
  --eval_mode "${EVAL_MODE}" \
  --save_dir "${RUN_DIR}"

