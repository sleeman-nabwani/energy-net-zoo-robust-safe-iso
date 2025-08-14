#!/bin/bash
#SBATCH --job-name=safeiso_baseline
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
# NOTE: Partition is cluster-specific; pass with 'sbatch -p <partition>' if required
#SBATCH --array=0-29%4

# ===== User knobs (override with --export or env) =====
CONDA_ENV="${CONDA_ENV:-energy-net-zoo}"
SAVE_ROOT="${SAVE_ROOT:-runs/baseline_static}"
STEPS="${STEPS:-48000}"            # multiple of 48
COST_LIMIT="${COST_LIMIT:-0.10}"
PRESET="${PRESET:-default}"
HORIZON="${HORIZON:-48}"
DEVICE="${DEVICE:-cuda}"            # default to cuda; override with DEVICE=cpu if needed
EVAL_MODE="${EVAL_MODE:-policy}"    # random|policy|none

# Grid
ALGOS=(PPOLag CUP CPO FOCOPS SautePPO)
MODES=(cmdp mdp)                    # mdp == ISOOnly
SEEDS=(0 1 2)

N_A=${#ALGOS[@]}
N_M=${#MODES[@]}
N_S=${#SEEDS[@]}

TID=${SLURM_ARRAY_TASK_ID:-0}
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "SLURM_ARRAY_TASK_ID not set; running single job with TID=0 (use --array=0-29%N to control concurrency)."
fi
AIDX=$(( TID % N_A ))
MIDX=$(( (TID / N_A) % N_M ))
SIDX=$(( (TID / (N_A * N_M)) % N_S ))

ALGO="${ALGOS[$AIDX]}"
MODE="${MODES[$MIDX]}"
SEED="${SEEDS[$SIDX]}"

if [[ "$MODE" == "cmdp" ]]; then
  MODE_FLAG="--cmdp"
  MODE_TAG="cmdp"
else
  MODE_FLAG="--no-cmdp"
  MODE_TAG="mdp"
fi

PCS="static:0.0"
PCS_TAG="static-0.0"
RUN_DIR="${SAVE_ROOT}/${ALGO}_${MODE_TAG}_${PRESET}_${PCS_TAG}/seed${SEED}"

echo "== SafeISO baseline =="
echo "ALGO=${ALGO} MODE=${MODE_TAG} SEED=${SEED} PCS=${PCS}"
echo "SAVE_DIR=${RUN_DIR}"
mkdir -p "$(dirname "$RUN_DIR")" logs

# Activate env (use absolute paths; compute nodes may not have interactive shell init)
CONDABASE="/home/sleemann/miniconda3"
# Try sourcing mamba first, then conda
if [[ -f "$CONDABASE/etc/profile.d/mamba.sh" ]]; then
  source "$CONDABASE/etc/profile.d/mamba.sh"
elif [[ -f "$CONDABASE/etc/profile.d/conda.sh" ]]; then
  source "$CONDABASE/etc/profile.d/conda.sh"
else
  echo "WARNING: Could not find conda/mamba init scripts under $CONDABASE/etc/profile.d"
fi
# Fallback to user's bashrc (harmless if absent)
source /home/sleemann/.bashrc >/dev/null 2>&1 || true
# Prefer mamba if available
if command -v mamba >/dev/null 2>&1; then
mamba activate "$CONDA_ENV" >/dev/null 2>&1 || conda activate "$CONDA_ENV"
else
  conda activate "$CONDA_ENV"
fi

# Ensure we use the repo's Python modules (eval_mode, eval helpers, etc.)
export PYTHONPATH="/home/sleemann/energy-net-zoo-robust-safe-iso:${PYTHONPATH}"

# Quick CUDA diagnostics on the compute node (optional but helpful)
python - <<'PY'
import os
try:
    import torch
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"torch.cuda.is_available()={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"num_gpus={torch.cuda.device_count()}")
        try:
            print(f"gpu_name={torch.cuda.get_device_name(0)}")
        except Exception:
            pass
except Exception as e:
    print(f"Torch/CUDA check failed: {e}")
PY

# Ensure SafeISO env IDs registered once per task (no-op if already imported)
python - <<'PY'
import safeiso.utils.register_envs  # noqa: F401
print("SafeISO Omni IDs registered.")
PY

# Train + quick eval
python -u -m safeiso.train.train_omnisafe \
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
  --eval_episodes 3 \
  --eval_mode "${EVAL_MODE}" \
  --save_dir "${RUN_DIR}" 