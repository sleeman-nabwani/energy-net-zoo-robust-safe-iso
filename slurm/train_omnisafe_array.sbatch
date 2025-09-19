#!/usr/bin/env bash
#SBATCH --job-name=omnisafe-train
#SBATCH --output=slurm_logs/train_%A_%a.out
#SBATCH --error=slurm_logs/train_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16

set -Eeo pipefail

# Tip: set --array at submit time (30 jobs basic, ~135 with PCS-only SB3 models)
# e.g. sbatch --array=0-29   slurm/train_omnisafe_array.sbatch  # basic (static+responsive only)
#      sbatch --array=0-134  slurm/train_omnisafe_array.sbatch  # with PCS-only SB3 models
#      TRAIN_STEPS=10000 sbatch --array=0-5 slurm/train_omnisafe_array.sbatch  # pilot run

# Repo & PYTHONPATH
REPO="/home/sleemann/energy-net-zoo-robust-safe-iso"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$REPO/slurm_logs"
cd "$REPO"

# Optional: SB3 PCS zip (set before sbatch): SB3_PCS_ZIP=/abs/path/to/best_model.zip
SB3_ZIP="${SB3_PCS_ZIP:-}"

module -q avail python 2>/dev/null || true
module -q load python/3.8 2>/dev/null || true
module -q load cuda/11.8 2>/dev/null || true

# Activate env (conda preferred, fallback to venv)
if command -v conda >/dev/null 2>&1; then
  source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
  conda activate energy-net-zoo 2>/dev/null || echo "[warn] conda env missing, trying venv"
fi
if [[ -z "${CONDA_PREFIX:-}" && -d "venv" ]]; then source venv/bin/activate; fi

# Config arrays
ALGOS=(PPOLag CPO CUP SautePPO FOCOPS)
PCS_STRINGS=("static:10.0" "responsive:spread_prop?gain=4.0&clip=25.0")
PCS_LABELS=("static" "responsive")
SEEDS=(0 42 123)

# Auto-discover trained PCS models (excluding ISO models)
if [[ -d "$REPO/trained_pcs_models" ]]; then
  echo "[info] Auto-discovering trained PCS models (excluding ISO models)..."
  
  # Find all SB3 model zip files
  while IFS= read -r -d '' model_zip; do
    model_dir="$(dirname "$model_zip")"
    model_basename="$(basename "$model_zip" .zip)"
    setting_name="$(basename "$(dirname "$model_zip")")"
    
    # CRITICAL FIX: Only include PCS models, skip ISO models
    # ISO models control the ISO directly and should not be used as PCS controllers
    if [[ "$model_basename" == *iso* ]]; then
      echo "[skip] Excluding ISO model: $model_basename (ISO agents shouldn't control PCS)"
      continue
    fi
    
    # Only process PCS models - determine appropriate vecnormalize file
    vecnorm_file=""
    if [[ "$model_basename" == pcs_* ]]; then
      # PCS models: look for PCS-specific normalization
      vecnorm_file="$(find "$model_dir" -name "*vec_normalize_pcs*.pkl" -o -name "*pcs*norm*.pkl" 2>/dev/null | head -1)"
    else
      # Other non-ISO models: look for model-specific or general normalization
      vecnorm_file="$(find "$model_dir" -name "${model_basename}_norm.pkl" -o -name "*vecnormalize*.pkl" 2>/dev/null | head -1)"
    fi
    
    if [[ -f "$vecnorm_file" ]]; then
      # Create clean label: SettingFiveL/pcs_td3_400.zip -> SettingFiveL_pcs_td3_400
      # or SettingFiveL/td3_iso_best.zip -> SettingFiveL_td3_iso_best
      label="${setting_name}_${model_basename}"
      
      PCS_STRINGS+=("sb3:${model_zip}")
      PCS_LABELS+=("$label")
      echo "[info] Added PCS model: $label -> $model_zip (vecnorm: $(basename "$vecnorm_file"))"
    else
      echo "[warn] No compatible vecnormalize found for $model_zip, skipping"
    fi
  done < <(find "$REPO/trained_pcs_models" -name "*.zip" -type f -print0 | sort -z)
fi

# Optional SB3 PCS as additional mode
if [[ -n "$SB3_ZIP" ]]; then
  if [[ ! -f "$SB3_ZIP" ]]; then echo "[fatal] SB3 zip not found: $SB3_ZIP"; exit 1; fi
  VN="$(dirname "$SB3_ZIP")/vecnormalize.pkl"
  if [[ ! -f "$VN" ]]; then echo "[fatal] vecnormalize.pkl not found next to SB3 zip: $VN"; exit 1; fi
  PCS_STRINGS+=("sb3:${SB3_ZIP}")
  PCS_LABELS+=("sb3")
fi

NUM_ALGOS=${#ALGOS[@]}
NUM_PCS=${#PCS_STRINGS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL=$((NUM_ALGOS * NUM_PCS * NUM_SEEDS))

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[fatal] SLURM_ARRAY_TASK_ID is not set. Submit with --array=0-$((TOTAL-1))"; exit 2
fi
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "[fatal] SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID out of range [0,$((TOTAL-1))] for TOTAL=$TOTAL"; exit 3
fi

ALGO_IDX=$(( SLURM_ARRAY_TASK_ID / (NUM_PCS * NUM_SEEDS) ))
PCS_IDX=$(( (SLURM_ARRAY_TASK_ID / NUM_SEEDS) % NUM_PCS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_SEEDS ))

ALGO="${ALGOS[$ALGO_IDX]}"
PCS="${PCS_STRINGS[$PCS_IDX]}"
PCS_LABEL="${PCS_LABELS[$PCS_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

# Create batch timestamp (shared across all jobs in this batch)
# Use SLURM_ARRAY_JOB_ID (shared across array) instead of SLURM_JOB_ID (unique per task)
BATCH_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
SAVE_DIR="$REPO/runs/omnisafe_trained_${BATCH_ID}/${ALGO}/${PCS_LABEL}/seed_${SEED}"
mkdir -p "$SAVE_DIR"

echo "=========================================="
echo "OmniSafe Training Configuration"
echo "=========================================="
echo "Array Idx / Total : ${SLURM_ARRAY_TASK_ID} / ${TOTAL}"
echo "Algorithm         : ${ALGO}"
echo "PCS (spec/label)  : ${PCS} / ${PCS_LABEL}"
echo "Seed              : ${SEED}"
echo "Save Directory    : ${SAVE_DIR}"
echo "Host / GPU        : $(hostname) / $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo cpu)"
echo "Start Time        : $(date)"
echo "=========================================="

# Enhanced configuration with adaptive cost limits to prevent NaN errors
TRAIN_STEPS=${TRAIN_STEPS:-100000}  # Default 100K for faster iteration

# Adaptive cost limit based on training steps to ensure numerical stability
if [ "$TRAIN_STEPS" -lt 1000 ]; then
    COST_LIMIT=${COST_LIMIT:-0.50}   # Very relaxed for exploration
elif [ "$TRAIN_STEPS" -lt 10000 ]; then
    COST_LIMIT=${COST_LIMIT:-0.35}   # Moderate for initial learning
elif [ "$TRAIN_STEPS" -lt 100000 ]; then
    COST_LIMIT=${COST_LIMIT:-0.25}   # Tightening constraints
else
    COST_LIMIT=${COST_LIMIT:-0.15}   # Production target
fi

# Algorithm-specific cost limit adjustments
if [ "$ALGO" == "PPOLag" ]; then
    # PPOLag needs extra care with cost limits due to Lagrange multiplier sensitivity
    COST_LIMIT=$(python3 -c "print(min(0.5, ${COST_LIMIT} * 1.5))")
fi

# Adaptive evaluation frequency based on training steps
EVAL_EVERY=$(python3 -c "print(max(1000, min(50000, ${TRAIN_STEPS} // 10)))")

echo "Enhanced Training Configuration:"
echo "  Training Steps : ${TRAIN_STEPS}"
echo "  Cost Limit     : ${COST_LIMIT} (adaptive)"
echo "  Eval Every     : ${EVAL_EVERY} steps"
echo "  Algorithm      : ${ALGO}"
echo "  PCS Mode       : ${PCS}"

# Run training with enhanced parameters
python3 -m safeiso.train.train_omnisafe \
    --algo ${ALGO} \
    --steps ${TRAIN_STEPS} \
    --seed ${SEED} \
    --pcs "${PCS}" \
    --save_dir ${SAVE_DIR} \
    --device cuda \
    --cost_limit ${COST_LIMIT} \
    --cmdp \
    --preset training \
    --eval_every ${EVAL_EVERY} \
    --eval_episodes 5

echo "[info] Training finished: $(date)"

# Post-train verification: prefer EvalPack, else SB3+vecnormalize
EP_ACTOR="${SAVE_DIR}/evalpack/actor.ts"
SB3_ZIP_ISO="${SAVE_DIR}/best_model.zip"
VN_ISO="${SAVE_DIR}/vecnormalize.pkl"

verified=false
model_files=()
[[ -f "$EP_ACTOR" ]] && verified=true && model_files+=("evalpack/actor.ts")
[[ -f "$SB3_ZIP_ISO" && -f "$VN_ISO" ]] && verified=true && model_files+=("best_model.zip" "vecnormalize.pkl")

python3 -c "
import os, json
model_files = [$(printf '"%s",' "${model_files[@]}" | sed 's/,$//')]
vf = {'algo': '${ALGO}', 'pcs': '${PCS}', 'seed': ${SEED}, 'model_files': model_files, 'verified': $([ "$verified" = true ] && echo True || echo False)}
os.makedirs('${SAVE_DIR}', exist_ok=True)
with open(os.path.join('${SAVE_DIR}', 'verification.json'), 'w') as f:
    json.dump(vf, f, indent=2)
print(json.dumps(vf, indent=2))
"

if [[ "$verified" != true ]]; then
  echo "[fatal] No EvalPack or SB3+vecnormalize artifacts found in ${SAVE_DIR}"
  exit 4
fi

echo "[ok] Verified artifacts for ${ALGO} / ${PCS_LABEL} / seed ${SEED} at $(date)"
