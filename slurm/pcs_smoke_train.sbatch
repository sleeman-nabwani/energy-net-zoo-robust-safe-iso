#!/bin/bash
#SBATCH --job-name=safeiso_pcs_smoke_train
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-29%6

ALGOS=(PPOLag CUP CPO FOCOPS SautePPO)
MODES=(cmdp mdp)
PCS_LIST=("static:0.0" "static:10.0" "responsive:spread_prop?gain=1.0&clip=15")

# Resolve index
a=$(( SLURM_ARRAY_TASK_ID % ${#ALGOS[@]} ))
m=$(( (SLURM_ARRAY_TASK_ID / ${#ALGOS[@]}) % ${#MODES[@]} ))
p=$(( SLURM_ARRAY_TASK_ID / (${#ALGOS[@]}*${#MODES[@]}) ))

ALGO="${ALGOS[$a]}"; MODE="${MODES[$m]}"; PCS="${PCS_LIST[$p]}"

CONDA_ENV="${CONDA_ENV:-energy-net-zoo}"
SAVE_ROOT="${SAVE_ROOT:-runs/pcs_smoke_fixcheck}"
STEPS="${STEPS:-48000}"
COST_LIMIT="${COST_LIMIT:-0.10}"
HORIZON="${HORIZON:-48}"
PRESET="${PRESET:-default}"
DEVICE="${DEVICE:-cuda}"

if [[ "$MODE" == "cmdp" ]]; then MF="--cmdp"; MT="cmdp"; else MF="--no-cmdp"; MT="mdp"; fi
TAG=$(echo "$PCS" | sed 's|/|-|g' | sed 's|:|_|g' | sed 's|?||g' | sed 's|&|_|g')
RUN_DIR="${SAVE_ROOT}/${ALGO}_${MT}_${PRESET}_${TAG}/seed0"

source "$HOME/miniconda3/etc/profile.d/conda.sh" || true
conda activate "$CONDA_ENV"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$HOME/energy-net-zoo-robust-safe-iso:$PYTHONPATH"

echo "Starting: algo=$ALGO mode=$MODE pcs=$PCS device=$DEVICE"

stdbuf -oL -eL python -u -m safeiso.train.train_omnisafe \
  --algo "$ALGO" --steps "$STEPS" --seed 0 --device "$DEVICE" $MF \
  --cost_limit "$COST_LIMIT" --pcs "$PCS" --preset "$PRESET" \
  --max_episode_steps "$HORIZON" --eval_every "$STEPS" --eval_episodes 0 \
  --save_dir "$RUN_DIR"

echo "Completed: $ALGO $MODE $PCS"
