#!/usr/bin/env bash
#SBATCH --job-name=omnisafe-analysis
#SBATCH --output=slurm_logs/analysis_%j.out
#SBATCH --error=slurm_logs/analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

set -Eeo pipefail

# Usage:
# sbatch slurm/analyze_omnisafe.sbatch                           # analyze all trained models
# ALGORITHMS="PPOLag CPO" sbatch slurm/analyze_omnisafe.sbatch   # analyze specific algorithms
# PCS_MODES="static responsive" sbatch slurm/analyze_omnisafe.sbatch  # specific PCS modes
# SUITES="standard stress" sbatch slurm/analyze_omnisafe.sbatch   # specific test suites

# Repo & PYTHONPATH
REPO="/home/sleemann/energy-net-zoo-robust-safe-iso"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
mkdir -p "$REPO/slurm_logs"
cd "$REPO"

# Load modules
module -q avail python 2>/dev/null || true
module -q load python/3.8 2>/dev/null || true
module -q load cuda/11.8 2>/dev/null || true

# Activate environment
if command -v conda >/dev/null 2>&1; then
  source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
  conda activate energy-net-zoo 2>/dev/null || echo "[warn] conda env missing, trying venv"
fi
if [[ -z "${CONDA_PREFIX:-}" && -d "venv" ]]; then source venv/bin/activate; fi

# Configuration (can be overridden via environment variables)
ALGORITHMS="${ALGORITHMS:-PPOLag CPO CUP SautePPO FOCOPS}"
PCS_MODES="${PCS_MODES:-static responsive}"
SUITES="${SUITES:-standard stress}"
# Auto-detect latest batch or use specified one
if [[ -n "${BATCH_ID:-}" ]]; then
    RESULTS_DIR="${RESULTS_DIR:-runs/omnisafe_trained_${BATCH_ID}}"
    echo "[info] Using specified batch: $BATCH_ID"
else
    # Find the latest batch directory
    LATEST_BATCH=$(find runs -name "omnisafe_trained_*" -type d 2>/dev/null | sort | tail -1)
    if [[ -n "$LATEST_BATCH" ]]; then
        RESULTS_DIR="$LATEST_BATCH"
        BATCH_ID=$(basename "$LATEST_BATCH" | sed 's/omnisafe_trained_//')
        echo "[info] Auto-detected latest batch: $BATCH_ID"
    else
        RESULTS_DIR="${RESULTS_DIR:-runs}"
        echo "[info] No timestamped batches found, using legacy directory: $RESULTS_DIR"
    fi
fi
# OUTPUT_DIR will auto-detect batch if not specified
OUTPUT_DIR="${OUTPUT_DIR:-}"

echo "=========================================="
echo "OmniSafe Analysis Configuration"
echo "=========================================="
echo "Job ID            : ${SLURM_JOB_ID}"
echo "Algorithms        : ${ALGORITHMS}"
echo "PCS Modes         : ${PCS_MODES}"
echo "Test Suites       : ${SUITES}"
echo "Results Directory : ${RESULTS_DIR}"
echo "Output Directory  : ${OUTPUT_DIR}"
echo "Host / GPU        : $(hostname) / $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo cpu)"
echo "Start Time        : $(date)"
echo "=========================================="

# Check if any trained models exist (handle both legacy and new structure)
if [[ -d "$RESULTS_DIR/omnisafe_trained" ]]; then
    # Legacy structure: runs/omnisafe_trained/
    MODELS_DIR="$RESULTS_DIR/omnisafe_trained"
elif [[ -d "$RESULTS_DIR" && $(find "$RESULTS_DIR" -maxdepth 1 -name "*" -type d | wc -l) -gt 1 ]]; then
    # New structure: runs/omnisafe_trained_BATCH/ALGO/PCS/seed/
    MODELS_DIR="$RESULTS_DIR"
else
    echo "[fatal] No trained models directory found"
    echo "Checked: $RESULTS_DIR/omnisafe_trained and $RESULTS_DIR"
    echo "Please run training first:"
    echo "  sbatch --array=0-194 slurm/train_omnisafe_array.sbatch"
    exit 1
fi

# Count available models
model_count=$(find "$MODELS_DIR" -name "evalpack" -type d 2>/dev/null | wc -l)
sb3_count=$(find "$MODELS_DIR" -name "best_model.zip" -type f 2>/dev/null | wc -l)

echo "[info] Found $model_count EvalPack models and $sb3_count SB3 models"

if [[ $model_count -eq 0 && $sb3_count -eq 0 ]]; then
    echo "[fatal] No trained models found!"
    echo "Please ensure training has completed successfully."
    exit 1
fi

echo "[info] Starting OmniSafe analysis..."

# Run the unified analysis
if [ -n "$OUTPUT_DIR" ]; then
    # Use specified output directory
    python3 scripts/omnisafe_analysis.py \
        --enhanced-analysis \
        --episodes 1000 \
        --results-dir "$MODELS_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --algorithms $ALGORITHMS \
        --pcs-modes $PCS_MODES \
        --suites $SUITES \
        --use-cache
else
    # Auto-detect output directory based on batch
    python3 scripts/omnisafe_analysis.py \
        --enhanced-analysis \
        --episodes 1000 \
        --results-dir "$MODELS_DIR" \
        --algorithms $ALGORITHMS \
        --pcs-modes $PCS_MODES \
        --suites $SUITES \
        --use-cache
fi

analysis_status=$?

echo "[info] Analysis finished: $(date)"

if [[ $analysis_status -eq 0 ]]; then
    echo "=========================================="
    echo "Analysis Complete - Results Summary"
    echo "=========================================="
    
    # Show what was generated
    if [[ -d "$OUTPUT_DIR" ]]; then
        echo "Generated files:"
        find "$OUTPUT_DIR" -type f -name "*.png" -o -name "*.pdf" -o -name "*.md" -o -name "*.csv" -o -name "*.tex" | sort
        
        echo ""
        echo "Key outputs:"
        echo "  📊 Plots: $OUTPUT_DIR/plots/"
        echo "  📋 Tables: $OUTPUT_DIR/tables/"
        echo "  📝 Report: $OUTPUT_DIR/benchmark_report.md"
        
        # Show brief results if available
        if [[ -f "$OUTPUT_DIR/tables/metrics_summary.csv" ]]; then
            echo ""
            echo "Top 3 Algorithms:"
            head -4 "$OUTPUT_DIR/tables/metrics_summary.csv" | tail -3 | while IFS=, read -r algo n_runs pass_rate_mean pass_rate_std pass_rate_ci cost_mean cost_std cost_ci reward_mean reward_std reward_ci safety_score rank_safety rank_reward rank_overall; do
                printf "  %s: Pass Rate=%.1f%%, Cost=%.3f, Reward=%.2f\n" "$algo" "$(echo "$pass_rate_mean * 100" | bc -l)" "$cost_mean" "$reward_mean" 2>/dev/null || echo "  $algo"
            done
        fi
    fi
    
    echo ""
    echo "✅ Analysis completed successfully!"
else
    echo "❌ Analysis failed with exit code $analysis_status"
    echo "Check the log files for details:"
    echo "  stdout: slurm_logs/analysis_${SLURM_JOB_ID}.out"
    echo "  stderr: slurm_logs/analysis_${SLURM_JOB_ID}.err"
    exit $analysis_status
fi

# Optional: Send notification (if configured)
if command -v notify-send >/dev/null 2>&1; then
    notify-send "OmniSafe Analysis" "Analysis job ${SLURM_JOB_ID} completed successfully"
fi

echo "[ok] Job completed at $(date)"
