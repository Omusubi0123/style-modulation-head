#!/bin/bash

# run.sh
# Steering position analysis pipeline
#
# This script runs the full analysis pipeline:
# 1. Transform log file to structured CSV
# 2. Extract and format scores/variances
# 3. Generate Pareto plots
#
# Usage:
#   ./run.sh                           # Use default settings
#   model=llama TRAIT=evil ./run.sh    # Override model and trait

set -e
set -o pipefail

# =============================================================================
# Configuration
# =============================================================================

# Model configuration (can be overridden via environment variables)
model="${model:-qwen}"

# Model name mapping
case "$model" in
    qwen)
        MODEL="${MODEL:-Qwen2.5-7B-Instruct}"
        ;;
    llama)
        MODEL="${MODEL:-Llama-3.1-8B-Instruct}"
        ;;
    *)
        echo "Error: Unknown model '$model'. Use 'qwen' or 'llama'."
        exit 1
        ;;
esac

# Trait to analyze (can be overridden via environment variable)
TRAIT="${TRAIT:-evil}"

# Directory configuration
DATA_DIR="${DATA_DIR:-data/steering_position_plot}"
LOG_DIR="${LOG_DIR:-logs/steering_position_comparison}"

# Input/output paths
INPUT_LOG_FILE="${INPUT_LOG_FILE:-${LOG_DIR}/${model}_${TRAIT}.log}"
FINAL_CSV_FILE="${FINAL_CSV_FILE:-${DATA_DIR}/${MODEL}/steering_position_comparison_${model}_${TRAIT}_formatted.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATA_DIR}/${MODEL}/plots_${TRAIT}}"

# Temporary files
TMP_DIR="${TMP_DIR:-logs/tmp}"
TMP_CSV_FILE="${TMP_DIR}/${model}_${TRAIT}_raw.csv"

# =============================================================================
# Setup
# =============================================================================

echo "=============================================="
echo "Steering Position Analysis Pipeline"
echo "=============================================="
echo "Model: ${MODEL} (${model})"
echo "Trait: ${TRAIT}"
echo "Input log: ${INPUT_LOG_FILE}"
echo "Output CSV: ${FINAL_CSV_FILE}"
echo "Output plots: ${OUTPUT_DIR}"
echo "----------------------------------------------"

# Create directories
mkdir -p "${TMP_DIR}"
mkdir -p "$(dirname "${FINAL_CSV_FILE}")"
mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# Step 1: Transform log to CSV
# =============================================================================

echo ""
echo "[Step 1/3] Transforming log file to CSV..."

if [ ! -f "${INPUT_LOG_FILE}" ]; then
    echo "Error: Input log file not found: ${INPUT_LOG_FILE}"
    exit 1
fi

uv run python src/pareto_analysis/transform_log_to_csv.py \
    --input_path "${INPUT_LOG_FILE}" \
    --output_path "${TMP_CSV_FILE}" \
    --extract_trait "${TRAIT}"

# =============================================================================
# Step 2: Extract and format scores
# =============================================================================

echo ""
echo "[Step 2/3] Extracting and formatting scores..."

uv run python src/pareto_analysis/split_scores_variances.py \
    --input_file "${TMP_CSV_FILE}" \
    --output_file "${FINAL_CSV_FILE}"

# =============================================================================
# Step 3: Generate Pareto plots
# =============================================================================

echo ""
echo "[Step 3/3] Generating Pareto plots..."

uv run python src/pareto_analysis/plot_pareto_curve.py single \
    --input_file "${FINAL_CSV_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --trait "${TRAIT}"

# =============================================================================
# Cleanup
# =============================================================================

echo ""
echo "[Cleanup] Removing temporary files..."
rm -f "${TMP_CSV_FILE}"

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================="
echo "Pipeline completed successfully!"
echo "=============================================="
echo "Output files:"
echo "  CSV: ${FINAL_CSV_FILE}"
echo "  Plots: ${OUTPUT_DIR}/"
echo ""
