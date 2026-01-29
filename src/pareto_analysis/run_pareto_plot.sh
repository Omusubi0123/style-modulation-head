#!/bin/bash

# run_pareto_plot.sh
# Generate Pareto curve plots from pre-processed CSV data
#
# Usage:
#   ./run_pareto_plot.sh                           # All models, all traits
#   ./run_pareto_plot.sh --model llama             # Llama only
#   ./run_pareto_plot.sh --model qwen              # Qwen only
#   ./run_pareto_plot.sh --traits evil,sycophantic # Specific traits only
#   ./run_pareto_plot.sh --show-coef-labels        # Show coefficient labels on points

set -e
set -o pipefail

# =============================================================================
# Default Configuration
# =============================================================================

MODELS=("llama" "qwen")
TRAITS="evil,sycophantic,hallucinating,humorous,passionate"
SHOW_COEF_LABELS="False"
DATA_DIR="data/steering_position_plot"
SAVE_PDF="True"

# =============================================================================
# Parse Command Line Arguments
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODELS=("$2")
            shift 2
            ;;
        --traits)
            TRAITS="$2"
            shift 2
            ;;
        --show-coef-labels)
            SHOW_COEF_LABELS="True"
            shift
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --no-pdf)
            SAVE_PDF="False"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL          Model to process (llama or qwen)"
            echo "  --traits TRAITS        Comma-separated list of traits"
            echo "  --show-coef-labels     Show coefficient labels on each point"
            echo "  --data-dir DIR         Data directory (default: data/steering_position_plot)"
            echo "  --no-pdf               Skip PDF generation"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # All models, all traits"
            echo "  $0 --model qwen                 # Qwen only"
            echo "  $0 --traits evil,sycophantic    # Specific traits only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# =============================================================================
# Main Execution
# =============================================================================

echo "===== Pareto Curve Plot Generation ====="
echo "Models: ${MODELS[*]}"
echo "Traits: $TRAITS"
echo "Show coefficient labels: $SHOW_COEF_LABELS"
echo "Data directory: $DATA_DIR"
echo ""

# Navigate to repository root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Process each model
for model in "${MODELS[@]}"; do
    echo "=========================================="
    echo ">>> Processing model: $model"
    echo "=========================================="
    
    uv run python src/pareto_analysis/plot_pareto_curve.py all \
        --data_dir "$DATA_DIR" \
        --output_base_dir "$DATA_DIR" \
        --model "$model" \
        --traits "$TRAITS" \
        --show_coef_labels "$SHOW_COEF_LABELS" \
        --save_pdf "$SAVE_PDF"
    
    echo ""
    echo ">>> Completed: $model"
    echo ""
done

# =============================================================================
# Summary
# =============================================================================

echo "===== All Plots Generated ====="
echo ""
echo "Output directories:"
for model in "${MODELS[@]}"; do
    if [ "$model" = "llama" ]; then
        MODEL_NAME="Llama-3.1-8B-Instruct"
    elif [ "$model" = "qwen" ]; then
        MODEL_NAME="Qwen2.5-7B-Instruct"
    fi
    echo "  - ${DATA_DIR}/${MODEL_NAME}/pareto_plots/"
done

exit 0
