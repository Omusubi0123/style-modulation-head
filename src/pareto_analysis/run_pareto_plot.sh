#!/bin/bash

# run_pareto_plot.sh
# Generate Pareto curve plots from pre-processed CSV data
#
# Usage:
#   ./run_pareto_plot.sh                           # All models, all traits, linear scale
#   ./run_pareto_plot.sh --model llama             # Llama only
#   ./run_pareto_plot.sh --model qwen              # Qwen only
#   ./run_pareto_plot.sh --traits evil,sycophantic # Specific traits only
#   ./run_pareto_plot.sh --linear-only             # Linear scale only
#   ./run_pareto_plot.sh --log-only                # Log scale only
#   ./run_pareto_plot.sh --both                    # Both linear and log scales
#   ./run_pareto_plot.sh --show-coef-labels        # Show coefficient labels on points

set -e
set -o pipefail

# =============================================================================
# Default Configuration
# =============================================================================

MODELS=("llama" "qwen")
TRAITS="evil,sycophantic,hallucinating,humorous,passionate"
SHOW_COEF_LABELS="False"
PLOT_MODE="linear"  # linear, log, or both
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
        --linear-only)
            PLOT_MODE="linear"
            shift
            ;;
        --log-only)
            PLOT_MODE="log"
            shift
            ;;
        --both)
            PLOT_MODE="both"
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
            echo "  --linear-only          Generate only linear scale plots (default)"
            echo "  --log-only             Generate only log scale plots"
            echo "  --both                 Generate both linear and log scale plots"
            echo "  --data-dir DIR         Data directory (default: data/steering_position_plot)"
            echo "  --no-pdf               Skip PDF generation"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # All models, all traits, linear scale"
            echo "  $0 --model qwen                 # Qwen only"
            echo "  $0 --traits evil,sycophantic    # Specific traits only"
            echo "  $0 --both                       # Both linear and log scales"
            echo "  $0 --log-only --show-coef-labels"
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
echo "Plot mode: $PLOT_MODE"
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
    
    case $PLOT_MODE in
        both)
            uv run python src/pareto_analysis/plot_pareto_curve.py both \
                --data_dir "$DATA_DIR" \
                --output_base_dir "$DATA_DIR" \
                --model "$model" \
                --traits "$TRAITS" \
                --show_coef_labels "$SHOW_COEF_LABELS" \
                --save_pdf "$SAVE_PDF"
            ;;
        linear)
            uv run python src/pareto_analysis/plot_pareto_curve.py all \
                --data_dir "$DATA_DIR" \
                --output_base_dir "$DATA_DIR" \
                --model "$model" \
                --traits "$TRAITS" \
                --show_coef_labels "$SHOW_COEF_LABELS" \
                --use_log_scale "False" \
                --save_pdf "$SAVE_PDF"
            ;;
        log)
            uv run python src/pareto_analysis/plot_pareto_curve.py all \
                --data_dir "$DATA_DIR" \
                --output_base_dir "$DATA_DIR" \
                --model "$model" \
                --traits "$TRAITS" \
                --show_coef_labels "$SHOW_COEF_LABELS" \
                --use_log_scale "True" \
                --save_pdf "$SAVE_PDF"
            ;;
    esac
    
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
    
    case $PLOT_MODE in
        both)
            echo "  - ${DATA_DIR}/${MODEL_NAME}/pareto_plots/"
            echo "  - ${DATA_DIR}/${MODEL_NAME}/pareto_plots_log/"
            ;;
        linear)
            echo "  - ${DATA_DIR}/${MODEL_NAME}/pareto_plots/"
            ;;
        log)
            echo "  - ${DATA_DIR}/${MODEL_NAME}/pareto_plots_log/"
            ;;
    esac
done

exit 0
