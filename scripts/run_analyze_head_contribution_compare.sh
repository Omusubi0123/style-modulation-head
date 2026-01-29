#!/bin/bash

# Script to analyze head contribution across multiple traits at specific layers
# Generates heatmaps with traits on Y-axis and heads on X-axis
# This helps identify if the same heads are important across different personas

# set -e  # Exit on any error
set -o pipefail  # Ensure pipelines fail if any command fails

# Configuration
# TRAITS=("evil" "apathetic" "hallucinating" "humorous" "impolite" "optimistic" "sycophantic" "tangential")
# TRAITS=("evil" "apathetic" "hallucinating" "humorous" "impolite" "optimistic" "sycophantic" "tangential" "angel" "polite" "passionate" "conservative" "liberal" "betrayal" "loyalty" "loser" "eco-friendly" "anti-environment")
TRAITS=("evil" "sycophantic" "hallucinating" "humorous" "passionate" "loser")

# MODELS=("Qwen/Qwen2.5-7B-Instruct")
# MODELS=("Qwen/Qwen3-30B-A3B-Instruct-2507")
MODELS=("meta-llama/Llama-3.1-8B-Instruct")

# Vector type to analyze
VECTOR_TYPE="response_avg"

# Layers to analyze (specify multiple layers for comparison)
# For Qwen2.5-7B-Instruct: 28 layers (0-27)
# For Llama-3.1-8B-Instruct: 32 layers (0-31)
# LAYERS=(9 10 14 18 19 20 27) # Qwen2.5-7B-Instruct
# LAYERS=(19)
# LAYERS=(15 16 17 21 22 23) # Qwen2.5-7B-Instruct 他
# LAYERS=(12 13 14 31)  # Llama-3.1-8B-Instruct
LAYERS=(13)
# LAYERS=(26 30 31 37 38) # Qwen3/-30B-A3B-Instruct-2507
# LAYERS=(38)

# GPU assignment
GPU=0

# Create output directories
mkdir -p logs
mkdir -p data/head_analysis/

# Log file
LOG_FILE="logs/analyze_head_contribution_compare_$(date +%Y%m%d_%H%M%S).log"

echo "Starting cross-trait head contribution analysis at $(date)" | tee -a $LOG_FILE
echo "Traits: ${TRAITS[*]}" | tee -a $LOG_FILE
echo "Models: ${MODELS[*]}" | tee -a $LOG_FILE
echo "Layers: ${LAYERS[*]}" | tee -a $LOG_FILE
echo "Vector type: ${VECTOR_TYPE}" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

# Function to check if output file already exists
check_output_exists() {
    local model=$1
    local layer=$2
    
    local output_file="data/head_analysis//$model/paper_no_log_no_zscore/traits_comparison_inner_product_no_log_no_zscore_layer${layer}_${VECTOR_TYPE}.png"
    
    if [ -f "$output_file" ]; then
        echo "Output file already exists: $output_file" | tee -a $LOG_FILE
        return 0
    fi
    return 1
}

# Function to run analyze_head_contribution.py in compare mode
run_analyze_compare() {
    local model=$1
    local layer=$2
    local gpu=$3
    
    # Check if output already exists (uncomment to enable skip)
    # if check_output_exists $model $layer; then
    #     echo "Skipping analysis for model: $model, layer: $layer (output already exists)" | tee -a $LOG_FILE
    #     return 0
    # fi
    
    echo "Running cross-trait analysis for model: $model, layer: $layer" | tee -a $LOG_FILE
    
    # Prepare arguments
    local vector_dir="data/persona_vectors/$model/"
    local output_dir="data/head_analysis/$model/paper_no_log_no_zscore"
    mkdir -p $output_dir
    
    # Build traits argument
    local traits_arg="${TRAITS[@]}"
    
    # Capture output
    analysis_output=$(CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. uv run python src/analyze_head_contribution.py \
        --model_name $model \
        --vector_dir $vector_dir \
        --output_dir $output_dir \
        --traits $traits_arg \
        --layer $layer \
        --vector_type $VECTOR_TYPE \
        --mode compare \
        --no_log \
        --no_zscore \
        2>&1)
        # --no_log \
    analysis_exit_code=$?
    
    # Always echo full output to terminal for user visibility
    echo "$analysis_output"
    # Append important lines to the log file
    echo "$analysis_output" | awk '/Heatmap saved/ || /Head [0-9]+:/ || /vs/ || /Top 10 heads/ {print}' >> $LOG_FILE
    
    if [ $analysis_exit_code -eq 0 ]; then
        echo "✓ Successfully completed cross-trait analysis for $model at layer $layer" | tee -a $LOG_FILE
        return 0
    else
        echo "✗ Failed cross-trait analysis for $model at layer $layer" | tee -a $LOG_FILE
        return 1
    fi
}

# Main execution
echo "=== RUNNING CROSS-TRAIT HEAD CONTRIBUTION ANALYSIS ===" | tee -a $LOG_FILE
failed_analyses=()
completed_analyses=0
total_analyses=0

for model in "${MODELS[@]}"; do
    echo "Processing model: $model" | tee -a $LOG_FILE
    
    for layer in "${LAYERS[@]}"; do
        total_analyses=$((total_analyses + 1))
        echo "Processing layer: $layer" | tee -a $LOG_FILE
        
        if run_analyze_compare $model $layer $GPU; then
            completed_analyses=$((completed_analyses + 1))
        else
            failed_analyses+=("$model-layer$layer")
        fi
        
        echo "----------------------------------------" | tee -a $LOG_FILE
    done
done

# Final summary
echo "" | tee -a $LOG_FILE
echo "=== FINAL SUMMARY ===" | tee -a $LOG_FILE
echo "Total traits compared: ${#TRAITS[@]}" | tee -a $LOG_FILE
echo "Total models: ${#MODELS[@]}" | tee -a $LOG_FILE
echo "Total layers: ${#LAYERS[@]}" | tee -a $LOG_FILE
echo "Total analyses: $total_analyses" | tee -a $LOG_FILE
echo "Completed analyses: $completed_analyses" | tee -a $LOG_FILE
echo "Failed analyses: ${#failed_analyses[@]}" | tee -a $LOG_FILE
echo "Vector type: ${VECTOR_TYPE}" | tee -a $LOG_FILE
echo "Output directory: data/head_analysis/" | tee -a $LOG_FILE
echo "Completed at $(date)" | tee -a $LOG_FILE

if [ ${#failed_analyses[@]} -eq 0 ]; then
    echo "🎉 All cross-trait head contribution analyses completed successfully!" | tee -a $LOG_FILE
    echo "" | tee -a $LOG_FILE
    echo "Output files:" | tee -a $LOG_FILE
    for model in "${MODELS[@]}"; do
        echo "  data/head_analysis/$model/" | tee -a $LOG_FILE
        for layer in "${LAYERS[@]}"; do
            echo "    - traits_comparison_inner_product_layer${layer}_${VECTOR_TYPE}.png" | tee -a $LOG_FILE
        done
    done
    exit 0
else
    echo "⚠️  Some cross-trait analyses failed. Check the log file for details." | tee -a $LOG_FILE
    echo "Failed analyses:" | tee -a $LOG_FILE
    for failed in "${failed_analyses[@]}"; do
        echo "  - $failed" | tee -a $LOG_FILE
    done
    exit 1
fi

