#!/bin/bash

# Script to evaluate Style Head Zero Ablation
# Investigate how generated text changes when cumulatively zero-ablating style heads
# 
# Usage:
#   bash scripts/run_eval_style_head_ablation.sh

set -o pipefail  # Ensure pipelines fail if any command fails

# ==============================================================================
# Configuration
# ==============================================================================

# Models to evaluate
MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
)

# Traits to evaluate
TRAITS=("evil" "sycophantic" "hallucinating" "humorous" "passionate" "loser")

# Judge model
JUDGE_MODEL="gpt-4.1-mini-2025-04-14"

# Generation parameters
BATCH_SIZE=100
N_PER_QUESTION=5
MAX_TOKENS=1000
POSITIONS="all"  # "all", "prompt", "response"

# Persona instruction type: "pos", "neg", or leave empty for no system prompt
PERSONA_INSTRUCTION_TYPE="pos"

# GPU assignment (modify based on available GPUs)
GPU=0

# Output base directory
OUTPUT_BASE_DIR="data/eval_persona_eval/style_head_ablation_${POSITIONS}"

# Data version
VERSION="eval"

# ==============================================================================
# Setup
# ==============================================================================

# Create output directories
mkdir -p logs
mkdir -p "$OUTPUT_BASE_DIR"

# Log file
LOG_FILE="logs/eval_style_head_ablation_${POSITIONS}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting Style Head Ablation Evaluation at $(date)" | tee -a $LOG_FILE
echo "Models: ${MODELS[*]}" | tee -a $LOG_FILE
echo "Traits: ${TRAITS[*]}" | tee -a $LOG_FILE
echo "Persona instruction type: ${PERSONA_INSTRUCTION_TYPE:-none}" | tee -a $LOG_FILE
echo "Positions: ${POSITIONS}" | tee -a $LOG_FILE
echo "Judge model: ${JUDGE_MODEL}" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

# ==============================================================================
# Helper Functions
# ==============================================================================

# Get model short name for CSV path
get_model_short_name() {
    local model=$1
    # Extract the last part after '/' and convert to lowercase
    echo "${model##*/}" | tr '[:upper:]' '[:lower:]'
}

# Check if style head CSV exists
check_style_head_csv() {
    local model=$1
    local short_name=$(get_model_short_name "$model")
    local csv_path="style_head/${short_name}.csv"
    
    if [ -f "$csv_path" ]; then
        echo "Style head CSV found: $csv_path" | tee -a $LOG_FILE
        return 0
    else
        echo "Style head CSV not found: $csv_path" | tee -a $LOG_FILE
        return 1
    fi
}

# Run evaluation for a single model-trait combination
run_evaluation() {
    local model=$1
    local trait=$2
    local gpu=$3
    
    local short_name=$(get_model_short_name "$model")
    local output_dir="${OUTPUT_BASE_DIR}/${short_name}/${trait}"
    
    echo "Running evaluation for model: $model, trait: $trait" | tee -a $LOG_FILE
    echo "Output directory: $output_dir" | tee -a $LOG_FILE
    
    # Build command
    local cmd="CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. uv run python src/eval/eval_style_head_ablation.py"
    cmd+=" --model \"$model\""
    cmd+=" --trait \"$trait\""
    cmd+=" --output_dir \"$output_dir\""
    cmd+=" --positions \"$POSITIONS\""
    cmd+=" --max_tokens $MAX_TOKENS"
    cmd+=" --n_per_question $N_PER_QUESTION"
    cmd+=" --batch_size $BATCH_SIZE"
    cmd+=" --judge_model \"$JUDGE_MODEL\""
    cmd+=" --version \"$VERSION\""
    
    if [ -n "$PERSONA_INSTRUCTION_TYPE" ]; then
        cmd+=" --persona_instruction_type \"$PERSONA_INSTRUCTION_TYPE\""
    fi
    
    # Execute
    echo "Command: $cmd" | tee -a $LOG_FILE
    eval_output=$(eval "$cmd" 2>&1)
    eval_exit_code=$?
    
    # Log output
    echo "$eval_output" | tee -a $LOG_FILE
    
    if [ $eval_exit_code -eq 0 ]; then
        echo "Successfully completed: $model, $trait" | tee -a $LOG_FILE
        return 0
    else
        echo "Failed: $model, $trait" | tee -a $LOG_FILE
        return 1
    fi
}

# ==============================================================================
# Main Execution
# ==============================================================================

echo "=== STARTING STYLE HEAD ABLATION EVALUATIONS ===" | tee -a $LOG_FILE

failed_evaluations=()
completed_evaluations=()
skipped_models=()

for model in "${MODELS[@]}"; do
    echo "" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
    echo "Processing model: $model" | tee -a $LOG_FILE
    echo "========================================" | tee -a $LOG_FILE
    
    # Check if style head CSV exists
    if ! check_style_head_csv "$model"; then
        echo "Skipping model $model (no style head CSV)" | tee -a $LOG_FILE
        skipped_models+=("$model")
        continue
    fi
    
    for trait in "${TRAITS[@]}"; do
        echo "" | tee -a $LOG_FILE
        echo "Processing trait: $trait" | tee -a $LOG_FILE
        echo "----------------------------------------" | tee -a $LOG_FILE
        
        if run_evaluation "$model" "$trait" "$GPU"; then
            completed_evaluations+=("$model:$trait")
        else
            failed_evaluations+=("$model:$trait")
        fi
    done
done

# ==============================================================================
# Summary
# ==============================================================================

echo "" | tee -a $LOG_FILE
echo "=== FINAL SUMMARY ===" | tee -a $LOG_FILE
echo "Total models: ${#MODELS[@]}" | tee -a $LOG_FILE
echo "Total traits: ${#TRAITS[@]}" | tee -a $LOG_FILE
echo "Completed evaluations: ${#completed_evaluations[@]}" | tee -a $LOG_FILE
echo "Failed evaluations: ${#failed_evaluations[@]}" | tee -a $LOG_FILE
echo "Skipped models: ${#skipped_models[@]}" | tee -a $LOG_FILE
echo "Completed at $(date)" | tee -a $LOG_FILE

if [ ${#skipped_models[@]} -gt 0 ]; then
    echo "" | tee -a $LOG_FILE
    echo "Skipped models (no style head CSV):" | tee -a $LOG_FILE
    for skipped in "${skipped_models[@]}"; do
        echo "  - $skipped" | tee -a $LOG_FILE
    done
fi

if [ ${#failed_evaluations[@]} -gt 0 ]; then
    echo "" | tee -a $LOG_FILE
    echo "Failed evaluations:" | tee -a $LOG_FILE
    for failed in "${failed_evaluations[@]}"; do
        echo "  - $failed" | tee -a $LOG_FILE
    done
    exit 1
else
    echo "" | tee -a $LOG_FILE
    echo "All style head ablation evaluations completed successfully!" | tee -a $LOG_FILE
    exit 0
fi
