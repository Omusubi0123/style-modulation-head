#!/bin/bash

# Script to evaluate attn_output and mlp_output steering for Qwen2.5-7B-Instruct
# Tests layers 15-25 (Python indices 14-24) with coefficient 1.0
# Captures scores and outputs them to a log file

# set -e  # Exit on any error
set -o pipefail  # Ensure pipelines fail if any command fails

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model> <trait1> <trait2> ..."
    exit 1
fi
MODEL="$1"    # 1番目の引数を MODEL に入れる
shift         # $1 を破棄し、$2以降を $1, $2... にずらす
TRAITS=("$@") # 残った引数すべて（旧$2以降）を配列に入れる
if [ ${#TRAITS[@]} -eq 0 ]; then
    echo "Usage: $0 <model> <trait1> <trait2> ..."
    exit 1
fi
echo "Model: $MODEL"
echo "Traits: ${TRAITS[*]}"

# Layers to test: 15-25 (Python indices 14-24)
# Python index 14 = transformer block layer 15
# Python index 24 = transformer block layer 25
if [ "${MODEL}" == "Qwen/Qwen2.5-7B-Instruct" ]; then
    LAYERS=($(seq 0 27))
elif [ "${MODEL}" == "meta-llama/Llama-3.1-8B-Instruct" ]; then
    LAYERS=($(seq 0 31))
elif [ "${MODEL}" == "Qwen/Qwen3-30B-A3B-Instruct-2507" ]; then
    # LAYERS=($(seq 0 47))
    LAYERS=($(seq 23 42))
else
    echo "Unsupported model: $MODEL"
    exit 1
fi
echo "Layers: ${LAYERS[*]}"

judge_model="gpt-4.1-mini-2025-04-14"
batch_size=100

# Block steering types to test
BLOCK_STEERING_TYPES=("attn_output")
# BLOCK_STEERING_TYPES=("mlp_output")

# GPU assignment (you can modify this based on available GPUs)
GPU=0

# Steering parameters
# COEF=2.5  # Fixed coefficient
COEF=7.5  # Fixed coefficient
# COEF=1.0  # Fixed coefficient
STEERING_TYPE="response"
PERSONA_INSTRUCTION_TYPE="neg"

# Create output directories
mkdir -p logs
mkdir -p data/eval_persona_eval/steering_results_block_output_${PERSONA_INSTRUCTION_TYPE}_${COEF}

# Log file
LOG_FILE="logs/eval_steering_block_output_layers_${PERSONA_INSTRUCTION_TYPE}_${COEF}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting eval_persona_steer_block.py for attn_output and mlp_output layers at $(date)" | tee -a $LOG_FILE
echo "Traits: ${TRAITS[*]}" | tee -a $LOG_FILE
echo "Model: ${MODEL}" | tee -a $LOG_FILE
echo "Layers: ${LAYERS[*]} (transformer blocks $((LAYERS[0]+1))-$((LAYERS[-1]+1)))" | tee -a $LOG_FILE
echo "Block steering types: ${BLOCK_STEERING_TYPES[*]}" | tee -a $LOG_FILE
echo "Coefficient: ${COEF}" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

# Function to check if vector file exists (for block version)
check_vector_file_exists() {
    local trait=$1
    local model=$2
    local block_steering_type=$3
    
    # Map block steering type to vector file suffix
    local vector_file="data/persona_vectors/$model/${trait}_response_avg_diff_${block_steering_type}.pt"
    if [ -f "$vector_file" ]; then
        return 0
    else
        echo "Vector file not found: $vector_file" | tee -a $LOG_FILE
        return 1
    fi
}

# Function to check if evaluation file already exists
check_eval_file_exists() {
    local trait=$1
    local model=$2
    local layer=$3
    local coef=$4
    local persona_instruction_type=$5
    local block_steering_type=$6
    
    local output_path="data/eval_persona_eval/steering_results_block_output_${persona_instruction_type}_${coef}/$model/${trait}_steer_block_${block_steering_type}_${STEERING_TYPE}_${persona_instruction_type}_layer${layer}_coef${coef}.csv"
    if [ -f "$output_path" ]; then
        echo "Evaluation file already exists: $output_path" | tee -a $LOG_FILE
        return 0
    fi
    return 1
}

# Function to run eval_persona_steer_block.py for a trait, model, layer, coefficient, and block steering type
run_eval_steering_block() {
    local trait=$1
    local model=$2
    local layer=$3
    local coef=$4
    local gpu=$5
    local persona_instruction_type=$6
    local block_steering_type=$7

    # Check if vector file exists
    if ! check_vector_file_exists $trait $model $block_steering_type; then
        echo "Skipping eval_persona_steer_block.py for trait: $trait, model: $model, block_steering_type: $block_steering_type (vector file not found)" | tee -a $LOG_FILE
        return 1
    fi
    
    # Check if evaluation file already exists
    if check_eval_file_exists $trait $model $layer $coef $persona_instruction_type $block_steering_type; then
        echo "Skipping eval_persona_steer_block.py for trait: $trait, model: $model, layer: $layer, coef: $coef, block_steering_type: $block_steering_type (file already exists)" | tee -a $LOG_FILE
        return 0
    fi
    
    echo "Running eval_persona_steer_block.py for trait: $trait, model: $model, layer: $layer (transformer block $(($layer + 1))), coef: $coef, block_steering_type: $block_steering_type" | tee -a $LOG_FILE
    
    # Prepare arguments for eval_persona_steer_block.py
    local vector_path="data/persona_vectors/$model/${trait}_response_avg_diff_${block_steering_type}.pt"
    local output_path="data/eval_persona_eval/steering_results_block_output_${persona_instruction_type}_${coef}/$model/${trait}_steer_block_${block_steering_type}_${STEERING_TYPE}_${persona_instruction_type}_layer${layer}_coef${coef}.csv"
    mkdir -p $(dirname $output_path)
    
    # Capture output to filter only CSV path and score lines for the log
    eval_output=$(CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. uv run python eval/eval_persona_steer_block.py \
        --model $model \
        --trait $trait \
        --output_path $output_path \
        --version eval \
        --steering_type $STEERING_TYPE \
        --block_steering_type $block_steering_type \
        --coef $coef \
        --vector_path $vector_path \
        --persona_instruction_type $persona_instruction_type \
        --layer $layer \
        --judge_model $judge_model \
        --batch_size $batch_size \
        2>&1)
    eval_exit_code=$?
    
    # Always echo full output to terminal for user visibility
    echo "$eval_output"
    # Append only CSV path and score lines (e.g., "trait:  97.44 +- 2.62") to the log file
    echo "$eval_output" | awk '/\.csv$/ || /:  [0-9]+\.[0-9]+ \+\- [0-9]+\.[0-9]+/ {print}' >> $LOG_FILE
    
    if [ $eval_exit_code -eq 0 ]; then
        echo "✓ Successfully completed eval_persona_steer_block.py for $trait at layer $layer (transformer block $(($layer + 1))) with block_steering_type $block_steering_type" | tee -a $LOG_FILE
        return 0
    else
        echo "✗ Failed eval_persona_steer_block.py for $trait at layer $layer (transformer block $(($layer + 1))) with block_steering_type $block_steering_type" | tee -a $LOG_FILE
        return 1
    fi
}

# Main execution: Run steering evaluation
echo "=== RUNNING BLOCK OUTPUT STEERING EVALUATIONS ===" | tee -a $LOG_FILE
failed_evaluations=()
skipped_evaluations=()
total_combinations=0
completed_combinations=0

# Generate responses and score
for trait in "${TRAITS[@]}"; do
    echo "Processing trait: $trait, model: $MODEL" | tee -a $LOG_FILE
    
    for block_steering_type in "${BLOCK_STEERING_TYPES[@]}"; do
        # Check if vector file exists for this block steering type
        if ! check_vector_file_exists $trait $MODEL $block_steering_type; then
            echo "Skipping all evaluations for trait: $trait, block_steering_type: $block_steering_type (vector file not found)" | tee -a $LOG_FILE
            continue
        fi
        
        for layer in "${LAYERS[@]}"; do
            total_combinations=$((total_combinations + 1))
            echo "Processing trait: $trait, layer: $layer (transformer block $(($layer + 1))), coef: $COEF, block_steering_type: $block_steering_type" | tee -a $LOG_FILE
            
            if check_eval_file_exists $trait $MODEL $layer $COEF $PERSONA_INSTRUCTION_TYPE $block_steering_type; then
                skipped_evaluations+=("$trait-layer$layer-block$block_steering_type")
            fi
            
            if run_eval_steering_block $trait $MODEL $layer $COEF $GPU $PERSONA_INSTRUCTION_TYPE $block_steering_type; then
                completed_combinations=$((completed_combinations + 1))
            else
                failed_evaluations+=("$trait-layer$layer-block$block_steering_type")
            fi
            
            echo "----------------------------------------" | tee -a $LOG_FILE
        done
    done
done

# Report evaluation results
if [ ${#skipped_evaluations[@]} -gt 0 ]; then
    echo "⏭️  Skipped steering evaluations (files already exist):" | tee -a $LOG_FILE
    for skipped in "${skipped_evaluations[@]}"; do
        echo "  - $skipped" | tee -a $LOG_FILE
    done
fi

# Final summary
echo "" | tee -a $LOG_FILE
echo "=== FINAL SUMMARY ===" | tee -a $LOG_FILE
echo "Total traits: ${#TRAITS[@]}" | tee -a $LOG_FILE
echo "Model: ${MODEL}" | tee -a $LOG_FILE
echo "Total layers: ${#LAYERS[@]} (transformer blocks $((LAYERS[0]+1))-$((LAYERS[-1]+1)))" | tee -a $LOG_FILE
echo "Total block steering types: ${#BLOCK_STEERING_TYPES[@]}" | tee -a $LOG_FILE
echo "Coefficient: ${COEF}" | tee -a $LOG_FILE
echo "Total combinations attempted: $total_combinations" | tee -a $LOG_FILE
echo "Completed combinations: $completed_combinations" | tee -a $LOG_FILE
echo "Skipped evaluations: ${#skipped_evaluations[@]}" | tee -a $LOG_FILE
echo "Failed evaluations: ${#failed_evaluations[@]}" | tee -a $LOG_FILE
echo "Completed at $(date)" | tee -a $LOG_FILE

if [ ${#failed_evaluations[@]} -eq 0 ]; then
    echo "🎉 All block output steering evaluations completed successfully!" | tee -a $LOG_FILE
    exit 0
else
    echo "⚠️  Some block output steering evaluations failed. Check the log file for details." | tee -a $LOG_FILE
    echo "Failed evaluations:" | tee -a $LOG_FILE
    for failed in "${failed_evaluations[@]}"; do
        echo "  - $failed" | tee -a $LOG_FILE
    done
    exit 1
fi

