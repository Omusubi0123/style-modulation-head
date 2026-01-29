#!/bin/bash

# Script to evaluate head-specific steering
# Tests all heads in specified layers to identify which heads contribute most to persona steering
# Uses attn_pre_o_proj vectors (O projection前のattention出力)

# set -e  # Exit on any error
set -o pipefail  # Ensure pipelines fail if any command fails

# Configuration
# TRAITS=("evil" "sycophantic" "hallucinating" "humorous" "passionate" "loser")
# TRAITS=("angel" "polite" "passionate" "conservative" "liberal" "betrayal" "loyalty" "loser" "eco-friendly" "anti-environment")
TRAITS=("loser")

# MODEL="Qwen/Qwen2.5-7B-Instruct"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

judge_model="gpt-4.1-mini-2025-04-14"
batch_size=100

# Layers to test (specify the layers you want to analyze)
# For Qwen2.5-7B-Instruct: 28 layers (0-27)
# For Llama-3.1-8B-Instruct: 32 layers (0-31)
# LAYERS=(19)
# LAYERS=(13)
LAYERS=(38)

HEADS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27)   # Qwen2.5-7B-Instruct
# HEADS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31)   # Llama-3.1-8B-Instruct
# HEADS=(1 2 3 4 5 27)
# HEADS=(23 28 29 30 31)

# Head indices to test
# For Qwen2.5-7B-Instruct: 28 Q heads (0-27)
# For Llama-3.1-8B-Instruct: 32 Q heads (0-31)
# Set to "all" to test all heads, or specify individual heads like "0,1,2,3"
HEAD_MODE="individual"  # "individual" or "all"
# For individual mode, each head is tested separately
# For all mode, all heads are tested together

# If testing specific heads only (comma-separated)
# SPECIFIC_HEADS="0,5,10,15,20"

# GPU assignment (you can modify this based on available GPUs)
GPU=0

# Steering parameters
COEF=7.0  # Steering coefficient
STEERING_TYPE="response"
PERSONA_INSTRUCTION_TYPE="neg"

# Create output directories
mkdir -p logs
mkdir -p data/eval_persona_eval/steering_results_head_${0
PERSONA_INSTRUCTION_TYPE}_${COEF}

# Log file
LOG_FILE="logs/eval_steering_head_${TRAITS[0]}_${PERSONA_INSTRUCTION_TYPE}_${COEF}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting eval_persona_steer_head.py for head-specific steering at $(date)" | tee -a $LOG_FILE
echo "Traits: ${TRAITS[*]}" | tee -a $LOG_FILE
echo "Model: ${MODEL}" | tee -a $LOG_FILE
echo "Layers: ${LAYERS[*]}" | tee -a $LOG_FILE
echo "Head mode: ${HEAD_MODE}" | tee -a $LOG_FILE
echo "Coefficient: ${COEF}" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

# Get number of attention heads from config
get_num_heads() {
    local model=$1
    local config_file="data/persona_vectors/$model/attn_config.json"
    
    if [ -f "$config_file" ]; then
        num_heads=$(cat "$config_file" | python -c "import json,sys; print(json.load(sys.stdin)['num_attention_heads'])")
        echo $num_heads
    else
        echo "Warning: attn_config.json not found for $model" | tee -a $LOG_FILE
        # Default values
        if [[ "$model" == *"Qwen"* ]]; then
            echo "28"
        elif [[ "$model" == *"Llama"* ]]; then
            echo "32"
        else
            echo "32"
        fi
    fi
}

# Function to check if vector file exists
check_vector_file_exists() {
    local trait=$1
    local model=$2
    
    local vector_file="data/persona_vectors/$model/${trait}_response_avg_diff_attn_pre_o_proj.pt"
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
    local head_indices=$6
    
    # Sanitize head_indices for filename (replace comma with dash)
    local head_str=$(echo $head_indices | tr ',' '-')
    local output_path="data/eval_persona_eval/steering_results_head_${persona_instruction_type}_${coef}/$model/${trait}_steer_head_${STEERING_TYPE}_${persona_instruction_type}_layer${layer}_head${head_str}_coef${coef}.csv"
    if [ -f "$output_path" ]; then
        echo "Evaluation file already exists: $output_path" | tee -a $LOG_FILE
        return 0
    fi
    return 1
}

# Function to run eval_persona_steer_head.py
run_eval_steering_head() {
    local trait=$1
    local model=$2
    local layer=$3
    local coef=$4
    local gpu=$5
    local persona_instruction_type=$6
    local head_indices=$7

    # Check if vector file exists
    if ! check_vector_file_exists $trait $model; then
        echo "Skipping eval_persona_steer_head.py for trait: $trait, model: $model (vector file not found)" | tee -a $LOG_FILE
        return 1
    fi
    
    # Check if evaluation file already exists
    if check_eval_file_exists $trait $model $layer $coef $persona_instruction_type $head_indices; then
        echo "Skipping eval_persona_steer_head.py for trait: $trait, model: $model, layer: $layer, heads: $head_indices (file already exists)" | tee -a $LOG_FILE
        return 0
    fi
    
    echo "Running eval_persona_steer_head.py for trait: $trait, model: $model, layer: $layer, heads: $head_indices, coef: $coef" | tee -a $LOG_FILE
    
    # Prepare arguments
    local vector_path="data/persona_vectors/$model/${trait}_response_avg_diff_attn_pre_o_proj.pt"
    
    # Sanitize head_indices for filename
    local head_str=$(echo $head_indices | tr ',' '-')
    local output_path="data/eval_persona_eval/steering_results_head_${persona_instruction_type}_${coef}/$model/${trait}_steer_head_${STEERING_TYPE}_${persona_instruction_type}_layer${layer}_head${head_str}_coef${coef}.csv"
    mkdir -p $(dirname $output_path)
    
    # Capture output
    eval_output=$(CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. uv run python eval/eval_persona_steer_head.py \
        --model $model \
        --trait $trait \
        --output_path $output_path \
        --version eval \
        --steering_type $STEERING_TYPE \
        --coef $coef \
        --vector_path $vector_path \
        --persona_instruction_type $persona_instruction_type \
        --layer $layer \
        --head_indices "$head_indices" \
        --judge_model $judge_model \
        --batch_size $batch_size \
        2>&1)
    eval_exit_code=$?
    
    # Always echo full output to terminal for user visibility
    echo "$eval_output"
    # Append only CSV path and score lines to the log file
    echo "$eval_output" | awk '/\.csv$/ || /:  [0-9]+\.[0-9]+ \+\- [0-9]+\.[0-9]+/ {print}' >> $LOG_FILE
    
    if [ $eval_exit_code -eq 0 ]; then
        echo "✓ Successfully completed eval_persona_steer_head.py for $trait at layer $layer heads $head_indices" | tee -a $LOG_FILE
        return 0
    else
        echo "✗ Failed eval_persona_steer_head.py for $trait at layer $layer heads $head_indices" | tee -a $LOG_FILE
        return 1
    fi
}

# Get number of heads for the model
NUM_HEADS=$(get_num_heads $MODEL)
echo "Number of attention heads: $NUM_HEADS" | tee -a $LOG_FILE

# Main execution: Run steering evaluation
echo "=== RUNNING HEAD-SPECIFIC STEERING EVALUATIONS ===" | tee -a $LOG_FILE
failed_evaluations=()
skipped_evaluations=()
total_combinations=0
completed_combinations=0

# Generate responses and score
for trait in "${TRAITS[@]}"; do
    echo "Processing trait: $trait, model: $MODEL" | tee -a $LOG_FILE
    
    # Check if vector file exists
    if ! check_vector_file_exists $trait $MODEL; then
        echo "Skipping all evaluations for trait: $trait (vector file not found)" | tee -a $LOG_FILE
        continue
    fi
    
    for layer in "${LAYERS[@]}"; do
        if [ "$HEAD_MODE" == "individual" ]; then
            # Test each head individually
            # for ((head=0; head<$NUM_HEADS; head++)); do
            for head in "${HEADS[@]}"; do
                total_combinations=$((total_combinations + 1))
                echo "Processing trait: $trait, layer: $layer, head: $head, coef: $COEF" | tee -a $LOG_FILE
                
                if check_eval_file_exists $trait $MODEL $layer $COEF $PERSONA_INSTRUCTION_TYPE "$head"; then
                    skipped_evaluations+=("$trait-layer$layer-head$head")
                fi
                
                if run_eval_steering_head $trait $MODEL $layer $COEF $GPU $PERSONA_INSTRUCTION_TYPE "$head"; then
                    completed_combinations=$((completed_combinations + 1))
                else
                    failed_evaluations+=("$trait-layer$layer-head$head")
                fi
                
                echo "----------------------------------------" | tee -a $LOG_FILE
            done
        elif [ "$HEAD_MODE" == "all" ]; then
            # Test all heads together
            total_combinations=$((total_combinations + 1))
            all_heads=$(seq -s, 0 $((NUM_HEADS - 1)))
            echo "Processing trait: $trait, layer: $layer, all heads, coef: $COEF" | tee -a $LOG_FILE
            
            if check_eval_file_exists $trait $MODEL $layer $COEF $PERSONA_INSTRUCTION_TYPE "$all_heads"; then
                skipped_evaluations+=("$trait-layer$layer-all_heads")
            fi
            
            if run_eval_steering_head $trait $MODEL $layer $COEF $GPU $PERSONA_INSTRUCTION_TYPE "$all_heads"; then
                completed_combinations=$((completed_combinations + 1))
            else
                failed_evaluations+=("$trait-layer$layer-all_heads")
            fi
            
            echo "----------------------------------------" | tee -a $LOG_FILE
        fi
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
echo "Total layers: ${#LAYERS[@]}" | tee -a $LOG_FILE
echo "Number of attention heads: $NUM_HEADS" | tee -a $LOG_FILE
echo "Head mode: ${HEAD_MODE}" | tee -a $LOG_FILE
echo "Coefficient: ${COEF}" | tee -a $LOG_FILE
echo "Total combinations attempted: $total_combinations" | tee -a $LOG_FILE
echo "Completed combinations: $completed_combinations" | tee -a $LOG_FILE
echo "Skipped evaluations: ${#skipped_evaluations[@]}" | tee -a $LOG_FILE
echo "Failed evaluations: ${#failed_evaluations[@]}" | tee -a $LOG_FILE
echo "Completed at $(date)" | tee -a $LOG_FILE

if [ ${#failed_evaluations[@]} -eq 0 ]; then
    echo "🎉 All head-specific steering evaluations completed successfully!" | tee -a $LOG_FILE
    exit 0
else
    echo "⚠️  Some head-specific steering evaluations failed. Check the log file for details." | tee -a $LOG_FILE
    echo "Failed evaluations:" | tee -a $LOG_FILE
    for failed in "${failed_evaluations[@]}"; do
        echo "  - $failed" | tee -a $LOG_FILE
    done
    exit 1
fi

