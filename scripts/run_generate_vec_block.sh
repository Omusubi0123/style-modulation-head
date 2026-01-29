#!/bin/bash

# Full evaluation script for persona vectors (block version)
# Runs generate_vec_block.py for all traits and models

# set -e  # Exit on any error
set -o pipefail  # Ensure pipelines fail if any command fails

# Configuration
# TRAITS=("evil" "apathetic" "hallucinating" "humorous" "impolite" "optimistic" "sycophantic" "tangential")
# TRAITS=("angel" "polite" "passionate" "conservative" "liberal" "betrayal" "loyalty" "loser" "eco-friendly" "anti-environment")
TRAITS=("loser")

MODELS=("Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")
# MODELS=("meta-llama/Llama-3.1-8B-Instruct")
# MODELS=("Qwen/Qwen3-30B-A3B-Instruct-2507")
# MODELS=("google/gemma-3-27b-it")

# GPU assignment (you can modify this based on available GPUs)
GPU=0

# Create output directories
mkdir -p logs
mkdir -p data/eval_persona_extract
mkdir -p data/persona_vectors

# Log file
LOG_FILE="logs/generate_vec_block_${TRAITS[0]}_${MODELS[0]}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting block vector generation at $(date)" | tee -a $LOG_FILE
echo "Traits: ${TRAITS[*]}" | tee -a $LOG_FILE
echo "Models: ${MODELS[*]}" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

# Function to check if vector files already exist (for block version)
check_vector_files_exist() {
    local trait=$1
    local model=$2
    
    # Check if at least one of the block vector files exists
    local vector_file_attn="data/persona_vectors/$model/${trait}_response_avg_diff_attn.pt"
    local vector_file_mlp="data/persona_vectors/$model/${trait}_response_avg_diff_mlp.pt"
    local vector_file_attn_ln="data/persona_vectors/$model/${trait}_response_avg_diff_attn_layernorm.pt"
    local vector_file_mlp_ln="data/persona_vectors/$model/${trait}_response_avg_diff_mlp_layernorm.pt"
    
    if [ -f "$vector_file_attn" ] || [ -f "$vector_file_mlp" ] || [ -f "$vector_file_attn_ln" ] || [ -f "$vector_file_mlp_ln" ]; then
        echo "Block vector files already exist for trait: $trait, model: $model" | tee -a $LOG_FILE
        return 0
    fi
    return 1
}

# Function to run generate_vec_block.py for a trait and model
run_generate_vec_block() {
    local trait=$1
    local model=$2
    local gpu=$3
    
    # Check if vector files already exist
    # if check_vector_files_exist $trait $model; then
    #     echo "Skipping generate_vec_block.py for trait: $trait, model: $model (files already exist)" | tee -a $LOG_FILE
    #     return 0
    # fi
    
    echo "Running generate_vec_block.py for trait: $trait, model: $model" | tee -a $LOG_FILE
    
    # First, run eval_persona.py to generate pos and neg CSV files if they don't exist
    local pos_path="data/eval_persona_extract/$model/${trait}_pos_instruct.csv"
    local neg_path="data/eval_persona_extract/$model/${trait}_neg_instruct.csv"
    
    # Now run generate_vec_block.py
    echo "Running generate_vec_block.py for trait: $trait, model: $model" | tee -a $LOG_FILE
    
    # Capture output to filter only important lines for the log
    gen_output=$(PYTHONPATH=. CUDA_VISIBLE_DEVICES=$gpu uv run python src/generate_vec_block.py \
        --model_name $model \
        --pos_path $pos_path \
        --neg_path $neg_path \
        --trait $trait \
        --save_dir data/persona_vectors/$model/ \
        --threshold 50 \
        2>&1)
    gen_exit_code=$?
    # Always echo full output to terminal for user visibility
    echo "$gen_output"
    # Append only important lines to the log file
    echo "$gen_output" | awk '/Persona vectors saved/ || /Processing/ || /Filtered effective/ || /Loaded model/ {print}' >> $LOG_FILE
    if [ $gen_exit_code -eq 0 ]; then
        echo "✓ Successfully completed generate_vec_block.py for $trait with $model" | tee -a $LOG_FILE
        return 0
    else
        echo "✗ Failed generate_vec_block.py for $trait with $model" | tee -a $LOG_FILE
        return 1
    fi
}

# Phase 1: Generate vectors for all trait-model combinations
echo "=== PHASE 1: Generating block vectors ===" | tee -a $LOG_FILE
failed_generations=()
skipped_generations=()

for trait in "${TRAITS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Processing trait: $trait, model: $model" | tee -a $LOG_FILE
        
        if ! run_generate_vec_block $trait $model $GPU; then
            failed_generations+=("$trait-$model")
        fi
        
        echo "----------------------------------------" | tee -a $LOG_FILE
    done
done

# Report generation results
if [ ${#skipped_generations[@]} -gt 0 ]; then
    echo "⏭️  Skipped vector generations (files already exist):" | tee -a $LOG_FILE
    for skipped in "${skipped_generations[@]}"; do
        echo "  - $skipped" | tee -a $LOG_FILE
    done
fi

if [ ${#failed_generations[@]} -eq 0 ]; then
    echo "✓ All block vector generations completed successfully!" | tee -a $LOG_FILE
else
    echo "✗ Some block vector generations failed:" | tee -a $LOG_FILE
    for failed in "${failed_generations[@]}"; do
        echo "  - $failed" | tee -a $LOG_FILE
    done
fi

# Final summary
echo "" | tee -a $LOG_FILE
echo "=== FINAL SUMMARY ===" | tee -a $LOG_FILE
echo "Total traits: ${#TRAITS[@]}" | tee -a $LOG_FILE
echo "Total models: ${#MODELS[@]}" | tee -a $LOG_FILE
echo "Total combinations: $((${#TRAITS[@]} * ${#MODELS[@]}))" | tee -a $LOG_FILE
echo "Skipped generations: ${#skipped_generations[@]}" | tee -a $LOG_FILE
echo "Failed generations: ${#failed_generations[@]}" | tee -a $LOG_FILE
echo "Completed at $(date)" | tee -a $LOG_FILE

if [ ${#failed_generations[@]} -eq 0 ]; then
    echo "🎉 All block vector generations completed successfully!" | tee -a $LOG_FILE
    exit 0
else
    echo "⚠️  Some block vector generations failed. Check the log file for details." | tee -a $LOG_FILE
    exit 1
fi

