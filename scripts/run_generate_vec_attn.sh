#!/bin/bash

# Full evaluation script for persona vectors (attn pre-O projection version)
# Runs generate_vec_attn.py for all traits and models
# This generates vectors from attn_weights @ V (before O projection)

# set -e  # Exit on any error
set -o pipefail  # Ensure pipelines fail if any command fails

# Configuration
# TRAITS=("evil" "apathetic" "hallucinating" "humorous" "impolite" "optimistic" "sycophantic" "tangential")
# TRAITS=("angel" "polite" "passionate" "conservative" "liberal" "betrayal" "loyalty" "loser" "eco-friendly" "anti-environment")
TRAITS=("loser")

MODELS=("Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")
# MODELS=("Qwen/Qwen3-30B-A3B-Instruct-2507")
# MODELS=("meta-llama/Llama-3.1-8B-Instruct")

# GPU assignment (you can modify this based on available GPUs)
GPU=0

# Create output directories
mkdir -p logs
mkdir -p data/eval_persona_extract
mkdir -p data/persona_vectors

# Log file
LOG_FILE="logs/generate_vec_attn_${TRAITS[0]}_${MODELS[0]}_$(date +%Y%m%d_%H%M%S).log"

echo "Starting attn pre-O projection vector generation at $(date)" | tee -a $LOG_FILE
echo "Traits: ${TRAITS[*]}" | tee -a $LOG_FILE
echo "Models: ${MODELS[*]}" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

# Function to check if vector files already exist
check_vector_files_exist() {
    local trait=$1
    local model=$2
    
    # Check if the attn_pre_o_proj vector files exist
    local vector_file="data/persona_vectors/$model/${trait}_response_avg_diff_attn_pre_o_proj.pt"
    
    if [ -f "$vector_file" ]; then
        echo "Attn pre-O projection vector file already exists for trait: $trait, model: $model" | tee -a $LOG_FILE
        return 0
    fi
    return 1
}

# Function to run generate_vec_attn.py for a trait and model
run_generate_vec_attn() {
    local trait=$1
    local model=$2
    local gpu=$3
    
    # Check if vector files already exist (uncomment to enable skip)
    # if check_vector_files_exist $trait $model; then
    #     echo "Skipping generate_vec_attn.py for trait: $trait, model: $model (files already exist)" | tee -a $LOG_FILE
    #     return 0
    # fi
    
    echo "Running generate_vec_attn.py for trait: $trait, model: $model" | tee -a $LOG_FILE
    
    # First, check if pos and neg CSV files exist
    local pos_path="data/eval_persona_extract/$model/${trait}_pos_instruct.csv"
    local neg_path="data/eval_persona_extract/$model/${trait}_neg_instruct.csv"
    
    if [ ! -f "$pos_path" ] || [ ! -f "$neg_path" ]; then
        echo "Warning: CSV files not found for trait: $trait, model: $model" | tee -a $LOG_FILE
        echo "  Expected: $pos_path" | tee -a $LOG_FILE
        echo "  Expected: $neg_path" | tee -a $LOG_FILE
        echo "  Please run eval_persona.py first to generate these files." | tee -a $LOG_FILE
        return 1
    fi
    
    # Now run generate_vec_attn.py
    echo "Running generate_vec_attn.py for trait: $trait, model: $model" | tee -a $LOG_FILE
    
    # Capture output to filter only important lines for the log
    gen_output=$(PYTHONPATH=. CUDA_VISIBLE_DEVICES=$gpu uv run python src/generate_vec_attn.py \
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
    echo "$gen_output" | awk '/Persona vectors saved/ || /Processing/ || /Filtered effective/ || /Loaded model/ || /Attention config/ {print}' >> $LOG_FILE
    if [ $gen_exit_code -eq 0 ]; then
        echo "✓ Successfully completed generate_vec_attn.py for $trait with $model" | tee -a $LOG_FILE
        return 0
    else
        echo "✗ Failed generate_vec_attn.py for $trait with $model" | tee -a $LOG_FILE
        return 1
    fi
}

# Function to save model attention config
save_attn_config() {
    local model=$1
    local gpu=$2
    
    local config_file="data/persona_vectors/$model/attn_config.json"
    
    if [ -f "$config_file" ]; then
        echo "Attention config already exists for model: $model" | tee -a $LOG_FILE
        return 0
    fi
    
    echo "Saving attention config for model: $model" | tee -a $LOG_FILE
    
    config_output=$(PYTHONPATH=. CUDA_VISIBLE_DEVICES=$gpu uv run python src/save_model_attn_config.py \
        --model_name $model \
        --save_dir data/persona_vectors/$model/ \
        2>&1)
    config_exit_code=$?
    
    echo "$config_output"
    
    if [ $config_exit_code -eq 0 ]; then
        echo "✓ Successfully saved attention config for $model" | tee -a $LOG_FILE
        return 0
    else
        echo "✗ Failed to save attention config for $model" | tee -a $LOG_FILE
        return 1
    fi
}

# Phase 0: Save attention config for all models
echo "=== PHASE 0: Saving attention configs ===" | tee -a $LOG_FILE
for model in "${MODELS[@]}"; do
    mkdir -p "data/persona_vectors/$model/"
    save_attn_config $model $GPU
done
echo "----------------------------------------" | tee -a $LOG_FILE

# Phase 1: Generate vectors for all trait-model combinations
echo "=== PHASE 1: Generating attn pre-O projection vectors ===" | tee -a $LOG_FILE
failed_generations=()
skipped_generations=()

for trait in "${TRAITS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Processing trait: $trait, model: $model" | tee -a $LOG_FILE
        
        if ! run_generate_vec_attn $trait $model $GPU; then
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
    echo "✓ All attn pre-O projection vector generations completed successfully!" | tee -a $LOG_FILE
else
    echo "✗ Some attn pre-O projection vector generations failed:" | tee -a $LOG_FILE
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
    echo "🎉 All attn pre-O projection vector generations completed successfully!" | tee -a $LOG_FILE
    exit 0
else
    echo "⚠️  Some attn pre-O projection vector generations failed. Check the log file for details." | tee -a $LOG_FILE
    exit 1
fi

