#!/bin/bash

# Script to run eval_steering.sh for all models, traits, and layers
# Captures scores and outputs them to a log file

# set -e  # Exit on any error
set -o pipefail  # Ensure pipelines fail if any command fails

# Configuration
# TRAITS=("evil" "sycophantic" "hallucinating" "humorous" "optimistic" "apathetic" "impolite" "tangential")
# TRAITS=${TRAITS:-"angel" "polite" "passionate" "conservative" "liberal" "betrayal" "loyalty" "loser" "eco-friendly" "anti-environment"}
TRAITS=("evil" "sycophantic" "hallucinating" "humorous" "passionate" "loser")
MODELS=("Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")
judge_model=${judge_model:-"gpt-4.1-mini-2025-04-14"}
batch_size=${batch_size:-100}

# Trait to layer mapping for each model
declare -A TRAIT_LAYERS
# TRAIT_LAYERS["openai/gpt-oss-20b-evil"]=16
# TRAIT_LAYERS["openai/gpt-oss-20b-sycophantic"]=16
# TRAIT_LAYERS["openai/gpt-oss-20b-hallucinating"]=16
# TRAIT_LAYERS["openai/gpt-oss-20b-humorous"]=16
# TRAIT_LAYERS["openai/gpt-oss-20b-optimistic"]=16
# TRAIT_LAYERS["openai/gpt-oss-20b-tangential"]=16
# TRAIT_LAYERS["openai/gpt-oss-20b-apathetic"]=16
# TRAIT_LAYERS["openai/gpt-oss-20b-impolite"]=16

TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-evil"]=16
TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-sycophantic"]=16
TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-hallucinating"]=16
TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-humorous"]=16
TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-optimistic"]=16
TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-apathetic"]=16
TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-impolite"]=16

TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-angel"]=16
TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-polite"]=16
TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-passionate"]=16
TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-conservative"]=16
TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-liberal"]=16
TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-betrayal"]=16
TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-loyalty"]=16
TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-loser"]=16
TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-eco-friendly"]=16
TRAIT_LAYERS["meta-llama/Llama-3.1-8B-Instruct-anti-environment"]=16

TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-evil"]=20
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-sycophantic"]=20
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-hallucinating"]=16
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-humorous"]=20
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-optimistic"]=20
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-apathetic"]=20
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-impolite"]=20
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-tangential"]=20

TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-angel"]=20
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-polite"]=20
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-passionate"]=20
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-conservative"]=20
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-liberal"]=20
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-betrayal"]=20
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-loyalty"]=20
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-loser"]=20
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-eco-friendly"]=20
TRAIT_LAYERS["Qwen/Qwen2.5-7B-Instruct-anti-environment"]=20

# TRAIT_LAYERS["Qwen/Qwen3-30B-A3B-Instruct-2507-evil"]=32
# TRAIT_LAYERS["Qwen/Qwen3-30B-A3B-Instruct-2507-sycophantic"]=32
# TRAIT_LAYERS["Qwen/Qwen3-30B-A3B-Instruct-2507-hallucinating"]=32
# TRAIT_LAYERS["Qwen/Qwen3-30B-A3B-Instruct-2507-humorous"]=32
# TRAIT_LAYERS["Qwen/Qwen3-30B-A3B-Instruct-2507-optimistic"]=32
# TRAIT_LAYERS["Qwen/Qwen3-30B-A3B-Instruct-2507-apathetic"]=32
# TRAIT_LAYERS["Qwen/Qwen3-30B-A3B-Instruct-2507-impolite"]=32
# TRAIT_LAYERS["Qwen/Qwen3-30B-A3B-Instruct-2507-tangential"]=32

# Model checkpoint paths (you may need to adjust these based on your actual checkpoint structure)
# declare -A MODEL_CHECKPOINTS
# MODEL_CHECKPOINTS["Qwen/Qwen2.5-7B-Instruct"]="./ckpt/Qwen2.5-7B-Instruct/qwen-steering"
# MODEL_CHECKPOINTS["meta-llama/Llama-3.1-8B-Instruct"]="./ckpt/Llama-3.1-8B-Instruct/llama-steering"

# GPU assignment (you can modify this based on available GPUs)
GPU=0

# Steering parameters
# COEFS=(0.0)
# COEFS=(-2.5 -2.0 -1.5 -1.0 -0.5 0.0 0.5 1.0 1.5 2.0 2.5)  # Different coefficients to test
# COEFS=(-2.5 -2.0 -1.5 -1.0 -0.5 0.0)  # Different coefficients to test
# COEFS=(0.5 1.0 1.5 2.0 2.5)  # Different coefficients to test
COEFS=(-6.0 -5.0 -4.0 -3.0)
# COEFS=(-10.0 -7.5 -5.0 -3.0 3.0 5.0 7.5 10.0)  # Different coefficients to test
STEERING_TYPE="response"
PERSONA_INSTRUCTION_TYPE=("pos" "neg")

persona_instruction_type=${PERSONA_INSTRUCTION_TYPE[1]} # neg

# Create output directories
mkdir -p logs
mkdir -p data/eval_persona_eval_neg_0121/steering_results

# Log file
LOG_FILE="logs/eval_steering_all_neg_0121$(date +%Y%m%d_%H%M%S).log"

echo "Starting eval_steering.sh execution for all models, traits, and layers at $(date)" | tee -a $LOG_FILE
echo "Traits: ${TRAITS[*]}" | tee -a $LOG_FILE
echo "Models: ${MODELS[*]}" | tee -a $LOG_FILE
echo "Coefficients: ${COEFS[*]}" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

# Function to check if vector file exists
check_vector_file_exists() {
    local trait=$1
    local model=$2
    
    local vector_file="data/persona_vectors/$model/${trait}_response_avg_diff.pt"
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
    
    local output_path="data/eval_persona_eval_neg_0121/steering_results/$model/${trait}_steer_${STEERING_TYPE}_${persona_instruction_type}_layer${layer}_coef${coef}.csv"
    if [ -f "$output_path" ]; then
        echo "Evaluation file already exists: $output_path" | tee -a $LOG_FILE
        return 0
    fi
    return 1
}

# Function to get intermediate responses file path
get_intermediate_path() {
    local trait=$1
    local model=$2
    local layer=$3
    local coef=$4
    local persona_instruction_type=$5

    local output_path="data/eval_persona_eval_neg_0121/steering_results/$model/${trait}_steer_${STEERING_TYPE}_${persona_instruction_type}_layer${layer}_coef${coef}.csv"
    local dir_path=$(dirname "$output_path")
    local file_name=$(basename "$output_path")
    local name_no_ext=${file_name%.*}
    local ext=${file_name##*.}
    echo "$dir_path/${name_no_ext}_responses.$ext"
}

# Function to check if intermediate responses file exists
check_intermediate_file_exists() {
    local trait=$1
    local model=$2
    local layer=$3
    local coef=$4
    local persona_instruction_type=$5

    local intermediate_path=$(get_intermediate_path "$trait" "$model" "$layer" "$coef" "$persona_instruction_type")
    if [ -f "$intermediate_path" ]; then
        echo "Intermediate responses file already exists: $intermediate_path" | tee -a $LOG_FILE
        return 0
    fi
    return 1
}

# Function to run eval_steering.sh for a trait, model, layer, and coefficient
run_eval_steering() {
    local trait=$1
    local model=$2
    local layer=$3
    local coef=$4
    local gpu=$5

    local persona_instruction_type=$6
    local phase_mode=$7  # "skip_phase3" to run Phase 1 only, "skip_phase1" to run Phase 3 only
    
    # Check if vector file exists
    if ! check_vector_file_exists $trait $model; then
        echo "Skipping eval_steering.sh for trait: $trait, model: $model (vector file not found)" | tee -a $LOG_FILE
        return 1
    fi
    
    # Skip checks depending on phase mode
    if [ "$phase_mode" == "skip_phase3" ]; then
        # Phase 1 only: skip if intermediate already exists
        if check_intermediate_file_exists $trait $model $layer $coef $persona_instruction_type; then
            echo "Skipping Phase 1 (responses already generated) for trait: $trait, model: $model, layer: $layer, coef: $coef, instruction type: $persona_instruction_type" | tee -a $LOG_FILE
            return 0
        fi
    else
        # Default and Phase 3 only: skip if final evaluation file exists
        if check_eval_file_exists $trait $model $layer $coef $persona_instruction_type; then
            echo "Skipping eval_steering.sh for trait: $trait, model: $model, layer: $layer, coef: $coef, instruction type: $persona_instruction_type (file already exists)" | tee -a $LOG_FILE
            return 0
        fi
    fi
    
    echo "Running eval_steering.sh for trait: $trait, model: $model, layer: $layer, coef: $coef, instruction type: $persona_instruction_type, phase_mode: ${phase_mode:-full}" | tee -a $LOG_FILE
    
    # Prepare arguments for eval_steering.sh
    local vector_path="data/persona_vectors/$model/${trait}_response_avg_diff.pt"
    local output_path="data/eval_persona_eval_neg_0121/steering_results/$model/${trait}_steer_${STEERING_TYPE}_${persona_instruction_type}_layer${layer}_coef${coef}.csv"
    mkdir -p $(dirname $output_path)
    
    # Phase flags
    local phase_flags=""
    if [ "$phase_mode" == "skip_phase3" ]; then
        phase_flags="--skip_phase3"
    elif [ "$phase_mode" == "skip_phase1" ]; then
        phase_flags="--skip_phase1"
    fi

    # Capture output to filter only CSV path and score lines for the log
    eval_output=$(CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. uv run python eval/eval_persona_no_vllm.py \
        --model $model \
        --trait $trait \
        --output_path $output_path \
        --version eval \
        --steering_type $STEERING_TYPE \
        --coef $coef \
        --vector_path $vector_path \
        --persona_instruction_type $persona_instruction_type \
        --layer $layer \
        --judge_model $judge_model \
        2>&1)
    eval_exit_code=$?
    
    # Always echo full output to terminal for user visibility
    echo "$eval_output"
    # Append only CSV path and score lines (e.g., "trait:  97.44 +- 2.62") to the log file
    echo "$eval_output" | awk '/\.csv$/ || /:  [0-9]+\.[0-9]+ \+\- [0-9]+\.[0-9]+/ {print}' >> $LOG_FILE
    
    if [ $eval_exit_code -eq 0 ]; then
        echo "✓ Successfully completed eval_steering.sh for $trait with $model at layer $layer with coef $coef and instruction type $persona_instruction_type" | tee -a $LOG_FILE
        return 0
    else
        echo "✗ Failed eval_steering.sh for $trait with $model at layer $layer with coef $coef and instruction type $persona_instruction_type" | tee -a $LOG_FILE
        return 1
    fi
}

# Main execution: Run steering evaluation in two passes
echo "=== RUNNING STEERING EVALUATIONS (PASS 1: Phase 1 only, skip_phase3) ===" | tee -a $LOG_FILE
failed_evaluations=()
skipped_evaluations=()
total_combinations=0
completed_combinations=0

# Generate responses and score
    for trait in "${TRAITS[@]}"; do
        for model in "${MODELS[@]}"; do
            if ! check_vector_file_exists $trait $model; then
                echo "Skipping all evaluations for trait: $trait, model: $model (vector file not found)" | tee -a $LOG_FILE
                continue
            fi

            layer_key="${model}-${trait}"
            layer=${TRAIT_LAYERS[$layer_key]}

            echo "Processing trait: $trait, model: $model, layer: $layer" | tee -a $LOG_FILE

            for coef in "${COEFS[@]}"; do
                total_combinations=$((total_combinations + 1))
                echo "Processing trait: $trait, model: $model, layer: $layer, coef: $coef, persona_instruction_type: $persona_instruction_type (PASS 1)" | tee -a $LOG_FILE

                if check_intermediate_file_exists $trait $model $layer $coef $persona_instruction_type; then
                    skipped_evaluations+=("$trait-$model-layer$layer-coef$coef-pass1")
                fi

                if ! check_vector_file_exists $trait $model; then
                    echo "Skipping run_eval_steering for trait: $trait, model: $model (vector file not found)" | tee -a $LOG_FILE
                    continue
                fi

                if run_eval_steering $trait $model $layer $coef $GPU $persona_instruction_type; then
                    completed_combinations=$((completed_combinations + 1))
                else
                    failed_evaluations+=("$trait-$model-layer$layer-coef$coef-persona_instruction_type$persona_instruction_type-pass1")
                fi

                echo "----------------------------------------" | tee -a $LOG_FILE
            done
        done
    done

echo "=== RUNNING STEERING EVALUATIONS (PASS 2: Phase 3 only, skip_phase1) ===" | tee -a $LOG_FILE

# Report evaluation results
if [ ${#skipped_evaluations[@]} -gt 0 ]; then
    echo "⏭️  Skipped steering evaluations (files already exist):" | tee -a $LOG_FILE
    for skipped in "${skipped_evaluations[@]}"; do
        echo "  - $skipped" | tee -a $LOG_FILE
    done
fi

if [ ${#failed_evaluations[@]} -eq 0 ]; then
    echo "🎉 All steering evaluations completed successfully!" | tee -a $LOG_FILE
    exit 0
else
    echo "⚠️  Some steering evaluations failed. Check the log file for details." | tee -a $LOG_FILE
    exit 1
fi
