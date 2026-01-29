#!/bin/bash
#
# eval_common.sh - Common functions for steering evaluation scripts
#
# Usage:
#   source scripts/lib/eval_common.sh

# ========== Default Configuration ==========
: ${GPU:=0}
: ${STEERING_TYPE:="response"}
: ${PERSONA_INSTRUCTION_TYPE:="neg"}
: ${JUDGE_MODEL:="gpt-4.1-mini-2025-04-14"}
: ${BATCH_SIZE:=100}
: ${N_PER_QUESTION:=5}

# ========== Logging ==========
setup_logging() {
    local prefix=$1
    mkdir -p logs
    LOG_FILE="logs/${prefix}_$(date +%Y%m%d_%H%M%S).log"
    echo "Log file: $LOG_FILE"
}

log() { echo "$1" | tee -a "$LOG_FILE"; }
log_separator() { log "----------------------------------------"; }

# ========== Common Functions ==========
run_python() {
    local script=$1
    shift
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. uv run python "$script" "$@" 2>&1
}

# Check if vector file exists
check_vector_exists() {
    local path=$1
    if [[ -f "$path" ]]; then
        return 0
    else
        log "Vector file not found: $path"
        return 1
    fi
}

# Check if output file exists
check_output_exists() {
    local path=$1
    if [[ -f "$path" ]]; then
        log "Output file already exists: $path"
        return 0
    fi
    return 1
}

# Get model layers count
get_model_layers() {
    local model=$1
    case "$model" in
        *"Qwen2.5-7B"*) echo "28" ;;
        *"Llama-3.1-8B"*) echo "32" ;;
        *"Qwen3-30B"*) echo "48" ;;
        *) echo "32" ;;
    esac
}

# Get model num_heads
get_num_heads() {
    local model=$1
    local config_file="data/persona_vectors/$model/attn_config.json"
    
    if [[ -f "$config_file" ]]; then
        python3 -c "import json; print(json.load(open('$config_file'))['num_attention_heads'])"
    else
        case "$model" in
            *"Qwen"*) echo "28" ;;
            *"Llama"*) echo "32" ;;
            *) echo "32" ;;
        esac
    fi
}

# ========== Trait-Layer Mapping ==========
# Get best layer for trait-model combination
get_trait_layer() {
    local model=$1
    local trait=$2
    
    # Default layers based on model
    case "$model" in
        *"Llama-3.1-8B"*) echo "16" ;;
        *"Qwen2.5-7B"*) 
            case "$trait" in
                "hallucinating") echo "16" ;;
                *) echo "20" ;;
            esac
            ;;
        *"Qwen3-30B"*) echo "32" ;;
        *) echo "16" ;;
    esac
}

# ========== Summary Functions ==========
print_summary() {
    local total=$1
    local completed=$2
    local skipped=$3
    local failed_count=$4
    shift 4
    local failed=("$@")
    
    log ""
    log "=== FINAL SUMMARY ==="
    log "Total combinations: $total"
    log "Completed: $completed"
    log "Skipped: $skipped"
    log "Failed: $failed_count"
    
    if [[ $failed_count -gt 0 ]]; then
        log "Failed items:"
        for f in "${failed[@]}"; do
            log "  - $f"
        done
    fi
    log "Completed at $(date)"
}

# ========== Eval Runner ==========
# Run standard steering evaluation
run_eval_steering() {
    local model=$1
    local trait=$2
    local layer=$3
    local coef=$4
    local vector_path=$5
    local output_path=$6
    local extra_args="${7:-}"
    
    mkdir -p "$(dirname "$output_path")"
    
    if check_output_exists "$output_path"; then
        return 0
    fi
    
    if ! check_vector_exists "$vector_path"; then
        return 1
    fi
    
    log "Running: trait=$trait layer=$layer coef=$coef"
    
    local output
    output=$(run_python src/eval/eval_persona.py \
        --model "$model" \
        --trait "$trait" \
        --output_path "$output_path" \
        --version eval \
        --steering_type "$STEERING_TYPE" \
        --coef "$coef" \
        --vector_path "$vector_path" \
        --persona_instruction_type "$PERSONA_INSTRUCTION_TYPE" \
        --layer "$layer" \
        --judge_model "$JUDGE_MODEL" \
        --batch_size "$BATCH_SIZE" \
        --n_per_question "$N_PER_QUESTION" \
        $extra_args)
    local exit_code=$?
    
    echo "$output"
    echo "$output" | awk '/\.csv$/ || /:  [0-9]+\.[0-9]+ \+\- [0-9]+\.[0-9]+/ {print}' >> "$LOG_FILE"
    
    return $exit_code
}

# Run block steering evaluation
run_eval_steering_block() {
    local model=$1
    local trait=$2
    local layer=$3
    local coef=$4
    local block_type=$5
    local vector_path=$6
    local output_path=$7
    
    mkdir -p "$(dirname "$output_path")"
    
    if check_output_exists "$output_path"; then
        return 0
    fi
    
    if ! check_vector_exists "$vector_path"; then
        return 1
    fi
    
    log "Running: trait=$trait layer=$layer coef=$coef block_type=$block_type"
    
    local output
    output=$(run_python src/eval/eval_persona_steer_block.py \
        --model "$model" \
        --trait "$trait" \
        --output_path "$output_path" \
        --version eval \
        --steering_type "$STEERING_TYPE" \
        --block_steering_type "$block_type" \
        --coef "$coef" \
        --vector_path "$vector_path" \
        --persona_instruction_type "$PERSONA_INSTRUCTION_TYPE" \
        --layer "$layer" \
        --judge_model "$JUDGE_MODEL" \
        --batch_size "$BATCH_SIZE" \
        --n_per_question "$N_PER_QUESTION")
    local exit_code=$?
    
    echo "$output"
    echo "$output" | awk '/\.csv$/ || /:  [0-9]+\.[0-9]+ \+\- [0-9]+\.[0-9]+/ {print}' >> "$LOG_FILE"
    
    return $exit_code
}

# Run head steering evaluation
run_eval_steering_head() {
    local model=$1
    local trait=$2
    local layer=$3
    local coef=$4
    local head_indices=$5
    local vector_path=$6
    local output_path=$7
    
    mkdir -p "$(dirname "$output_path")"
    
    if check_output_exists "$output_path"; then
        return 0
    fi
    
    if ! check_vector_exists "$vector_path"; then
        return 1
    fi
    
    log "Running: trait=$trait layer=$layer coef=$coef heads=$head_indices"
    
    local output
    output=$(run_python src/eval/eval_persona_steer_head.py \
        --model "$model" \
        --trait "$trait" \
        --output_path "$output_path" \
        --version eval \
        --steering_type "$STEERING_TYPE" \
        --coef "$coef" \
        --vector_path "$vector_path" \
        --persona_instruction_type "$PERSONA_INSTRUCTION_TYPE" \
        --layer "$layer" \
        --head_indices "$head_indices" \
        --judge_model "$JUDGE_MODEL" \
        --batch_size "$BATCH_SIZE" \
        --n_per_question "$N_PER_QUESTION")
    local exit_code=$?
    
    echo "$output"
    echo "$output" | awk '/\.csv$/ || /:  [0-9]+\.[0-9]+ \+\- [0-9]+\.[0-9]+/ {print}' >> "$LOG_FILE"
    
    return $exit_code
}
