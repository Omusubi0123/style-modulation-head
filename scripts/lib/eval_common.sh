#!/bin/bash
#
# eval_common.sh - Common functions for evaluation scripts
#
# Usage:
#   source scripts/lib/eval_common.sh
#
# Layer index convention:
#   --layer k = model.layers[k] (0-indexed transformer block)
#   All vector files use the same convention: tensor[k] = model.layers[k]

# ========== Default Configuration (override via environment variables) ==========
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
# Get best layer for trait-model combination (0-indexed transformer block)
get_trait_layer() {
    local model=$1
    local trait=$2
    
    case "$model" in
        *"Llama-3.1-8B"*) echo "15" ;;
        *"Qwen2.5-7B"*) 
            case "$trait" in
                "hallucinating") echo "18" ;;
                *) echo "19" ;;
            esac
            ;;
        *"Qwen3-30B"*) echo "31" ;;
        *) echo "15" ;;
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

# ========== Unified Eval Runner ==========
# Run persona evaluation with any steering mode.
#
# All steering modes use src/eval/eval_persona.py with --steering_target.
#
# Args:
#   $1 - model name
#   $2 - trait
#   $3 - layer (0-indexed transformer block)
#   $4 - coef
#   $5 - vector_path
#   $6 - output_path
#   $7 - steering_target ("layer_output", "attn_output", "mlp_output", "head", etc.)
#   $8 - extra args (optional, e.g., "--head_indices 2,4,27")
run_eval() {
    local model=$1
    local trait=$2
    local layer=$3
    local coef=$4
    local vector_path=$5
    local output_path=$6
    local steering_target=${7:-"layer_output"}
    local extra_args="${8:-}"
    
    mkdir -p "$(dirname "$output_path")"
    
    if check_output_exists "$output_path"; then
        return 0
    fi
    
    if ! check_vector_exists "$vector_path"; then
        return 1
    fi
    
    log "Running: trait=$trait layer=$((layer+1)) (1-indexed) coef=$coef target=$steering_target"
    
    local output
    output=$(run_python src/eval/eval_persona.py \
        --model "$model" \
        --trait "$trait" \
        --output_path "$output_path" \
        --version eval \
        --steering_type "$STEERING_TYPE" \
        --steering_target "$steering_target" \
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

# ========== Convenience Wrappers ==========

# Run layer output (residual stream) steering
run_eval_steering() {
    local model=$1
    local trait=$2
    local layer=$3
    local coef=$4
    local vector_path=$5
    local output_path=$6
    local extra_args="${7:-}"
    
    run_eval "$model" "$trait" "$layer" "$coef" "$vector_path" "$output_path" \
        "layer_output" "$extra_args"
}

# Run block steering
run_eval_steering_block() {
    local model=$1
    local trait=$2
    local layer=$3
    local coef=$4
    local block_type=$5
    local vector_path=$6
    local output_path=$7
    
    run_eval "$model" "$trait" "$layer" "$coef" "$vector_path" "$output_path" \
        "$block_type"
}

# Run head steering
run_eval_steering_head() {
    local model=$1
    local trait=$2
    local layer=$3
    local coef=$4
    local head_indices=$5
    local vector_path=$6
    local output_path=$7
    
    run_eval "$model" "$trait" "$layer" "$coef" "$vector_path" "$output_path" \
        "head" "--head_indices $head_indices"
}
