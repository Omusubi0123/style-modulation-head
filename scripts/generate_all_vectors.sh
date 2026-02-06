#!/bin/bash
#
# generate_all_vectors.sh - Extract persona vectors for all models and traits
#
# Generates:
#   1. Pos/neg CSV files (via eval_persona.py with no steering)
#   2. Residual stream vectors (generate_vec.py)
#   3. Attention pre-O-projection vectors (generate_vec_head.py)
#   4. Block-level vectors (generate_vec_block.py)
#
# Usage:
#   ./scripts/generate_all_vectors.sh <model> <trait1> [trait2 ...]
#
# Example:
#   ./scripts/generate_all_vectors.sh "Qwen/Qwen2.5-7B-Instruct" evil humorous
#
# Configuration is set in the script (can be overridden via environment variables):
#   GPU, THRESHOLD

set -o pipefail

# ========== Arguments ==========
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model> <trait1> [trait2 ...]"
    exit 1
fi

MODEL="$1"
shift
TRAITS=("$@")

# ========== Default Configuration ==========
# Set default values (can be overridden via environment variables)
GPU="${GPU:-0}"
THRESHOLD="${THRESHOLD:-50}"

# ========== Setup ==========
mkdir -p logs data/eval_persona_extract data/persona_vectors
LOG_FILE="logs/generate_all_vectors_$(date +%Y%m%d_%H%M%S).log"

log() { echo "$1" | tee -a "$LOG_FILE"; }
log_separator() { log "----------------------------------------"; }

run_python() {
    local script=$1
    shift
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=$GPU uv run python "$script" "$@" 2>&1
}

check_csv_exists() {
    local model=$1 trait=$2
    local pos="data/eval_persona_extract/$model/${trait}_pos_instruct.csv"
    local neg="data/eval_persona_extract/$model/${trait}_neg_instruct.csv"
    [[ -f "$pos" && -f "$neg" ]]
}

log "Starting at $(date)"
log "Model: $MODEL"
log "Traits: ${TRAITS[*]}"
log "GPU: $GPU"
log_separator

# ========== Main Processing ==========
failed=()
completed=0
total=0

mkdir -p "data/persona_vectors/$MODEL/"

# Save attention config (for head-level analysis)
if [[ ! -f "data/persona_vectors/$MODEL/attn_config.json" ]]; then
    log "Saving attention config..."
    run_python src/save_model_attn_config.py \
        --model_name "$MODEL" \
        --save_dir "data/persona_vectors/$MODEL/" | tee -a "$LOG_FILE"
fi

for trait in "${TRAITS[@]}"; do
    total=$((total + 1))
    log "--- Processing trait: $trait ---"

    pos_path="data/eval_persona_extract/$MODEL/${trait}_pos_instruct.csv"
    neg_path="data/eval_persona_extract/$MODEL/${trait}_neg_instruct.csv"

    # Step 1: Generate pos/neg CSV if not exist
    if ! check_csv_exists "$MODEL" "$trait"; then
        log "Generating pos/neg CSV files..."

        run_python src/eval/eval_persona.py \
            --model "$MODEL" \
            --trait "$trait" \
            --output_path "$pos_path" \
            --persona_instruction_type pos \
            --assistant_name "$trait" \
            --judge_model gpt-4.1-mini-2025-04-14 \
            --version extract \
            --max_concurrent_judges 4 | tee -a "$LOG_FILE"

        run_python src/eval/eval_persona.py \
            --model "$MODEL" \
            --trait "$trait" \
            --output_path "$neg_path" \
            --persona_instruction_type neg \
            --assistant_name helpful \
            --judge_model gpt-4.1-mini-2025-04-14 \
            --version extract \
            --max_concurrent_judges 4 | tee -a "$LOG_FILE"
    fi

    if ! check_csv_exists "$MODEL" "$trait"; then
        log "CSV files not found for $trait"
        failed+=("$trait-csv")
        continue
    fi

    # Step 2: Generate all vector types
    log "Generating residual stream vectors..."
    run_python src/generate_vec/generate_vec.py \
        --model_name "$MODEL" \
        --pos_path "$pos_path" \
        --neg_path "$neg_path" \
        --trait "$trait" \
        --save_dir "data/persona_vectors/$MODEL/" \
        --threshold "$THRESHOLD" | tee -a "$LOG_FILE"

    log "Generating attention pre-O-projection vectors..."
    run_python src/generate_vec/generate_vec_head.py \
        --model_name "$MODEL" \
        --pos_path "$pos_path" \
        --neg_path "$neg_path" \
        --trait "$trait" \
        --save_dir "data/persona_vectors/$MODEL/" \
        --threshold "$THRESHOLD" | tee -a "$LOG_FILE"

    log "Generating block-level vectors..."
    run_python src/generate_vec/generate_vec_block.py \
        --model_name "$MODEL" \
        --pos_path "$pos_path" \
        --neg_path "$neg_path" \
        --trait "$trait" \
        --save_dir "data/persona_vectors/$MODEL/" \
        --threshold "$THRESHOLD" | tee -a "$LOG_FILE"

    completed=$((completed + 1))
    log "Completed $trait"
    log_separator
done

# ========== Summary ==========
log ""
log "=== FINAL SUMMARY ==="
log "Total traits: $total"
log "Completed: $completed"
log "Failed: ${#failed[@]}"
if [[ ${#failed[@]} -gt 0 ]]; then
    log "Failed items:"
    for f in "${failed[@]}"; do log "  - $f"; done
    exit 1
else
    log "All vector generations completed successfully!"
    exit 0
fi
