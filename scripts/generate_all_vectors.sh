#!/bin/bash
#
# generate_all_vectors.sh - 全モデル・全トレイトのPersona Vector抽出
#
# 使い方:
#   ./scripts/generate_all_vectors.sh [GPU_ID]
#
# 設定:
#   TRAITS, MODELSを下記で指定

set -o pipefail

# ========== Configuration ==========
TRAITS=("evil" "sycophantic" "hallucinating" "humorous" "loser" "passionate")
MODELS=("Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")
GPU=${1:-0}

# ========== Setup ==========
mkdir -p logs data/eval_persona_extract data/persona_vectors
LOG_FILE="logs/generate_all_vectors_$(date +%Y%m%d_%H%M%S).log"

log() { echo "$1" | tee -a "$LOG_FILE"; }
log "Starting at $(date)"
log "Traits: ${TRAITS[*]}"
log "Models: ${MODELS[*]}"
log "GPU: $GPU"
log "----------------------------------------"

# ========== Helper Functions ==========
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

# ========== Main Processing ==========
failed=()

for model in "${MODELS[@]}"; do
    log "=== Processing model: $model ==="
    mkdir -p "data/persona_vectors/$model/"
    
    # Save attention config (for head-level analysis)
    if [[ ! -f "data/persona_vectors/$model/attn_config.json" ]]; then
        log "Saving attention config..."
        run_python src/save_model_attn_config.py \
            --model_name "$model" \
            --save_dir "data/persona_vectors/$model/" | tee -a "$LOG_FILE"
    fi
    
    for trait in "${TRAITS[@]}"; do
        log "--- Processing trait: $trait ---"
        
        pos_path="data/eval_persona_extract/$model/${trait}_pos_instruct.csv"
        neg_path="data/eval_persona_extract/$model/${trait}_neg_instruct.csv"
        
        # Step 1: Generate pos/neg CSV if not exist
        if ! check_csv_exists "$model" "$trait"; then
            log "Generating pos/neg CSV files..."
            
            run_python src/eval/eval_persona.py \
                --model "$model" \
                --trait "$trait" \
                --output_path "$pos_path" \
                --persona_instruction_type pos \
                --assistant_name "$trait" \
                --judge_model gpt-4.1-mini-2025-04-14 \
                --version extract \
                --max_concurrent_judges 4 | tee -a "$LOG_FILE"
            
            run_python src/eval/eval_persona.py \
                --model "$model" \
                --trait "$trait" \
                --output_path "$neg_path" \
                --persona_instruction_type neg \
                --assistant_name helpful \
                --judge_model gpt-4.1-mini-2025-04-14 \
                --version extract \
                --max_concurrent_judges 4 | tee -a "$LOG_FILE"
        fi
        
        if ! check_csv_exists "$model" "$trait"; then
            log "✗ CSV files not found for $trait-$model"
            failed+=("$trait-$model-csv")
            continue
        fi
        
        # Step 2: Generate all vector types
        # Transformer block output
        log "Generating block output vectors..."
        run_python src/generate_vec.py \
            --model_name "$model" \
            --pos_path "$pos_path" \
            --neg_path "$neg_path" \
            --trait "$trait" \
            --save_dir "data/persona_vectors/$model/" \
            --threshold 50 | tee -a "$LOG_FILE"
        
        # Attention pre-O projection
        log "Generating attention pre-O projection vectors..."
        run_python src/generate_vec_attn.py \
            --model_name "$model" \
            --pos_path "$pos_path" \
            --neg_path "$neg_path" \
            --trait "$trait" \
            --save_dir "data/persona_vectors/$model/" \
            --threshold 50 | tee -a "$LOG_FILE"
        
        # Block-level (attn/mlp input/output, layernorms)
        log "Generating block-level vectors..."
        run_python src/generate_vec_block.py \
            --model_name "$model" \
            --pos_path "$pos_path" \
            --neg_path "$neg_path" \
            --trait "$trait" \
            --save_dir "data/persona_vectors/$model/" \
            --threshold 50 | tee -a "$LOG_FILE"
        
        log "✓ Completed $trait for $model"
        log "----------------------------------------"
    done
done

# ========== Summary ==========
log ""
log "=== FINAL SUMMARY ==="
log "Total traits: ${#TRAITS[@]}"
log "Total models: ${#MODELS[@]}"
log "Failed: ${#failed[@]}"
if [[ ${#failed[@]} -gt 0 ]]; then
    log "Failed items:"
    for f in "${failed[@]}"; do log "  - $f"; done
    exit 1
else
    log "🎉 All vector generations completed successfully!"
    exit 0
fi

