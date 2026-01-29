#!/bin/bash
#
# run_eval_steering_block.sh - Block位置別のステアリング評価
#
# 使い方:
#   ./scripts/run_eval_steering_block.sh <model> <trait1> [trait2 ...]
#
# 例:
#   ./scripts/run_eval_steering_block.sh "Qwen/Qwen2.5-7B-Instruct" evil humorous

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/eval_common.sh"

# ========== Arguments ==========
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <model> <trait1> [trait2 ...]"
    exit 1
fi

MODEL="$1"
shift
TRAITS=("$@")

# ========== Configuration ==========
# Layer range based on model
num_layers=$(get_model_layers "$MODEL")
LAYERS=($(seq 0 $((num_layers - 1))))

# Block steering types
BLOCK_TYPES=("attn_output" "mlp_output")
COEF=2.5

OUTPUT_DIR="data/eval_persona_eval/steering_results_block_${PERSONA_INSTRUCTION_TYPE}_${COEF}"

# ========== Setup ==========
setup_logging "eval_steering_block"
log "Starting block steering evaluation at $(date)"
log "Model: $MODEL"
log "Traits: ${TRAITS[*]}"
log "Layers: 0-$((num_layers - 1))"
log "Block types: ${BLOCK_TYPES[*]}"
log "Coefficient: $COEF"
log_separator

# ========== Main Loop ==========
failed=()
skipped=0
completed=0
total=0

for trait in "${TRAITS[@]}"; do
    for block_type in "${BLOCK_TYPES[@]}"; do
        vector_path="data/persona_vectors/$MODEL/${trait}_response_avg_diff_${block_type}.pt"
        
        if ! check_vector_exists "$vector_path"; then
            log "Skipping $trait $block_type (vector not found)"
            continue
        fi
        
        for layer in "${LAYERS[@]}"; do
            total=$((total + 1))
            output_path="$OUTPUT_DIR/$MODEL/${trait}_steer_block_${block_type}_${STEERING_TYPE}_${PERSONA_INSTRUCTION_TYPE}_layer${layer}_coef${COEF}.csv"
            
            if run_eval_steering_block "$MODEL" "$trait" "$layer" "$COEF" "$block_type" "$vector_path" "$output_path"; then
                if check_output_exists "$output_path" 2>/dev/null; then
                    skipped=$((skipped + 1))
                else
                    completed=$((completed + 1))
                fi
                log "✓ Completed: $trait $block_type layer$layer"
            else
                failed+=("$trait-$block_type-layer$layer")
                log "✗ Failed: $trait $block_type layer$layer"
            fi
            log_separator
        done
    done
done

# ========== Summary ==========
print_summary "$total" "$completed" "$skipped" "${#failed[@]}" "${failed[@]}"

[[ ${#failed[@]} -eq 0 ]] && exit 0 || exit 1

