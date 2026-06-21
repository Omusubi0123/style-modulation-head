#!/bin/bash
#
# run_eval_steering_block.sh - Block position steering evaluation
#
# Usage:
#   ./scripts/run_eval_steering_block.sh <model> <trait1> [trait2 ...]
#
# Example:
#   ./scripts/run_eval_steering_block.sh "Qwen/Qwen2.5-7B-Instruct" evil humorous
#
# Configuration is set in the script (can be overridden via environment variables):
#   GPU, STEERING_TYPE, PERSONA_INSTRUCTION_TYPE, JUDGE_MODEL, BATCH_SIZE, N_PER_QUESTION

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========== Default Configuration ==========
# Set default values (can be overridden via environment variables)
GPU="${GPU:-0}"
STEERING_TYPE="${STEERING_TYPE:-response}"
PERSONA_INSTRUCTION_TYPE="${PERSONA_INSTRUCTION_TYPE:-neg}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini-2025-04-14}"
BATCH_SIZE="${BATCH_SIZE:-100}"
N_PER_QUESTION="${N_PER_QUESTION:-5}"

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
num_layers=$(get_model_layers "$MODEL")
LAYERS=($(seq 0 $((num_layers - 1))))
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
        # For mlp_output steering, the post-MLP residual stream direction is captured at
        # the next layer's attn_layernorm input (layer N+1), not at layer N's mlp_output.
        # Hook remains at layer N's mlp_output; only the vector source changes.
        if [[ "$block_type" == "mlp_output" ]]; then
            vector_path="data/persona_vectors/$MODEL/${trait}_response_avg_diff_attn_layernorm.pt"
        else
            vector_path="data/persona_vectors/$MODEL/${trait}_response_avg_diff_${block_type}.pt"
        fi
        
        if ! check_vector_exists "$vector_path"; then
            log "Skipping $trait $block_type (vector not found)"
            continue
        fi
        
        for layer in "${LAYERS[@]}"; do
            total=$((total + 1))
            output_path="$OUTPUT_DIR/$MODEL/${trait}_steer_block_${block_type}_${STEERING_TYPE}_${PERSONA_INSTRUCTION_TYPE}_layer$((layer+1))_coef${COEF}.csv"
            
            # For mlp_output, load vector at layer+1 (post-MLP residual = next layer's attn input)
            if [[ "$block_type" == "mlp_output" ]]; then
                extra_args="--vector_layer $((layer+1))"
            else
                extra_args=""
            fi
            
            if run_eval_steering_block "$MODEL" "$trait" "$layer" "$COEF" "$block_type" "$vector_path" "$output_path" "$extra_args"; then
                if check_output_exists "$output_path" 2>/dev/null; then
                    skipped=$((skipped + 1))
                else
                    completed=$((completed + 1))
                fi
                log "Completed: $trait $block_type layer$((layer+1))"
            else
                failed+=("$trait-$block_type-layer$((layer+1))")
                log "Failed: $trait $block_type layer$((layer+1))"
            fi
            log_separator
        done
    done
done

# ========== Summary ==========
print_summary "$total" "$completed" "$skipped" "${#failed[@]}" "${failed[@]}"

[[ ${#failed[@]} -eq 0 ]] && exit 0 || exit 1
