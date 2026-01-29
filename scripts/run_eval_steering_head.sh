#!/bin/bash
#
# run_eval_steering_head.sh - Head-level steering evaluation
#
# Usage:
#   ./scripts/run_eval_steering_head.sh <model> <trait1> [trait2 ...]
#
# Example:
#   ./scripts/run_eval_steering_head.sh "Qwen/Qwen2.5-7B-Instruct" evil humorous

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
# Get model-specific config
NUM_HEADS=$(get_num_heads "$MODEL")
LAYERS=(19)  # Specify layers to analyze
COEF=7.0
HEAD_MODE="individual"  # "individual" or "all"

OUTPUT_DIR="data/eval_persona_eval/steering_results_head_${PERSONA_INSTRUCTION_TYPE}_${COEF}"

# ========== Setup ==========
setup_logging "eval_steering_head"
log "Starting head steering evaluation at $(date)"
log "Model: $MODEL"
log "Traits: ${TRAITS[*]}"
log "Layers: ${LAYERS[*]}"
log "Number of heads: $NUM_HEADS"
log "Head mode: $HEAD_MODE"
log "Coefficient: $COEF"
log_separator

# ========== Main Loop ==========
failed=()
skipped=0
completed=0
total=0

for trait in "${TRAITS[@]}"; do
    vector_path="data/persona_vectors/$MODEL/${trait}_response_avg_diff_attn_pre_o_proj.pt"
    
    if ! check_vector_exists "$vector_path"; then
        log "Skipping $trait (vector not found)"
        continue
    fi
    
    for layer in "${LAYERS[@]}"; do
        if [[ "$HEAD_MODE" == "individual" ]]; then
            # Test each head individually
            for ((head=0; head<NUM_HEADS; head++)); do
                total=$((total + 1))
                output_path="$OUTPUT_DIR/$MODEL/${trait}_steer_head_${STEERING_TYPE}_${PERSONA_INSTRUCTION_TYPE}_layer${layer}_head${head}_coef${COEF}.csv"
                
                if run_eval_steering_head "$MODEL" "$trait" "$layer" "$COEF" "$head" "$vector_path" "$output_path"; then
                    if check_output_exists "$output_path" 2>/dev/null; then
                        skipped=$((skipped + 1))
                    else
                        completed=$((completed + 1))
                    fi
                    log "Completed: $trait layer$layer head$head"
                else
                    failed+=("$trait-layer$layer-head$head")
                    log "Failed: $trait layer$layer head$head"
                fi
                log_separator
            done
        else
            # Test all heads together
            total=$((total + 1))
            all_heads=$(seq -s, 0 $((NUM_HEADS - 1)))
            head_str="0-$((NUM_HEADS - 1))"
            output_path="$OUTPUT_DIR/$MODEL/${trait}_steer_head_${STEERING_TYPE}_${PERSONA_INSTRUCTION_TYPE}_layer${layer}_headall_coef${COEF}.csv"
            
            if run_eval_steering_head "$MODEL" "$trait" "$layer" "$COEF" "$all_heads" "$vector_path" "$output_path"; then
                if check_output_exists "$output_path" 2>/dev/null; then
                    skipped=$((skipped + 1))
                else
                    completed=$((completed + 1))
                fi
                log "Completed: $trait layer$layer all_heads"
            else
                failed+=("$trait-layer$layer-all_heads")
                log "Failed: $trait layer$layer all_heads"
            fi
            log_separator
        fi
    done
done

# ========== Summary ==========
print_summary "$total" "$completed" "$skipped" "${#failed[@]}" "${failed[@]}"

[[ ${#failed[@]} -eq 0 ]] && exit 0 || exit 1
