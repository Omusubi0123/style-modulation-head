#!/bin/bash
#
# run_eval_steering.sh - Residual stream (layer output) steering evaluation
#
# Usage:
#   ./scripts/run_eval_steering.sh <model> <trait1> [trait2 ...]
#
# Example:
#   ./scripts/run_eval_steering.sh "Qwen/Qwen2.5-7B-Instruct" evil humorous
#
# Environment variables (override defaults):
#   GPU, STEERING_TYPE, PERSONA_INSTRUCTION_TYPE, JUDGE_MODEL, BATCH_SIZE

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
# COEFS=(-3.0 -2.0 -1.0 0.0 1.0 2.0 3.0)
COEFS=(2.0)
OUTPUT_DIR="data/eval_persona_eval/steering_results"

# ========== Setup ==========
setup_logging "eval_steering"
log "Starting steering evaluation at $(date)"
log "Model: $MODEL"
log "Traits: ${TRAITS[*]}"
log "Coefficients: ${COEFS[*]}"
log_separator

# ========== Main Loop ==========
failed=()
skipped=0
completed=0
total=0

for trait in "${TRAITS[@]}"; do
    layer=$(get_trait_layer "$MODEL" "$trait")
    vector_path="data/persona_vectors/$MODEL/${trait}_response_avg_diff.pt"
    
    for coef in "${COEFS[@]}"; do
        total=$((total + 1))
        output_path="$OUTPUT_DIR/$MODEL/${trait}_steer_${STEERING_TYPE}_${PERSONA_INSTRUCTION_TYPE}_layer$((layer+1))_coef${coef}.csv"
        
        if run_eval_steering "$MODEL" "$trait" "$layer" "$coef" "$vector_path" "$output_path"; then
            if check_output_exists "$output_path" 2>/dev/null; then
                skipped=$((skipped + 1))
            else
                completed=$((completed + 1))
            fi
            log "Completed: $trait layer$((layer+1)) (1-indexed) coef$coef"
        else
            failed+=("$trait-layer$layer-coef$coef")
            log "Failed: $trait layer$((layer+1)) (1-indexed) coef$coef"
        fi
        log_separator
    done
done

# ========== Summary ==========
print_summary "$total" "$completed" "$skipped" "${#failed[@]}" "${failed[@]}"

[[ ${#failed[@]} -eq 0 ]] && exit 0 || exit 1
