#!/bin/bash
#
# run_eval_steering.sh - 標準的なステアリング評価
#
# 使い方:
#   ./scripts/run_eval_steering.sh
#
# 環境変数で設定可能:
#   GPU, STEERING_TYPE, PERSONA_INSTRUCTION_TYPE, JUDGE_MODEL

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/eval_common.sh"

# ========== Configuration ==========
TRAITS=("evil" "sycophantic" "hallucinating" "humorous" "passionate" "loser")
MODELS=("Qwen/Qwen2.5-7B-Instruct" "meta-llama/Llama-3.1-8B-Instruct")
COEFS=(-3.0 -2.0 -1.0 0.0 1.0 2.0 3.0)

OUTPUT_DIR="data/eval_persona_eval/steering_results"

# ========== Setup ==========
setup_logging "eval_steering"
log "Starting steering evaluation at $(date)"
log "Traits: ${TRAITS[*]}"
log "Models: ${MODELS[*]}"
log "Coefficients: ${COEFS[*]}"
log_separator

# ========== Main Loop ==========
failed=()
skipped=0
completed=0
total=0

for model in "${MODELS[@]}"; do
    for trait in "${TRAITS[@]}"; do
        layer=$(get_trait_layer "$model" "$trait")
        vector_path="data/persona_vectors/$model/${trait}_response_avg_diff.pt"
        
        for coef in "${COEFS[@]}"; do
            total=$((total + 1))
            output_path="$OUTPUT_DIR/$model/${trait}_steer_${STEERING_TYPE}_${PERSONA_INSTRUCTION_TYPE}_layer${layer}_coef${coef}.csv"
            
            if run_eval_steering "$model" "$trait" "$layer" "$coef" "$vector_path" "$output_path"; then
                if check_output_exists "$output_path" 2>/dev/null; then
                    skipped=$((skipped + 1))
                else
                    completed=$((completed + 1))
                fi
                log "✓ Completed: $trait layer$layer coef$coef"
            else
                failed+=("$trait-$model-layer$layer-coef$coef")
                log "✗ Failed: $trait layer$layer coef$coef"
            fi
            log_separator
        done
    done
done

# ========== Summary ==========
print_summary "$total" "$completed" "$skipped" "${#failed[@]}" "${failed[@]}"

[[ ${#failed[@]} -eq 0 ]] && exit 0 || exit 1

