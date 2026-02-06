#!/bin/bash
#
# run_steering_position_comparison.sh - Steering position comparison experiments
#
# Compares steering at different positions within a transformer block:
#   1. Post-attention residual stream (vector: mlp_layernorm, apply: attn_output)
#   2. Post-MLP residual stream (vector: attn_layernorm, apply: mlp_output)
#   3. Attention output (vector: attn_output, apply: attn_output)
#   4. Correlated attention heads only
#   5. Correlated + anti-correlated attention heads
#
# Usage:
#   Set required environment variables, then run:
#   MODEL="Qwen/Qwen2.5-7B-Instruct" LAYER=19 ... ./scripts/run_steering_position_comparison.sh
#
# Required environment variables:
#   MODEL, LAYER, NUM_HEADS, CORRELATED_HEADS, CORRELATED_ANTI_HEADS,
#   NUM_CORRELATED_HEADS, NUM_CORRELATED_ANTI_HEADS
#
# Environment variables (override defaults):
#   GPU, STEERING_TYPE, JUDGE_MODEL, BATCH_SIZE,
#   TRAITS, BASE_COEFS, HEAD_ADDITIONAL_COEFS,
#   POSITIONS, INSTRUCTION_TYPES

set -o pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/lib/eval_common.sh"

# ========== Validate Required Variables ==========
for var in MODEL LAYER NUM_HEADS CORRELATED_HEADS CORRELATED_ANTI_HEADS \
           NUM_CORRELATED_HEADS NUM_CORRELATED_ANTI_HEADS; do
    if [[ -z "${!var}" ]]; then
        echo "Error: $var is not set"
        exit 1
    fi
done

# ========== Configuration ==========
TRAITS="${TRAITS:-evil sycophantic hallucinating impolite apathetic humorous optimistic passionate betrayal anti-environment}"
BASE_COEFS="${BASE_COEFS:-0.5 1.0 1.5 2.0 2.5 3.0 4.0 5.0 6.0 8.0 10.0}"
HEAD_ADDITIONAL_COEFS="${HEAD_ADDITIONAL_COEFS:-12.0 14.0 16.0}"
POSITIONS="${POSITIONS:-1 2 3 4 5}"
INSTRUCTION_TYPES="${INSTRUCTION_TYPES:-neg pos}"

VECTOR_DIR="${VECTOR_DIR:-data/persona_vectors/$MODEL}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-data/steering_position_comparison/$MODEL}"

# ========== Setup ==========
setup_logging "steering_position_comparison_${MODEL//\//-}"
log "============================================================"
log "Steering Position Comparison Experiment - ${MODEL}"
log "============================================================"
log "Layer: $LAYER (transformer block $((LAYER + 1)), 1-indexed)"
log "Traits: $TRAITS"
log "Positions: $POSITIONS"
log "Instruction types: $INSTRUCTION_TYPES"
log "Base coefficients: $BASE_COEFS"
log "Head additional coefficients: $HEAD_ADDITIONAL_COEFS"
log "Correlated heads: $CORRELATED_HEADS"
log "Correlated+anti heads: $CORRELATED_ANTI_HEADS"
log "Vector directory: $VECTOR_DIR"
log "Output directory: $OUTPUT_BASE_DIR"
log_separator

# ========== Helper Functions ==========
check_vector_file() {
    local trait=$1 vector_suffix=$2
    local vector_path="$VECTOR_DIR/${trait}_prompt_avg_diff_${vector_suffix}.pt"
    [[ -f "$vector_path" ]] || { log "Vector not found: $vector_path"; return 1; }
}

# ========== Position Runner Functions ==========
# Each function calls the unified eval_persona.py with appropriate --steering_target.

# Position 1: Post-attention residual stream
# Vector: mlp_layernorm direction, Applied at: attn_output
run_position_1() {
    local trait=$1 instruction_type=$2 coef=$3 coef_label=$4 coef_sign=$5
    local actual_coef=$(echo "scale=6; $coef * $coef_sign" | bc)
    local output_dir="$OUTPUT_BASE_DIR/post_attention_residual"
    local output_path="$output_dir/${trait}_${instruction_type}_coef${coef_label}_layer${LAYER}.csv"
    local vector_path="$VECTOR_DIR/${trait}_prompt_avg_diff_mlp_layernorm.pt"

    mkdir -p "$output_dir"
    check_vector_file "$trait" "mlp_layernorm" || return 1
    check_output_exists "$output_path" && return 0

    PERSONA_INSTRUCTION_TYPE="$instruction_type" \
    run_eval "$MODEL" "$trait" "$LAYER" "$actual_coef" "$vector_path" "$output_path" "attn_output"
}

# Position 2: Post-MLP residual stream
# Vector: attn_layernorm direction, Applied at: mlp_output
run_position_2() {
    local trait=$1 instruction_type=$2 coef=$3 coef_label=$4 coef_sign=$5
    local actual_coef=$(echo "scale=6; $coef * $coef_sign" | bc)
    local output_dir="$OUTPUT_BASE_DIR/post_mlp_residual"
    local output_path="$output_dir/${trait}_${instruction_type}_coef${coef_label}_layer${LAYER}.csv"
    local vector_path="$VECTOR_DIR/${trait}_prompt_avg_diff_attn_layernorm.pt"

    mkdir -p "$output_dir"
    check_vector_file "$trait" "attn_layernorm" || return 1
    check_output_exists "$output_path" && return 0

    PERSONA_INSTRUCTION_TYPE="$instruction_type" \
    run_eval "$MODEL" "$trait" "$LAYER" "$actual_coef" "$vector_path" "$output_path" "mlp_output"
}

# Position 3: Attention output (before residual addition)
run_position_3() {
    local trait=$1 instruction_type=$2 coef=$3 coef_label=$4 coef_sign=$5
    local actual_coef=$(echo "scale=6; $coef * $coef_sign" | bc)
    local output_dir="$OUTPUT_BASE_DIR/attention_output"
    local output_path="$output_dir/${trait}_${instruction_type}_coef${coef_label}_layer${LAYER}.csv"
    local vector_path="$VECTOR_DIR/${trait}_prompt_avg_diff_attn_output.pt"

    mkdir -p "$output_dir"
    check_vector_file "$trait" "attn_output" || return 1
    check_output_exists "$output_path" && return 0

    PERSONA_INSTRUCTION_TYPE="$instruction_type" \
    run_eval "$MODEL" "$trait" "$LAYER" "$actual_coef" "$vector_path" "$output_path" "attn_output"
}

# Position 4: Correlated heads only
run_position_4() {
    local trait=$1 instruction_type=$2 base_coef=$3 coef_sign=$4
    local actual_coef=$(echo "scale=6; $base_coef * $coef_sign" | bc)
    local coef_label; [[ "$coef_sign" == "-1" ]] && coef_label="-${base_coef}" || coef_label="${base_coef}"
    local head_str=$(echo $CORRELATED_HEADS | tr ',' '-')
    local output_dir="$OUTPUT_BASE_DIR/correlated_heads"
    local output_path="$output_dir/${trait}_${instruction_type}_coef${coef_label}_normal_layer${LAYER}_heads${head_str}.csv"
    local vector_path="$VECTOR_DIR/${trait}_prompt_avg_diff_attn_pre_o_proj.pt"

    mkdir -p "$output_dir"
    check_vector_file "$trait" "attn_pre_o_proj" || return 1
    check_output_exists "$output_path" && return 0

    PERSONA_INSTRUCTION_TYPE="$instruction_type" \
    run_eval "$MODEL" "$trait" "$LAYER" "$actual_coef" "$vector_path" "$output_path" \
        "head" "--head_indices $CORRELATED_HEADS"
}

# Position 5: Correlated + anti-correlated heads
run_position_5() {
    local trait=$1 instruction_type=$2 base_coef=$3 coef_sign=$4
    local actual_coef=$(echo "scale=6; $base_coef * $coef_sign" | bc)
    local coef_label; [[ "$coef_sign" == "-1" ]] && coef_label="-${base_coef}" || coef_label="${base_coef}"
    local head_str=$(echo $CORRELATED_ANTI_HEADS | tr ',' '-')
    local output_dir="$OUTPUT_BASE_DIR/correlated_anti_heads"
    local output_path="$output_dir/${trait}_${instruction_type}_coef${coef_label}_normal_layer${LAYER}_heads${head_str}.csv"
    local vector_path="$VECTOR_DIR/${trait}_prompt_avg_diff_attn_pre_o_proj.pt"

    mkdir -p "$output_dir"
    check_vector_file "$trait" "attn_pre_o_proj" || return 1
    check_output_exists "$output_path" && return 0

    PERSONA_INSTRUCTION_TYPE="$instruction_type" \
    run_eval "$MODEL" "$trait" "$LAYER" "$actual_coef" "$vector_path" "$output_path" \
        "head" "--head_indices $CORRELATED_ANTI_HEADS"
}

# ========== Main Execution ==========
log "=== STARTING STEERING POSITION COMPARISON ==="

total_experiments=0
completed_experiments=0
failed_experiments=()

read -ra TRAITS_ARR <<< "$TRAITS"
read -ra BASE_COEFS_ARR <<< "$BASE_COEFS"
read -ra HEAD_ADDITIONAL_COEFS_ARR <<< "$HEAD_ADDITIONAL_COEFS"

for trait in "${TRAITS_ARR[@]}"; do
    log ""
    log "============================================================"
    log "Processing trait: $trait"
    log "============================================================"
    
    for position in $POSITIONS; do
        log ""
        log "--- Position $position ---"
        
        for instruction_type in $INSTRUCTION_TYPES; do
            if [[ "$instruction_type" == "neg" ]]; then
                coef_signs=(1)
                conditions=("add")
            else
                coef_signs=(1 -1)
                conditions=("add" "subtract")
            fi
            
            for idx in "${!coef_signs[@]}"; do
                coef_sign=${coef_signs[$idx]}
                condition=${conditions[$idx]}
                
                log "Condition: ${instruction_type}_${condition}"
                
                if [[ "$position" -le 3 ]]; then
                    for base_coef in "${BASE_COEFS_ARR[@]}"; do
                        total_experiments=$((total_experiments + 1))
                        local_label; [[ "$coef_sign" == "-1" ]] && local_label="-${base_coef}" || local_label="${base_coef}"
                        
                        case $position in
                            1) run_position_1 "$trait" "$instruction_type" "$base_coef" "$local_label" "$coef_sign" ;;
                            2) run_position_2 "$trait" "$instruction_type" "$base_coef" "$local_label" "$coef_sign" ;;
                            3) run_position_3 "$trait" "$instruction_type" "$base_coef" "$local_label" "$coef_sign" ;;
                        esac
                        [[ $? -eq 0 ]] && completed_experiments=$((completed_experiments + 1)) || \
                            failed_experiments+=("$trait-pos${position}-${instruction_type}_${condition}-coef${local_label}")
                    done
                else
                    for base_coef in "${BASE_COEFS_ARR[@]}" "${HEAD_ADDITIONAL_COEFS_ARR[@]}"; do
                        total_experiments=$((total_experiments + 1))
                        
                        case $position in
                            4) run_position_4 "$trait" "$instruction_type" "$base_coef" "$coef_sign" ;;
                            5) run_position_5 "$trait" "$instruction_type" "$base_coef" "$coef_sign" ;;
                        esac
                        [[ $? -eq 0 ]] && completed_experiments=$((completed_experiments + 1)) || \
                            failed_experiments+=("$trait-pos${position}-${instruction_type}_${condition}-coef${base_coef}")
                    done
                fi
            done
        done
    done
done

# ========== Summary ==========
print_summary "$total_experiments" "$completed_experiments" "0" "${#failed_experiments[@]}" "${failed_experiments[@]}"

[[ ${#failed_experiments[@]} -eq 0 ]] && exit 0 || exit 1
