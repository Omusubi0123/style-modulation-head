#!/bin/bash

# =============================================================================
# run_steering_position_comparison.sh
# Steering位置比較実験の共通スクリプト
#
# 対照実験:
#   1. attention加算後residual stream (post_attention)
#   2. mlp加算後residual stream (post_mlp)
#   3. attention output（residual stream加算前）
#   4. attention head出力と相関の強いheadのみ
#   5. attention head出力と相関の強い・反対に強いheadのみ
#
# 実験条件:
#   - neg system prompt + steering add → ペルソナ増幅
#   - pos system prompt + steering add → ペルソナ増幅
#   - pos system prompt + steering subtract → ペルソナ抑制
#
# 使用方法:
#   環境変数で設定を指定してから実行
#   例: MODEL="Qwen/Qwen2.5-7B-Instruct" LAYER=19 ... ./scripts/run_steering_position_comparison.sh
# =============================================================================

set -o pipefail

# =============================================================================
# Required Environment Variables (must be set before calling)
# =============================================================================
# MODEL              - モデル名 (e.g., "Qwen/Qwen2.5-7B-Instruct")
# LAYER              - Steering層 (0-indexed)
# NUM_HEADS          - Attention head数
# CORRELATED_HEADS   - 相関の強いhead (e.g., "2,4,27")
# CORRELATED_ANTI_HEADS - 相関の強い・反対に強いhead (e.g., "0,2,4,26,27")
# NUM_CORRELATED_HEADS - 相関の強いhead数
# NUM_CORRELATED_ANTI_HEADS - 相関の強い・反対に強いhead数

# =============================================================================
# Validate Required Variables
# =============================================================================

if [ -z "$MODEL" ]; then
    echo "Error: MODEL is not set"
    exit 1
fi

if [ -z "$LAYER" ]; then
    echo "Error: LAYER is not set"
    exit 1
fi

if [ -z "$NUM_HEADS" ]; then
    echo "Error: NUM_HEADS is not set"
    exit 1
fi

if [ -z "$CORRELATED_HEADS" ]; then
    echo "Error: CORRELATED_HEADS is not set"
    exit 1
fi

if [ -z "$CORRELATED_ANTI_HEADS" ]; then
    echo "Error: CORRELATED_ANTI_HEADS is not set"
    exit 1
fi

if [ -z "$NUM_CORRELATED_HEADS" ]; then
    echo "Error: NUM_CORRELATED_HEADS is not set"
    exit 1
fi

if [ -z "$NUM_CORRELATED_ANTI_HEADS" ]; then
    echo "Error: NUM_CORRELATED_ANTI_HEADS is not set"
    exit 1
fi

# =============================================================================
# Optional Configuration (with defaults)
# =============================================================================

# Traits to evaluate
TRAITS="${TRAITS:-evil sycophantic hallucinating impolite apathetic humorous optimistic passionate betrayal anti-environment}"

# Base steering coefficients
BASE_COEFS="${BASE_COEFS:-0.5 1.0 1.5 2.0 2.5 3.0 4.0 5.0 6.0 8.0 10.0}"
HEAD_ADDITIONAL_COEFS="${HEAD_ADDITIONAL_COEFS:-12.0 14.0 16.0}"

# Judge model and batch settings
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1-mini-2025-04-14}"
BATCH_SIZE="${BATCH_SIZE:-100}"
STEERING_TYPE="${STEERING_TYPE:-response}"

# GPU assignment
GPU="${GPU:-0}"

# Vector and output directories
VECTOR_DIR="${VECTOR_DIR:-data/persona_vectors/$MODEL}"
OUTPUT_BASE_DIR="${OUTPUT_BASE_DIR:-data/steering_position_comparison/$MODEL}"

# Steering positions to run (space-separated)
POSITIONS="${POSITIONS:-1 2 3 4 5}"

# Persona instruction types to run (space-separated)
INSTRUCTION_TYPES="${INSTRUCTION_TYPES:-neg pos}"

# =============================================================================
# Setup
# =============================================================================

mkdir -p logs
LOG_FILE="logs/steering_position_comparison_${MODEL//\//-}_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================" | tee -a $LOG_FILE
echo "Steering Position Comparison Experiment - ${MODEL}" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "Started at: $(date)" | tee -a $LOG_FILE
echo "Model: $MODEL" | tee -a $LOG_FILE
echo "Layer: $LAYER (transformer block $((LAYER + 1)))" | tee -a $LOG_FILE
echo "Traits: $TRAITS" | tee -a $LOG_FILE
echo "Positions: $POSITIONS" | tee -a $LOG_FILE
echo "Instruction types: $INSTRUCTION_TYPES" | tee -a $LOG_FILE
echo "Base coefficients: $BASE_COEFS" | tee -a $LOG_FILE
echo "Head additional coefficients: $HEAD_ADDITIONAL_COEFS" | tee -a $LOG_FILE
echo "Correlated heads: $CORRELATED_HEADS" | tee -a $LOG_FILE
echo "Correlated+anti heads: $CORRELATED_ANTI_HEADS" | tee -a $LOG_FILE
echo "Vector directory: $VECTOR_DIR" | tee -a $LOG_FILE
echo "Output directory: $OUTPUT_BASE_DIR" | tee -a $LOG_FILE
echo "Log file: $LOG_FILE" | tee -a $LOG_FILE
echo "------------------------------------------------------------" | tee -a $LOG_FILE

# =============================================================================
# Helper Functions
# =============================================================================

# Check if vector file exists
check_vector_file() {
    local trait=$1
    local vector_suffix=$2
    local vector_path="$VECTOR_DIR/${trait}_prompt_avg_diff_${vector_suffix}.pt"
    
    if [ -f "$vector_path" ]; then
        return 0
    else
        echo "Vector file not found: $vector_path" | tee -a $LOG_FILE
        return 1
    fi
}

# Check if output file exists
check_output_exists() {
    local output_path=$1
    if [ -f "$output_path" ]; then
        echo "Output already exists: $output_path" | tee -a $LOG_FILE
        return 0
    fi
    return 1
}

# Note: calc_scaled_coef is no longer needed since we only use normal (1x) scaling

# =============================================================================
# Evaluation Functions
# =============================================================================

# Position 1: Attention加算後residual stream (post_attention)
# ベクトル: mlp_layernorm.pt（attention後residual streamの方向）
# 加算位置: attn_output（residual加算で反映させる）
run_position_1() {
    local trait=$1
    local instruction_type=$2
    local coef=$3
    local coef_label=$4
    local coef_sign=$5
    
    local actual_coef=$(echo "scale=6; $coef * $coef_sign" | bc)
    local vector_suffix="mlp_layernorm"
    local output_dir="$OUTPUT_BASE_DIR/post_attention_residual"
    local output_path="$output_dir/${trait}_${instruction_type}_coef${coef_label}_layer${LAYER}.csv"
    
    mkdir -p "$output_dir"
    
    if ! check_vector_file "$trait" "$vector_suffix"; then
        return 1
    fi
    
    if check_output_exists "$output_path"; then
        return 0
    fi
    
    echo "Running Position 1 (post_attention): trait=$trait, instruction=$instruction_type, coef=$actual_coef" | tee -a $LOG_FILE
    
    local vector_path="$VECTOR_DIR/${trait}_prompt_avg_diff_${vector_suffix}.pt"
    
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. uv run python eval/eval_persona_steer_residual_stream.py \
        --model "$MODEL" \
        --trait "$trait" \
        --output_path "$output_path" \
        --version eval \
        --steering_type "$STEERING_TYPE" \
        --residual_position "post_attention" \
        --coef "$actual_coef" \
        --vector_path "$vector_path" \
        --persona_instruction_type "$instruction_type" \
        --layer "$LAYER" \
        --judge_model "$JUDGE_MODEL" \
        --batch_size "$BATCH_SIZE" \
        2>&1 | tee -a $LOG_FILE
    
    return ${PIPESTATUS[0]}
}

# Position 2: MLP加算後residual stream (post_mlp)
# ベクトル: attn_layernorm.pt（MLP後residual stream = 次層入力の方向）
# 加算位置: mlp_output（residual加算で反映させる）
run_position_2() {
    local trait=$1
    local instruction_type=$2
    local coef=$3
    local coef_label=$4
    local coef_sign=$5
    
    local actual_coef=$(echo "scale=6; $coef * $coef_sign" | bc)
    local vector_suffix="attn_layernorm"
    local output_dir="$OUTPUT_BASE_DIR/post_mlp_residual"
    local output_path="$output_dir/${trait}_${instruction_type}_coef${coef_label}_layer${LAYER}.csv"
    
    mkdir -p "$output_dir"
    
    if ! check_vector_file "$trait" "$vector_suffix"; then
        return 1
    fi
    
    if check_output_exists "$output_path"; then
        return 0
    fi
    
    echo "Running Position 2 (post_mlp): trait=$trait, instruction=$instruction_type, coef=$actual_coef" | tee -a $LOG_FILE
    
    local vector_path="$VECTOR_DIR/${trait}_prompt_avg_diff_${vector_suffix}.pt"
    
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. uv run python eval/eval_persona_steer_residual_stream.py \
        --model "$MODEL" \
        --trait "$trait" \
        --output_path "$output_path" \
        --version eval \
        --steering_type "$STEERING_TYPE" \
        --residual_position "post_mlp" \
        --coef "$actual_coef" \
        --vector_path "$vector_path" \
        --persona_instruction_type "$instruction_type" \
        --layer "$LAYER" \
        --judge_model "$JUDGE_MODEL" \
        --batch_size "$BATCH_SIZE" \
        2>&1 | tee -a $LOG_FILE
    
    return ${PIPESTATUS[0]}
}

# Position 3: Attention output（residual stream加算前）
run_position_3() {
    local trait=$1
    local instruction_type=$2
    local coef=$3
    local coef_label=$4
    local coef_sign=$5
    
    local actual_coef=$(echo "scale=6; $coef * $coef_sign" | bc)
    local vector_suffix="attn_output"
    local output_dir="$OUTPUT_BASE_DIR/attention_output"
    local output_path="$output_dir/${trait}_${instruction_type}_coef${coef_label}_layer${LAYER}.csv"
    
    mkdir -p "$output_dir"
    
    if ! check_vector_file "$trait" "$vector_suffix"; then
        return 1
    fi
    
    if check_output_exists "$output_path"; then
        return 0
    fi
    
    echo "Running Position 3 (attention_output): trait=$trait, instruction=$instruction_type, coef=$actual_coef" | tee -a $LOG_FILE
    
    local vector_path="$VECTOR_DIR/${trait}_prompt_avg_diff_${vector_suffix}.pt"
    
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. uv run python eval/eval_persona_steer_block.py \
        --model "$MODEL" \
        --trait "$trait" \
        --output_path "$output_path" \
        --version eval \
        --steering_type "$STEERING_TYPE" \
        --block_steering_type "attn_output" \
        --coef "$actual_coef" \
        --vector_path "$vector_path" \
        --persona_instruction_type "$instruction_type" \
        --layer "$LAYER" \
        --judge_model "$JUDGE_MODEL" \
        --batch_size "$BATCH_SIZE" \
        2>&1 | tee -a $LOG_FILE
    
    return ${PIPESTATUS[0]}
}

# Position 4: 相関の強いheadのみ
run_position_4() {
    local trait=$1
    local instruction_type=$2
    local base_coef=$3
    local coef_sign=$4
    
    local actual_coef=$(echo "scale=6; $base_coef * $coef_sign" | bc)
    # Include sign in coef_label to distinguish add vs subtract
    if [ "$coef_sign" == "-1" ]; then
        local coef_label="-${base_coef}"
    else
        local coef_label="${base_coef}"
    fi
    local vector_suffix="attn_pre_o_proj"
    local output_dir="$OUTPUT_BASE_DIR/correlated_heads"
    local head_str=$(echo $CORRELATED_HEADS | tr ',' '-')
    local output_path="$output_dir/${trait}_${instruction_type}_coef${coef_label}_normal_layer${LAYER}_heads${head_str}.csv"
    
    mkdir -p "$output_dir"
    
    if ! check_vector_file "$trait" "$vector_suffix"; then
        return 1
    fi
    
    if check_output_exists "$output_path"; then
        return 0
    fi
    
    echo "Running Position 4 (correlated_heads): trait=$trait, instruction=$instruction_type, coef=$actual_coef, heads=$CORRELATED_HEADS" | tee -a $LOG_FILE
    
    local vector_path="$VECTOR_DIR/${trait}_prompt_avg_diff_${vector_suffix}.pt"
    
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. uv run python eval/eval_persona_steer_head.py \
        --model "$MODEL" \
        --trait "$trait" \
        --output_path "$output_path" \
        --version eval \
        --steering_type "$STEERING_TYPE" \
        --coef "$actual_coef" \
        --vector_path "$vector_path" \
        --persona_instruction_type "$instruction_type" \
        --layer "$LAYER" \
        --head_indices "$CORRELATED_HEADS" \
        --judge_model "$JUDGE_MODEL" \
        --batch_size "$BATCH_SIZE" \
        2>&1 | tee -a $LOG_FILE
    
    return ${PIPESTATUS[0]}
}

# Position 5: 相関の強い・反対に強いhead
run_position_5() {
    local trait=$1
    local instruction_type=$2
    local base_coef=$3
    local coef_sign=$4
    
    local actual_coef=$(echo "scale=6; $base_coef * $coef_sign" | bc)
    # Include sign in coef_label to distinguish add vs subtract
    if [ "$coef_sign" == "-1" ]; then
        local coef_label="-${base_coef}"
    else
        local coef_label="${base_coef}"
    fi
    local vector_suffix="attn_pre_o_proj"
    local output_dir="$OUTPUT_BASE_DIR/correlated_anti_heads"
    local head_str=$(echo $CORRELATED_ANTI_HEADS | tr ',' '-')
    local output_path="$output_dir/${trait}_${instruction_type}_coef${coef_label}_normal_layer${LAYER}_heads${head_str}.csv"
    
    mkdir -p "$output_dir"
    
    if ! check_vector_file "$trait" "$vector_suffix"; then
        return 1
    fi
    
    if check_output_exists "$output_path"; then
        return 0
    fi
    
    echo "Running Position 5 (correlated_anti_heads): trait=$trait, instruction=$instruction_type, coef=$actual_coef, heads=$CORRELATED_ANTI_HEADS" | tee -a $LOG_FILE
    
    local vector_path="$VECTOR_DIR/${trait}_prompt_avg_diff_${vector_suffix}.pt"
    
    CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. uv run python eval/eval_persona_steer_head.py \
        --model "$MODEL" \
        --trait "$trait" \
        --output_path "$output_path" \
        --version eval \
        --steering_type "$STEERING_TYPE" \
        --coef "$actual_coef" \
        --vector_path "$vector_path" \
        --persona_instruction_type "$instruction_type" \
        --layer "$LAYER" \
        --head_indices "$CORRELATED_ANTI_HEADS" \
        --judge_model "$JUDGE_MODEL" \
        --batch_size "$BATCH_SIZE" \
        2>&1 | tee -a $LOG_FILE
    
    return ${PIPESTATUS[0]}
}

# =============================================================================
# Main Execution
# =============================================================================

echo "=== STARTING STEERING POSITION COMPARISON ===" | tee -a $LOG_FILE

total_experiments=0
completed_experiments=0
skipped_experiments=0
failed_experiments=()

# Convert space-separated strings to arrays
read -ra TRAITS_ARR <<< "$TRAITS"
read -ra BASE_COEFS_ARR <<< "$BASE_COEFS"
read -ra HEAD_ADDITIONAL_COEFS_ARR <<< "$HEAD_ADDITIONAL_COEFS"

for trait in "${TRAITS_ARR[@]}"; do
    echo "" | tee -a $LOG_FILE
    echo "============================================================" | tee -a $LOG_FILE
    echo "Processing trait: $trait" | tee -a $LOG_FILE
    echo "============================================================" | tee -a $LOG_FILE
    
    for position in $POSITIONS; do
        echo "" | tee -a $LOG_FILE
        echo "--- Position $position ---" | tee -a $LOG_FILE
        
        for instruction_type in $INSTRUCTION_TYPES; do
            # Determine coefficient sign based on condition
            # neg + add → positive coef (amplify)
            # pos + add → positive coef (amplify)
            # pos + subtract → negative coef (suppress) - only for pos
            
            if [ "$instruction_type" == "neg" ]; then
                # neg system prompt + add → ペルソナ増幅
                coef_signs=(1)
                conditions=("add")
            else
                # pos system prompt + add → ペルソナ増幅
                # pos system prompt + subtract → ペルソナ抑制
                coef_signs=(1 -1)
                conditions=("add" "subtract")
            fi
            
            for idx in "${!coef_signs[@]}"; do
                coef_sign=${coef_signs[$idx]}
                condition=${conditions[$idx]}
                condition_name="${instruction_type}_${condition}"
                
                echo "" | tee -a $LOG_FILE
                echo "Condition: $condition_name" | tee -a $LOG_FILE
                
                # For positions 1-3: use base coefficients directly
                if [[ "$position" -le 3 ]]; then
                    for base_coef in "${BASE_COEFS_ARR[@]}"; do
                        total_experiments=$((total_experiments + 1))
                        # Include sign in coef_label to distinguish add vs subtract
                        if [ "$coef_sign" == "-1" ]; then
                            coef_label="-${base_coef}"
                        else
                            coef_label="${base_coef}"
                        fi
                        
                        case $position in
                            1) run_position_1 "$trait" "$instruction_type" "$base_coef" "$coef_label" "$coef_sign" ;;
                            2) run_position_2 "$trait" "$instruction_type" "$base_coef" "$coef_label" "$coef_sign" ;;
                            3) run_position_3 "$trait" "$instruction_type" "$base_coef" "$coef_label" "$coef_sign" ;;
                        esac
                        
                        exit_code=$?
                        if [ $exit_code -eq 0 ]; then
                            completed_experiments=$((completed_experiments + 1))
                        else
                            failed_experiments+=("$trait-pos${position}-${condition_name}-coef${coef_label}")
                        fi
                    done
                
                # For positions 4-5: head steering (normal scaling only)
                # Position 4-5では12.0と14.0も追加
                else
                    # Position 4-5用の係数配列（12.0と14.0を追加）
                    for base_coef in "${BASE_COEFS_ARR[@]}" "${HEAD_ADDITIONAL_COEFS_ARR[@]}"; do
                        total_experiments=$((total_experiments + 1))
                        # Include sign in coef_label to distinguish add vs subtract
                        if [ "$coef_sign" == "-1" ]; then
                            coef_label="-${base_coef}"
                        else
                            coef_label="${base_coef}"
                        fi
                        
                        case $position in
                            4) run_position_4 "$trait" "$instruction_type" "$base_coef" "$coef_sign" ;;
                            5) run_position_5 "$trait" "$instruction_type" "$base_coef" "$coef_sign" ;;
                        esac
                        
                        exit_code=$?
                        if [ $exit_code -eq 0 ]; then
                            completed_experiments=$((completed_experiments + 1))
                        else
                            failed_experiments+=("$trait-pos${position}-${condition_name}-coef${coef_label}")
                        fi
                    done
                fi
            done
        done
    done
done

# =============================================================================
# Summary
# =============================================================================

echo "" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "FINAL SUMMARY" | tee -a $LOG_FILE
echo "============================================================" | tee -a $LOG_FILE
echo "Model: $MODEL" | tee -a $LOG_FILE
echo "Total experiments: $total_experiments" | tee -a $LOG_FILE
echo "Completed: $completed_experiments" | tee -a $LOG_FILE
echo "Failed: ${#failed_experiments[@]}" | tee -a $LOG_FILE
echo "Completed at: $(date)" | tee -a $LOG_FILE

if [ ${#failed_experiments[@]} -gt 0 ]; then
    echo "" | tee -a $LOG_FILE
    echo "Failed experiments:" | tee -a $LOG_FILE
    for failed in "${failed_experiments[@]}"; do
        echo "  - $failed" | tee -a $LOG_FILE
    done
    exit 1
else
    echo "" | tee -a $LOG_FILE
    echo "🎉 All experiments completed successfully!" | tee -a $LOG_FILE
    exit 0
fi
