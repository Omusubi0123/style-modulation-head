#!/bin/bash
#
# run_eval_style_head_ablation.sh - Style head zero-ablation evaluation
#
# Investigate how generated text changes when cumulatively
# zero-ablating style heads.
#
# Usage:
#   ./scripts/run_eval_style_head_ablation.sh <model> <trait1> [trait2 ...]
#
# Example:
#   ./scripts/run_eval_style_head_ablation.sh "Qwen/Qwen2.5-7B-Instruct" evil humorous
#
# Environment variables (override defaults):
#   GPU, PERSONA_INSTRUCTION_TYPE, JUDGE_MODEL, BATCH_SIZE,
#   N_PER_QUESTION, MAX_TOKENS, POSITIONS, VERSION

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
: ${MAX_TOKENS:=1000}
: ${N_PER_QUESTION:=5}
: ${POSITIONS:="all"}      # "all", "prompt", "response"
: ${VERSION:="eval"}

OUTPUT_BASE_DIR="data/eval_persona_eval/style_head_ablation_${POSITIONS}"

# ========== Helper ==========
get_model_short_name() {
    local model=$1
    echo "${model##*/}" | tr '[:upper:]' '[:lower:]'
}

check_style_head_csv() {
    local model=$1
    local short_name=$(get_model_short_name "$model")
    local csv_path="style_head/${short_name}.csv"

    if [[ -f "$csv_path" ]]; then
        log "Style head CSV found: $csv_path"
        return 0
    else
        log "Style head CSV not found: $csv_path"
        return 1
    fi
}

# ========== Setup ==========
setup_logging "eval_style_head_ablation_${POSITIONS}"
log "Starting style head ablation evaluation at $(date)"
log "Model: $MODEL"
log "Traits: ${TRAITS[*]}"
log "Persona instruction type: ${PERSONA_INSTRUCTION_TYPE:-none}"
log "Positions: ${POSITIONS}"
log "Judge model: ${JUDGE_MODEL}"
log_separator

# ========== Main Loop ==========
failed=()
skipped=0
completed=0
total=0

if ! check_style_head_csv "$MODEL"; then
    log "Style head CSV not found for $MODEL. Exiting."
    exit 1
fi

short_name=$(get_model_short_name "$MODEL")

for trait in "${TRAITS[@]}"; do
    total=$((total + 1))
    output_dir="${OUTPUT_BASE_DIR}/${short_name}/${trait}"

    log "Processing trait: $trait"
    log "Output directory: $output_dir"

    cmd="CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=. uv run python src/eval/eval_style_head_ablation.py"
    cmd+=" --model \"$MODEL\""
    cmd+=" --trait \"$trait\""
    cmd+=" --output_dir \"$output_dir\""
    cmd+=" --positions \"$POSITIONS\""
    cmd+=" --max_tokens $MAX_TOKENS"
    cmd+=" --n_per_question $N_PER_QUESTION"
    cmd+=" --batch_size $BATCH_SIZE"
    cmd+=" --judge_model \"$JUDGE_MODEL\""
    cmd+=" --version \"$VERSION\""

    if [[ -n "$PERSONA_INSTRUCTION_TYPE" ]]; then
        cmd+=" --persona_instruction_type \"$PERSONA_INSTRUCTION_TYPE\""
    fi

    log "Command: $cmd"
    eval_output=$(eval "$cmd" 2>&1)
    eval_exit_code=$?

    echo "$eval_output" | tee -a "$LOG_FILE"

    if [[ $eval_exit_code -eq 0 ]]; then
        completed=$((completed + 1))
        log "Completed: $trait"
    else
        failed+=("$trait")
        log "Failed: $trait"
    fi
    log_separator
done

# ========== Summary ==========
print_summary "$total" "$completed" "$skipped" "${#failed[@]}" "${failed[@]}"

[[ ${#failed[@]} -eq 0 ]] && exit 0 || exit 1
