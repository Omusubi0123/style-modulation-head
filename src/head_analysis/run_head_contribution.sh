#!/bin/bash
#
# run_head_contribution.sh - Sample execution for Head Contribution analysis
#
# Usage:
#   ./src/head_analysis/run_head_contribution.sh

set -e

# Configuration
MODEL="Qwen/Qwen2.5-7B-Instruct"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
VECTOR_DIR="data/persona_vectors/Qwen/Qwen2.5-7B-Instruct"
OUTPUT_DIR="data/head_analysis"
VECTOR_TYPE="response_avg"

echo "=== Head Contribution Analysis ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"

# Single trait analysis
PYTHONPATH=. uv run python -m src.head_analysis.head_contribution.main analyze_trait \
    --model_name "$MODEL" \
    --vector_dir "$VECTOR_DIR" \
    --trait "evil" \
    --output_dir "$OUTPUT_DIR" \
    --vector_type "$VECTOR_TYPE"

# Multi-trait comparison (Layer 20 = index 19)
PYTHONPATH=. uv run python -m src.head_analysis.head_contribution.main compare_traits \
    --model_name "$MODEL" \
    --vector_dir "$VECTOR_DIR" \
    --traits "evil,humorous,sycophantic,hallucinating" \
    --layer 19 \
    --output_dir "$OUTPUT_DIR" \
    --vector_type "$VECTOR_TYPE"

echo "=== Done ==="
