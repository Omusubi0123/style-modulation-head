#!/bin/bash
#
# run_cosine_similarity.sh - Residual Stream Cosine Similarity分析のサンプル実行
#
# 使い方:
#   ./src/layer_analysis/run_cosine_similarity.sh

set -e

# 設定
MODEL="Qwen/Qwen2.5-7B-Instruct"
# MODEL="meta-llama/Llama-3.1-8B-Instruct"
VECTORS_DIR="data/persona_vectors"
OUTPUT_DIR="data/layer_analysis"
VECTOR_TYPE="response_avg_diff"

echo "=== Cosine Similarity Analysis ==="
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"

# 全traitの分析
PYTHONPATH=. uv run python -m src.layer_analysis.cosine_similarity.main analyze_all \
    --model_name "$MODEL" \
    --persona_vectors_dir "$VECTORS_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --vector_type "$VECTOR_TYPE"

# 単一traitの分析例
# PYTHONPATH=. uv run python -m src.layer_analysis.cosine_similarity.main analyze_trait \
#     --model_name "$MODEL" \
#     --trait "evil" \
#     --stream_type "input"

echo "=== Done ==="

