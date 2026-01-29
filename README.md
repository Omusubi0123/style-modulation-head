# Steering at the Source: Style Modulation Heads for Robust Persona Control

A research toolkit for analyzing and steering persona-related behaviors in Large Language Models through activation engineering at various positions (residual stream, attention output, MLP output, and individual attention heads).

## Overview

This repository provides tools for:

1. **Persona Vector Extraction**: Extract steering vectors from model activations that represent specific personality traits
2. **Steering Evaluation**: Evaluate how steering vectors affect model behavior and output coherency
3. **Layer-wise Analysis**: Analyze cosine similarity patterns across transformer layers
4. **Head-wise Analysis**: Identify which attention heads contribute most to persona-related behaviors
5. **Steering Position Comparison**: Compare the effectiveness of steering at different positions within transformer blocks

## Supported Models

- `Qwen/Qwen2.5-7B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`

## Supported Traits

Default traits include: `evil`, `sycophantic`, `hallucinating`, `humorous`, `passionate`, `loser`

Custom traits can be added by creating new JSON files in `data_generation/trait_data_eval/` and `data_generation/trait_data_extract/`.

## Environment Setup

### Prerequisites

- Python 3.10 (required: `>=3.10, <3.11`)
- CUDA-compatible GPU with sufficient VRAM (recommended: 24GB+)
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. Clone the repository: (after the paper review)
```bash
git clone https://github.com/your-repo/style-modulation-head.git
cd style-modulation-head
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up environment variables by creating a `.env` file:
```bash
# .env
OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token
```

The OpenAI API key is required for the LLM-as-judge evaluation. The HuggingFace token is required for accessing gated models.

## Project Structure

```
style-modulation-head/
├── data_generation/           # Trait data for evaluation
│   ├── prompts.py            # System prompts for personas
│   ├── trait_data_eval/      # Questions for steering evaluation
│   └── trait_data_extract/   # Questions for vector extraction
├── scripts/                   # Shell scripts for running experiments
│   ├── generate_all_vectors.sh
│   ├── run_eval_steering.sh
│   ├── run_eval_steering_block.sh
│   ├── run_eval_steering_head.sh
│   ├── run_steering_position_comparison.sh
│   └── lib/
│       └── eval_common.sh    # Common functions for evaluation scripts
├── src/
│   ├── activation_steer/     # Activation steering classes
│   │   ├── base/             # Base classes (modifier, steerer, ablator)
│   │   ├── activation_steer.py
│   │   ├── activation_steer_head.py
│   │   └── activation_ablation.py
│   ├── eval/                 # Evaluation scripts
│   │   ├── common/           # Common evaluation utilities
│   │   ├── eval_persona.py
│   │   ├── eval_persona_steer_block.py
│   │   ├── eval_persona_steer_head.py
│   │   └── eval_persona_steer_residual_stream.py
│   ├── generate_vec/         # Vector generation scripts
│   │   ├── generate_vec.py
│   │   ├── generate_vec_attn.py
│   │   └── generate_vec_block.py
│   ├── layer_analysis/       # Layer-wise analysis
│   │   └── cosine_similarity/
│   ├── head_analysis/        # Head contribution analysis
│   │   └── head_contribution/
│   ├── pareto_analysis/      # Steering position trade-off analysis
│   ├── config.py             # Configuration management
│   └── chat_template_utils.py
└── pyproject.toml
```

## Usage

### Step 1: Persona Vector Extraction

Extract persona vectors from model activations. This step generates pos/neg instruction response data and computes difference vectors.

```bash
# Extract all vectors for all models and traits
./scripts/generate_all_vectors.sh

# Or run individual components
GPU=0 uv run python src/eval/eval_persona.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --trait evil \
    --output_path data/eval_persona_extract/Qwen_Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --persona_instruction_type pos \
    --version extract

# Generate transformer block output vectors
uv run python src/generate_vec/generate_vec.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --pos_path data/eval_persona_extract/.../evil_pos_instruct.csv \
    --neg_path data/eval_persona_extract/.../evil_neg_instruct.csv \
    --trait evil \
    --save_dir data/persona_vectors/Qwen_Qwen2.5-7B-Instruct/

# Generate attention pre-O projection vectors (for head-level steering)
uv run python src/generate_vec/generate_vec_attn.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --pos_path ... --neg_path ... --trait evil \
    --save_dir data/persona_vectors/Qwen_Qwen2.5-7B-Instruct/

# Generate block-level vectors (attn/mlp input/output, layernorms)
uv run python src/generate_vec/generate_vec_block.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --pos_path ... --neg_path ... --trait evil \
    --save_dir data/persona_vectors/Qwen_Qwen2.5-7B-Instruct/
```

**Output files:**
- `{trait}_response_avg_diff.pt` - Transformer block output difference vectors
- `{trait}_response_avg_diff_attn_pre_o_proj.pt` - Attention pre-O projection vectors
- `{trait}_response_avg_diff_attn_output.pt` - Attention output vectors
- `{trait}_response_avg_diff_mlp_output.pt` - MLP output vectors
- `{trait}_prompt_avg_diff_*.pt` - Prompt-based difference vectors

### Step 2: Steering Evaluation

Evaluate the effect of steering vectors on model behavior.

```bash
# Standard steering evaluation (residual stream)
./scripts/run_eval_steering.sh

# Or run manually
GPU=0 uv run python src/eval/eval_persona.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --trait evil \
    --output_path data/eval_persona_eval/.../evil_steer.csv \
    --version eval \
    --steering_type response \
    --coef 2.0 \
    --vector_path data/persona_vectors/.../evil_response_avg_diff.pt \
    --persona_instruction_type neg \
    --layer 20 \
    --judge_model gpt-4.1-mini-2025-04-14

# Block-level steering (attn_output or mlp_output)
./scripts/run_eval_steering_block.sh "Qwen/Qwen2.5-7B-Instruct" evil humorous

# Head-level steering
./scripts/run_eval_steering_head.sh "Qwen/Qwen2.5-7B-Instruct" evil humorous
```

**Key Parameters:**
- `--coef`: Steering coefficient (positive = amplify, negative = suppress)
- `--layer`: Target layer for steering (0-indexed)
- `--persona_instruction_type`: System prompt type (`pos` or `neg`)
- `--steering_type`: When to apply steering (`response`, `prompt`, or `all`)

### Step 3: Layer-wise Analysis

Analyze cosine similarity patterns across transformer layers.

```bash
# Run cosine similarity analysis
./src/layer_analysis/run_cosine_similarity.sh

# Or run manually
uv run python src/layer_analysis/cosine_similarity/main.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --persona_vectors_dir data/persona_vectors \
    --output_dir data/layer_analysis_results \
    --vector_type response_avg_diff
```

**Output:**
- Heatmaps showing cosine similarity between persona vectors across layers
- Adjacent layer difference visualizations

### Step 4: Head-wise Analysis

Identify which attention heads contribute most to persona-related behaviors.

```bash
# Run head contribution analysis
./src/head_analysis/run_head_contribution.sh

# Or run manually
uv run python src/head_analysis/head_contribution/main.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --vector_dir data/persona_vectors/Qwen_Qwen2.5-7B-Instruct \
    --trait evil \
    --output_dir data/head_analysis_results \
    --vector_type response_avg

# Cross-trait comparison at specific layers
./scripts/run_analyze_head_contribution_compare.sh
```

**Output:**
- Heatmaps showing head contribution scores
- Cross-trait comparison visualizations

### Step 5: Steering Position Comparison

Compare the effectiveness of steering at different positions within transformer blocks.

```bash
# Run steering position comparison experiments
MODEL="Qwen/Qwen2.5-7B-Instruct" \
LAYER=20 \
NUM_HEADS=28 \
CORRELATED_HEADS="2,4,27" \
CORRELATED_ANTI_HEADS="0,2,4,26,27" \
NUM_CORRELATED_HEADS=3 \
NUM_CORRELATED_ANTI_HEADS=5 \
./scripts/run_steering_position_comparison.sh
```

After experiments, analyze results with Pareto analysis:

```bash
# Transform logs to CSV and generate plots
./src/pareto_analysis/run.sh

# Generate Pareto curve plots
./src/pareto_analysis/run_pareto_plot.sh --model qwen --traits "evil,sycophantic"
```

**Steering Positions:**
1. Post-attention residual stream
2. Post-MLP residual stream
3. Attention output (before residual addition)
4. Correlated attention heads only
5. Correlated + anti-correlated attention heads

**Output:**
- CSV files with trait scores and coherency scores
- Pareto frontier visualizations showing trait-coherency trade-offs

## Evaluation Metrics

The evaluation uses an LLM-as-judge approach with two metrics:

1. **Trait Score (0-100)**: How strongly the model exhibits the target trait
2. **Coherency Score (0-100)**: How coherent and well-formed the response is

The goal is to maximize trait score while maintaining high coherency.


## License

This project is released under the MIT License.

