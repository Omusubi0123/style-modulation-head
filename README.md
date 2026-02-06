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

1. Clone the repository:
```bash
git clone https://github.com/your-repo/style-modulation-head.git
cd style-modulation-head
```

2. Install dependencies using uv:
```bash
uv sync
```

3. **Set up environment variables** (Step 0):

Create a `.env` file from the example template:
```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
HF_TOKEN=your_huggingface_token_here
```

**Required API Keys:**
- `OPENAI_API_KEY`: Required for LLM-as-judge evaluation (e.g., `gpt-4.1-mini-2025-04-14`)
- `HF_TOKEN`: Required for accessing gated models on HuggingFace

The application reads these values via `src/settings.py` using `pydantic-settings`.

## ⚠️ Important: Layer Index Convention

**All layer indices in this repository are 0-indexed transformer blocks.**

- `--layer k` corresponds to `model.layers[k]` (the k-th transformer block, 0-indexed)
- Vector files follow the same convention: `tensor[k]` corresponds to `model.layers[k]`
- For example, `--layer 19` targets the 20th transformer block (1-indexed)

This convention applies to:
- All `generate_vec*.py` scripts
- All `eval_persona.py` steering operations
- All shell scripts (`run_eval_steering*.sh`)

**Example:**
```python
# To steer at the 20th transformer block (1-indexed):
# Use --layer 19 (0-indexed)
vector = torch.load("evil_response_avg_diff.pt")[19]  # tensor[19] = model.layers[19]
```

## Project Structure

```
style-modulation-head/
├── .env.example              # Environment variable template
├── data_generation/           # Trait data for evaluation
│   ├── prompts.py            # System prompts for personas
│   ├── trait_data_eval/      # Questions for steering evaluation
│   └── trait_data_extract/   # Questions for vector extraction
├── scripts/                   # Shell scripts for running experiments
│   ├── generate_all_vectors.sh
│   ├── run_eval_steering.sh
│   ├── run_eval_steering_block.sh
│   ├── run_eval_steering_head.sh
│   ├── run_eval_style_head_ablation.sh
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
│   │   ├── eval_ablation_block.py # Ablation evaluation scripts
│   │   ├── eval_ablation_style_head.py
│   │   └── eval_persona.py   # Unified persona eval (layer_output/block/head steering)
│   ├── generate_vec/         # Vector generation scripts
│   │   ├── generate_vec.py   # Residual stream vector generation
│   │   ├── generate_vec_head.py # Attention pre-O-projection vector generation
│   │   └── generate_vec_block.py # Block-level vector generation
│   ├── layer_analysis/       # Layer-wise analysis
│   │   └── cosine_similarity/ # Cosine similarity analysis
│   ├── head_analysis/        # Head contribution analysis
│   │   └── head_contribution/ # Head contribution analysis
│   ├── pareto_analysis/     # Steering position trade-off analysis
│   ├── settings.py           # Environment variable management (pydantic-settings)
│   └── chat_template_utils.py # Chat template utilities
```

## Usage

### Step 1: Persona Vector Extraction

Extract persona vectors from model activations. This step generates pos/neg instruction response data and computes difference vectors.

#### Using Shell Scripts (Recommended)

```bash
# Extract all vectors for a model and traits
./scripts/generate_all_vectors.sh "Qwen/Qwen2.5-7B-Instruct" evil humorous passionate
```

This script automatically:
1. Generates pos/neg CSV files (if not exist)
2. Generates residual stream vectors (`generate_vec.py`)
3. Generates attention pre-O-projection vectors (`generate_vec_head.py`)
4. Generates block-level vectors (`generate_vec_block.py`)

#### Manual Execution

```bash
# Step 1a: Generate pos/neg CSV files
GPU=0 uv run python src/eval/eval_persona.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --trait evil \
    --output_path data/eval_persona_extract/Qwen_Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --persona_instruction_type pos \
    --assistant_name evil \
    --version extract

GPU=0 uv run python src/eval/eval_persona.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --trait evil \
    --output_path data/eval_persona_extract/Qwen_Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --persona_instruction_type neg \
    --assistant_name helpful \
    --version extract

# Step 1b: Generate residual stream vectors (transformer block outputs)
GPU=0 uv run python src/generate_vec/generate_vec.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --pos_path data/eval_persona_extract/Qwen_Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --neg_path data/eval_persona_extract/Qwen_Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --trait evil \
    --save_dir data/persona_vectors/Qwen_Qwen2.5-7B-Instruct/ \
    --threshold 50

# Step 1c: Generate attention pre-O-projection vectors (for head-level steering)
GPU=0 uv run python src/generate_vec/generate_vec_head.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --pos_path data/eval_persona_extract/Qwen_Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --neg_path data/eval_persona_extract/Qwen_Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --trait evil \
    --save_dir data/persona_vectors/Qwen_Qwen2.5-7B-Instruct/ \
    --threshold 50

# Step 1d: Generate block-level vectors (attn/mlp input/output, layernorms)
GPU=0 uv run python src/generate_vec/generate_vec_block.py \
    --model_name "Qwen/Qwen2.5-7B-Instruct" \
    --pos_path data/eval_persona_extract/Qwen_Qwen2.5-7B-Instruct/evil_pos_instruct.csv \
    --neg_path data/eval_persona_extract/Qwen_Qwen2.5-7B-Instruct/evil_neg_instruct.csv \
    --trait evil \
    --save_dir data/persona_vectors/Qwen_Qwen2.5-7B-Instruct/ \
    --threshold 50
```

**Output files:**
- `{trait}_response_avg_diff.pt` - Transformer block output difference vectors `[num_layers, hidden_size]`
- `{trait}_response_avg_diff_attn_pre_o_proj.pt` - Attention pre-O projection vectors `[num_layers, hidden_size]`
- `{trait}_response_avg_diff_attn_output.pt` - Attention output vectors `[num_layers, hidden_size]`
- `{trait}_response_avg_diff_mlp_output.pt` - MLP output vectors `[num_layers, hidden_size]`
- `{trait}_prompt_avg_diff_*.pt` - Prompt-based difference vectors

**Note:** All vector files use 0-indexed layer convention: `tensor[k]` = `model.layers[k]`.

### Step 2: Steering Evaluation

Evaluate the effect of steering vectors on model behavior.

#### Using Shell Scripts (Recommended)

```bash
# Residual stream (layer output) steering
./scripts/run_eval_steering.sh "Qwen/Qwen2.5-7B-Instruct" evil humorous passionate

# Block-level steering (attn_output, mlp_output, etc.)
./scripts/run_eval_steering_block.sh "Qwen/Qwen2.5-7B-Instruct" evil humorous

# Head-level steering
./scripts/run_eval_steering_head.sh "Qwen/Qwen2.5-7B-Instruct" evil humorous
```

**Environment Variables** (override defaults):
- `GPU`: GPU device ID (default: `0`)
- `STEERING_TYPE`: When to apply steering - `response`, `prompt`, or `all` (default: `response`)
- `PERSONA_INSTRUCTION_TYPE`: System prompt type - `pos` or `neg` (default: `neg`)
- `JUDGE_MODEL`: Judge model name (default: `gpt-4.1-mini-2025-04-14`)
- `BATCH_SIZE`: Batch size for generation (default: `100`)

Example with custom settings:
```bash
GPU=1 STEERING_TYPE=all PERSONA_INSTRUCTION_TYPE=pos \
./scripts/run_eval_steering.sh "Qwen/Qwen2.5-7B-Instruct" evil
```

#### Manual Execution

```bash
# Residual stream steering (layer output)
GPU=0 uv run python src/eval/eval_persona.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --trait evil \
    --output_path data/eval_persona_eval/steering_results/Qwen_Qwen2.5-7B-Instruct/evil_steer_response_neg_layer19_coef2.0.csv \
    --version eval \
    --steering_type response \
    --steering_target layer_output \
    --coef 2.0 \
    --vector_path data/persona_vectors/Qwen_Qwen2.5-7B-Instruct/evil_response_avg_diff.pt \
    --persona_instruction_type neg \
    --layer 19 \
    --judge_model gpt-4.1-mini-2025-04-14

# Block-level steering (attention output)
GPU=0 uv run python src/eval/eval_persona.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --trait evil \
    --output_path data/eval_persona_eval/steering_results_block/.../evil_steer_block_attn_output_response_neg_layer19_coef2.5.csv \
    --version eval \
    --steering_type response \
    --steering_target attn_output \
    --coef 2.5 \
    --vector_path data/persona_vectors/Qwen_Qwen2.5-7B-Instruct/evil_response_avg_diff_attn_output.pt \
    --persona_instruction_type neg \
    --layer 19 \
    --judge_model gpt-4.1-mini-2025-04-14

# Head-level steering (specific attention heads)
GPU=0 uv run python src/eval/eval_persona.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --trait evil \
    --output_path data/eval_persona_eval/steering_results_head/.../evil_steer_head_response_neg_layer19_head2_coef7.0.csv \
    --version eval \
    --steering_type response \
    --steering_target head \
    --coef 7.0 \
    --vector_path data/persona_vectors/Qwen_Qwen2.5-7B-Instruct/evil_response_avg_diff_attn_pre_o_proj.pt \
    --persona_instruction_type neg \
    --layer 19 \
    --head_indices "2,4,27" \
    --judge_model gpt-4.1-mini-2025-04-14
```

**Key Parameters:**
- `--coef`: Steering coefficient (positive = amplify, negative = suppress)
- `--layer`: Target layer for steering (**0-indexed transformer block**, e.g., `19` = 20th block)
- `--steering_target`: Steering position:
  - `layer_output` (default): Transformer block output (residual stream)
  - `attn_output`, `mlp_output`: Block submodule outputs
  - `attn`, `mlp`: Block submodule inputs (post-LayerNorm)
  - `attn_layernorm`, `mlp_layernorm`: LayerNorm inputs
  - `head`: Attention head O projection input (requires `--head_indices`)
- `--persona_instruction_type`: System prompt type (`pos` or `neg`)
- `--steering_type`: When to apply steering (`response`, `prompt`, or `all`)

### Step 3: Style Head Ablation

Investigate how generated text changes when cumulatively zero-ablating style heads.

```bash
# Run style head ablation evaluation
./scripts/run_eval_style_head_ablation.sh "Qwen/Qwen2.5-7B-Instruct" evil humorous

# With custom settings
POSITIONS=response MAX_TOKENS=2000 \
./scripts/run_eval_style_head_ablation.sh "Qwen/Qwen2.5-7B-Instruct" evil
```

### Step 4: Steering Position Comparison

Compare the effectiveness of steering at different positions within transformer blocks.

```bash
# Run steering position comparison experiments
MODEL="Qwen/Qwen2.5-7B-Instruct" \
LAYER=19 \
NUM_HEADS=28 \
CORRELATED_HEADS="2,4,27" \
CORRELATED_ANTI_HEADS="0,2,4,26,27" \
NUM_CORRELATED_HEADS=3 \
NUM_CORRELATED_ANTI_HEADS=5 \
./scripts/run_steering_position_comparison.sh
```

**Steering Positions:**
1. Post-attention residual stream (vector: `mlp_layernorm`, apply: `attn_output`)
2. Post-MLP residual stream (vector: `attn_layernorm`, apply: `mlp_output`)
3. Attention output (vector: `attn_output`, apply: `attn_output`)
4. Correlated attention heads only
5. Correlated + anti-correlated attention heads

**Note:** `LAYER` is 0-indexed (e.g., `LAYER=19` targets the 20th transformer block).

After experiments, analyze results with Pareto analysis:

```bash
# Transform logs to CSV and generate plots
./src/pareto_analysis/run.sh

# Generate Pareto curve plots
./src/pareto_analysis/run_pareto_plot.sh --model qwen --traits "evil,sycophantic"
```

**Output:**
- CSV files with trait scores and coherency scores
- Pareto frontier visualizations showing trait-coherency trade-offs

### Step 5: Layer-wise Analysis

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

### Step 6: Head-wise Analysis

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

## Evaluation Metrics

The evaluation uses an LLM-as-judge approach with two metrics:

1. **Trait Score (0-100)**: How strongly the model exhibits the target trait
2. **Coherency Score (0-100)**: How coherent and well-formed the response is

The goal is to maximize trait score while maintaining high coherency.

## License

This project is released under the MIT License.
