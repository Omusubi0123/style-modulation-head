# Pareto Analysis

This module provides tools for analyzing the trade-off between trait score and coherency in persona steering experiments.

## Overview

The analysis pipeline computes and visualizes Pareto frontiers for different steering positions (MLP residual, Attention residual, Attention output, and Head-level steering).

## Pipeline

```
Experiment Logs → CSV Transformation → Score Extraction → Pareto Plots / Pareto Scores
```

1. **transform_log_to_csv.py**: Parses experiment log files and extracts results into structured CSV
2. **split_scores_variances.py**: Transforms wide-format CSV to long-format with separate columns for trait/coherence scores
3. **plot_pareto_curve.py**: Generates Pareto frontier visualizations
4. **pareto_score.py**: Computes quantitative Pareto scores using envelope area metrics

## CSV Format

### Input Format (after `split_scores_variances.py`)

The analysis scripts expect CSV files with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `trait` | string | Name of the persona trait (e.g., "evil", "sycophantic") |
| `module` | string | Steering position module (see below) |
| `steering_method` | string | One of: `neg_add`, `pos_add`, `pos_subtract` |
| `multiplier` | float | Steering coefficient (e.g., 0.5, 1.0, 2.0, ...) |
| `value` | float | Trait score (0-100) |
| `value_std` | float | Standard deviation of trait score |
| `coherence` | float | Coherency score (0-100) |
| `coherence_std` | float | Standard deviation of coherency score |

### Module Values

| Module | Description |
|--------|-------------|
| `attn_residual` | Post-attention residual stream |
| `mlp_residual` | Post-MLP residual stream |
| `attn_output` | Attention output (before residual addition) |
| `head_cor` | Correlated attention heads only |
| `head_cor_anti` | Correlated + anti-correlated heads |

### Steering Methods

| Method | Description |
|--------|-------------|
| `neg_add` | Negative system prompt + steering vector addition (enhance trait) |
| `pos_add` | Positive system prompt + steering vector addition (enhance trait) |
| `pos_subtract` | Positive system prompt + steering vector subtraction (suppress trait) |

### Example CSV

```csv
trait,module,steering_method,multiplier,value,value_std,coherence,coherence_std
evil,attn_residual,neg_add,0.5,45.2,3.1,92.5,1.8
evil,attn_residual,neg_add,1.0,52.8,4.2,89.3,2.1
evil,attn_residual,neg_add,2.0,68.5,5.3,82.1,3.4
...
```

## Usage

### Complete Pipeline

```bash
# Run the full analysis pipeline for a specific trait
./src/pareto_analysis/run.sh
```

Edit `run.sh` to configure:
- `model`: "qwen" or "llama"
- `MODEL`: Full model name
- `TRAIT`: Target trait to analyze

### Individual Scripts

#### 1. Transform Log to CSV

```bash
uv run python src/pareto_analysis/transform_log_to_csv.py \
    --input_path logs/steering_position_comparison/qwen_evil.log \
    --output_path data/tmp.csv \
    --extract_trait evil
```

#### 2. Split Scores and Variances

```bash
uv run python src/pareto_analysis/split_scores_variances.py \
    --input_file data/tmp.csv \
    --output_file data/steering_position_plot/Qwen2.5-7B-Instruct/formatted.csv
```

#### 3. Generate Pareto Plots

```bash
# Single file
uv run python src/pareto_analysis/plot_pareto_curve.py single \
    --input_file data/steering_position_plot/Qwen2.5-7B-Instruct/formatted.csv \
    --output_dir data/plots \
    --trait evil

# All traits for a model
uv run python src/pareto_analysis/plot_pareto_curve.py all \
    --model qwen \
    --traits "evil,sycophantic,hallucinating"
```

#### 4. Compute Pareto Score

```bash
# Command line with sample data
uv run python src/pareto_analysis/pareto_score.py \
    --pareto_points "90,60;80,70;70,80;60,85" \
    --tau 50.0 \
    --x_max_common 90.0 \
    --envelope_type lower
```

**Parameters:**
- `tau`: Minimum coherency threshold (x-axis lower bound)
- `x_max_common`: Maximum coherency for integration (x-axis upper bound)
- `envelope_type`: 
  - `"upper"`: Use upper envelope (optimistic estimate)
  - `"lower"`: Use lower envelope (conservative estimate, default)

### Generate Pareto Plots Only

```bash
./src/pareto_analysis/run_pareto_plot.sh
./src/pareto_analysis/run_pareto_plot.sh --model qwen --traits "evil,sycophantic"
```

## Pareto Score Algorithm

The Pareto score is computed as the area under the envelope curve:

```
Score_τ(P) = (1 / (x_max - τ)) * ∫_τ^{x_max} y_P(x) dx
```

Where:
- `P` is the set of Pareto points (coherency, trait_score)
- `τ` is the minimum coherency constraint
- `x_max` is the maximum coherency across all compared methods
- `y_P(x)` is the envelope function:
  - Upper: `y_P(x) = max { y | (x', y) ∈ P, x' ≥ x }`
  - Lower: `y_P(x) = value at the right boundary of each segment`

Higher scores indicate better Pareto frontiers (achieving higher trait scores at given coherency levels).

## Output

### Pareto Plots

PNG and PDF files showing:
- X-axis: Trait score
- Y-axis: Coherency score
- Each steering module as a different colored line with arrows showing coefficient progression

### Pareto Scores

Numerical scores (0-100) quantifying the quality of each Pareto frontier for comparison across steering methods.
