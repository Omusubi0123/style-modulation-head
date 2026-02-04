"""
plot_pareto_curve.py

Plot steering position comparison results as Pareto frontier curves.

Visualization:
- X-axis: trait score
- Y-axis: coherency score
- Each steering position shown in different colors
- Arrows connect points from smaller to larger coefficients
"""

import os
from typing import List, Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Unified color map (consistent with other plots)
MODULE_ORDER = [
    "mlp_residual",
    "attn_residual",
    "attn_output",
    "head_cor",
    "head_cor_anti",
]
MODULE_COLORS = {
    "mlp_residual": "#27ae60",  # Green
    "attn_residual": "#2980b9",  # Blue
    "attn_output": "#c0392b",  # Red
    "head_cor": "#8e44ad",  # Purple
    "head_cor_anti": "#d35400",  # Orange
}

MODULE_LABELS = {
    "mlp_residual": "MLP Residual",
    "attn_residual": "Attn Residual",
    "attn_output": "Attn Output",
    "head_cor": "Head (Correlated)",
    "head_cor_anti": "Head (Cor+Anti)",
}

METHOD_LABELS = {
    "neg_add": "Negative + Add (Enhance)",
    "pos_add": "Positive + Add (Enhance)",
    "pos_subtract": "Positive + Subtract (Suppress)",
}


def draw_arrow_curve(
    ax,
    points: List[tuple],
    color: str,
    alpha: float = 0.7,
):
    """
    Draw a curve connecting points with arrows.

    Args:
        ax: matplotlib axes
        points: [(x1, y1), (x2, y2), ...] list of points in order
        color: line color
        alpha: transparency
    """
    if len(points) < 2:
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # Draw the curve
    ax.plot(xs, ys, color=color, alpha=alpha, linestyle="-", zorder=1)

    # Add arrows at midpoints
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2

        # Direction vector
        dx = x2 - x1
        dy = y2 - y1

        # Draw arrow only if length is non-zero
        if dx != 0 or dy != 0:
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                # Normalize
                dx_norm = dx / length * 2
                dy_norm = dy / length * 2

                ax.annotate(
                    "",
                    xy=(mid_x + dx_norm, mid_y + dy_norm),
                    xytext=(mid_x - dx_norm, mid_y - dy_norm),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=color,
                        alpha=alpha,
                        lw=8.5,
                        mutation_scale=22,
                    ),
                    zorder=2,
                )


def plot_pareto_curve(
    input_file: str,
    output_dir: str,
    trait: str = None,
    steering_method: str = None,
    filter_modules: str = "mlp_residual,attn_residual,attn_output,head_cor,head_cor_anti",
    figsize: tuple = (10, 8),
    show_coef_labels: bool = False,
    save_pdf: bool = True,
):
    """
    Create Pareto frontier-style plots.

    Args:
        input_file: Path to CSV file (requires value, coherence columns)
        output_dir: Output directory
        trait: Target trait (if None, extracted from CSV)
        steering_method: Target steering method (neg_add, pos_add, pos_subtract)
        filter_modules: Comma-separated list of modules to plot
        figsize: Figure size
        show_coef_labels: Whether to show coefficient labels on each point
        save_pdf: Whether to also save as PDF
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(input_file)

    # Get trait
    if trait is None:
        trait = df["trait"].iloc[0] if "trait" in df.columns else "unknown"

    # Filter modules
    if filter_modules:
        modules_filter = [m.strip() for m in filter_modules.split(",")]
        df = df[df["module"].isin(modules_filter)]

    # Exclude mul_h_div_s variants
    drop_modules = ["head_cor_mul_h_div_s", "head_cor_anti_mul_h_div_s"]
    df = df[~df["module"].isin(drop_modules)]

    if df.empty:
        print(f"No data to plot for trait={trait}")
        return

    # Determine steering methods to plot
    if steering_method:
        methods = [steering_method]
    else:
        methods = ["neg_add", "pos_add", "pos_subtract"]

    for method in methods:
        method_df = df[df["steering_method"] == method].copy()

        if method_df.empty:
            print(f"No data for trait={trait}, method={method}")
            continue

        # Create plot
        fig, ax = plt.subplots(figsize=figsize)

        # Background color
        ax.set_facecolor("#f8f9fa")

        legend_handles = []

        # Plot in module order
        ordered_modules = [m for m in MODULE_ORDER if m in method_df["module"].unique()]

        for module in ordered_modules:
            module_df = method_df[method_df["module"] == module].copy()

            if module_df.empty:
                continue

            # Sort by coefficient
            module_df = module_df.sort_values("multiplier")

            color = MODULE_COLORS.get(module, "#333333")
            label = MODULE_LABELS.get(module, module)

            # Get coordinates
            coherences = module_df["coherence"].values
            values = module_df["value"].values
            multipliers = module_df["multiplier"].values

            # X-axis: trait, Y-axis: coherency
            plot_x = values
            plot_y = coherences

            # Draw curve with arrows
            points = list(zip(plot_x, plot_y))
            draw_arrow_curve(ax, points, color=color, alpha=0.7)

            # Plot all points
            for i, (px, py, mult) in enumerate(zip(plot_x, plot_y, multipliers)):
                ax.scatter(
                    px, py, color=color, s=160, alpha=0.85, edgecolors="white", zorder=3
                )

                if show_coef_labels:
                    ax.annotate(
                        f"{mult:.1f}",
                        (px, py),
                        textcoords="offset points",
                        xytext=(5, 5),
                        fontsize=8,
                        alpha=0.8,
                        fontweight="medium",
                    )

            # Legend handle
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color=color,
                    markerfacecolor=color,
                    markersize=10,
                    label=f"{label}",
                    linestyle="-",
                    markeredgecolor="white",
                )
            )

        # Graph decoration
        ax.set_xlabel("Trait Score")
        ax.set_ylabel("Coherency Score")
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.4, linestyle="--")

        # Add legend
        if legend_handles:
            legend = ax.legend(
                handles=legend_handles,
                loc="best",
                framealpha=0.95,
                edgecolor="#bdc3c7",
                fancybox=True,
                shadow=False,
            )
            legend.get_frame().set_linewidth(1.5)

        plt.tight_layout()

        # Save
        output_path = os.path.join(output_dir, f"pareto_{trait}_{method}.png")

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        if save_pdf:
            plt.savefig(
                output_path.replace(".png", ".pdf"),
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
        plt.close()
        print(f"Saved: {output_path}")


def plot_all_pareto(
    data_dir: str = "data/steering_position_plot",
    output_base_dir: str = "data/steering_position_plot",
    model: str = "qwen",
    traits: str = "evil,sycophantic,hallucinating",
    filter_modules: str = "mlp_residual,attn_residual,attn_output,head_cor,head_cor_anti",
    show_coef_labels: bool = False,
    save_pdf: bool = True,
):
    """
    Plot Pareto curves for all personas and steering methods.

    Args:
        data_dir: Directory containing CSV data
        output_base_dir: Base output directory
        model: Model name ('llama' or 'qwen')
        traits: Comma-separated list of trait names
        filter_modules: Modules to include in plot
        show_coef_labels: Whether to show coefficient labels
        save_pdf: Whether to also save as PDF
    """
    # Model name mapping
    model_names = {
        "llama": "Llama-3.1-8B-Instruct",
        "qwen": "Qwen2.5-7B-Instruct",
    }

    model_prefixes = {
        "llama": "llama",
        "qwen": "qwen",
    }

    if model not in model_names:
        print(f"Error: Unknown model '{model}'. Use 'llama' or 'qwen'.")
        return

    model_name = model_names[model]
    model_prefix = model_prefixes[model]

    # Parse traits
    if isinstance(traits, (list, tuple)):
        trait_list = list(traits)
    else:
        trait_list = [t.strip() for t in traits.split(",")]

    print(f"=== Generating Pareto Curves ===")
    print(f"Model: {model_name}")
    print(f"Traits: {trait_list}")
    print()

    for trait in trait_list:
        # Input file path (formatted CSV without F-value)
        input_file = os.path.join(
            data_dir,
            model_name,
            f"steering_position_comparison_{model_prefix}_{trait}_formatted.csv",
        )

        # Check for F-value file for compatibility
        fvalue_file = os.path.join(
            data_dir,
            model_name,
            f"steering_position_comparison_{model_prefix}_{trait}_fvalue.csv",
        )
        if os.path.exists(fvalue_file):
            input_file = fvalue_file

        if not os.path.exists(input_file):
            print(f"Warning: File not found: {input_file}")
            continue

        # Output directory
        output_dir = os.path.join(output_base_dir, model_name, "pareto_plots")

        print(f"Processing: {trait}")

        # Plot all methods
        plot_pareto_curve(
            input_file=input_file,
            output_dir=output_dir,
            trait=trait,
            steering_method=None,  # All methods
            filter_modules=filter_modules,
            show_coef_labels=show_coef_labels,
            save_pdf=save_pdf,
        )

    print()
    print(f"=== Completed ===")
    print(
        f"Output directory: {os.path.join(output_base_dir, model_name, 'pareto_plots')}"
    )


def plot_single(
    input_file: str,
    output_dir: str,
    trait: str = None,
    steering_method: str = None,
    filter_modules: str = "mlp_residual,attn_residual,attn_output,head_cor,head_cor_anti",
    show_coef_labels: bool = False,
    save_pdf: bool = True,
):
    """
    Plot Pareto curves from a single file (convenient wrapper for CLI).
    """
    plot_pareto_curve(
        input_file=input_file,
        output_dir=output_dir,
        trait=trait,
        steering_method=steering_method,
        filter_modules=filter_modules,
        show_coef_labels=show_coef_labels,
        save_pdf=save_pdf,
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "single": plot_single,
            "all": plot_all_pareto,
        }
    )
