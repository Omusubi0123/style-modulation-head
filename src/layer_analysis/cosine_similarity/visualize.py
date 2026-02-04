"""
visualize.py - Cosine similarity visualization logic
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def create_residual_stream_labels(num_layers: int) -> list[str]:
    """Create labels for residual stream

    Args:
        num_layers: Number of layers

    Returns:
        list of labels in format ['attn_L1', 'mlp_L1', 'attn_L2', 'mlp_L2', ...]
    """
    labels = []
    for layer_idx in range(num_layers):
        labels.append(f"attn_L{layer_idx + 1}")
        labels.append(f"mlp_L{layer_idx + 1}")
    return labels


def visualize_layer_cosine_similarity(
    similarity_matrix: np.ndarray,
    save_path: Path,
    layer_labels: Optional[list[str]] = None,
    figsize: tuple = (10, 10),
) -> str:
    """Visualize layer-wise cosine similarity as heatmap

    Args:
        similarity_matrix: Similarity matrix [n, n]
        save_path: Path to save the figure
        layer_labels: Axis labels (1-indexed numbers if None)
        figsize: Figure size

    Returns:
        Save path
    """
    num_positions = similarity_matrix.shape[0]

    if layer_labels is None:
        layer_labels = [str(i + 1) for i in range(num_positions)]

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        similarity_matrix,
        annot=False,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        xticklabels=layer_labels,
        yticklabels=layer_labels,
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
    )

    plt.xlabel("Position")
    plt.ylabel("Position")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = str(save_path).replace(".png", ".pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")
    return str(save_path)


def visualize_residual_stream_similarity(
    similarity_matrix: np.ndarray,
    num_layers: int,
    save_path: Path,
    figsize: tuple = (12, 10),
    tick_interval: int = 4,
) -> str:
    """Visualize residual stream cosine similarity as heatmap

    Args:
        similarity_matrix: Similarity matrix [num_layers*2, num_layers*2]
        num_layers: Number of layers
        save_path: Path to save the figure
        figsize: Figure size
        tick_interval: Interval for displaying labels

    Returns:
        Save path
    """
    labels = create_residual_stream_labels(num_layers)
    total_positions = len(labels)

    # Thin out displayed labels (too many makes it unreadable)
    tick_positions = list(range(0, total_positions, tick_interval))
    tick_labels = [labels[i] for i in tick_positions]

    plt.figure(figsize=figsize)

    ax = sns.heatmap(
        similarity_matrix,
        annot=False,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        cbar_kws={"label": "Cosine Similarity", "shrink": 0.8},
    )

    # Set axis ticks
    ax.set_xticks([i + 0.5 for i in tick_positions])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks([i + 0.5 for i in tick_positions])
    ax.set_yticklabels(tick_labels, rotation=0, fontsize=8)

    plt.xlabel("Position in Residual Stream")
    plt.ylabel("Position in Residual Stream")
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = str(save_path).replace(".png", ".pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")
    return str(save_path)


def visualize_adjacent_difference_lineplot(
    differences: torch.Tensor,
    save_path: Path,
    figsize: tuple = (14, 5),
) -> str:
    """Visualize adjacent layer similarity differences as line plot

    Args:
        differences: Difference vector
        save_path: Path to save the figure
        figsize: Figure size

    Returns:
        Save path
    """
    num_layers = len(differences)
    x = list(range(2, num_layers + 2))
    y = differences.numpy()

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, y, marker="o", linewidth=2, markersize=4, color="#2563eb")
    ax.fill_between(
        x, y, 0, where=(y >= 0), alpha=0.3, color="#3b82f6", interpolate=True
    )
    ax.fill_between(
        x, y, 0, where=(y < 0), alpha=0.3, color="#ef4444", interpolate=True
    )
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    max_abs = max(abs(y.min()), abs(y.max()), 0.5)
    ax.set_ylim(-max_abs * 1.2, max_abs * 1.2)

    ax.grid(True, alpha=0.3, linestyle="-")
    ax.set_axisbelow(True)
    ax.set_xlabel("Layer i (sim(L_{i-1}→L_i) - sim(L_i→L_{i+1}))")
    ax.set_ylabel("Similarity Difference")

    # Statistics
    stats_text = (
        f"Mean: {y.mean():.3f} ± {y.std():.3f}\n"
        f"Min: {y.min():.3f} (L{y.argmin() + 2})\n"
        f"Max: {y.max():.3f} (L{y.argmax() + 2})"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = str(save_path).replace(".png", ".pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")
    return str(save_path)
