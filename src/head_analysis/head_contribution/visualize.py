"""
visualize.py - Head Contribution可視化ロジック
"""

import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def visualize_head_contribution_heatmap(
    matrix: np.ndarray,
    layer_list: List[int],
    num_heads: int,
    save_path: str,
    title: Optional[str] = None,
    xlabel: str = "Head Index",
    ylabel: str = "Layer Index",
    cbar_label: str = "Contribution Score",
    figsize: Optional[tuple] = None,
) -> str:
    """Head contributionヒートマップを可視化

    Args:
        matrix: 貢献度行列 [num_layers, num_heads]
        layer_list: 層インデックスのリスト
        num_heads: ヘッド数
        save_path: 保存パス
        title: タイトル（オプション）
        xlabel: X軸ラベル
        ylabel: Y軸ラベル
        cbar_label: カラーバーラベル
        figsize: 図のサイズ（Noneの場合は自動計算）

    Returns:
        保存パス
    """
    n_layers = len(layer_list)
    n_heads = num_heads

    # 図のサイズを自動計算
    if figsize is None:
        base_size = max(8, max(n_heads, n_layers) * 0.35)
        figsize = (base_size, base_size)

    plt.figure(figsize=figsize)

    # カラーマップの設定
    vmin = matrix.min()
    vmax = matrix.max()

    if vmin < 0:
        abs_max = max(abs(vmin), abs(vmax))
        cmap = "RdBu_r"
        vmin_plot, vmax_plot = -abs_max, abs_max
    else:
        cmap = "YlOrRd"
        vmin_plot, vmax_plot = vmin, vmax

    ax = sns.heatmap(
        matrix,
        xticklabels=[f"H{i+1}" for i in range(num_heads)],
        yticklabels=[f"L{l+1}" for l in layer_list],
        cmap=cmap,
        vmin=vmin_plot,
        vmax=vmax_plot,
        annot=False,
        cbar_kws={"label": cbar_label},
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold", rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")
    return save_path


def visualize_traits_comparison_heatmap(
    matrix: np.ndarray,
    trait_labels: List[str],
    num_heads: int,
    save_path: str,
    title: Optional[str] = None,
    xlabel: str = "Head Index",
    ylabel: str = "Trait",
    cbar_label: str = "Contribution Score",
    figsize: Optional[tuple] = None,
) -> str:
    """複数trait比較のヒートマップを可視化

    Args:
        matrix: 貢献度行列 [num_traits, num_heads]
        trait_labels: trait名のリスト
        num_heads: ヘッド数
        save_path: 保存パス
        title: タイトル（オプション）
        xlabel: X軸ラベル
        ylabel: Y軸ラベル
        cbar_label: カラーバーラベル
        figsize: 図のサイズ

    Returns:
        保存パス
    """
    n_traits = len(trait_labels)
    n_heads = num_heads

    if figsize is None:
        base_size = max(8, max(n_heads, n_traits) * 0.35)
        figsize = (base_size, int(base_size / 5 * 3))

    plt.figure(figsize=figsize)

    vmin = matrix.min()
    vmax = matrix.max()

    if vmin < 0:
        abs_max = max(abs(vmin), abs(vmax))
        cmap = "RdBu_r"
        vmin_plot, vmax_plot = -abs_max, abs_max
    else:
        cmap = "YlOrRd"
        vmin_plot, vmax_plot = vmin, vmax

    ax = sns.heatmap(
        matrix,
        xticklabels=[f"H{i+1}" for i in range(num_heads)],
        yticklabels=trait_labels,
        cmap=cmap,
        vmin=vmin_plot,
        vmax=vmax_plot,
        annot=False,
        cbar=True,
        cbar_kws={"label": cbar_label, "pad": 0.02},
    )

    ax.set_xticklabels(ax.get_xticklabels(), fontweight="bold", rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontweight="bold")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.savefig(save_path.replace(".png", ".pdf"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")
    return save_path


def print_top_heads(
    matrix: np.ndarray,
    layer_list: List[int],
    num_heads: int,
    top_k: int = 10,
    raw_matrix: Optional[np.ndarray] = None,
) -> None:
    """上位のヘッドを表示

    Args:
        matrix: 正規化された貢献度行列 [num_layers, num_heads]
        layer_list: 層インデックスのリスト
        num_heads: ヘッド数
        top_k: 表示する上位の数
        raw_matrix: 生の内積値行列（オプション）
    """
    print(f"\nTop {top_k} heads by contribution (normalized):")
    flat_idx = np.argsort(matrix.flatten())[::-1]

    for rank, idx in enumerate(flat_idx[:top_k]):
        layer_idx = layer_list[idx // num_heads]
        head_idx = idx % num_heads
        sim_norm = matrix[idx // num_heads, head_idx]

        if raw_matrix is not None:
            sim_raw = raw_matrix[idx // num_heads, head_idx]
            print(f"  {rank + 1}. Layer {layer_idx+1}, Head {head_idx+1}: {sim_norm:.4f} (raw: {sim_raw:.2e})")
        else:
            print(f"  {rank + 1}. Layer {layer_idx+1}, Head {head_idx+1}: {sim_norm:.4f}")

