"""
main.py - Head Contribution分析のエントリーポイント

使用例:
    # 単一traitの分析
    python -m src.head_analysis.head_contribution.main analyze_trait \
        --model_name "Qwen/Qwen2.5-7B-Instruct" \
        --vector_dir "data/persona_vectors/Qwen/Qwen2.5-7B-Instruct" \
        --trait "evil"

    # 特定層での複数trait比較
    python -m src.head_analysis.head_contribution.main compare_traits \
        --model_name "Qwen/Qwen2.5-7B-Instruct" \
        --vector_dir "data/persona_vectors/Qwen/Qwen2.5-7B-Instruct" \
        --traits "evil,humorous,sycophantic" \
        --layer 19
"""

import os
from typing import List, Optional

import fire
import numpy as np
import torch

from src.head_analysis.head_contribution.compute import (
    compute_head_contributions,
    load_attention_config,
    load_o_proj_weights,
    normalize_matrix,
)
from src.head_analysis.head_contribution.visualize import (
    print_top_heads,
    visualize_head_contribution_heatmap,
    visualize_traits_comparison_heatmap,
)


def analyze_trait(
    model_name: str,
    vector_dir: str,
    trait: str,
    layers: Optional[List[int]] = None,
    output_dir: Optional[str] = None,
    vector_type: str = "response_avg",
    use_log: bool = True,
    use_zscore: bool = True,
) -> np.ndarray:
    """単一traitのHead Contributionを分析

    Args:
        model_name: モデル名
        vector_dir: ベクトルディレクトリ
        trait: 特性名
        layers: 分析する層のリスト（Noneで全層）
        output_dir: 出力ディレクトリ（Noneでvector_dir）
        vector_type: ベクトルタイプ
        use_log: 対数スケーリング
        use_zscore: Z-score正規化

    Returns:
        正規化された貢献度行列 [num_layers, num_heads]
    """
    if output_dir is None:
        output_dir = vector_dir

    os.makedirs(output_dir, exist_ok=True)

    # ベクトルを読み込む
    attn_pre_o_proj_path = os.path.join(
        vector_dir, f"{trait}_{vector_type}_diff_attn_pre_o_proj.pt"
    )
    attn_output_path = os.path.join(
        vector_dir, f"{trait}_{vector_type}_diff_attn_output.pt"
    )

    if not os.path.exists(attn_pre_o_proj_path):
        raise FileNotFoundError(f"Not found: {attn_pre_o_proj_path}")
    if not os.path.exists(attn_output_path):
        raise FileNotFoundError(f"Not found: {attn_output_path}")

    attn_pre_o_proj = torch.load(attn_pre_o_proj_path, map_location="cpu")
    attn_output = torch.load(attn_output_path, map_location="cpu")

    # Attention設定を読み込む
    attn_config = load_attention_config(vector_dir, trait)
    if attn_config:
        num_heads = attn_config["num_attention_heads"]
        head_dim = attn_config["head_dim"]
    else:
        # ベクトル形状から推測
        hidden_size = attn_pre_o_proj.shape[-1]
        for n_heads in [32, 28, 40, 24, 16, 12, 8]:
            if hidden_size % n_heads == 0:
                num_heads = n_heads
                head_dim = hidden_size // n_heads
                print(f"Warning: guessing num_heads={num_heads}, head_dim={head_dim}")
                break
        else:
            raise ValueError("Could not determine num_heads")

    print(f"Attention config: num_heads={num_heads}, head_dim={head_dim}")

    num_layers = attn_pre_o_proj.shape[0]
    if layers is None:
        layers = list(range(num_layers))

    # O projection weightsを読み込む
    o_proj_weights = load_o_proj_weights(model_name)

    # 各層・各ヘッドの類似度を計算
    similarity_matrix = np.zeros((len(layers), num_heads))

    for i, layer_idx in enumerate(layers):
        if layer_idx not in o_proj_weights:
            print(f"Warning: O proj weight not found for layer {layer_idx}")
            continue

        similarities = compute_head_contributions(
            attn_pre_o_proj[layer_idx],
            attn_output[layer_idx],
            o_proj_weights[layer_idx],
            num_heads,
            head_dim,
        )
        similarity_matrix[i, :] = similarities

    # 正規化
    normalized_matrix = normalize_matrix(
        similarity_matrix, use_log=use_log, use_zscore=use_zscore, axis=0
    )

    # 可視化
    suffix = ""
    if not use_log:
        suffix += "_no_log"
    if not use_zscore:
        suffix += "_no_zscore"

    output_path = os.path.join(
        output_dir, f"{trait}_head_contribution_{vector_type}{suffix}.png"
    )

    visualize_head_contribution_heatmap(
        normalized_matrix,
        layers,
        num_heads,
        output_path,
    )

    # 上位ヘッドを表示
    print_top_heads(normalized_matrix, layers, num_heads, raw_matrix=similarity_matrix)

    # 行列を保存
    np.save(output_path.replace(".png", ".npy"), normalized_matrix)
    np.save(output_path.replace(".png", "_raw.npy"), similarity_matrix)

    return normalized_matrix


def compare_traits(
    model_name: str,
    vector_dir: str,
    traits: str,
    layer: int,
    output_dir: Optional[str] = None,
    vector_type: str = "response_avg",
    use_log: bool = True,
    use_zscore: bool = True,
) -> np.ndarray:
    """複数traitを特定層で比較

    Args:
        model_name: モデル名
        vector_dir: ベクトルディレクトリ
        traits: カンマ区切りのtrait名
        layer: 層インデックス（0-based）
        output_dir: 出力ディレクトリ
        vector_type: ベクトルタイプ
        use_log: 対数スケーリング
        use_zscore: Z-score正規化

    Returns:
        正規化された貢献度行列 [num_traits, num_heads]
    """
    trait_list = [t.strip() for t in traits.split(",")]

    if output_dir is None:
        output_dir = vector_dir

    os.makedirs(output_dir, exist_ok=True)

    # 最初のtraitからAttention設定を読み込む
    attn_config = load_attention_config(vector_dir, trait_list[0])
    if attn_config is None:
        raise ValueError("Could not load attention config")

    num_heads = attn_config["num_attention_heads"]
    head_dim = attn_config["head_dim"]

    print(f"Attention config: num_heads={num_heads}, head_dim={head_dim}")
    print(f"Analyzing layer {layer} for {len(trait_list)} traits")

    # O projection weightsを読み込む
    o_proj_weights = load_o_proj_weights(model_name)

    if layer not in o_proj_weights:
        raise ValueError(f"O proj weight not found for layer {layer}")

    o_proj_weight = o_proj_weights[layer]

    # 各traitのヘッド貢献度を計算
    similarity_matrix = np.zeros((len(trait_list), num_heads))
    valid_traits = []

    for i, trait in enumerate(trait_list):
        attn_pre_o_proj_path = os.path.join(
            vector_dir, f"{trait}_{vector_type}_diff_attn_pre_o_proj.pt"
        )
        attn_output_path = os.path.join(
            vector_dir, f"{trait}_{vector_type}_diff_attn_output.pt"
        )

        if not os.path.exists(attn_pre_o_proj_path):
            print(f"Warning: Skipping {trait} - attn_pre_o_proj not found")
            continue
        if not os.path.exists(attn_output_path):
            print(f"Warning: Skipping {trait} - attn_output not found")
            continue

        attn_pre_o_proj = torch.load(attn_pre_o_proj_path, map_location="cpu")
        attn_output = torch.load(attn_output_path, map_location="cpu")

        similarities = compute_head_contributions(
            attn_pre_o_proj[layer],
            attn_output[layer],
            o_proj_weight,
            num_heads,
            head_dim,
        )

        similarity_matrix[i, :] = similarities
        valid_traits.append(trait)

    if not valid_traits:
        raise ValueError("No valid traits found")

    # 有効なtraitのみに絞る
    valid_indices = [i for i, t in enumerate(trait_list) if t in valid_traits]
    similarity_matrix = similarity_matrix[valid_indices, :]

    # 正規化（traitごと = axis=0）
    normalized_matrix = normalize_matrix(
        similarity_matrix, use_log=use_log, use_zscore=use_zscore, axis=0
    )

    # 可視化
    suffix = ""
    if not use_log:
        suffix += "_no_log"
    if not use_zscore:
        suffix += "_no_zscore"

    output_path = os.path.join(
        output_dir, f"traits_comparison_layer{layer+1}_{vector_type}{suffix}.png"
    )

    visualize_traits_comparison_heatmap(
        normalized_matrix,
        valid_traits,
        num_heads,
        output_path,
    )

    # 行列を保存
    np.save(output_path.replace(".png", ".npy"), normalized_matrix)
    np.save(output_path.replace(".png", "_raw.npy"), similarity_matrix)

    # 各ヘッドの平均貢献度を表示
    print(f"\nTop 10 heads by average contribution (Layer {layer+1}):")
    head_means = normalized_matrix.mean(axis=0)
    top_heads = np.argsort(head_means)[::-1][:10]
    for rank, head_idx in enumerate(top_heads):
        print(f"  {rank + 1}. Head {head_idx+1}: {head_means[head_idx]:.4f}")

    return normalized_matrix


if __name__ == "__main__":
    fire.Fire({
        "analyze_trait": analyze_trait,
        "compare_traits": compare_traits,
    })

