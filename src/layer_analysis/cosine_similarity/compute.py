"""
compute.py - Cosine類似度の計算ロジック
"""

from pathlib import Path
from typing import Dict, List

import torch
from sklearn.metrics.pairwise import cosine_similarity


def load_vectors_for_positions(
    persona_vectors_dir: Path,
    model_name: str,
    vector_type: str,
    layer_positions: List[str],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """複数のlayer positionのベクトルを読み込む

    Args:
        persona_vectors_dir: ベクトルディレクトリのルート
        model_name: モデル名
        vector_type: 'response_avg_diff', 'prompt_avg_diff', 'prompt_last_diff'
        layer_positions: 読み込むlayer positionのリスト

    Returns:
        {trait_name: {layer_position: vector}} の辞書
    """
    vectors_dir = Path(persona_vectors_dir) / model_name
    vectors_dict: Dict[str, Dict[str, torch.Tensor]] = {}

    for layer_position in layer_positions:
        for vector_file in vectors_dir.glob(f"*_{vector_type}_{layer_position}.pt"):
            trait_name = vector_file.stem.replace(f"_{vector_type}_{layer_position}", "")

            try:
                vector = torch.load(vector_file, weights_only=False, map_location="cpu")

                if trait_name not in vectors_dict:
                    vectors_dict[trait_name] = {}
                vectors_dict[trait_name][layer_position] = vector

            except Exception as e:
                print(f"Warning: Failed to load {vector_file}: {e}")
                continue

    return vectors_dict


def compute_layerwise_cosine_similarity(vector: torch.Tensor) -> torch.Tensor:
    """各層ベクトル間のcosine類似度を計算

    Args:
        vector: [num_layers, hidden_dim] のベクトル

    Returns:
        similarity_matrix: [num_layers, num_layers] のcosine類似度行列
    """
    vectors_np = vector.numpy()
    similarity_matrix = cosine_similarity(vectors_np)
    return torch.tensor(similarity_matrix)


def compute_residual_stream_similarity(
    attn_vectors: torch.Tensor,
    mlp_vectors: torch.Tensor,
) -> torch.Tensor:
    """Residual streamのcosine類似度行列を計算

    Args:
        attn_vectors: Attention層のベクトル [num_layers, hidden_dim]
        mlp_vectors: MLP層のベクトル [num_layers, hidden_dim]

    Returns:
        similarity_matrix: [num_layers*2, num_layers*2] のcosine類似度行列
    """
    num_layers = attn_vectors.shape[0]

    # Residual streamの順序に従って結合: [attn_1, mlp_1, attn_2, mlp_2, ...]
    combined_vectors = []
    for layer_idx in range(num_layers):
        combined_vectors.append(attn_vectors[layer_idx])
        combined_vectors.append(mlp_vectors[layer_idx])

    combined_vectors = torch.stack(combined_vectors)  # [num_layers*2, hidden_dim]

    # Cosine類似度を計算
    vectors_np = combined_vectors.numpy()
    similarity_matrix = cosine_similarity(vectors_np)

    return torch.tensor(similarity_matrix)


def compute_adjacent_layer_similarity(vector: torch.Tensor) -> torch.Tensor:
    """隣接層間のcosine類似度を計算

    Args:
        vector: [num_layers, hidden_dim] のベクトル

    Returns:
        adjacent_similarities: [num_layers-1] の隣接層間cosine類似度
    """
    vectors_np = vector.numpy()
    num_layers = vector.shape[0]

    adjacent_similarities = []
    for i in range(num_layers - 1):
        sim = cosine_similarity(vectors_np[i : i + 1], vectors_np[i + 1 : i + 2])[0, 0]
        adjacent_similarities.append(sim)

    return torch.tensor(adjacent_similarities)


def compute_adjacent_layer_difference(vector: torch.Tensor) -> torch.Tensor:
    """隣接層間類似度の差分を計算

    層iについて: sim(L_{i-1}, L_i) - sim(L_i, L_{i+1})

    Args:
        vector: [num_layers, hidden_dim] のベクトル

    Returns:
        differences: [num_layers-2] の差分
    """
    adjacent_sims = compute_adjacent_layer_similarity(vector)
    differences = adjacent_sims[:-1] - adjacent_sims[1:]
    return differences

