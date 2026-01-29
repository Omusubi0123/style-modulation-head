"""
compute.py - Head Contribution計算ロジック

各Attentionヘッドの貢献度を計算する。
実験設定:
1. OV計算前のVのPersona Vector（attn_pre_o_proj）を読み込む
2. 各ヘッドごとにO projectionを計算（他のヘッドは0固定）
3. attn_outputのPersona Vectorとの内積類似度を計算
4. 対数スケーリング (s' = sign(s) * log(1 + |s|)) を適用
5. 層ごと/traitごとにZ-score正規化を適用
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM

from src.config import setup_credentials

config = setup_credentials()


def inner_product_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
    """内積類似度を計算"""
    vec1 = vec1.float()
    vec2 = vec2.float()
    return torch.dot(vec1, vec2)


def load_o_proj_weights(model_name: str, device: str = "cpu") -> Dict[int, torch.Tensor]:
    """モデルからO projection weightsを読み込む

    Args:
        model_name: モデル名
        device: 計算デバイス

    Returns:
        {layer_idx: o_proj_weight tensor}
    """
    print(f"Loading model for O projection weights: {model_name}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=config.hf_token,
        )
    except ValueError as e:
        if "Unrecognized configuration class" in str(e):
            model = AutoModel.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=config.hf_token,
            )
        else:
            raise

    # レイヤーリストを取得
    possible_attrs = [
        "transformer.h",
        "encoder.layer",
        "model.layers",
        "gpt_neox.layers",
        "block",
        "language_model.layers",
    ]

    layer_list = None
    for attr_path in possible_attrs:
        parts = attr_path.split(".")
        cur = model
        found = True
        for part in parts:
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                found = False
                break
        if found and hasattr(cur, "__getitem__"):
            layer_list = cur
            break

    if layer_list is None:
        raise ValueError("Could not find layer list in model")

    o_proj_weights = {}
    for layer_idx, layer in enumerate(layer_list):
        # Attention blockを探す
        attn_attrs = ["self_attn", "attention", "attn"]
        attn_block = None
        for attr in attn_attrs:
            if hasattr(layer, attr):
                attn_block = getattr(layer, attr)
                break

        if attn_block is None:
            continue

        # o_projを探す
        o_proj_attrs = ["o_proj", "out_proj", "dense"]
        o_proj = None
        for attr in o_proj_attrs:
            if hasattr(attn_block, attr):
                o_proj = getattr(attn_block, attr)
                break

        if o_proj is not None:
            o_proj_weights[layer_idx] = o_proj.weight.data.clone().float()

    # モデルを解放
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return o_proj_weights


def compute_head_contributions(
    attn_pre_o_proj_vec: torch.Tensor,
    attn_output_vec: torch.Tensor,
    o_proj_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> List[float]:
    """各ヘッドのO projection後の出力とattn_outputの内積類似度を計算

    Args:
        attn_pre_o_proj_vec: O projection前のPersona Vector [hidden_size]
        attn_output_vec: Attention出力のPersona Vector [hidden_size]
        o_proj_weight: O projection weight [hidden_size, hidden_size]
        num_heads: Attentionヘッド数
        head_dim: 各ヘッドの次元

    Returns:
        各ヘッドの内積類似度のリスト
    """
    similarities = []

    for head_idx in range(num_heads):
        # 該当ヘッド以外を0にしたベクトルを作成
        masked_vec = torch.zeros_like(attn_pre_o_proj_vec)
        start_idx = head_idx * head_dim
        end_idx = (head_idx + 1) * head_dim
        masked_vec[start_idx:end_idx] = attn_pre_o_proj_vec[start_idx:end_idx]

        # O projectionを適用
        projected_vec = torch.matmul(o_proj_weight, masked_vec)

        # 内積類似度を計算
        sim = inner_product_similarity(projected_vec, attn_output_vec)
        similarities.append(sim.item())

    return similarities


def load_attention_config(vector_dir: str, trait: str) -> Dict:
    """Attention設定を読み込む

    Args:
        vector_dir: ベクトルディレクトリ
        trait: 特性名

    Returns:
        {num_attention_heads, head_dim, ...}
    """
    attn_config_path = os.path.join(vector_dir, f"{trait}_attn_config.pt")
    attn_config_json_path = os.path.join(vector_dir, "attn_config.json")

    if os.path.exists(attn_config_path):
        return torch.load(attn_config_path, map_location="cpu")
    elif os.path.exists(attn_config_json_path):
        import json
        with open(attn_config_json_path, "r") as f:
            return json.load(f)

    return None


def normalize_matrix(
    matrix: np.ndarray,
    use_log: bool = True,
    use_zscore: bool = True,
    axis: int = 0,
) -> np.ndarray:
    """行列を正規化する

    Args:
        matrix: 入力行列
        use_log: 対数スケーリングを適用するか
        use_zscore: Z-score正規化を適用するか
        axis: 正規化の軸（0=行ごと、1=列ごと）

    Returns:
        正規化された行列
    """
    result = matrix.copy()

    # 対数スケーリング: s' = sign(s) * log(1 + |s|)
    if use_log:
        result = np.sign(result) * np.log1p(np.abs(result))

    # Z-score正規化
    if use_zscore:
        normalized = np.zeros_like(result)
        for i in range(result.shape[axis]):
            if axis == 0:
                values = result[i, :]
            else:
                values = result[:, i]

            mean = np.mean(values)
            std = np.std(values)
            if std > 0:
                normalized_values = (values - mean) / std
            else:
                normalized_values = values - mean

            if axis == 0:
                normalized[i, :] = normalized_values
            else:
                normalized[:, i] = normalized_values

        result = normalized

    return result

