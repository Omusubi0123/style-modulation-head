"""
common.py - Persona Vector生成用の共通ユーティリティ

generate_vec.py, generate_vec_attn.py, generate_vec_block.py で共通に使用する
機能を提供する。
"""

import gc
import json
import os
from typing import List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from src.config import setup_credentials

# Set up credentials and environment
config = setup_credentials()


def load_jsonl(file_path: str) -> list[dict]:
    """JSONLファイルを読み込んで辞書のリストとして返す

    Args:
        file_path: 読み込むJSONLファイルのパス

    Returns:
        各行がパースされたJSONオブジェクトのリスト
    """
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def get_max_layer(model) -> int:
    """モデルから最大レイヤー数を取得する

    Args:
        model: Transformerモデル

    Returns:
        最大レイヤー数

    Raises:
        AttributeError: レイヤー数が見つからない
    """
    possible_layer_attrs = ["num_hidden_layers", "n_layers", "num_layers", "n_layer"]
    max_layer = None

    # まずメインのconfigで試す
    for attr in possible_layer_attrs:
        if hasattr(model.config, attr):
            max_layer = getattr(model.config, attr)
            print(f"Using {attr} = {max_layer}")
            break

    # メインのconfigで見つからない場合、text_configを試す（マルチモーダルモデル用）
    if max_layer is None and hasattr(model.config, "text_config"):
        text_config = model.config.text_config
        print("Checking text_config for layer attributes...")
        for attr in possible_layer_attrs:
            if hasattr(text_config, attr):
                max_layer = getattr(text_config, attr)
                print(f"Using text_config.{attr} = {max_layer}")
                break

    if max_layer is None:
        print(
            "Available attributes in main config:",
            [attr for attr in dir(model.config) if not attr.startswith("_")],
        )
        if hasattr(model.config, "text_config"):
            print(
                "Available attributes in text_config:",
                [
                    attr
                    for attr in dir(model.config.text_config)
                    if not attr.startswith("_")
                ],
            )
        raise AttributeError(
            "Could not find layer count attribute in model config or text_config"
        )

    return max_layer


def locate_layer_list(model):
    """モデルからレイヤーリストを特定する

    Args:
        model: Transformerモデル

    Returns:
        レイヤーリスト（ModuleList）

    Raises:
        ValueError: レイヤーリストが見つからない
    """
    possible_attrs = [
        "transformer.h",  # GPT-2/Neo, Bloom, etc.
        "encoder.layer",  # BERT/RoBERTa
        "model.layers",  # Llama/Mistral/Qwen
        "gpt_neox.layers",  # GPT-NeoX
        "block",  # Flan-T5
        "language_model.layers",  # Multimodal Gemma-3
    ]

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
            return cur

    raise ValueError("Could not find layer list in model")


def get_attention_config(model) -> dict:
    """モデルからAttention関連の設定を取得する

    Args:
        model: Transformerモデル

    Returns:
        dict: Attention設定
            - num_attention_heads: アテンションヘッド数
            - num_key_value_heads: KVヘッド数（GQA用）
            - hidden_size: 隠れ層サイズ
            - head_dim: ヘッド次元
    """
    cfg = model.config
    if hasattr(cfg, "text_config"):
        cfg = cfg.text_config

    num_attention_heads = getattr(cfg, "num_attention_heads", None)
    num_key_value_heads = getattr(cfg, "num_key_value_heads", num_attention_heads)
    hidden_size = getattr(cfg, "hidden_size", None)

    if num_attention_heads is None or hidden_size is None:
        raise AttributeError("Could not find attention config in model")

    head_dim = hidden_size // num_attention_heads

    return {
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "hidden_size": hidden_size,
        "head_dim": head_dim,
    }


def load_model(model_name: str) -> Tuple:
    """モデルとトークナイザーを読み込む

    メモリ効率のためにbfloat16を使用。
    マルチモーダルモデルにも対応。

    Args:
        model_name: モデル名またはパス

    Returns:
        tuple: (model, tokenizer)
    """
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=config.hf_token,
        )
        print(f"Loaded model using AutoModelForCausalLM")
    except ValueError as e:
        if "Unrecognized configuration class" in str(e):
            print(
                f"AutoModelForCausalLM failed, trying AutoModel for multimodal model..."
            )
            model = AutoModel.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=config.hf_token,
            )
            print(f"Loaded model using AutoModel")
        else:
            raise

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=config.hf_token
    )

    torch.cuda.empty_cache()

    return model, tokenizer


def get_persona_effective(
    pos_path: str, neg_path: str, trait: str, threshold: int = 50
) -> Tuple:
    """効果的なペルソナデータをフィルタリングして抽出する

    Args:
        pos_path: ポジティブなペルソナデータのCSVファイルパス
        neg_path: ネガティブなペルソナデータのCSVファイルパス
        trait: 評価する特性名
        threshold: フィルタリングの閾値（デフォルト: 50）

    Returns:
        tuple: (pos_effective, neg_effective, pos_prompts, neg_prompts, pos_responses, neg_responses)

    Raises:
        FileNotFoundError: ファイルが見つからない
    """
    if not os.path.exists(pos_path):
        raise FileNotFoundError(
            f"Positive persona file not found: {pos_path}\n"
            f"Please run eval_persona.py first to generate the required CSV files."
        )
    if not os.path.exists(neg_path):
        raise FileNotFoundError(
            f"Negative persona file not found: {neg_path}\n"
            f"Please run eval_persona.py first to generate the required CSV files."
        )

    persona_pos = pd.read_csv(pos_path)
    persona_neg = pd.read_csv(neg_path)
    mask = (
        (persona_pos[trait] >= threshold)
        & (persona_neg[trait] < 100 - threshold)
        & (persona_pos["coherence"] >= 50)
        & (persona_neg["coherence"] >= 50)
    )

    persona_pos_effective = persona_pos[mask]
    persona_neg_effective = persona_neg[mask]

    persona_pos_effective_prompts = persona_pos_effective["prompt"].tolist()
    persona_neg_effective_prompts = persona_neg_effective["prompt"].tolist()

    persona_pos_effective_responses = persona_pos_effective["answer"].tolist()
    persona_neg_effective_responses = persona_neg_effective["answer"].tolist()

    return (
        persona_pos_effective,
        persona_neg_effective,
        persona_pos_effective_prompts,
        persona_neg_effective_prompts,
        persona_pos_effective_responses,
        persona_neg_effective_responses,
    )


def compute_persona_vector_diff(
    pos_vectors: dict, neg_vectors: dict, layer_list: List[int], hidden_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ポジティブとネガティブのベクトルからPersona Vectorの差分を計算する

    Args:
        pos_vectors: ポジティブデータのベクトル辞書（layer -> tensor）
        neg_vectors: ネガティブデータのベクトル辞書（layer -> tensor）
        layer_list: レイヤーインデックスのリスト
        hidden_size: 隠れ層サイズ

    Returns:
        tuple: (prompt_avg_diff, response_avg_diff, prompt_last_diff)
            各テンソルの形状は [num_layers, hidden_size]
    """
    prompt_avg_diff_list = []
    response_avg_diff_list = []
    prompt_last_diff_list = []

    for layer_idx in layer_list:
        pos_prompt_avg = pos_vectors.get("prompt_avg", {}).get(layer_idx)
        neg_prompt_avg = neg_vectors.get("prompt_avg", {}).get(layer_idx)
        pos_response_avg = pos_vectors.get("response_avg", {}).get(layer_idx)
        neg_response_avg = neg_vectors.get("response_avg", {}).get(layer_idx)
        pos_prompt_last = pos_vectors.get("prompt_last", {}).get(layer_idx)
        neg_prompt_last = neg_vectors.get("prompt_last", {}).get(layer_idx)

        if pos_prompt_avg is not None and neg_prompt_avg is not None:
            prompt_avg_diff_list.append(
                pos_prompt_avg.mean(0).float() - neg_prompt_avg.mean(0).float()
            )
        else:
            prompt_avg_diff_list.append(torch.zeros(hidden_size))

        if pos_response_avg is not None and neg_response_avg is not None:
            response_avg_diff_list.append(
                pos_response_avg.mean(0).float() - neg_response_avg.mean(0).float()
            )
        else:
            response_avg_diff_list.append(torch.zeros(hidden_size))

        if pos_prompt_last is not None and neg_prompt_last is not None:
            prompt_last_diff_list.append(
                pos_prompt_last.mean(0).float() - neg_prompt_last.mean(0).float()
            )
        else:
            prompt_last_diff_list.append(torch.zeros(hidden_size))

    prompt_avg_diff = torch.stack(prompt_avg_diff_list, dim=0)
    response_avg_diff = torch.stack(response_avg_diff_list, dim=0)
    prompt_last_diff = torch.stack(prompt_last_diff_list, dim=0)

    return prompt_avg_diff, response_avg_diff, prompt_last_diff


def save_persona_vectors(
    save_dir: str,
    trait: str,
    prompt_avg_diff: torch.Tensor,
    response_avg_diff: torch.Tensor,
    prompt_last_diff: torch.Tensor,
    suffix: str = "",
) -> None:
    """Persona Vectorを保存する

    Args:
        save_dir: 保存先ディレクトリ
        trait: 特性名
        prompt_avg_diff: プロンプト平均の差分ベクトル
        response_avg_diff: 応答平均の差分ベクトル
        prompt_last_diff: プロンプト最終トークンの差分ベクトル
        suffix: ファイル名の接尾辞（例: "_attn_pre_o_proj"）
    """
    os.makedirs(save_dir, exist_ok=True)

    torch.save(prompt_avg_diff, f"{save_dir}/{trait}_prompt_avg_diff{suffix}.pt")
    torch.save(response_avg_diff, f"{save_dir}/{trait}_response_avg_diff{suffix}.pt")
    torch.save(prompt_last_diff, f"{save_dir}/{trait}_prompt_last_diff{suffix}.pt")

    print(f"Persona vectors saved to {save_dir}")
    if suffix:
        print(f"  - {trait}_prompt_avg_diff{suffix}.pt")
        print(f"  - {trait}_response_avg_diff{suffix}.pt")
        print(f"  - {trait}_prompt_last_diff{suffix}.pt")


def clear_memory() -> None:
    """GPUメモリをクリアする"""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


def validate_effective_samples(
    pos_prompts: List[str], neg_prompts: List[str], threshold: int
) -> None:
    """フィルタリング後のサンプル数を検証する

    Args:
        pos_prompts: ポジティブなプロンプトリスト
        neg_prompts: ネガティブなプロンプトリスト
        threshold: フィルタリング閾値

    Raises:
        ValueError: 有効なサンプルが見つからない
    """
    print(f"Filtered effective samples:")
    print(f"  Positive: {len(pos_prompts)} samples")
    print(f"  Negative: {len(neg_prompts)} samples")

    if len(pos_prompts) == 0:
        raise ValueError(
            f"No effective positive samples found after filtering with threshold={threshold}. "
            f"Try lowering the threshold or check the input data."
        )

    if len(neg_prompts) == 0:
        raise ValueError(
            f"No effective negative samples found after filtering with threshold={threshold}. "
            f"Try lowering the threshold or check the input data."
        )

