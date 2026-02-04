"""
generate_vec_attn.py - OV計算前のV（Softmax(QK^T)V の結果）からPersona Vectorを生成

Attention内部のattn_weights @ Vの結果（O projection前）をキャプチャし、
各ヘッドごとのPersona Vectorを保存する。

保存形式:
- {trait}_attn_pre_o_proj_prompt_avg_diff.pt: [num_layers, hidden_size]
  （全ヘッドを結合した形状。各ヘッドへの分解は後処理で可能）
- 各層のindex 0が1層目のattention

注意点（GQA対応）:
- Qwen2.5-7B: num_attention_heads=28, num_key_value_heads=4
- Llama3.1-8B: num_attention_heads=32, num_key_value_heads=8
- O projection前の出力は常にnum_attention_heads * head_dimの次元
"""

import argparse
import gc
import os

import torch
from tqdm import tqdm

from src.generate_vec.common import (
    clear_memory,
    compute_persona_vector_diff,
    get_attention_config,
    get_max_layer,
    get_persona_effective,
    load_model,
    locate_layer_list,
    save_persona_vectors,
    validate_effective_samples,
)


def get_attn_pre_o_proj_vectors(
    model: object,
    tokenizer: object,
    prompts: list[str],
    responses: list[str],
    layer_list: list[int] | None = None,
):
    """OV計算前のV（O projection前のAttention出力）を各層から抽出する

    Args:
        model: Transformerモデル
        tokenizer: トークナイザー
        prompts: プロンプトのリスト
        responses: 応答のリスト
        layer_list: 抽出するレイヤーのリスト。Noneの場合は全レイヤー

    Returns:
        dict: {
            'prompt_avg': {layer: tensor [num_samples, hidden_size]},
            'response_avg': {layer: tensor [num_samples, hidden_size]},
            'prompt_last': {layer: tensor [num_samples, hidden_size]},
        }
    """
    if not prompts or not responses:
        raise ValueError(
            f"Empty data provided: prompts={len(prompts)}, responses={len(responses)}"
        )

    if len(prompts) != len(responses):
        raise ValueError(
            f"Prompts and responses must have the same length: "
            f"prompts={len(prompts)}, responses={len(responses)}"
        )

    print(f"Processing {len(prompts)} prompt-response pairs...")

    max_layer = get_max_layer(model)
    attn_config = get_attention_config(model)
    print(f"Attention config: {attn_config}")

    if layer_list is None:
        layer_list = list(range(max_layer))

    # 平均化済みベクトルを保存
    prompt_avg = {l: [] for l in layer_list}
    response_avg = {l: [] for l in layer_list}
    prompt_last = {l: [] for l in layer_list}

    # 現在処理中のprompt_lenを保持
    current_prompt_len = [0]

    # レイヤーリストを取得
    layer_list_module = locate_layer_list(model)

    # フックハンドルを保存
    handles = []

    def make_hook_fn(layer_idx: int):
        """o_proj入力をキャプチャするフック関数"""

        def hook_fn(module: object, input: object, output: object | None = None):
            hidden_input = None
            if isinstance(input, tuple) and len(input) > 0:
                hidden_input = input[0]
            elif not isinstance(input, tuple):
                hidden_input = input

            if hidden_input is not None and torch.is_tensor(hidden_input):
                prompt_len = current_prompt_len[0]

                if hidden_input.dim() == 3:
                    batch_size, seq_len, hidden_dim = hidden_input.shape
                    for b in range(batch_size):
                        if prompt_len > 0:
                            prompt_avg_vec = (
                                hidden_input[b, :prompt_len, :]
                                .float()
                                .mean(dim=0)
                                .detach()
                                .cpu()
                            )
                            prompt_last_vec = (
                                hidden_input[b, prompt_len - 1, :]
                                .float()
                                .detach()
                                .cpu()
                            )
                            prompt_avg[layer_idx].append(prompt_avg_vec)
                            prompt_last[layer_idx].append(prompt_last_vec)

                        if seq_len > prompt_len:
                            response_avg_vec = (
                                hidden_input[b, prompt_len:, :]
                                .float()
                                .mean(dim=0)
                                .detach()
                                .cpu()
                            )
                            response_avg[layer_idx].append(response_avg_vec)

                del hidden_input

        return hook_fn

    # 各レイヤーのo_projにpre_hookを登録
    for layer_idx in layer_list:
        layer = layer_list_module[layer_idx]

        # Attention blockを探す
        attn_attrs = ["self_attn", "attention", "attn"]
        attn_block = None
        for attr in attn_attrs:
            if hasattr(layer, attr):
                attn_block = getattr(layer, attr)
                break

        if attn_block is None:
            print(f"Warning: Could not find attention block for layer {layer_idx}")
            continue

        # o_projを探す
        o_proj_attrs = ["o_proj", "out_proj", "dense"]
        o_proj = None
        for attr in o_proj_attrs:
            if hasattr(attn_block, attr):
                o_proj = getattr(attn_block, attr)
                break

        if o_proj is None:
            print(f"Warning: Could not find o_proj for layer {layer_idx}")
            continue

        handle = o_proj.register_forward_pre_hook(make_hook_fn(layer_idx))
        handles.append(handle)

    texts = [p + a for p, a in zip(prompts, responses)]

    try:
        for idx, (text, prompt) in enumerate(
            tqdm(zip(texts, prompts), total=len(texts), desc="Processing samples")
        ):
            with torch.no_grad():
                inputs = tokenizer(
                    text, return_tensors="pt", add_special_tokens=False
                ).to(model.device)
                prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
                current_prompt_len[0] = prompt_len

                outputs = model(**inputs, use_cache=False)

                del outputs
                del inputs
                torch.cuda.synchronize()

            if (idx + 1) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    finally:
        for handle in handles:
            handle.remove()

    # 結果を整理
    result = {
        "prompt_avg": {},
        "response_avg": {},
        "prompt_last": {},
    }

    for layer_idx in layer_list:
        if prompt_avg[layer_idx]:
            result["prompt_avg"][layer_idx] = torch.stack(prompt_avg[layer_idx], dim=0)
        else:
            result["prompt_avg"][layer_idx] = None

        if response_avg[layer_idx]:
            result["response_avg"][layer_idx] = torch.stack(
                response_avg[layer_idx], dim=0
            )
        else:
            result["response_avg"][layer_idx] = None

        if prompt_last[layer_idx]:
            result["prompt_last"][layer_idx] = torch.stack(
                prompt_last[layer_idx], dim=0
            )
        else:
            result["prompt_last"][layer_idx] = None

    gc.collect()
    torch.cuda.empty_cache()

    return result


def save_persona_vector(
    model_name: str,
    pos_path: str,
    neg_path: str,
    trait: str,
    save_dir: str,
    threshold: int = 50,
):
    """OV計算前のVからペルソナベクトルを生成して保存する

    Args:
        model_name: 使用するTransformerモデル名
        pos_path: ポジティブなペルソナデータのCSVファイルパス
        neg_path: ネガティブなペルソナデータのCSVファイルパス
        trait: 特性名
        save_dir: ベクトルを保存するディレクトリ
        threshold: フィルタリング閾値（デフォルト: 50）
    """
    model, tokenizer = load_model(model_name)

    attn_config = get_attention_config(model)
    print(f"Model attention config:")
    print(f"  num_attention_heads: {attn_config['num_attention_heads']}")
    print(f"  num_key_value_heads: {attn_config['num_key_value_heads']}")
    print(f"  hidden_size: {attn_config['hidden_size']}")
    print(f"  head_dim: {attn_config['head_dim']}")

    (
        _,
        _,
        persona_pos_effective_prompts,
        persona_neg_effective_prompts,
        persona_pos_effective_responses,
        persona_neg_effective_responses,
    ) = get_persona_effective(pos_path, neg_path, trait, threshold)

    validate_effective_samples(
        persona_pos_effective_prompts, persona_neg_effective_prompts, threshold
    )

    # ポジティブデータから抽出
    pos_result = get_attn_pre_o_proj_vectors(
        model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses
    )

    clear_memory()

    # ネガティブデータから抽出
    neg_result = get_attn_pre_o_proj_vectors(
        model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses
    )

    clear_memory()

    max_layer = get_max_layer(model)
    layer_list = list(range(max_layer))

    # Persona Vectorを計算
    prompt_avg_diff, response_avg_diff, prompt_last_diff = compute_persona_vector_diff(
        pos_result, neg_result, layer_list, attn_config["hidden_size"]
    )

    os.makedirs(save_dir, exist_ok=True)

    save_persona_vectors(
        save_dir,
        trait,
        prompt_avg_diff,
        response_avg_diff,
        prompt_last_diff,
        suffix="_attn_pre_o_proj",
    )

    # Attention設定も保存
    torch.save(attn_config, f"{save_dir}/{trait}_attn_config.pt")
    print(f"  - {trait}_attn_config.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Persona Vector from OV pre-projection (attn_weights @ V)"
    )
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--pos_path", type=str, required=True)
    parser.add_argument("--neg_path", type=str, required=True)
    parser.add_argument("--trait", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--threshold", type=int, default=50)
    args = parser.parse_args()

    save_persona_vector(
        args.model_name,
        args.pos_path,
        args.neg_path,
        args.trait,
        args.save_dir,
        args.threshold,
    )
