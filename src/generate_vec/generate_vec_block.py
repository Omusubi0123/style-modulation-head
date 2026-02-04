"""
generate_vec_block.py - Block入力・出力とLayerNorm入力からPersona Vectorを生成

各レイヤーの以下の位置からベクトルを抽出:
- attn_layernorm: Attention前のLayerNormへの入力
- attn: Attention Blockへの入力（LayerNormの出力）
- attn_output: Attention Blockの出力
- mlp_layernorm: MLP前のLayerNormへの入力
- mlp: MLP Blockへの入力（LayerNormの出力）
- mlp_output: MLP Blockの出力
"""

import argparse
import gc
import os

import torch
from tqdm import tqdm

from src.generate_vec.common import (
    clear_memory,
    get_max_layer,
    get_persona_effective,
    load_model,
    locate_layer_list,
    validate_effective_samples,
)


def get_hidden_block_inputs(
    model: object,
    tokenizer: object,
    prompts: list[str],
    responses: list[str],
    layer_list: list[int] | None = None,
):
    """プロンプトと応答から各レイヤーのblock入力・出力とlayer norm入力を抽出する

    Args:
        model: Transformerモデル
        tokenizer: トークナイザー
        prompts: プロンプトのリスト
        responses: 応答のリスト
        layer_list: 抽出するレイヤーのリスト。Noneの場合は全レイヤー

    Returns:
        dict: 各hook typeに対するレイヤーごとのデータ
    """
    if not prompts or not responses:
        raise ValueError(
            f"Empty data provided: prompts={len(prompts)}, responses={len(responses)}"
        )

    if len(prompts) != len(responses):
        raise ValueError(f"Prompts and responses must have the same length")

    print(f"Processing {len(prompts)} prompt-response pairs...")

    max_layer = get_max_layer(model)
    if layer_list is None:
        layer_list = list(range(max_layer))

    # Hook types
    hook_types = [
        "attn_layernorm_input",
        "attn_input",
        "attn_output",
        "mlp_layernorm_input",
        "mlp_input",
        "mlp_output",
    ]

    # 平均化済みベクトルを保存
    data = {
        hook_type: {
            "prompt_avg": {l: [] for l in layer_list},
            "response_avg": {l: [] for l in layer_list},
            "prompt_last": {l: [] for l in layer_list},
        }
        for hook_type in hook_types
    }

    current_prompt_len = [0]
    layer_list_module = locate_layer_list(model)
    handles = []

    def make_hook_fn(layer_idx: int, hook_type: str, is_pre_hook: bool = False):
        """フック関数を生成"""

        def hook_fn(module: object, input: object, output: object | None = None):
            # 対象テンソルを決定
            target_tensor = None

            if is_pre_hook:
                # pre_hookの場合は入力を使用
                if isinstance(input, tuple) and len(input) > 0:
                    target_tensor = input[0]
                elif not isinstance(input, tuple):
                    target_tensor = input
            else:
                # forward hookの場合は出力を使用
                if output is not None:
                    if isinstance(output, tuple):
                        target_tensor = output[0]
                    else:
                        target_tensor = output

            if target_tensor is not None and torch.is_tensor(target_tensor):
                prompt_len = current_prompt_len[0]

                if target_tensor.dim() == 3:
                    batch_size, seq_len, _ = target_tensor.shape
                    for b in range(batch_size):
                        if prompt_len > 0:
                            prompt_avg_vec = (
                                target_tensor[b, :prompt_len, :]
                                .float()
                                .mean(dim=0)
                                .detach()
                                .cpu()
                            )
                            prompt_last_vec = (
                                target_tensor[b, prompt_len - 1, :]
                                .float()
                                .detach()
                                .cpu()
                            )
                            data[hook_type]["prompt_avg"][layer_idx].append(
                                prompt_avg_vec
                            )
                            data[hook_type]["prompt_last"][layer_idx].append(
                                prompt_last_vec
                            )

                        if seq_len > prompt_len:
                            response_avg_vec = (
                                target_tensor[b, prompt_len:, :]
                                .float()
                                .mean(dim=0)
                                .detach()
                                .cpu()
                            )
                            data[hook_type]["response_avg"][layer_idx].append(
                                response_avg_vec
                            )

                del target_tensor

        return hook_fn

    # 各レイヤーにフックを登録
    for layer_idx in layer_list:
        layer = layer_list_module[layer_idx]

        # Attention前のlayer normを探す
        attn_ln_attrs = [
            "input_layernorm",
            "ln_1",
            "layer_norm",
            "pre_attention_layernorm",
        ]
        attn_ln = None
        for attr in attn_ln_attrs:
            if hasattr(layer, attr):
                attn_ln = getattr(layer, attr)
                break

        # MLP前のlayer normを探す
        mlp_ln_attrs = ["post_attention_layernorm", "ln_2", "mlp_layernorm"]
        mlp_ln = None
        for attr in mlp_ln_attrs:
            if hasattr(layer, attr):
                mlp_ln = getattr(layer, attr)
                break

        # Attention blockを探す
        attn_attrs = ["self_attn", "attention", "attn"]
        attn_block = None
        for attr in attn_attrs:
            if hasattr(layer, attr):
                attn_block = getattr(layer, attr)
                break

        # MLP blockを探す
        mlp_attrs = ["mlp", "feed_forward", "ffn"]
        mlp_block = None
        for attr in mlp_attrs:
            if hasattr(layer, attr):
                mlp_block = getattr(layer, attr)
                break

        # フックを登録
        if attn_ln is not None:
            handles.append(
                attn_ln.register_forward_pre_hook(
                    make_hook_fn(layer_idx, "attn_layernorm_input", is_pre_hook=True)
                )
            )
            handles.append(
                attn_ln.register_forward_hook(
                    make_hook_fn(layer_idx, "attn_input", is_pre_hook=False)
                )
            )
        elif attn_block is not None:
            handles.append(
                attn_block.register_forward_pre_hook(
                    make_hook_fn(layer_idx, "attn_input", is_pre_hook=True)
                )
            )

        if attn_block is not None:
            handles.append(
                attn_block.register_forward_hook(
                    make_hook_fn(layer_idx, "attn_output", is_pre_hook=False)
                )
            )

        if mlp_ln is not None:
            handles.append(
                mlp_ln.register_forward_pre_hook(
                    make_hook_fn(layer_idx, "mlp_layernorm_input", is_pre_hook=True)
                )
            )
            handles.append(
                mlp_ln.register_forward_hook(
                    make_hook_fn(layer_idx, "mlp_input", is_pre_hook=False)
                )
            )
        elif mlp_block is not None:
            handles.append(
                mlp_block.register_forward_pre_hook(
                    make_hook_fn(layer_idx, "mlp_input", is_pre_hook=True)
                )
            )

        if mlp_block is not None:
            handles.append(
                mlp_block.register_forward_hook(
                    make_hook_fn(layer_idx, "mlp_output", is_pre_hook=False)
                )
            )

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
    result = {}
    for hook_type in hook_types:
        result[hook_type] = {}
        for layer_idx in layer_list:
            if data[hook_type]["prompt_avg"][layer_idx]:
                result[hook_type][layer_idx] = {
                    "prompt_avg": torch.stack(
                        data[hook_type]["prompt_avg"][layer_idx], dim=0
                    ),
                    "response_avg": (
                        torch.stack(data[hook_type]["response_avg"][layer_idx], dim=0)
                        if data[hook_type]["response_avg"][layer_idx]
                        else None
                    ),
                    "prompt_last": (
                        torch.stack(data[hook_type]["prompt_last"][layer_idx], dim=0)
                        if data[hook_type]["prompt_last"][layer_idx]
                        else None
                    ),
                }
            else:
                result[hook_type][layer_idx] = None

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
    """ペルソナベクトルを生成して保存する

    Args:
        model_name: 使用するTransformerモデル名
        pos_path: ポジティブなペルソナデータのCSVファイルパス
        neg_path: ネガティブなペルソナデータのCSVファイルパス
        trait: 特性名
        save_dir: ベクトルを保存するディレクトリ
        threshold: フィルタリング閾値（デフォルト: 50）
    """
    model, tokenizer = load_model(model_name)

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
    pos_result = get_hidden_block_inputs(
        model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses
    )

    clear_memory()

    # ネガティブデータから抽出
    neg_result = get_hidden_block_inputs(
        model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses
    )

    clear_memory()

    max_layer = get_max_layer(model)
    layer_list = list(range(max_layer))
    hidden_size = model.config.hidden_size

    os.makedirs(save_dir, exist_ok=True)

    # 各タイプに対してベクトルを計算・保存
    type_mapping = {
        "attn_layernorm_input": "attn_layernorm",
        "attn_input": "attn",
        "attn_output": "attn_output",
        "mlp_layernorm_input": "mlp_layernorm",
        "mlp_input": "mlp",
        "mlp_output": "mlp_output",
    }

    for hook_type, suffix in type_mapping.items():
        prompt_avg_diff_list = []
        response_avg_diff_list = []
        prompt_last_diff_list = []

        for layer_idx in layer_list:
            pos_data = pos_result[hook_type].get(layer_idx)
            neg_data = neg_result[hook_type].get(layer_idx)

            if pos_data is not None and neg_data is not None:
                prompt_avg_diff_list.append(
                    pos_data["prompt_avg"].mean(0).float()
                    - neg_data["prompt_avg"].mean(0).float()
                )
                if (
                    pos_data["response_avg"] is not None
                    and neg_data["response_avg"] is not None
                ):
                    response_avg_diff_list.append(
                        pos_data["response_avg"].mean(0).float()
                        - neg_data["response_avg"].mean(0).float()
                    )
                else:
                    response_avg_diff_list.append(torch.zeros(hidden_size))
                if (
                    pos_data["prompt_last"] is not None
                    and neg_data["prompt_last"] is not None
                ):
                    prompt_last_diff_list.append(
                        pos_data["prompt_last"].mean(0).float()
                        - neg_data["prompt_last"].mean(0).float()
                    )
                else:
                    prompt_last_diff_list.append(torch.zeros(hidden_size))
            else:
                prompt_avg_diff_list.append(torch.zeros(hidden_size))
                response_avg_diff_list.append(torch.zeros(hidden_size))
                prompt_last_diff_list.append(torch.zeros(hidden_size))

        if prompt_avg_diff_list:
            prompt_avg_diff = torch.stack(prompt_avg_diff_list, dim=0)
            response_avg_diff = torch.stack(response_avg_diff_list, dim=0)
            prompt_last_diff = torch.stack(prompt_last_diff_list, dim=0)

            torch.save(
                prompt_avg_diff, f"{save_dir}/{trait}_prompt_avg_diff_{suffix}.pt"
            )
            torch.save(
                response_avg_diff, f"{save_dir}/{trait}_response_avg_diff_{suffix}.pt"
            )
            torch.save(
                prompt_last_diff, f"{save_dir}/{trait}_prompt_last_diff_{suffix}.pt"
            )

    print(f"Persona vectors saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
