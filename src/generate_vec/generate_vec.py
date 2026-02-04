"""
generate_vec.py - Transformer Block出力からPersona Vectorを生成

プロンプトと応答から各レイヤーの隠れ状態を抽出し、
ポジティブとネガティブの差分としてPersona Vectorを計算する。
"""

import argparse

import torch
from tqdm import tqdm

from src.generate_vec.common import (
    clear_memory,
    get_max_layer,
    get_persona_effective,
    load_model,
    save_persona_vectors,
    validate_effective_samples,
)


def get_hidden_p_and_r(
    model: object,
    tokenizer: object,
    prompts: list[str],
    responses: list[str],
    layer_list: list[int] | None = None,
):
    """プロンプトと応答から各レイヤーの隠れ状態を抽出する

    Args:
        model: Transformerモデル
        tokenizer: トークナイザー
        prompts: プロンプトのリスト
        responses: 応答のリスト
        layer_list: 抽出するレイヤーのリスト。Noneの場合は全レイヤー

    Returns:
        tuple: (プロンプト平均, プロンプト最終, 応答平均)の隠れ状態タプル
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
    if layer_list is None:
        layer_list = list(range(max_layer + 1))

    prompt_avg = [[] for _ in range(max_layer + 1)]
    response_avg = [[] for _ in range(max_layer + 1)]
    prompt_last = [[] for _ in range(max_layer + 1)]
    texts = [p + a for p, a in zip(prompts, responses)]

    for idx, (text, prompt) in enumerate(tqdm(zip(texts, prompts), total=len(texts))):
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(
                model.device
            )
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)

            for layer in layer_list:
                prompt_avg[layer].append(
                    outputs.hidden_states[layer][:, :prompt_len, :]
                    .float()
                    .mean(dim=1)
                    .detach()
                    .cpu()
                )
                response_avg[layer].append(
                    outputs.hidden_states[layer][:, prompt_len:, :]
                    .float()
                    .mean(dim=1)
                    .detach()
                    .cpu()
                )
                prompt_last[layer].append(
                    outputs.hidden_states[layer][:, prompt_len - 1, :]
                    .float()
                    .detach()
                    .cpu()
                )

            del outputs
            del inputs

            if (idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

    for layer in layer_list:
        if not prompt_avg[layer]:
            raise RuntimeError(
                f"No data collected for layer {layer}. "
                f"This might indicate an issue with layer_list or model output. "
                f"layer_list={layer_list}, max_layer={max_layer}"
            )
        prompt_avg[layer] = torch.cat(prompt_avg[layer], dim=0)
        prompt_last[layer] = torch.cat(prompt_last[layer], dim=0)
        response_avg[layer] = torch.cat(response_avg[layer], dim=0)

    return prompt_avg, prompt_last, response_avg


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

    persona_effective_prompt_avg = {}
    persona_effective_prompt_last = {}
    persona_effective_response_avg = {}

    (
        persona_effective_prompt_avg["pos"],
        persona_effective_prompt_last["pos"],
        persona_effective_response_avg["pos"],
    ) = get_hidden_p_and_r(
        model, tokenizer, persona_pos_effective_prompts, persona_pos_effective_responses
    )

    clear_memory()

    (
        persona_effective_prompt_avg["neg"],
        persona_effective_prompt_last["neg"],
        persona_effective_response_avg["neg"],
    ) = get_hidden_p_and_r(
        model, tokenizer, persona_neg_effective_prompts, persona_neg_effective_responses
    )

    clear_memory()

    # Persona Vectorの差分を計算
    persona_effective_prompt_avg_diff = torch.stack(
        [
            persona_effective_prompt_avg["pos"][l].mean(0).float()
            - persona_effective_prompt_avg["neg"][l].mean(0).float()
            for l in range(len(persona_effective_prompt_avg["pos"]))
        ],
        dim=0,
    )
    persona_effective_response_avg_diff = torch.stack(
        [
            persona_effective_response_avg["pos"][l].mean(0).float()
            - persona_effective_response_avg["neg"][l].mean(0).float()
            for l in range(len(persona_effective_response_avg["pos"]))
        ],
        dim=0,
    )
    persona_effective_prompt_last_diff = torch.stack(
        [
            persona_effective_prompt_last["pos"][l].mean(0).float()
            - persona_effective_prompt_last["neg"][l].mean(0).float()
            for l in range(len(persona_effective_prompt_last["pos"]))
        ],
        dim=0,
    )

    save_persona_vectors(
        save_dir,
        trait,
        persona_effective_prompt_avg_diff,
        persona_effective_response_avg_diff,
        persona_effective_prompt_last_diff,
    )


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
