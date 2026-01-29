"""
sampling.py - ステアリングを使用したサンプリングユーティリティ

モデルの応答生成のための共通サンプリング機能を提供する。
ActivationSteerer系のクラスを直接使用する。
"""

from typing import List, Tuple

import torch
from tqdm import trange
from vllm import SamplingParams

from src.activation_steer import (
    ActivationSteerer,
    ActivationSteererBlock,
    ActivationSteererHead,
)
from src.chat_template_utils import apply_chat_template_safe


def sample_vllm(
    model,
    tokenizer,
    conversations: List[List[dict]],
    top_p: float = 1,
    max_tokens: int = 1000,
    temperature: float = 1,
    min_tokens: int = 1,
) -> Tuple[List[str], List[str]]:
    """VLLMを使用した通常のサンプリングでモデルの応答を生成する

    Args:
        model: VLLMモデル
        tokenizer: トークナイザー
        conversations: 会話データのリスト
        top_p: nucleus samplingパラメータ（デフォルト: 1）
        max_tokens: 最大生成トークン数（デフォルト: 1000）
        temperature: サンプリング温度（デフォルト: 1）
        min_tokens: 最小生成トークン数（デフォルト: 1）

    Returns:
        tuple: (プロンプトリスト, 応答リスト)
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        skip_special_tokens=True,
        stop=[tokenizer.eos_token],
        min_tokens=min_tokens,
    )

    texts = []
    for i, messages in enumerate(conversations):
        text = apply_chat_template_safe(
            tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    generate_kwargs = {"sampling_params": sampling_params, "use_tqdm": True}
    completions = model.generate(texts, **generate_kwargs)
    answers = [completion.outputs[0].text for completion in completions]
    return texts, answers


def sample_with_steering(
    model,
    tokenizer,
    conversations: List[List[dict]],
    vector: torch.Tensor,
    layer: int,
    coef: float,
    bs: int = 100,
    top_p: float = 1.0,
    max_tokens: int = 1000,
    temperature: float = 1.0,
    min_tokens: int = 1,
    steering_type: str = "response",
    renorm_after_steering: bool = False,
) -> Tuple[List[str], List[str]]:
    """ステアリングベクトルを使用してモデルの応答を生成する

    Args:
        model: 使用する言語モデル（Transformers）
        tokenizer: トークナイザー
        conversations: 会話データのリスト
        vector: ステアリングベクトル
        layer: 適用するレイヤー番号（0-based index）
        coef: ステアリング係数
        bs: バッチサイズ（デフォルト: 100）
        top_p: nucleus samplingパラメータ（デフォルト: 1）
        max_tokens: 最大生成トークン数（デフォルト: 1000）
        temperature: サンプリング温度（デフォルト: 1）
        min_tokens: 最小生成トークン数（デフォルト: 1）
        steering_type: ステアリングタイプ（"response", "prompt", "all"）
        renorm_after_steering: ステアリング後にノルムをリノームするか

    Returns:
        tuple: (プロンプトリスト, 応答リスト)
    """
    # トークナイザーの設定
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # プロンプトを準備
    prompts = []
    for messages in conversations:
        text = apply_chat_template_safe(
            tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    outputs = []

    for i in trange(0, len(prompts), bs, desc="Generating"):
        batch = prompts[i : i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}

        steerer = ActivationSteerer(
            model,
            vector,
            coeff=coef,
            layer_idx=layer,
            positions=steering_type,
            renorm_to_original_norm=renorm_after_steering,
        )

        with steerer:
            with torch.no_grad():
                output = model.generate(
                    **tokenized_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    min_new_tokens=min_tokens,
                )

        prompt_len = tokenized_batch["input_ids"].shape[1]
        batch_outputs = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output
        ]
        outputs.extend(batch_outputs)

    return prompts, outputs


def sample_with_block_steering(
    model,
    tokenizer,
    conversations: List[List[dict]],
    vector: torch.Tensor,
    layer: int,
    coef: float,
    block_steering_type: str = "attn",
    bs: int = 100,
    top_p: float = 1.0,
    max_tokens: int = 1000,
    temperature: float = 1.0,
    min_tokens: int = 1,
    steering_type: str = "response",
    renorm_after_steering: bool = False,
) -> Tuple[List[str], List[str]]:
    """Block位置へのステアリングベクトルを使用してモデルの応答を生成する

    Args:
        model: 使用する言語モデル（Transformers）
        tokenizer: トークナイザー
        conversations: 会話データのリスト
        vector: ステアリングベクトル
        layer: 適用するレイヤー番号（0-based index）
        coef: ステアリング係数
        block_steering_type: ブロックステアリングタイプ
            - "attn": attention blockへの入力
            - "mlp": MLP blockへの入力
            - "attn_layernorm": attention前のlayer normへの入力
            - "mlp_layernorm": MLP前のlayer normへの入力
            - "attn_output": attention blockの出力
            - "mlp_output": MLP blockの出力
        bs: バッチサイズ
        top_p: nucleus samplingパラメータ
        max_tokens: 最大生成トークン数
        temperature: サンプリング温度
        min_tokens: 最小生成トークン数
        steering_type: 位置ステアリングタイプ（"response", "prompt", "all"）
        renorm_after_steering: ステアリング後にノルムをリノームするか

    Returns:
        tuple: (プロンプトリスト, 応答リスト)
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = []
    for messages in conversations:
        text = apply_chat_template_safe(
            tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    outputs = []

    for i in trange(0, len(prompts), bs, desc="Generating"):
        batch = prompts[i : i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}

        steerer = ActivationSteererBlock(
            model,
            vector,
            coeff=coef,
            layer_idx=layer,
            positions=steering_type,
            steering_type=block_steering_type,
            renorm_to_original_norm=renorm_after_steering,
        )

        with steerer:
            with torch.no_grad():
                output = model.generate(
                    **tokenized_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    min_new_tokens=min_tokens,
                )

        prompt_len = tokenized_batch["input_ids"].shape[1]
        batch_outputs = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output
        ]
        outputs.extend(batch_outputs)

    return prompts, outputs


def sample_with_head_steering(
    model,
    tokenizer,
    conversations: List[List[dict]],
    vector: torch.Tensor,
    layer: int,
    head_indices: List[int],
    coef: float,
    bs: int = 100,
    top_p: float = 1.0,
    max_tokens: int = 1000,
    temperature: float = 1.0,
    min_tokens: int = 1,
    steering_type: str = "response",
) -> Tuple[List[str], List[str]]:
    """ヘッド単位のステアリングベクトルを使用してモデルの応答を生成する

    Args:
        model: 使用する言語モデル（Transformers）
        tokenizer: トークナイザー
        conversations: 会話データのリスト
        vector: ステアリングベクトル
        layer: 適用するレイヤー番号（0-based index）
        head_indices: ステアリング対象のヘッドインデックスのリスト
        coef: ステアリング係数
        bs: バッチサイズ
        top_p: nucleus samplingパラメータ
        max_tokens: 最大生成トークン数
        temperature: サンプリング温度
        min_tokens: 最小生成トークン数
        steering_type: ステアリングタイプ（"response", "prompt", "all"）

    Returns:
        tuple: (プロンプトリスト, 応答リスト)
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = []
    for messages in conversations:
        text = apply_chat_template_safe(
            tokenizer, messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(text)

    outputs = []

    for i in trange(0, len(prompts), bs, desc="Generating"):
        batch = prompts[i : i + bs]
        tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
        tokenized_batch = {k: v.to(model.device) for k, v in tokenized_batch.items()}

        steerer = ActivationSteererHead(
            model,
            vector,
            coeff=coef,
            layer_idx=layer,
            head_indices=head_indices,
            positions=steering_type,
        )

        with steerer:
            with torch.no_grad():
                output = model.generate(
                    **tokenized_batch,
                    do_sample=(temperature > 0),
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    use_cache=True,
                    min_new_tokens=min_tokens,
                )

        prompt_len = tokenized_batch["input_ids"].shape[1]
        batch_outputs = [
            tokenizer.decode(o[prompt_len:], skip_special_tokens=True) for o in output
        ]
        outputs.extend(batch_outputs)

    return prompts, outputs

