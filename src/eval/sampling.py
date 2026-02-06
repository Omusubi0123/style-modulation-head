"""
sampling.py - Sampling utilities with steering

Provides common sampling functionality for model response generation.
Uses ActivationSteerer classes directly.
"""

from typing import Tuple

import torch
from tqdm import trange
from vllm import SamplingParams

from src.activation_steer.activation_steer import (
    ActivationSteerer,
    ActivationSteererBlock,
)
from src.activation_steer.activation_steer_head import ActivationSteererHead
from src.chat_template_utils import apply_chat_template_safe


def sample_vllm(
    model,
    tokenizer,
    conversations: list[list[dict]],
    top_p: float = 1,
    max_tokens: int = 1000,
    temperature: float = 1,
    min_tokens: int = 1,
) -> Tuple[list[str], list[str]]:
    """Generate model responses using standard VLLM sampling

    Args:
        model: VLLM model
        tokenizer: Tokenizer
        conversations: list of conversation data
        top_p: Nucleus sampling parameter (default: 1)
        max_tokens: Maximum generation tokens (default: 1000)
        temperature: Sampling temperature (default: 1)
        min_tokens: Minimum generation tokens (default: 1)

    Returns:
        tuple: (list of prompts, list of responses)
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
    conversations: list[list[dict]],
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
) -> Tuple[list[str], list[str]]:
    """Generate model responses using steering vector

    Args:
        model: Language model (Transformers)
        tokenizer: Tokenizer
        conversations: list of conversation data
        vector: Steering vector
        layer: Layer number to apply (0-based index)
        coef: Steering coefficient
        bs: Batch size (default: 100)
        top_p: Nucleus sampling parameter (default: 1)
        max_tokens: Maximum generation tokens (default: 1000)
        temperature: Sampling temperature (default: 1)
        min_tokens: Minimum generation tokens (default: 1)
        steering_type: Steering type ("response", "prompt", "all")
        renorm_after_steering: Whether to renormalize after steering

    Returns:
        tuple: (list of prompts, list of responses)
    """
    # Configure tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare prompts
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
    conversations: list[list[dict]],
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
) -> Tuple[list[str], list[str]]:
    """Generate model responses using block position steering vector

    Args:
        model: Language model (Transformers)
        tokenizer: Tokenizer
        conversations: list of conversation data
        vector: Steering vector
        layer: Layer number to apply (0-based index)
        coef: Steering coefficient
        block_steering_type: Block steering type
            - "attn": Input to attention block
            - "mlp": Input to MLP block
            - "attn_layernorm": Input to layer norm before attention
            - "mlp_layernorm": Input to layer norm before MLP
            - "attn_output": Output of attention block
            - "mlp_output": Output of MLP block
        bs: Batch size
        top_p: Nucleus sampling parameter
        max_tokens: Maximum generation tokens
        temperature: Sampling temperature
        min_tokens: Minimum generation tokens
        steering_type: Position steering type ("response", "prompt", "all")

    Returns:
        tuple: (list of prompts, list of responses)
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
    conversations: list[list[dict]],
    vector: torch.Tensor,
    layer: int,
    head_indices: list[int],
    coef: float,
    bs: int = 100,
    top_p: float = 1.0,
    max_tokens: int = 1000,
    temperature: float = 1.0,
    min_tokens: int = 1,
    steering_type: str = "response",
) -> Tuple[list[str], list[str]]:
    """Generate model responses using head-level steering vector

    Args:
        model: Language model (Transformers)
        tokenizer: Tokenizer
        conversations: list of conversation data
        vector: Steering vector
        layer: Layer number to apply (0-based index)
        head_indices: list of head indices to steer
        coef: Steering coefficient
        bs: Batch size
        top_p: Nucleus sampling parameter
        max_tokens: Maximum generation tokens
        temperature: Sampling temperature
        min_tokens: Minimum generation tokens
        steering_type: Steering type ("response", "prompt", "all")

    Returns:
        tuple: (list of prompts, list of responses)
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
