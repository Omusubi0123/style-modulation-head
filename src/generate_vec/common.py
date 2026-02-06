"""
common.py - Common utilities for persona vector generation

Provides common functionality used by generate_vec.py, generate_vec_attn.py,
and generate_vec_block.py.
"""

import gc
import json
import os
from typing import Tuple

import pandas as pd
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from src.settings import settings


def load_jsonl(file_path: str) -> list[dict]:
    with open(file_path, "r") as f:
        return [json.loads(line) for line in f]


def get_max_layer(model) -> int:
    """Get maximum layer count from model

    Args:
        model: Transformer model

    Returns:
        Maximum layer count

    Raises:
        AttributeError: If layer count not found
    """
    possible_layer_attrs = ["num_hidden_layers", "n_layers", "num_layers", "n_layer"]
    max_layer = None

    # First try main config
    for attr in possible_layer_attrs:
        if hasattr(model.config, attr):
            max_layer = getattr(model.config, attr)
            print(f"Using {attr} = {max_layer}")
            break

    # If not found in main config, try text_config (for multimodal models)
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
    """Locate layer list from model

    Args:
        model: Transformer model

    Returns:
        Layer list (Modulelist)

    Raises:
        ValueError: If layer list not found
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
    """Get attention-related configuration from model

    Args:
        model: Transformer model

    Returns:
        dict: Attention configuration
            - num_attention_heads: Number of attention heads
            - num_key_value_heads: Number of KV heads (for GQA)
            - hidden_size: Hidden size
            - head_dim: Head dimension
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
    """Load model and tokenizer

    Uses bfloat16 for memory efficiency.
    Also supports multimodal models.

    Args:
        model_name: Model name or path

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
            token=settings.hf_token,
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
                token=settings.hf_token,
            )
            print(f"Loaded model using AutoModel")
        else:
            raise

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=settings.hf_token
    )

    torch.cuda.empty_cache()

    return model, tokenizer


def get_persona_effective(
    pos_path: str, neg_path: str, trait: str, threshold: int = 50
) -> Tuple:
    """Filter and extract effective persona data

    Args:
        pos_path: CSV file path for positive persona data
        neg_path: CSV file path for negative persona data
        trait: Trait name to evaluate
        threshold: Filtering threshold (default: 50)

    Returns:
        tuple: (pos_effective, neg_effective, pos_prompts, neg_prompts, pos_responses, neg_responses)

    Raises:
        FileNotFoundError: If file not found
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
    pos_vectors: dict, neg_vectors: dict, layer_list: list[int], hidden_size: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute persona vector differences from positive and negative vectors

    Args:
        pos_vectors: Positive data vector dictionary (layer -> tensor)
        neg_vectors: Negative data vector dictionary (layer -> tensor)
        layer_list: list of layer indices
        hidden_size: Hidden size

    Returns:
        tuple: (prompt_avg_diff, response_avg_diff, prompt_last_diff)
            Each tensor has shape [num_layers, hidden_size]
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
    """Save persona vectors

    Args:
        save_dir: Save directory
        trait: Trait name
        prompt_avg_diff: Prompt average difference vector
        response_avg_diff: Response average difference vector
        prompt_last_diff: Prompt last token difference vector
        suffix: Filename suffix (e.g., "_attn_pre_o_proj")
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
    """Clear GPU memory"""
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()


def validate_effective_samples(
    pos_prompts: list[str], neg_prompts: list[str], threshold: int
) -> None:
    """Validate sample count after filtering

    Args:
        pos_prompts: list of positive prompts
        neg_prompts: list of negative prompts
        threshold: Filtering threshold

    Raises:
        ValueError: If no valid samples found
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
