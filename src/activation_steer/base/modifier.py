"""
modifier.py - Abstract base class for activation modification

Provides common functionality for ActivationSteerer, ActivationAblator, etc.
"""

from abc import ABC, abstractmethod
from typing import Iterable, Optional

import torch


class BaseActivationModifier(ABC):
    """Abstract base class for activation modification

    Provides common infrastructure for registering hooks on specific
    model layers and modifying activations.
    """

    # Layer list paths for different model architectures
    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "transformer.h",  # GPT-2/Neo, Bloom, etc.
        "encoder.layer",  # BERT/RoBERTa
        "model.layers",  # Llama/Mistral/Qwen
        "gpt_neox.layers",  # GPT-NeoX
        "block",  # Flan-T5
        "language_model.layers",  # Multimodal Gemma-3
    )

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        layer_idx: int = -1,
        positions: str = "all",
        debug: bool = False,
    ):
        """Constructor

        Args:
            model: Target model
            layer_idx: Target layer index (0-based, default: -1)
            positions: Application position ("all"|"prompt"|"response")
            debug: Enable debug output
        """
        self.model = model
        self.layer_idx = layer_idx
        self.positions = positions.lower()
        self.debug = debug
        self._handle = None

        # Check if positions is valid
        valid_positions = {"all", "prompt", "response"}
        if self.positions not in valid_positions:
            raise ValueError(f"positions must be one of {valid_positions}")

    def _get_max_layer(self) -> int:
        """Get maximum layer count from model

        Returns:
            int: Maximum layer count

        Raises:
            AttributeError: If layer count not found
        """
        possible_layer_attrs = [
            "num_hidden_layers",
            "n_layers",
            "num_layers",
            "n_layer",
        ]
        max_layer = None

        # First try main config
        for attr in possible_layer_attrs:
            if hasattr(self.model.config, attr):
                max_layer = getattr(self.model.config, attr)
                if self.debug:
                    print(f"Using {attr} = {max_layer}")
                break

        # If not found in main config, try text_config (for multimodal models)
        if max_layer is None and hasattr(self.model.config, "text_config"):
            text_config = self.model.config.text_config
            if self.debug:
                print("Checking text_config for layer attributes...")
            for attr in possible_layer_attrs:
                if hasattr(text_config, attr):
                    max_layer = getattr(text_config, attr)
                    if self.debug:
                        print(f"Using text_config.{attr} = {max_layer}")
                    break

        if max_layer is None:
            if self.debug:
                print(
                    "Available attributes in main config:",
                    [
                        attr
                        for attr in dir(self.model.config)
                        if not attr.startswith("_")
                    ],
                )
                if hasattr(self.model.config, "text_config"):
                    print(
                        "Available attributes in text_config:",
                        [
                            attr
                            for attr in dir(self.model.config.text_config)
                            if not attr.startswith("_")
                        ],
                    )
            raise AttributeError(
                "Could not find layer count attribute in model config or text_config"
            )

        return max_layer

    def _locate_layer_list(self) -> torch.nn.ModuleList:
        """Locate layer list from model

        Returns:
            torch.nn.Modulelist: Layer list

        Raises:
            ValueError: If layer list not found
        """
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            path_parts = path.split(".")
            found = True

            for part in path_parts:
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    found = False
                    break

            if found and hasattr(cur, "__getitem__"):
                return cur

        raise ValueError("Could not find layer list in model")

    def _get_layer(self, layer_idx: Optional[int] = None) -> torch.nn.Module:
        """Get layer at specified index

        Args:
            layer_idx: Layer index (uses self.layer_idx if None)

        Returns:
            torch.nn.Module: Target layer

        Raises:
            IndexError: If layer index out of range
        """
        if layer_idx is None:
            layer_idx = self.layer_idx

        max_layer = self._get_max_layer()
        if not (-max_layer <= layer_idx < max_layer):
            raise IndexError(
                f"layer_idx {layer_idx} out of range [{-max_layer}, {max_layer})"
            )

        layer_list = self._locate_layer_list()
        return layer_list[layer_idx]

    def _get_attention_config(self) -> dict:
        """Get attention-related configuration from model

        Returns:
            dict: Attention configuration
                - num_attention_heads: Number of attention heads
                - num_key_value_heads: Number of KV heads (for GQA)
                - hidden_size: Hidden size
                - head_dim: Head dimension

        Raises:
            AttributeError: If configuration not found
        """
        cfg = self.model.config
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

    def _find_submodule(
        self, layer: torch.nn.Module, attr_names: list[str]
    ) -> Optional[torch.nn.Module]:
        """Find submodule by attribute names from layer

        Args:
            layer: Target layer
            attr_names: list of attribute names to try

        Returns:
            torch.nn.Module: Found submodule, or None if not found
        """
        for attr in attr_names:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        return None

    def _find_attention_block(
        self, layer: torch.nn.Module
    ) -> Optional[torch.nn.Module]:
        """Find attention block from layer"""
        return self._find_submodule(layer, ["self_attn", "attention", "attn"])

    def _find_mlp_block(self, layer: torch.nn.Module) -> Optional[torch.nn.Module]:
        """Find MLP block from layer"""
        return self._find_submodule(layer, ["mlp", "feed_forward", "ffn"])

    def _find_attn_layernorm(self, layer: torch.nn.Module) -> Optional[torch.nn.Module]:
        """Find LayerNorm before attention from layer"""
        return self._find_submodule(
            layer,
            ["input_layernorm", "ln_1", "layer_norm", "pre_attention_layernorm"],
        )

    def _find_mlp_layernorm(self, layer: torch.nn.Module) -> Optional[torch.nn.Module]:
        """Find LayerNorm before MLP from layer"""
        return self._find_submodule(
            layer, ["post_attention_layernorm", "ln_2", "mlp_layernorm"]
        )

    def _find_o_proj(self, attn_block: torch.nn.Module) -> Optional[torch.nn.Module]:
        """Find o_proj from attention block"""
        return self._find_submodule(attn_block, ["o_proj", "out_proj", "dense"])

    @abstractmethod
    def _register_hooks(self) -> None:
        """Register hooks (to be implemented by subclasses)"""
        pass

    def __enter__(self):
        """Register hooks when entering context manager"""
        self._register_hooks()
        return self

    def __exit__(self, *exc):
        """Remove hooks when exiting context manager"""
        self.remove()

    def remove(self):
        """Remove registered hooks"""
        if self._handle:
            self._handle.remove()
            self._handle = None
