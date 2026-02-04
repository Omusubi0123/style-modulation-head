"""
activation_steer_head.py - Steering for specific attention heads only

Apply steering to specific heads' O projection input only.
"""

from typing import Sequence, Union

import torch

from src.activation_steer.base.steerer import BaseActivationSteerer


class ActivationSteererHead(BaseActivationSteerer):
    """Steer the O projection input of specific attention heads

    Apply steering vector to specific head dimensions of the
    attn_weights @ V result (before O projection).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        head_indices: list[int] | None = None,
        positions: str = "all",
        debug: bool = False,
    ):
        """Constructor

        Args:
            model: Target model
            steering_vector: Steering vector (1-dimensional, hidden_size)
            coeff: Coefficient (default: 1.0)
            layer_idx: Target layer index (0-based)
            head_indices: List of head indices to steer
            positions: Application position ("all"|"prompt"|"response")
            debug: Enable debug output
        """
        super().__init__(
            model,
            steering_vector,
            coeff=coeff,
            layer_idx=layer_idx,
            positions=positions,
            renorm_to_original_norm=False,
            debug=debug,
        )

        self.head_indices = head_indices if head_indices is not None else []

        # Get attention configuration
        attn_config = self._get_attention_config()
        self.num_heads = attn_config["num_attention_heads"]
        self.head_dim = attn_config["head_dim"]

        # Validate head indices
        for h_idx in self.head_indices:
            if h_idx < 0 or h_idx >= self.num_heads:
                raise ValueError(
                    f"head_index {h_idx} out of range [0, {self.num_heads})"
                )

        # Create mask that only enables specified head dimensions
        self.head_mask = torch.zeros_like(self.vector)
        for h_idx in self.head_indices:
            start_idx = h_idx * self.head_dim
            end_idx = (h_idx + 1) * self.head_dim
            self.head_mask[start_idx:end_idx] = 1.0

        # Masked steering vector
        self.masked_vector = self.vector * self.head_mask

        if self.debug:
            print(f"[ActivationSteererHead] num_heads: {self.num_heads}")
            print(f"[ActivationSteererHead] head_dim: {self.head_dim}")
            print(f"[ActivationSteererHead] head_indices: {self.head_indices}")
            print(
                f"[ActivationSteererHead] masked_vector norm: {self.masked_vector.norm():.4f}"
            )

    def _locate_o_proj(self) -> torch.nn.Module:
        """Locate the o_proj module for the target layer"""
        layer = self._get_layer()
        attn_block = self._find_attention_block(layer)

        if attn_block is None:
            raise ValueError(
                f"Could not find attention block for layer {self.layer_idx}"
            )

        o_proj = self._find_o_proj(attn_block)
        if o_proj is None:
            raise ValueError(f"Could not find o_proj for layer {self.layer_idx}")

        if self.debug:
            print(f"[ActivationSteererHead] Found o_proj: {type(o_proj).__name__}")

        return o_proj

    def _register_hooks(self) -> None:
        """Register hooks"""
        o_proj = self._locate_o_proj()
        self._handle = o_proj.register_forward_pre_hook(self._hook_fn)

    def _hook_fn(self, module: object, input: object):
        """Pre-hook: apply steering to specified heads only on o_proj input"""
        steer = self.coeff * self.masked_vector

        def _add_steering(t: torch.Tensor) -> torch.Tensor:
            if self.positions == "all":
                return t + steer.to(t.device)
            elif self.positions == "prompt":
                if t.shape[1] == 1:
                    return t
                t2 = t.clone()
                t2 += steer.to(t.device)
                return t2
            elif self.positions == "response":
                t2 = t.clone()
                t2[:, -1, :] += steer.to(t.device)
                return t2
            else:
                raise ValueError(f"Invalid positions: {self.positions}")

        if isinstance(input, tuple):
            if len(input) > 0 and torch.is_tensor(input[0]):
                new_input = (_add_steering(input[0]), *input[1:])
                return new_input
            return input
        elif torch.is_tensor(input):
            return _add_steering(input)
        return input


class ActivationSteererHeadMultiple:
    """Apply multiple head steering to different layers simultaneously"""

    def __init__(
        self,
        model: torch.nn.Module,
        instructions: Sequence[dict],
        *,
        debug: bool = False,
    ):
        """Constructor

        Args:
            model: Target model
            instructions: List of steering instructions
                Each dict has the following keys:
                - steering_vector: Steering vector
                - coeff: Coefficient (optional, default: 1.0)
                - layer_idx: Layer index (optional, default: -1)
                - head_indices: List of head indices (optional)
                - positions: Application position (optional, default: "all")
            debug: Enable debug output
        """
        self.model = model
        self.instructions = instructions
        self.debug = debug
        self._steerers = []

        for inst in self.instructions:
            steerer = ActivationSteererHead(
                model,
                inst["steering_vector"],
                coeff=inst.get("coeff", 1.0),
                layer_idx=inst.get("layer_idx", -1),
                head_indices=inst.get("head_indices", []),
                positions=inst.get("positions", "all"),
                debug=debug,
            )
            self._steerers.append(steerer)

    def __enter__(self):
        """Register hooks for all steerers"""
        for steerer in self._steerers:
            steerer._register_hooks()
        return self

    def __exit__(self, *exc):
        """Remove all hooks"""
        self.remove()

    def remove(self):
        """Remove all registered hooks"""
        for steerer in self._steerers:
            steerer.remove()


def create_head_steering_instructions(
    steering_vector: torch.Tensor,
    layer_idx: int,
    head_indices: list[int],
    coeff: float = 1.0,
    positions: str = "all",
) -> dict:
    """Helper function to create head steering instructions"""
    return {
        "steering_vector": steering_vector,
        "coeff": coeff,
        "layer_idx": layer_idx,
        "head_indices": head_indices,
        "positions": positions,
    }
