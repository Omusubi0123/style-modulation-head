"""
activation_ablation.py - Removal of persona vector direction components (ablation)

Remove persona vector direction components from specific layer's attn/mlp output.
ablated_output = output - (output · unit_persona_vector) * unit_persona_vector
"""

from typing import Sequence, Union

import torch

from src.activation_steer.base.ablator import BaseActivationAblator


class ActivationAblator(BaseActivationAblator):
    """Remove persona vector direction components from specific block output (ablation)

    Remove the persona vector direction component from attn or mlp output
    at the specified layer to eliminate that persona's contribution.
    """

    VALID_ABLATION_TYPES = {"attn_output", "mlp_output"}

    def __init__(
        self,
        model: torch.nn.Module,
        persona_vector: Union[torch.Tensor, Sequence[float]],
        *,
        layer_idx: int = -1,
        ablation_type: str = "attn_output",
        positions: str = "all",
        debug: bool = False,
    ):
        """Constructor

        Args:
            model: Target model
            persona_vector: Persona vector (direction to be removed)
            layer_idx: Target layer index (0-based, default: -1)
            ablation_type: Ablation type ("attn_output"|"mlp_output")
            positions: Application position ("all"|"prompt"|"response")
            debug: Enable debug output
        """
        super().__init__(
            model,
            persona_vector,
            layer_idx=layer_idx,
            positions=positions,
            debug=debug,
        )
        self.ablation_type = ablation_type.lower()

        if self.ablation_type not in self.VALID_ABLATION_TYPES:
            raise ValueError(
                f"ablation_type must be one of {self.VALID_ABLATION_TYPES}"
            )

    def _locate_target_module(self) -> torch.nn.Module:
        """Identify target module based on ablation type"""
        layer = self._get_layer()

        if self.ablation_type == "attn_output":
            attn_block = self._find_attention_block(layer)
            if attn_block:
                return attn_block
        elif self.ablation_type == "mlp_output":
            mlp_block = self._find_mlp_block(layer)
            if mlp_block:
                return mlp_block

        raise ValueError(
            f"Could not find target module for ablation_type={self.ablation_type}"
        )

    def _register_hooks(self) -> None:
        """Register hooks"""
        target_module = self._locate_target_module()
        self._handle = target_module.register_forward_hook(self._hook_fn)
        if self.debug:
            print(
                f"[ActivationAblator] Registered hook at layer {self.layer_idx}, "
                f"type={self.ablation_type}"
            )

    def _hook_fn(self, module: object, ins: object, out: object):
        """Forward hook: remove persona vector direction from output tensor"""
        if torch.is_tensor(out):
            new_out = self._remove_persona_direction(out)
        elif isinstance(out, (tuple, list)):
            if not torch.is_tensor(out[0]):
                return out
            head = self._remove_persona_direction(out[0])
            new_out = (head, *out[1:])
        else:
            return out

        if self.debug:
            with torch.no_grad():
                original = out[0] if isinstance(out, (tuple, list)) else out
                modified = new_out[0] if isinstance(new_out, tuple) else new_out
                delta = (original - modified).abs().mean()
                print(
                    f"[ActivationAblator] Removed persona direction at layer {self.layer_idx}, "
                    f"type={self.ablation_type}, |delta|={delta:.4g}"
                )
        return new_out


class ActivationAblatorMultiple:
    """Apply multiple ablations to different layers simultaneously"""

    def __init__(
        self,
        model: torch.nn.Module,
        instructions: list[dict],
        *,
        debug: bool = False,
    ):
        """Constructor

        Args:
            model: Target model
            instructions: List of ablation instructions
                Each dict has the following keys:
                - persona_vector: Persona vector
                - layer_idx: Target layer index
                - ablation_type: "attn_output" or "mlp_output"
                - positions: "all", "prompt", "response" (optional)
            debug: Enable debug output
        """
        self.model = model
        self.instructions = instructions
        self.debug = debug
        self._ablators = []

        for inst in self.instructions:
            ablator = ActivationAblator(
                model,
                persona_vector=inst["persona_vector"],
                layer_idx=inst["layer_idx"],
                ablation_type=inst["ablation_type"],
                positions=inst.get("positions", "all"),
                debug=debug,
            )
            self._ablators.append(ablator)

    def __enter__(self):
        """Register hooks for all ablators"""
        for ablator in self._ablators:
            ablator._register_hooks()
        return self

    def __exit__(self, *exc):
        """Remove all hooks"""
        self.remove()

    def remove(self):
        """Remove all registered hooks"""
        for ablator in self._ablators:
            ablator.remove()
