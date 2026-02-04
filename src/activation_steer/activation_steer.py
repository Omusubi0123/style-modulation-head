"""
activation_steer.py - Steering for transformer block outputs

Add steering vectors to specific layer outputs or specific positions
(attention input, MLP input, etc.).
"""

from typing import Sequence, Union

import torch

from src.activation_steer.base.steerer import BaseActivationSteerer


class ActivationSteerer(BaseActivationSteerer):
    """Add (coeff * steering_vector) to specific transformer block output"""

    def _register_hooks(self) -> None:
        """Register hooks"""
        layer = self._get_layer()
        if self.debug:
            print(f"[ActivationSteerer] hooking layer {self.layer_idx}")
        self._handle = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module: object, ins: object, out: object):
        """Forward hook: apply steering to output tensor"""

        def _process(t: torch.Tensor) -> torch.Tensor:
            return self._apply_steering(t)

        # out may be tensor or tuple/list
        if torch.is_tensor(out):
            new_out = _process(out)
        elif isinstance(out, (tuple, list)):
            if not torch.is_tensor(out[0]):
                return out
            head = _process(out[0])
            new_out = (head, *out[1:])
        else:
            return out

        if self.debug:
            with torch.no_grad():
                delta = (new_out[0] if isinstance(new_out, tuple) else new_out) - (
                    out[0] if isinstance(out, (tuple, list)) else out
                )
                print(
                    "[ActivationSteerer] |delta| (mean ± std): "
                    f"{delta.abs().mean():.4g} ± {delta.std():.4g}"
                )
        return new_out


class ActivationSteererBlock(BaseActivationSteerer):
    """Add (coeff * steering_vector) to specific block input or layer norm input

    steering_type must be one of:
    - "attn": Input to attention block (output of layer norm)
    - "mlp": Input to MLP block (output of layer norm)
    - "attn_layernorm": Input to layer norm before attention
    - "mlp_layernorm": Input to layer norm before MLP
    - "attn_output": Output of attention block (before residual addition)
    - "mlp_output": Output of MLP block (before residual addition)
    """

    VALID_STEERING_TYPES = {
        "attn",
        "mlp",
        "attn_layernorm",
        "mlp_layernorm",
        "attn_output",
        "mlp_output",
    }

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        positions: str = "all",
        steering_type: str = "attn",
        renorm_to_original_norm: bool = False,
        debug: bool = False,
    ):
        """Constructor

        Args:
            model: Target model
            steering_vector: Steering vector (1-dimensional)
            coeff: Coefficient (default: 1.0)
            layer_idx: Target layer index (0-based, default: -1)
            positions: Application position ("all"|"prompt"|"response")
            steering_type: Steering type
            renorm_to_original_norm: Whether to restore norm after steering
            debug: Enable debug output
        """
        super().__init__(
            model,
            steering_vector,
            coeff=coeff,
            layer_idx=layer_idx,
            positions=positions,
            renorm_to_original_norm=renorm_to_original_norm,
            debug=debug,
        )
        self.steering_type = steering_type.lower()

        if self.steering_type not in self.VALID_STEERING_TYPES:
            raise ValueError(
                f"steering_type must be one of {self.VALID_STEERING_TYPES}"
            )

    def _locate_target_module(self) -> tuple[torch.nn.Module, str]:
        """Identify target module based on steering type

        Returns:
            tuple: (target module, hook type "input" or "output")
        """
        layer = self._get_layer()

        if self.steering_type == "attn":
            # Output of layer norm before attention
            attn_ln = self._find_attn_layernorm(layer)
            if attn_ln:
                return attn_ln, "output"
            # If layer norm not found, hook directly to attention block input
            attn_block = self._find_attention_block(layer)
            if attn_block:
                return attn_block, "input"

        elif self.steering_type == "mlp":
            # Output of layer norm before MLP
            mlp_ln = self._find_mlp_layernorm(layer)
            if mlp_ln:
                return mlp_ln, "output"
            # If layer norm not found, hook directly to MLP block input
            mlp_block = self._find_mlp_block(layer)
            if mlp_block:
                return mlp_block, "input"

        elif self.steering_type == "attn_layernorm":
            # Input to layer norm before attention
            attn_ln = self._find_attn_layernorm(layer)
            if attn_ln:
                return attn_ln, "input"

        elif self.steering_type == "mlp_layernorm":
            # Input to layer norm before MLP
            mlp_ln = self._find_mlp_layernorm(layer)
            if mlp_ln:
                return mlp_ln, "input"

        elif self.steering_type == "attn_output":
            # Output of attention block
            attn_block = self._find_attention_block(layer)
            if attn_block:
                return attn_block, "output"

        elif self.steering_type == "mlp_output":
            # Output of MLP block
            mlp_block = self._find_mlp_block(layer)
            if mlp_block:
                return mlp_block, "output"

        raise ValueError(
            f"Could not find target module for steering_type={self.steering_type}"
        )

    def _register_hooks(self) -> None:
        """Register hooks"""
        target_module, hook_type = self._locate_target_module()

        if hook_type == "input":
            self._handle = target_module.register_forward_pre_hook(self._pre_hook_fn)
        else:
            self._handle = target_module.register_forward_hook(self._forward_hook_fn)

    def _pre_hook_fn(self, module: object, ins: object):
        """Pre-hook: apply steering to input tensor"""
        if isinstance(ins, tuple):
            if torch.is_tensor(ins[0]):
                new_ins = (self._apply_steering(ins[0]), *ins[1:])
                return new_ins
            return ins
        elif torch.is_tensor(ins):
            return self._apply_steering(ins)
        return ins

    def _forward_hook_fn(self, module: object, ins: object, out: object):
        """Forward hook: apply steering to output tensor"""
        if torch.is_tensor(out):
            new_out = self._apply_steering(out)
        elif isinstance(out, (tuple, list)):
            if not torch.is_tensor(out[0]):
                return out
            head = self._apply_steering(out[0])
            new_out = (head, *out[1:])
        else:
            return out

        if self.debug:
            with torch.no_grad():
                delta = (new_out[0] if isinstance(new_out, tuple) else new_out) - (
                    out[0] if isinstance(out, (tuple, list)) else out
                )
                print(
                    "[ActivationSteererBlock] |delta| (mean ± std): "
                    f"{delta.abs().mean():.4g} ± {delta.std():.4g}"
                )
        return new_out


class ActivationSteererMultiple:
    """Apply multiple steering vectors to different layers simultaneously"""

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
            instructions: list of steering instructions
                Each dict has the following keys:
                - steering_vector: Steering vector
                - coeff: Coefficient (optional, default: 1.0)
                - layer_idx: Layer index (optional, default: -1)
                - positions: Application position (optional, default: "all")
                - renorm_to_original_norm: Whether to renormalize (optional)
            debug: Enable debug output
        """
        self.model = model
        self.instructions = instructions
        self.debug = debug
        self._steerers = []

        for inst in self.instructions:
            steerer = ActivationSteerer(
                model,
                inst["steering_vector"],
                coeff=inst.get("coeff", 1.0),
                layer_idx=inst.get("layer_idx", -1),
                positions=inst.get("positions", "all"),
                renorm_to_original_norm=inst.get("renorm_to_original_norm", False),
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
