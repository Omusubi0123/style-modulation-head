"""
steerer.py - Base class for adding steering vectors
"""

from typing import Sequence, Union

import torch

from src.activation_steer.base.modifier import BaseActivationModifier


class BaseActivationSteerer(BaseActivationModifier):
    """Base class for adding steering vectors"""

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        positions: str = "all",
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
            renorm_to_original_norm: Whether to restore norm after steering
            debug: Enable debug output
        """
        super().__init__(model, layer_idx=layer_idx, positions=positions, debug=debug)
        self.coeff = float(coeff)
        self.renorm_to_original_norm = renorm_to_original_norm

        # Build vector
        p = next(model.parameters())
        self.vector = torch.as_tensor(steering_vector, dtype=p.dtype, device=p.device)
        if self.vector.ndim != 1:
            raise ValueError("steering_vector must be 1-D")

        hidden = getattr(model.config, "hidden_size", None)
        if hidden and self.vector.numel() != hidden:
            raise ValueError(
                f"Vector length {self.vector.numel()} ≠ model hidden_size {hidden}"
            )

    def _apply_steering(self, t: torch.Tensor) -> torch.Tensor:
        """Apply steering to tensor

        Args:
            t: Input tensor [batch, seq_len, hidden]

        Returns:
            Tensor after steering
        """
        steer = self.coeff * self.vector

        if self.positions == "all":
            result = t + steer.to(t.device)
            if self.renorm_to_original_norm:
                with torch.no_grad():
                    orig_norm = t.norm(dim=-1, keepdim=True)
                    new_norm = result.norm(dim=-1, keepdim=True)
                    scale = orig_norm / (new_norm + 1e-8)
                result = result * scale
            return result

        elif self.positions == "prompt":
            if t.shape[1] == 1:
                return t
            t2 = t.clone()
            t2 += steer.to(t.device)
            if self.renorm_to_original_norm:
                with torch.no_grad():
                    orig_norm = t.norm(dim=-1, keepdim=True)
                    new_norm = t2.norm(dim=-1, keepdim=True)
                    scale = orig_norm / (new_norm + 1e-8)
                t2 = t2 * scale
            return t2

        elif self.positions == "response":
            t2 = t.clone()
            t2[:, -1, :] += steer.to(t.device)
            if self.renorm_to_original_norm:
                with torch.no_grad():
                    orig_norm = t[:, -1, :].norm(dim=-1, keepdim=True)
                    new_norm = t2[:, -1, :].norm(dim=-1, keepdim=True)
                    scale = orig_norm / (new_norm + 1e-8)
                t2[:, -1, :] = t2[:, -1, :] * scale
            return t2

        else:
            raise ValueError(f"Invalid positions: {self.positions}")
