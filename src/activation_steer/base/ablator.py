"""
ablator.py - Base class for removing direction components
"""

from typing import Sequence, Union

import torch

from src.activation_steer.base.modifier import BaseActivationModifier


class BaseActivationAblator(BaseActivationModifier):
    """Base class for removing direction components"""

    def __init__(
        self,
        model: torch.nn.Module,
        persona_vector: Union[torch.Tensor, Sequence[float]],
        *,
        layer_idx: int = -1,
        positions: str = "all",
        debug: bool = False,
    ):
        """Constructor

        Args:
            model: Target model
            persona_vector: Persona vector (direction to be removed)
            layer_idx: Target layer index (0-based, default: -1)
            positions: Application position ("all"|"prompt"|"response")
            debug: Enable debug output
        """
        super().__init__(model, layer_idx=layer_idx, positions=positions, debug=debug)

        # Build persona vector
        p = next(model.parameters())
        self.persona_vector = torch.as_tensor(
            persona_vector, dtype=p.dtype, device=p.device
        )
        if self.persona_vector.ndim != 1:
            raise ValueError("persona_vector must be 1-D")

        hidden = getattr(model.config, "hidden_size", None)
        if hidden and self.persona_vector.numel() != hidden:
            raise ValueError(
                f"Vector length {self.persona_vector.numel()} ≠ model hidden_size {hidden}"
            )

        # Pre-compute unit vector
        self.unit_vector = self.persona_vector / (self.persona_vector.norm() + 1e-8)

    def _remove_persona_direction(self, t: torch.Tensor) -> torch.Tensor:
        """Remove persona vector direction component from tensor

        ablated = t - (t · unit_vec) * unit_vec

        Args:
            t: Input tensor [batch, seq_len, hidden]

        Returns:
            Tensor with persona vector direction removed
        """
        unit_vec = self.unit_vector

        if self.positions == "all":
            projection = torch.sum(t * unit_vec.to(t.device), dim=-1, keepdim=True)
            return t - projection * unit_vec.to(t.device)

        elif self.positions == "prompt":
            if t.shape[1] == 1:
                return t
            projection = torch.sum(t * unit_vec.to(t.device), dim=-1, keepdim=True)
            return t - projection * unit_vec.to(t.device)

        elif self.positions == "response":
            t2 = t.clone()
            last_hidden = t2[:, -1, :]
            projection = torch.sum(
                last_hidden * unit_vec.to(t.device), dim=-1, keepdim=True
            )
            t2[:, -1, :] = last_hidden - projection * unit_vec.to(t.device)
            return t2

        else:
            raise ValueError(f"Invalid positions: {self.positions}")
