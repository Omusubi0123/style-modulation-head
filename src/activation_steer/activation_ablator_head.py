"""
activation_ablator_head.py - Zero ablation of specific attention heads

For Style Head zero ablation experiments:
Zero out the O projection input of specific heads at specific layers
to remove their contribution.
"""

from typing import Sequence

import torch

from src.activation_steer.base.modifier import BaseActivationModifier


class ActivationAblatorHead(BaseActivationModifier):
    """Zero out the O projection input of specific attention heads (Zero Ablation)

    Zero out specific head dimensions of the attn_weights @ V result
    (before O projection) to remove that head's contribution.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        layer_idx: int = -1,
        head_indices: list[int] | None = None,
        positions: str = "all",
        debug: bool = False,
    ):
        """Constructor

        Args:
            model: Target model
            layer_idx: Target layer index (0-based)
            head_indices: List of head indices to ablate (0-based)
            positions: Application position ("all"|"prompt"|"response")
            debug: Enable debug output
        """
        super().__init__(model, layer_idx=layer_idx, positions=positions, debug=debug)
        self.head_indices = head_indices if head_indices is not None else []

        # Get attention configuration
        attn_config = self._get_attention_config()
        self.num_heads = attn_config["num_attention_heads"]
        self.head_dim = attn_config["head_dim"]
        self.hidden_size = attn_config["hidden_size"]

        # Validate head indices
        for h_idx in self.head_indices:
            if h_idx < 0 or h_idx >= self.num_heads:
                raise ValueError(
                    f"head_index {h_idx} out of range [0, {self.num_heads})"
                )

        # Create mask to zero out specified head dimensions (1=keep, 0=zero)
        p = next(model.parameters())
        self.head_mask = torch.ones(self.hidden_size, dtype=p.dtype, device=p.device)
        for h_idx in self.head_indices:
            start_idx = h_idx * self.head_dim
            end_idx = (h_idx + 1) * self.head_dim
            self.head_mask[start_idx:end_idx] = 0.0

        if self.debug:
            print(f"[ActivationAblatorHead] num_heads: {self.num_heads}")
            print(f"[ActivationAblatorHead] head_dim: {self.head_dim}")
            print(f"[ActivationAblatorHead] head_indices: {self.head_indices}")
            print(f"[ActivationAblatorHead] layer_idx: {self.layer_idx}")

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
            print(f"[ActivationAblatorHead] Found o_proj: {type(o_proj).__name__}")

        return o_proj

    def _register_hooks(self) -> None:
        """Register hooks"""
        o_proj = self._locate_o_proj()
        self._handle = o_proj.register_forward_pre_hook(self._hook_fn)

    def _hook_fn(self, module: object, input: object):
        """Pre-hook: zero out specified heads on o_proj input"""
        mask = self.head_mask

        def _apply_zero_ablation(t: torch.Tensor) -> torch.Tensor:
            if self.positions == "all":
                return t * mask.to(t.device)
            elif self.positions == "prompt":
                if t.shape[1] == 1:
                    return t
                return t * mask.to(t.device)
            elif self.positions == "response":
                t2 = t.clone()
                t2[:, -1, :] = t2[:, -1, :] * mask.to(t.device)
                return t2
            else:
                raise ValueError(f"Invalid positions: {self.positions}")

        if isinstance(input, tuple):
            if len(input) > 0 and torch.is_tensor(input[0]):
                new_input = (_apply_zero_ablation(input[0]), *input[1:])
                return new_input
            return input
        elif torch.is_tensor(input):
            return _apply_zero_ablation(input)
        return input


class ActivationAblatorHeadMultiple:
    """Apply multiple head ablations to different layers simultaneously"""

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
            instructions: List of ablation instructions
                Each dict has the following keys:
                - layer_idx: Layer index (optional, default: -1)
                - head_indices: List of head indices (optional)
                - positions: Application position (optional, default: "all")
            debug: Enable debug output
        """
        self.model = model
        self.instructions = instructions
        self.debug = debug
        self._ablators = []

        for inst in self.instructions:
            ablator = ActivationAblatorHead(
                model,
                layer_idx=inst.get("layer_idx", -1),
                head_indices=inst.get("head_indices", []),
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


def create_head_ablation_instructions(
    layer_idx: int,
    head_indices: list[int],
    positions: str = "all",
) -> dict:
    """Helper function to create head ablation instructions"""
    return {
        "layer_idx": layer_idx,
        "head_indices": head_indices,
        "positions": positions,
    }


def load_style_heads_from_csv(csv_path: str) -> list[dict]:
    """Load style head information from CSV file

    CSV format:
    layer,cor_head,anti_head
    20,"3,5,28","1,27"
    ...

    Args:
        csv_path: Path to CSV file

    Returns:
        List of style head information. Each element is:
        {
            "layer": int (0-based index),
            "cor_heads": List[int] (0-based indices),
            "anti_heads": List[int] (0-based indices),
        }
    """
    import csv

    style_heads = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # layer is 1-indexed, convert to 0-indexed
            layer_1based = int(row["layer"])
            layer_0based = layer_1based - 1

            # Parse head indices (1-indexed, convert to 0-indexed)
            cor_heads = []
            if row["cor_head"].strip():
                cor_heads = [int(h.strip()) - 1 for h in row["cor_head"].split(",")]

            anti_heads = []
            if row["anti_head"].strip():
                anti_heads = [int(h.strip()) - 1 for h in row["anti_head"].split(",")]

            style_heads.append(
                {
                    "layer": layer_0based,
                    "cor_heads": cor_heads,
                    "anti_heads": anti_heads,
                }
            )

    return style_heads
