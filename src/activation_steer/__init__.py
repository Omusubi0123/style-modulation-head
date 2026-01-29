"""
activation_steer - Module for activation steering/ablation

Provides functionality to add steering vectors to specific model layers
or remove specific direction components.

Usage (direct import recommended):
    from src.activation_steer.activation_steer import ActivationSteerer
    from src.activation_steer.activation_steer import ActivationSteererBlock
    from src.activation_steer.activation_steer_head import ActivationSteererHead
    from src.activation_steer.base.modifier import BaseActivationModifier
"""

# Exports for backward compatibility
from src.activation_steer.activation_ablation import (
    ActivationAblator,
    ActivationAblatorMultiple,
)
from src.activation_steer.activation_ablator_head import (
    ActivationAblatorHead,
    ActivationAblatorHeadMultiple,
    create_head_ablation_instructions,
    load_style_heads_from_csv,
)
from src.activation_steer.activation_steer import (
    ActivationSteerer,
    ActivationSteererBlock,
    ActivationSteererMultiple,
)
from src.activation_steer.activation_steer_head import (
    ActivationSteererHead,
    ActivationSteererHeadMultiple,
    create_head_steering_instructions,
)
from src.activation_steer.base import (
    BaseActivationAblator,
    BaseActivationModifier,
    BaseActivationSteerer,
)

__all__ = [
    # Base classes
    "BaseActivationModifier",
    "BaseActivationSteerer",
    "BaseActivationAblator",
    # Steerers
    "ActivationSteerer",
    "ActivationSteererBlock",
    "ActivationSteererMultiple",
    "ActivationSteererHead",
    "ActivationSteererHeadMultiple",
    "create_head_steering_instructions",
    # Ablators
    "ActivationAblator",
    "ActivationAblatorMultiple",
    "ActivationAblatorHead",
    "ActivationAblatorHeadMultiple",
    "create_head_ablation_instructions",
    "load_style_heads_from_csv",
]
