"""
activation_ablation.py - Persona Vector方向成分の除去（アブレーション）

特定の層のattn/mlp出力からPersona Vector方向の成分を除去する。
ablated_output = output - (output · unit_persona_vector) * unit_persona_vector
"""

from typing import Sequence, Union

import torch

from .base import BaseActivationAblator


class ActivationAblator(BaseActivationAblator):
    """特定のblock出力からPersona Vector方向の成分を除去する（アブレーション）

    指定した層のattnまたはmlp出力から、Persona Vectorの方向成分を除去することで、
    そのペルソナ特性の寄与を除去する。
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
        """コンストラクタ

        Args:
            model: 対象のモデル
            persona_vector: Persona Vector（この方向成分を除去）
            layer_idx: 対象レイヤーのインデックス（0-based、デフォルト: -1）
            ablation_type: アブレーションタイプ（"attn_output"|"mlp_output"）
            positions: 反映位置（"all"|"prompt"|"response"）
            debug: デバッグ出力を有効化
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
        """アブレーションタイプに応じて対象モジュールを特定する"""
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
        """フックを登録"""
        target_module = self._locate_target_module()
        self._handle = target_module.register_forward_hook(self._hook_fn)
        if self.debug:
            print(
                f"[ActivationAblator] Registered hook at layer {self.layer_idx}, "
                f"type={self.ablation_type}"
            )

    def _hook_fn(self, module, ins, out):
        """forwardフック：出力テンソルからPersona Vector方向の成分を除去"""
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
    """複数のアブレーションを異なるレイヤーに同時適用する"""

    def __init__(
        self,
        model: torch.nn.Module,
        instructions: list[dict],
        *,
        debug: bool = False,
    ):
        """コンストラクタ

        Args:
            model: 対象のモデル
            instructions: アブレーション指示のリスト
                各dictは以下のキーを持つ:
                - persona_vector: Persona Vector
                - layer_idx: 対象レイヤーのインデックス
                - ablation_type: "attn_output" または "mlp_output"
                - positions: "all", "prompt", "response"（オプション）
            debug: デバッグ出力を有効化
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
        """全アブレーターにフック登録"""
        for ablator in self._ablators:
            ablator._register_hooks()
        return self

    def __exit__(self, *exc):
        """すべてのフックを解除"""
        self.remove()

    def remove(self):
        """すべての登録済みフックを解除"""
        for ablator in self._ablators:
            ablator.remove()
