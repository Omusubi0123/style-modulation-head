"""
ablator.py - 特定方向の成分を除去するための基底クラス
"""

from typing import Sequence, Union

import torch

from src.activation_steer.base.modifier import BaseActivationModifier


class BaseActivationAblator(BaseActivationModifier):
    """特定方向の成分を除去するための基底クラス"""

    def __init__(
        self,
        model: torch.nn.Module,
        persona_vector: Union[torch.Tensor, Sequence[float]],
        *,
        layer_idx: int = -1,
        positions: str = "all",
        debug: bool = False,
    ):
        """コンストラクタ

        Args:
            model: 対象のモデル
            persona_vector: Persona Vector（この方向成分を除去）
            layer_idx: 対象レイヤーのインデックス（0-based、デフォルト: -1）
            positions: 反映位置（"all"|"prompt"|"response"）
            debug: デバッグ出力を有効化
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

        # 単位ベクトルを事前計算
        self.unit_vector = self.persona_vector / (self.persona_vector.norm() + 1e-8)

    def _remove_persona_direction(self, t: torch.Tensor) -> torch.Tensor:
        """テンソルからPersona Vector方向の成分を除去

        ablated = t - (t · unit_vec) * unit_vec

        Args:
            t: 入力テンソル [batch, seq_len, hidden]

        Returns:
            Persona Vector方向成分を除去後のテンソル
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

