"""
activation_steer_head.py - 特定のAttentionヘッドのみをsteeringする

特定の層の特定のヘッドのO projection前の出力部分のみを増大させてsteering。
"""

from typing import List, Sequence, Union

import torch

from .base import BaseActivationSteerer


class ActivationSteererHead(BaseActivationSteerer):
    """特定のAttentionヘッドのO projection前の出力をsteeringする

    Attention内部のattn_weights @ Vの結果（O projection前）に対して、
    特定のヘッドの次元のみにsteeringベクトルを加算する。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        head_indices: List[int] = None,
        positions: str = "all",
        debug: bool = False,
    ):
        """コンストラクタ

        Args:
            model: 対象のモデル
            steering_vector: ステアリングベクトル（1次元、hidden_size）
            coeff: 係数（デフォルト: 1.0）
            layer_idx: 対象レイヤーのインデックス（0-based）
            head_indices: ステアリング対象のヘッドインデックスのリスト
            positions: 反映位置（"all"|"prompt"|"response"）
            debug: デバッグ出力を有効化
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

        # Attention設定を取得
        attn_config = self._get_attention_config()
        self.num_heads = attn_config["num_attention_heads"]
        self.head_dim = attn_config["head_dim"]

        # Validate head indices
        for h_idx in self.head_indices:
            if h_idx < 0 or h_idx >= self.num_heads:
                raise ValueError(
                    f"head_index {h_idx} out of range [0, {self.num_heads})"
                )

        # 指定されたヘッドの次元のみを有効にしたマスクを作成
        self.head_mask = torch.zeros_like(self.vector)
        for h_idx in self.head_indices:
            start_idx = h_idx * self.head_dim
            end_idx = (h_idx + 1) * self.head_dim
            self.head_mask[start_idx:end_idx] = 1.0

        # マスクされたsteeringベクトル
        self.masked_vector = self.vector * self.head_mask

        if self.debug:
            print(f"[ActivationSteererHead] num_heads: {self.num_heads}")
            print(f"[ActivationSteererHead] head_dim: {self.head_dim}")
            print(f"[ActivationSteererHead] head_indices: {self.head_indices}")
            print(
                f"[ActivationSteererHead] masked_vector norm: {self.masked_vector.norm():.4f}"
            )

    def _locate_o_proj(self) -> torch.nn.Module:
        """対象レイヤーのo_projモジュールを特定する"""
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
        """フックを登録"""
        o_proj = self._locate_o_proj()
        self._handle = o_proj.register_forward_pre_hook(self._hook_fn)

    def _hook_fn(self, module, input):
        """pre_hookフック：o_projへの入力に対して指定ヘッドのみsteeringを適用"""
        steer = self.coeff * self.masked_vector

        def _add_steering(t):
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
    """複数のヘッドsteeringを異なるレイヤーに同時適用する"""

    def __init__(
        self,
        model: torch.nn.Module,
        instructions: Sequence[dict],
        *,
        debug: bool = False,
    ):
        """コンストラクタ

        Args:
            model: 対象のモデル
            instructions: ステアリング指示のリスト
                各dictは以下のキーを持つ:
                - steering_vector: ステアリングベクトル
                - coeff: 係数（オプション、デフォルト: 1.0）
                - layer_idx: レイヤーインデックス（オプション、デフォルト: -1）
                - head_indices: ヘッドインデックスのリスト（オプション）
                - positions: 反映位置（オプション、デフォルト: "all"）
            debug: デバッグ出力を有効化
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
        """全ステアラーにフック登録"""
        for steerer in self._steerers:
            steerer._register_hooks()
        return self

    def __exit__(self, *exc):
        """すべてのフックを解除"""
        self.remove()

    def remove(self):
        """すべての登録済みフックを解除"""
        for steerer in self._steerers:
            steerer.remove()


def create_head_steering_instructions(
    steering_vector: torch.Tensor,
    layer_idx: int,
    head_indices: List[int],
    coeff: float = 1.0,
    positions: str = "all",
) -> dict:
    """ヘッドsteeringの指示を作成するヘルパー関数"""
    return {
        "steering_vector": steering_vector,
        "coeff": coeff,
        "layer_idx": layer_idx,
        "head_indices": head_indices,
        "positions": positions,
    }
