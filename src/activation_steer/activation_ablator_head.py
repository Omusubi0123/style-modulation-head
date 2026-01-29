"""
activation_ablator_head.py - 特定のAttentionヘッドをZero Ablationする

Style HeadのZero Ablation実験用：
特定の層の特定のヘッドのO projection前の出力部分をゼロにして、
そのヘッドの寄与を除去する。
"""

from typing import List, Sequence

import torch

from .base import BaseActivationModifier


class ActivationAblatorHead(BaseActivationModifier):
    """特定のAttentionヘッドのO projection前の出力をゼロにする（Zero Ablation）

    Attention内部のattn_weights @ Vの結果（O projection前）に対して、
    特定のヘッドの次元をゼロにすることで、そのヘッドの寄与を除去する。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        layer_idx: int = -1,
        head_indices: List[int] = None,
        positions: str = "all",
        debug: bool = False,
    ):
        """コンストラクタ

        Args:
            model: 対象のモデル
            layer_idx: 対象レイヤーのインデックス（0-based）
            head_indices: アブレーション対象のヘッドインデックスのリスト（0-based）
            positions: 反映位置（"all"|"prompt"|"response"）
            debug: デバッグ出力を有効化
        """
        super().__init__(model, layer_idx=layer_idx, positions=positions, debug=debug)
        self.head_indices = head_indices if head_indices is not None else []

        # Attention設定を取得
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

        # 指定されたヘッドの次元をゼロにするマスクを作成（1=保持、0=ゼロ化）
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
            print(f"[ActivationAblatorHead] Found o_proj: {type(o_proj).__name__}")

        return o_proj

    def _register_hooks(self) -> None:
        """フックを登録"""
        o_proj = self._locate_o_proj()
        self._handle = o_proj.register_forward_pre_hook(self._hook_fn)

    def _hook_fn(self, module, input):
        """pre_hookフック：o_projへの入力に対して指定ヘッドをゼロ化する"""
        mask = self.head_mask

        def _apply_zero_ablation(t):
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
    """複数のヘッドアブレーションを異なるレイヤーに同時適用する"""

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
            instructions: アブレーション指示のリスト
                各dictは以下のキーを持つ:
                - layer_idx: レイヤーインデックス（オプション、デフォルト: -1）
                - head_indices: ヘッドインデックスのリスト（オプション）
                - positions: 反映位置（オプション、デフォルト: "all"）
            debug: デバッグ出力を有効化
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


def create_head_ablation_instructions(
    layer_idx: int,
    head_indices: List[int],
    positions: str = "all",
) -> dict:
    """ヘッドアブレーションの指示を作成するヘルパー関数"""
    return {
        "layer_idx": layer_idx,
        "head_indices": head_indices,
        "positions": positions,
    }


def load_style_heads_from_csv(csv_path: str) -> List[dict]:
    """CSVファイルからStyle Head情報を読み込む

    CSVフォーマット:
    layer,cor_head,anti_head
    20,"3,5,28","1,27"
    ...

    Args:
        csv_path: CSVファイルのパス

    Returns:
        Style Head情報のリスト。各要素は:
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
            # layerは1-indexなので0-indexに変換
            layer_1based = int(row["layer"])
            layer_0based = layer_1based - 1

            # ヘッドインデックスをパース（1-indexなので0-indexに変換）
            cor_heads = []
            if row["cor_head"].strip():
                cor_heads = [int(h.strip()) - 1 for h in row["cor_head"].split(",")]

            anti_heads = []
            if row["anti_head"].strip():
                anti_heads = [int(h.strip()) - 1 for h in row["anti_head"].split(",")]

            style_heads.append({
                "layer": layer_0based,
                "cor_heads": cor_heads,
                "anti_heads": anti_heads,
            })

    return style_heads
