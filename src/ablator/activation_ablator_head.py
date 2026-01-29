"""
activation_ablator_head.py - 特定のAttentionヘッドをZero Ablationする

Style HeadのZero Ablation実験用：
特定の層の特定のヘッドのO projection前の出力部分をゼロにして、
そのヘッドの寄与を除去する。

使い方:
1. O projection前の状態で、指定されたヘッドに対応する次元をゼロにする
2. 他のヘッドの次元は変更しない
3. 単一ヘッドまたは複数ヘッドを指定可能
4. 複数層にわたって複数ヘッドをアブレーション可能
"""

from typing import Iterable, List, Sequence

import torch


class ActivationAblatorHead:
    """特定のAttentionヘッドのO projection前の出力をゼロにする（Zero Ablation）

    Attention内部のattn_weights @ Vの結果（O projection前）に対して、
    特定のヘッドの次元をゼロにすることで、そのヘッドの寄与を除去する。

    Args:
        model: 対象のモデル
        layer_idx: 対象レイヤーのインデックス（0-based）
        head_indices: アブレーション対象のヘッドインデックスのリスト（0-based）
        positions: 反映位置（"all"|"prompt"|"response"）
        debug: デバッグ出力を有効化
    """

    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "transformer.h",
        "encoder.layer",
        "model.layers",
        "gpt_neox.layers",
        "block",
        "language_model.layers",
    )

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        layer_idx: int = -1,
        head_indices: List[int] = None,
        positions: str = "all",
        debug: bool = False,
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.head_indices = head_indices if head_indices is not None else []
        self.positions = positions.lower()
        self.debug = debug
        self._handle = None

        # Attention設定を取得
        self.attn_config = self._get_attention_config()
        self.num_heads = self.attn_config["num_attention_heads"]
        self.head_dim = self.attn_config["head_dim"]
        self.hidden_size = self.attn_config["hidden_size"]

        # Validate head indices
        for h_idx in self.head_indices:
            if h_idx < 0 or h_idx >= self.num_heads:
                raise ValueError(
                    f"head_index {h_idx} out of range [0, {self.num_heads})"
                )

        # Check if positions is valid
        valid_positions = {"all", "prompt", "response"}
        if self.positions not in valid_positions:
            raise ValueError("positions must be 'all', 'prompt', 'response'")

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

    def _get_attention_config(self):
        """モデルからAttention関連の設定を取得する"""
        cfg = self.model.config
        if hasattr(cfg, "text_config"):
            cfg = cfg.text_config

        num_attention_heads = getattr(cfg, "num_attention_heads", None)
        num_key_value_heads = getattr(cfg, "num_key_value_heads", num_attention_heads)
        hidden_size = getattr(cfg, "hidden_size", None)

        if num_attention_heads is None or hidden_size is None:
            raise AttributeError("Could not find attention config in model")

        head_dim = hidden_size // num_attention_heads

        return {
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "hidden_size": hidden_size,
            "head_dim": head_dim,
        }

    def _get_max_layer(self):
        """モデルから最大レイヤー数を取得する"""
        possible_layer_attrs = [
            "num_hidden_layers",
            "n_layers",
            "num_layers",
            "n_layer",
        ]
        max_layer = None

        for attr in possible_layer_attrs:
            if hasattr(self.model.config, attr):
                max_layer = getattr(self.model.config, attr)
                break

        if max_layer is None and hasattr(self.model.config, "text_config"):
            text_config = self.model.config.text_config
            for attr in possible_layer_attrs:
                if hasattr(text_config, attr):
                    max_layer = getattr(text_config, attr)
                    break

        if max_layer is None:
            raise AttributeError("Could not find layer count attribute in model config")

        return max_layer

    def _locate_layer_list(self):
        """モデルからレイヤーリストを特定する"""
        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            path_parts = path.split(".")
            found = True

            for part in path_parts:
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    found = False
                    break

            if found and hasattr(cur, "__getitem__"):
                return cur

        raise ValueError("Could not find layer list in model")

    def _locate_o_proj(self):
        """対象レイヤーのo_projモジュールを特定する"""
        max_layer = self._get_max_layer()
        if not (-max_layer <= self.layer_idx < max_layer):
            raise IndexError(
                f"layer_idx {self.layer_idx} out of range [{-max_layer}, {max_layer})"
            )

        layer_list = self._locate_layer_list()
        layer = layer_list[self.layer_idx]

        # Attention blockを探す
        attn_attrs = ["self_attn", "attention", "attn"]
        attn_block = None
        for attr in attn_attrs:
            if hasattr(layer, attr):
                attn_block = getattr(layer, attr)
                break

        if attn_block is None:
            raise ValueError(
                f"Could not find attention block for layer {self.layer_idx}"
            )

        # o_projを探す
        o_proj_attrs = ["o_proj", "out_proj", "dense"]
        o_proj = None
        for attr in o_proj_attrs:
            if hasattr(attn_block, attr):
                o_proj = getattr(attn_block, attr)
                if self.debug:
                    print(f"[ActivationAblatorHead] Found o_proj at {attr}")
                break

        if o_proj is None:
            raise ValueError(f"Could not find o_proj for layer {self.layer_idx}")

        return o_proj

    def _hook_fn(self, module, input):
        """pre_hookフック：o_projへの入力に対して指定ヘッドをゼロ化する"""
        mask = self.head_mask

        def _apply_zero_ablation(t):
            if self.positions == "all":
                # すべての位置でマスクを適用
                result = t * mask.to(t.device)
                return result
            elif self.positions == "prompt":
                # promptのみでマスクを適用（seq_len > 1の場合）
                if t.shape[1] == 1:
                    return t
                else:
                    result = t * mask.to(t.device)
                    return result
            elif self.positions == "response":
                # 最後の位置のみでマスクを適用
                t2 = t.clone()
                t2[:, -1, :] = t2[:, -1, :] * mask.to(t.device)
                return t2
            else:
                raise ValueError(f"Invalid positions: {self.positions}")

        # inputはタプルの場合があるので処理
        if isinstance(input, tuple):
            if len(input) > 0 and torch.is_tensor(input[0]):
                new_input = (_apply_zero_ablation(input[0]), *input[1:])
                return new_input
            return input
        elif torch.is_tensor(input):
            return _apply_zero_ablation(input)
        return input

    def __enter__(self):
        """コンテキストマネージャ開始時にフックを登録"""
        o_proj = self._locate_o_proj()
        self._handle = o_proj.register_forward_pre_hook(self._hook_fn)
        return self

    def __exit__(self, *exc):
        """コンテキストマネージャ終了時にフックを解除"""
        self.remove()

    def remove(self):
        """登録済みのフックを解除"""
        if self._handle:
            self._handle.remove()
            self._handle = None


class ActivationAblatorHeadMultiple:
    """複数のヘッドアブレーションを異なるレイヤーに同時適用する

    各指示は辞書形式で指定:
    {
        "layer_idx": int,
        "head_indices": List[int],
        "positions": str,
    }
    """

    def __init__(
        self,
        model: torch.nn.Module,
        instructions: Sequence[dict],
        *,
        debug: bool = False,
    ):
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
        """全アブレーターにフック登録し、コンテキストを開始"""
        for ablator in self._ablators:
            o_proj = ablator._locate_o_proj()
            ablator._handle = o_proj.register_forward_pre_hook(ablator._hook_fn)
        return self

    def __exit__(self, *exc):
        """コンテキストマネージャ終了時にすべてのフックを解除"""
        self.remove()

    def remove(self):
        """すべての登録済みフックを解除"""
        for ablator in self._ablators:
            ablator.remove()


# 便利関数
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

