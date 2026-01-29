"""
activation_steer_head.py - 特定のAttentionヘッドのみをsteeringする

実験設定5のためのスクリプト:
特定の層の特定のヘッドのOV計算前の出力部分のみを増大させてsteering

使い方:
1. O projection前の状態で、指定されたヘッドに対応する次元のみにsteeringベクトルを加算
2. 他のヘッドの次元は変更しない
3. 単一ヘッドまたは複数ヘッドを指定可能
"""

from typing import Iterable, List, Sequence, Union

import torch


class ActivationSteererHead:
    """特定のAttentionヘッドのO projection前の出力をsteeringする

    Attention内部のattn_weights @ Vの結果（O projection前）に対して、
    特定のヘッドの次元のみにsteeringベクトルを加算する。

    Args:
        model: 対象のモデル
        steering_vector: ステアリングベクトル（1次元、hidden_size）
        coeff: 係数（デフォルト: 1.0）
        layer_idx: 対象レイヤーのインデックス（0-based）
        head_indices: ステアリング対象のヘッドインデックスのリスト
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
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        head_indices: List[int] = None,
        positions: str = "all",
        debug: bool = False,
    ):
        self.model = model
        self.coeff = float(coeff)
        self.layer_idx = layer_idx
        self.head_indices = head_indices if head_indices is not None else []
        self.positions = positions.lower()
        self.debug = debug
        self._handle = None

        # Attention設定を取得
        self.attn_config = self._get_attention_config()
        self.num_heads = self.attn_config["num_attention_heads"]
        self.head_dim = self.attn_config["head_dim"]

        # Validate head indices
        for h_idx in self.head_indices:
            if h_idx < 0 or h_idx >= self.num_heads:
                raise ValueError(
                    f"head_index {h_idx} out of range [0, {self.num_heads})"
                )

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

        # Check if positions is valid
        valid_positions = {"all", "prompt", "response"}
        if self.positions not in valid_positions:
            raise ValueError("positions must be 'all', 'prompt', 'response'")

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
                    print(f"[ActivationSteererHead] Found o_proj at {attr}")
                break

        if o_proj is None:
            raise ValueError(f"Could not find o_proj for layer {self.layer_idx}")

        return o_proj

    def _hook_fn(self, module, input):
        """pre_hookフック：o_projへの入力に対して指定ヘッドのみsteeringを適用"""
        steer = self.coeff * self.masked_vector

        def _add_steering(t):
            if self.positions == "all":
                result = t + steer.to(t.device)
                return result
            elif self.positions == "prompt":
                if t.shape[1] == 1:
                    return t
                else:
                    t2 = t.clone()
                    t2 += steer.to(t.device)
                    return t2
            elif self.positions == "response":
                t2 = t.clone()
                t2[:, -1, :] += steer.to(t.device)
                return t2
            else:
                raise ValueError(f"Invalid positions: {self.positions}")

        # inputはタプルの場合があるので処理
        if isinstance(input, tuple):
            if len(input) > 0 and torch.is_tensor(input[0]):
                new_input = (_add_steering(input[0]), *input[1:])
                return new_input
            return input
        elif torch.is_tensor(input):
            return _add_steering(input)
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


class ActivationSteererHeadMultiple:
    """複数のヘッドsteeringを異なるレイヤーに同時適用する

    各指示は辞書形式で指定:
    {
        "steering_vector": tensor,
        "coeff": float,
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
        """全ステアラーにフック登録し、コンテキストを開始"""
        for steerer in self._steerers:
            o_proj = steerer._locate_o_proj()
            steerer._handle = o_proj.register_forward_pre_hook(steerer._hook_fn)
        return self

    def __exit__(self, *exc):
        """コンテキストマネージャ終了時にすべてのフックを解除"""
        self.remove()

    def remove(self):
        """すべての登録済みフックを解除"""
        for steerer in self._steerers:
            steerer.remove()


# 便利関数
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
