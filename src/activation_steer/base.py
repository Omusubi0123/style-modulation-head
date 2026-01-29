"""
base.py - Activation modification用の抽象基底クラス

ActivationSteererやActivationAblatorなどの共通機能を提供する。
"""

from abc import ABC, abstractmethod
from typing import Iterable, List, Optional, Sequence, Union

import torch


class BaseActivationModifier(ABC):
    """Activation modification用の抽象基底クラス

    モデルの特定レイヤーに対してフックを登録し、
    activationを変更する機能の共通基盤を提供する。
    """

    # モデルアーキテクチャごとのレイヤーリストのパス
    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "transformer.h",  # GPT-2/Neo, Bloom, etc.
        "encoder.layer",  # BERT/RoBERTa
        "model.layers",  # Llama/Mistral/Qwen
        "gpt_neox.layers",  # GPT-NeoX
        "block",  # Flan-T5
        "language_model.layers",  # Multimodal Gemma-3
    )

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        layer_idx: int = -1,
        positions: str = "all",
        debug: bool = False,
    ):
        """コンストラクタ

        Args:
            model: 対象のモデル
            layer_idx: 対象レイヤーのインデックス（0-based、デフォルト: -1）
            positions: 反映位置（"all"|"prompt"|"response"）
            debug: デバッグ出力を有効化
        """
        self.model = model
        self.layer_idx = layer_idx
        self.positions = positions.lower()
        self.debug = debug
        self._handle = None

        # Check if positions is valid
        valid_positions = {"all", "prompt", "response"}
        if self.positions not in valid_positions:
            raise ValueError(f"positions must be one of {valid_positions}")

    def _get_max_layer(self) -> int:
        """モデルから最大レイヤー数を取得する

        Returns:
            int: 最大レイヤー数

        Raises:
            AttributeError: レイヤー数が見つからない
        """
        possible_layer_attrs = [
            "num_hidden_layers",
            "n_layers",
            "num_layers",
            "n_layer",
        ]
        max_layer = None

        # まずメインのconfigで試す
        for attr in possible_layer_attrs:
            if hasattr(self.model.config, attr):
                max_layer = getattr(self.model.config, attr)
                if self.debug:
                    print(f"Using {attr} = {max_layer}")
                break

        # メインのconfigで見つからない場合、text_configを試す（マルチモーダルモデル用）
        if max_layer is None and hasattr(self.model.config, "text_config"):
            text_config = self.model.config.text_config
            if self.debug:
                print("Checking text_config for layer attributes...")
            for attr in possible_layer_attrs:
                if hasattr(text_config, attr):
                    max_layer = getattr(text_config, attr)
                    if self.debug:
                        print(f"Using text_config.{attr} = {max_layer}")
                    break

        if max_layer is None:
            if self.debug:
                print(
                    "Available attributes in main config:",
                    [
                        attr
                        for attr in dir(self.model.config)
                        if not attr.startswith("_")
                    ],
                )
                if hasattr(self.model.config, "text_config"):
                    print(
                        "Available attributes in text_config:",
                        [
                            attr
                            for attr in dir(self.model.config.text_config)
                            if not attr.startswith("_")
                        ],
                    )
            raise AttributeError(
                "Could not find layer count attribute in model config or text_config"
            )

        return max_layer

    def _locate_layer_list(self) -> torch.nn.ModuleList:
        """モデルからレイヤーリストを特定する

        Returns:
            torch.nn.ModuleList: レイヤーリスト

        Raises:
            ValueError: レイヤーリストが見つからない
        """
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

    def _get_layer(self, layer_idx: Optional[int] = None) -> torch.nn.Module:
        """指定されたインデックスのレイヤーを取得する

        Args:
            layer_idx: レイヤーインデックス（Noneの場合はself.layer_idxを使用）

        Returns:
            torch.nn.Module: 対象レイヤー

        Raises:
            IndexError: レイヤーインデックスが範囲外
        """
        if layer_idx is None:
            layer_idx = self.layer_idx

        max_layer = self._get_max_layer()
        if not (-max_layer <= layer_idx < max_layer):
            raise IndexError(
                f"layer_idx {layer_idx} out of range [{-max_layer}, {max_layer})"
            )

        layer_list = self._locate_layer_list()
        return layer_list[layer_idx]

    def _get_attention_config(self) -> dict:
        """モデルからAttention関連の設定を取得する

        Returns:
            dict: Attention設定
                - num_attention_heads: アテンションヘッド数
                - num_key_value_heads: KVヘッド数（GQA用）
                - hidden_size: 隠れ層サイズ
                - head_dim: ヘッド次元

        Raises:
            AttributeError: 設定が見つからない
        """
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

    def _find_submodule(
        self, layer: torch.nn.Module, attr_names: List[str]
    ) -> Optional[torch.nn.Module]:
        """レイヤーから指定された属性名のサブモジュールを探す

        Args:
            layer: 対象レイヤー
            attr_names: 試す属性名のリスト

        Returns:
            torch.nn.Module: 見つかったサブモジュール、見つからない場合はNone
        """
        for attr in attr_names:
            if hasattr(layer, attr):
                return getattr(layer, attr)
        return None

    def _find_attention_block(
        self, layer: torch.nn.Module
    ) -> Optional[torch.nn.Module]:
        """レイヤーからAttentionブロックを探す"""
        return self._find_submodule(layer, ["self_attn", "attention", "attn"])

    def _find_mlp_block(self, layer: torch.nn.Module) -> Optional[torch.nn.Module]:
        """レイヤーからMLPブロックを探す"""
        return self._find_submodule(layer, ["mlp", "feed_forward", "ffn"])

    def _find_attn_layernorm(
        self, layer: torch.nn.Module
    ) -> Optional[torch.nn.Module]:
        """レイヤーからAttention前のLayerNormを探す"""
        return self._find_submodule(
            layer,
            ["input_layernorm", "ln_1", "layer_norm", "pre_attention_layernorm"],
        )

    def _find_mlp_layernorm(self, layer: torch.nn.Module) -> Optional[torch.nn.Module]:
        """レイヤーからMLP前のLayerNormを探す"""
        return self._find_submodule(
            layer, ["post_attention_layernorm", "ln_2", "mlp_layernorm"]
        )

    def _find_o_proj(self, attn_block: torch.nn.Module) -> Optional[torch.nn.Module]:
        """Attentionブロックからo_projを探す"""
        return self._find_submodule(attn_block, ["o_proj", "out_proj", "dense"])

    @abstractmethod
    def _register_hooks(self) -> None:
        """フックを登録する（サブクラスで実装）"""
        pass

    def __enter__(self):
        """コンテキストマネージャ開始時にフックを登録"""
        self._register_hooks()
        return self

    def __exit__(self, *exc):
        """コンテキストマネージャ終了時にフックを解除"""
        self.remove()

    def remove(self):
        """登録済みのフックを解除"""
        if self._handle:
            self._handle.remove()
            self._handle = None


class BaseActivationSteerer(BaseActivationModifier):
    """Steeringベクトルを加算するための基底クラス"""

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
        """コンストラクタ

        Args:
            model: 対象のモデル
            steering_vector: ステアリングベクトル（1次元）
            coeff: 係数（デフォルト: 1.0）
            layer_idx: 対象レイヤーのインデックス（0-based、デフォルト: -1）
            positions: 反映位置（"all"|"prompt"|"response"）
            renorm_to_original_norm: ステアリング後にノルムを元に戻すか
            debug: デバッグ出力を有効化
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
        """テンソルにステアリングを適用する

        Args:
            t: 入力テンソル [batch, seq_len, hidden]

        Returns:
            ステアリング適用後のテンソル
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

