"""
activation_steer.py - Transformerブロック出力へのステアリング

特定のレイヤーの出力または特定の位置（Attention入力、MLP入力など）に
ステアリングベクトルを加算する。
"""

from typing import Sequence, Union

import torch

from .base import BaseActivationSteerer


class ActivationSteerer(BaseActivationSteerer):
    """特定のトランスフォーマーブロック出力に (coeff * steering_vector) を加算する"""

    def _register_hooks(self) -> None:
        """フックを登録"""
        layer = self._get_layer()
        if self.debug:
            print(f"[ActivationSteerer] hooking layer {self.layer_idx}")
        self._handle = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, ins, out):
        """forwardフック：出力テンソルへステアリングを適用"""

        def _process(t):
            return self._apply_steering(t)

        # out may be tensor or tuple/list
        if torch.is_tensor(out):
            new_out = _process(out)
        elif isinstance(out, (tuple, list)):
            if not torch.is_tensor(out[0]):
                return out
            head = _process(out[0])
            new_out = (head, *out[1:])
        else:
            return out

        if self.debug:
            with torch.no_grad():
                delta = (new_out[0] if isinstance(new_out, tuple) else new_out) - (
                    out[0] if isinstance(out, (tuple, list)) else out
                )
                print(
                    "[ActivationSteerer] |delta| (mean ± std): "
                    f"{delta.abs().mean():.4g} ± {delta.std():.4g}"
                )
        return new_out


class ActivationSteererBlock(BaseActivationSteerer):
    """特定のblock入力またはlayer norm入力に (coeff * steering_vector) を加算する

    steering_typeは以下のいずれか:
    - "attn": attention blockへの入力（layer normの出力）
    - "mlp": MLP blockへの入力（layer normの出力）
    - "attn_layernorm": attention前のlayer normへの入力
    - "mlp_layernorm": MLP前のlayer normへの入力
    - "attn_output": attention blockの出力（residual加算前）
    - "mlp_output": MLP blockの出力（residual加算前）
    """

    VALID_STEERING_TYPES = {
        "attn",
        "mlp",
        "attn_layernorm",
        "mlp_layernorm",
        "attn_output",
        "mlp_output",
    }

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        positions: str = "all",
        steering_type: str = "attn",
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
            steering_type: ステアリングタイプ
            renorm_to_original_norm: ステアリング後にノルムを元に戻すか
            debug: デバッグ出力を有効化
        """
        super().__init__(
            model,
            steering_vector,
            coeff=coeff,
            layer_idx=layer_idx,
            positions=positions,
            renorm_to_original_norm=renorm_to_original_norm,
            debug=debug,
        )
        self.steering_type = steering_type.lower()

        if self.steering_type not in self.VALID_STEERING_TYPES:
            raise ValueError(
                f"steering_type must be one of {self.VALID_STEERING_TYPES}"
            )

    def _locate_target_module(self):
        """ステアリングタイプに応じて対象モジュールを特定する

        Returns:
            tuple: (対象モジュール, フックタイプ "input" or "output")
        """
        layer = self._get_layer()

        if self.steering_type == "attn":
            # Attention前のlayer normの出力
            attn_ln = self._find_attn_layernorm(layer)
            if attn_ln:
                return attn_ln, "output"
            # Layer normが見つからない場合、attention blockへの入力として直接フック
            attn_block = self._find_attention_block(layer)
            if attn_block:
                return attn_block, "input"

        elif self.steering_type == "mlp":
            # MLP前のlayer normの出力
            mlp_ln = self._find_mlp_layernorm(layer)
            if mlp_ln:
                return mlp_ln, "output"
            # Layer normが見つからない場合、MLP blockへの入力として直接フック
            mlp_block = self._find_mlp_block(layer)
            if mlp_block:
                return mlp_block, "input"

        elif self.steering_type == "attn_layernorm":
            # Attention前のlayer normへの入力
            attn_ln = self._find_attn_layernorm(layer)
            if attn_ln:
                return attn_ln, "input"

        elif self.steering_type == "mlp_layernorm":
            # MLP前のlayer normへの入力
            mlp_ln = self._find_mlp_layernorm(layer)
            if mlp_ln:
                return mlp_ln, "input"

        elif self.steering_type == "attn_output":
            # Attention blockの出力
            attn_block = self._find_attention_block(layer)
            if attn_block:
                return attn_block, "output"

        elif self.steering_type == "mlp_output":
            # MLP blockの出力
            mlp_block = self._find_mlp_block(layer)
            if mlp_block:
                return mlp_block, "output"

        raise ValueError(
            f"Could not find target module for steering_type={self.steering_type}"
        )

    def _register_hooks(self) -> None:
        """フックを登録"""
        target_module, hook_type = self._locate_target_module()

        if hook_type == "input":
            self._handle = target_module.register_forward_pre_hook(self._pre_hook_fn)
        else:
            self._handle = target_module.register_forward_hook(self._forward_hook_fn)

    def _pre_hook_fn(self, module, ins):
        """pre_hookフック：入力テンソルへステアリングを適用"""
        if isinstance(ins, tuple):
            if torch.is_tensor(ins[0]):
                new_ins = (self._apply_steering(ins[0]), *ins[1:])
                return new_ins
            return ins
        elif torch.is_tensor(ins):
            return self._apply_steering(ins)
        return ins

    def _forward_hook_fn(self, module, ins, out):
        """forwardフック：出力テンソルへステアリングを適用"""
        if torch.is_tensor(out):
            new_out = self._apply_steering(out)
        elif isinstance(out, (tuple, list)):
            if not torch.is_tensor(out[0]):
                return out
            head = self._apply_steering(out[0])
            new_out = (head, *out[1:])
        else:
            return out

        if self.debug:
            with torch.no_grad():
                delta = (new_out[0] if isinstance(new_out, tuple) else new_out) - (
                    out[0] if isinstance(out, (tuple, list)) else out
                )
                print(
                    "[ActivationSteererBlock] |delta| (mean ± std): "
                    f"{delta.abs().mean():.4g} ± {delta.std():.4g}"
                )
        return new_out


class ActivationSteererMultiple:
    """複数のステアリングベクトルを異なるレイヤーに同時適用する"""

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
                - positions: 反映位置（オプション、デフォルト: "all"）
                - renorm_to_original_norm: リノーム有無（オプション）
            debug: デバッグ出力を有効化
        """
        self.model = model
        self.instructions = instructions
        self.debug = debug
        self._steerers = []

        for inst in self.instructions:
            steerer = ActivationSteerer(
                model,
                inst["steering_vector"],
                coeff=inst.get("coeff", 1.0),
                layer_idx=inst.get("layer_idx", -1),
                positions=inst.get("positions", "all"),
                renorm_to_original_norm=inst.get("renorm_to_original_norm", False),
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
