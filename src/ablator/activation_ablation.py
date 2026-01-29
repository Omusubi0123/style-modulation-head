# activation_ablation.py
"""
特定の層のattn/mlp出力からPersona Vector方向の成分を除去（アブレーション）するためのモジュール

Ablationとは：出力ベクトルからPersona Vectorの方向成分を除去すること
ablated_output = output - (output · unit_persona_vector) * unit_persona_vector
"""
from typing import Iterable, Sequence, Union

import torch


class ActivationAblator:
    """特定のblock出力からPersona Vector方向の成分を除去する（アブレーション）

    指定した層のattnまたはmlp出力から、Persona Vectorの方向成分を除去することで、
    そのペルソナ特性の寄与を除去する。
    """

    _POSSIBLE_LAYER_ATTRS: Iterable[str] = (
        "transformer.h",  # GPT‑2/Neo, Bloom, etc.
        "encoder.layer",  # BERT/RoBERTa
        "model.layers",  # Llama/Mistral
        "gpt_neox.layers",  # GPT‑NeoX
        "block",  # Flan‑T5
        "language_model.layers",  # Multimodal Gemma-3 (Gemma3TextModel)
    )

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
            model (torch.nn.Module): 対象のモデル
            persona_vector (Tensor|Sequence[float]): Persona Vector（1次元、この方向成分を除去）
            layer_idx (int): 対象レイヤーのインデックス（0-based、0=1層目、デフォルト: -1）
            ablation_type (str): アブレーションタイプ（"attn_output"|"mlp_output"）
                - "attn_output": attention blockの出力からPersona Vector方向成分を除去
                - "mlp_output": MLP blockの出力からPersona Vector方向成分を除去
            positions (str): 反映位置（"all"|"prompt"|"response"）
            debug (bool): デバッグ出力を有効化（デフォルト: False）
        """
        self.model = model
        self.layer_idx = layer_idx
        self.ablation_type = ablation_type.lower()
        self.positions = positions.lower()
        self.debug = debug
        self._handle = None

        # --- build persona vector ---
        p = next(model.parameters())
        self.persona_vector = torch.as_tensor(
            persona_vector, dtype=p.dtype, device=p.device
        )
        if self.persona_vector.ndim != 1:
            raise ValueError("persona_vector must be 1‑D")
        hidden = getattr(model.config, "hidden_size", None)
        if hidden and self.persona_vector.numel() != hidden:
            raise ValueError(
                f"Vector length {self.persona_vector.numel()} ≠ model hidden_size {hidden}"
            )

        # 単位ベクトルを事前計算
        self.unit_vector = self.persona_vector / (self.persona_vector.norm() + 1e-8)

        # Check if positions is valid
        valid_positions = {"all", "prompt", "response"}
        if self.positions not in valid_positions:
            raise ValueError("positions must be 'all', 'prompt', 'response'")

        # Check if ablation_type is valid
        valid_ablation_types = {"attn_output", "mlp_output"}
        if self.ablation_type not in valid_ablation_types:
            raise ValueError(f"ablation_type must be one of {valid_ablation_types}")

    def _get_max_layer(self):
        """モデルから最大レイヤー数を取得する"""
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
                print(f"Checking text_config for layer attributes...")
            for attr in possible_layer_attrs:
                if hasattr(text_config, attr):
                    max_layer = getattr(text_config, attr)
                    if self.debug:
                        print(f"Using text_config.{attr} = {max_layer}")
                    break

        if max_layer is None:
            raise AttributeError(
                f"Could not find layer count attribute in model config or text_config"
            )

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

    def _locate_target_module(self):
        """アブレーションタイプに応じて対象モジュールを特定する"""
        max_layer = self._get_max_layer()
        if not (-max_layer <= self.layer_idx < max_layer):
            raise IndexError(
                f"layer_idx {self.layer_idx} out of range [{-max_layer}, {max_layer})"
            )

        layer_list = self._locate_layer_list()
        layer = layer_list[self.layer_idx]

        # アブレーションタイプに応じて対象モジュールを探す
        if self.ablation_type == "attn_output":
            # Attention blockの出力
            attn_attrs = ["self_attn", "attention", "attn"]
            for attr in attn_attrs:
                if hasattr(layer, attr):
                    return getattr(layer, attr)
        elif self.ablation_type == "mlp_output":
            # MLP blockの出力
            mlp_attrs = ["mlp", "feed_forward", "ffn"]
            for attr in mlp_attrs:
                if hasattr(layer, attr):
                    return getattr(layer, attr)

        raise ValueError(
            f"Could not find target module for ablation_type={self.ablation_type}"
        )

    def _hook_fn(self, module, ins, out):
        """forwardフック：出力テンソルからPersona Vector方向の成分を除去

        Ablation: output - (output · unit_vector) * unit_vector

        Args:
            module: 対象サブモジュール
            ins: 入力（未使用）
            out: 出力テンソルまたはタプル

        Returns:
            出力と同型のテンソル/タプル（Persona Vector方向成分を除去後）
        """
        unit_vec = self.unit_vector  # (hidden,)

        def _remove_persona_direction(t):
            """テンソルからPersona Vector方向の成分を除去

            ablated = t - (t · unit_vec) * unit_vec
            """
            if self.positions == "all":
                # すべての位置からPersona Vector方向成分を除去
                # t: [batch, seq_len, hidden]
                # projection: (t · unit_vec) -> [batch, seq_len]
                projection = torch.sum(t * unit_vec.to(t.device), dim=-1, keepdim=True)
                # ablated: t - projection * unit_vec
                ablated = t - projection * unit_vec.to(t.device)
                return ablated
            elif self.positions == "prompt":
                # promptのみからPersona Vector方向成分を除去（seq_len > 1の場合）
                if t.shape[1] == 1:
                    return t
                else:
                    projection = torch.sum(
                        t * unit_vec.to(t.device), dim=-1, keepdim=True
                    )
                    ablated = t - projection * unit_vec.to(t.device)
                    return ablated
            elif self.positions == "response":
                # 最後の位置のみからPersona Vector方向成分を除去
                t2 = t.clone()
                last_hidden = t2[:, -1, :]  # [batch, hidden]
                projection = torch.sum(
                    last_hidden * unit_vec.to(t.device), dim=-1, keepdim=True
                )  # [batch, 1]
                t2[:, -1, :] = last_hidden - projection * unit_vec.to(t.device)
                return t2
            else:
                raise ValueError(f"Invalid positions: {self.positions}")

        # outを処理
        if torch.is_tensor(out):
            new_out = _remove_persona_direction(out)
        elif isinstance(out, (tuple, list)):
            if not torch.is_tensor(out[0]):
                return out
            head = _remove_persona_direction(out[0])
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

    # ---------- context manager ----------
    def __enter__(self):
        """コンテキストマネージャ開始時にフックを登録"""
        target_module = self._locate_target_module()
        self._handle = target_module.register_forward_hook(self._hook_fn)
        if self.debug:
            print(
                f"[ActivationAblator] Registered hook at layer {self.layer_idx}, "
                f"type={self.ablation_type}"
            )
        return self

    def __exit__(self, *exc):
        """コンテキストマネージャ終了時にフックを解除"""
        self.remove()

    def remove(self):
        """登録済みのフックを解除"""
        if self._handle:
            self._handle.remove()
            self._handle = None


class ActivationAblatorMultiple:
    """複数のアブレーションを異なるレイヤーに同時適用する

    各指示は辞書形式（persona_vector, layer_idx, ablation_type, positions）で指定。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        instructions: list[dict],
        *,
        debug: bool = False,
    ):
        """コンストラクタ

        Args:
            model (torch.nn.Module): 対象のモデル
            instructions (list[dict]): アブレーション指示のリスト
                各dictは以下のキーを持つ:
                - persona_vector (Tensor): Persona Vector
                - layer_idx (int): 対象レイヤーのインデックス
                - ablation_type (str): "attn_output" または "mlp_output"
                - positions (str, optional): "all", "prompt", "response"
            debug (bool): デバッグ出力を有効化
        """
        self.model = model
        self.instructions = instructions
        self.debug = debug
        self._ablators = []

        # Create individual ablators
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
        """全アブレーターにフック登録し、コンテキストを開始"""
        for ablator in self._ablators:
            ablator.__enter__()
        return self

    def __exit__(self, *exc):
        """コンテキスト終了時にすべてのフックを解除"""
        self.remove()

    def remove(self):
        """すべての登録済みフックを解除"""
        for ablator in self._ablators:
            ablator.remove()
