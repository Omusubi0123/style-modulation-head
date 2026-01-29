# activation_steering.py  – v0.2
from typing import Iterable, Sequence, Union

import torch


class ActivationSteerer:
    """特定のトランスフォーマーブロック出力に (coeff * steering_vector) を加算する

    タプルを返すブロックにも対応し、レイヤーリストが見つからない場合は明示的にエラーにする。
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
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        positions: str = "all",
        renorm_to_original_norm: bool = False,
        debug: bool = False,
        track_projections: bool = False,
    ):
        """コンストラクタ

        Args:
            model (torch.nn.Module): 対象のモデル
            steering_vector (Tensor|Sequence[float]): ステアリングベクトル（1次元）
            coeff (float): 係数（デフォルト: 1.0）
            layer_idx (int): 対象レイヤーのインデックス（0-based、0=1層目、デフォルト: -1）
            positions (str): 反映位置（"all"|"prompt"|"response"）
            renorm_to_original_norm (bool): 投影後のノルムを元のノルムにリノームするか（デフォルト: False）
            debug (bool): デバッグ出力を有効化（デフォルト: False）
            track_projections (bool): 投影値を追跡するか（デフォルト: False）
        """
        self.model, self.coeff, self.layer_idx = model, float(coeff), layer_idx
        self.positions = positions.lower()
        self.debug = debug
        self.track_projections = track_projections
        self.renorm_to_original_norm = renorm_to_original_norm
        self._handle = None
        self.projections = []  # Store projection values for each token

        # --- build vector ---
        p = next(model.parameters())
        self.vector = torch.as_tensor(steering_vector, dtype=p.dtype, device=p.device)
        if self.vector.ndim != 1:
            raise ValueError("steering_vector must be 1‑D")
        hidden = getattr(model.config, "hidden_size", None)
        if hidden and self.vector.numel() != hidden:
            raise ValueError(
                f"Vector length {self.vector.numel()} ≠ model hidden_size {hidden}"
            )
        # Check if positions is valid
        valid_positions = {"all", "prompt", "response"}
        if self.positions not in valid_positions:
            raise ValueError("positions must be 'all', 'prompt', 'response'")

    # ---------- helpers ----------
    def _locate_layer(self):
        """モデルから対象レイヤーを特定し返す

        Returns:
            torch.nn.Module: 対象レイヤーのモジュール

        Raises:
            IndexError: レイヤーインデックスが範囲外
            ValueError: レイヤーリストが見つからない
        """
        # まず、レイヤー数を確認してインデックスの妥当性をチェック
        max_layer = self._get_max_layer()
        if not (-max_layer <= self.layer_idx < max_layer):
            raise IndexError(
                f"layer_idx {self.layer_idx} out of range [{-max_layer}, {max_layer})"
            )

        # デバッグ情報を出力
        print(
            f"[ActivationSteerer] Looking for layer {self.layer_idx} in model with {max_layer} layers"
        )
        print(f"[ActivationSteerer] Model type: {type(self.model).__name__}")

        for path in self._POSSIBLE_LAYER_ATTRS:
            cur = self.model
            path_parts = path.split(".")
            found = True

            for i, part in enumerate(path_parts):
                if hasattr(cur, part):
                    cur = getattr(cur, part)
                else:
                    found = False
                    break

            if found:  # found a full match
                if not hasattr(cur, "__getitem__"):
                    continue  # not a list/ModuleList

                print(
                    f"[ActivationSteerer] Found layer list at {path} with {len(cur)} layers"
                )
                if self.debug:
                    print(f"[ActivationSteerer] hooking {path}[{self.layer_idx}]")
                return cur[self.layer_idx]

        # 利用可能な属性を詳しく調べる（デバッグ用）
        print(f"[ActivationSteerer] Available model attributes:")
        for attr in dir(self.model):
            if not attr.startswith("_"):
                try:
                    obj = getattr(self.model, attr)
                    if hasattr(obj, "__getitem__") and hasattr(obj, "__len__"):
                        print(
                            f"  {attr}: {type(obj).__name__} (indexable with {len(obj)} items)"
                        )
                except:
                    pass

        raise ValueError(
            "Could not find layer list on the model. "
            "Add the attribute name to _POSSIBLE_LAYER_ATTRS."
        )

    def _get_max_layer(self):
        """モデルから最大レイヤー数を取得する

        Returns:
            int: 最大レイヤー数

        Raises:
            AttributeError: レイヤー数が見つからない
        """
        # 異なるモデルでレイヤー数の属性名が異なる可能性があるため、複数の属性名を試す
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
                f"Could not find layer count attribute in model config or text_config"
            )

        return max_layer

    def _hook_fn(self, module, ins, out):
        """forwardフック：出力テンソルへステアリングを適用

        Args:
            module: 対象サブモジュール
            ins: 入力（未使用）
            out: 出力テンソルまたはタプル

        Returns:
            出力と同型のテンソル/タプル（ステアリング適用後）
        """
        steer = self.coeff * self.vector  # (hidden,)

        def _add_and_project(t):
            # First compute projection if tracking is enabled
            if self.track_projections:
                with torch.no_grad():
                    # Normalize vector for projection computation
                    norm_vector = self.vector / (self.vector.norm() + 1e-8)

                    if self.positions == "all":
                        # Project all positions
                        projections = torch.sum(
                            t * norm_vector.to(t.device), dim=-1
                        )  # [batch, seq_len]
                        self.projections.extend(projections.cpu().flatten().tolist())
                    elif self.positions == "prompt":
                        if t.shape[1] > 1:  # Only during prompt processing
                            projections = torch.sum(
                                t * norm_vector.to(t.device), dim=-1
                            )  # [batch, seq_len]
                            self.projections.extend(
                                projections.cpu().flatten().tolist()
                            )
                    elif self.positions == "response":
                        # Project only the last position (current token being generated)
                        last_hidden = t[:, -1, :]  # [batch, hidden]
                        projection = torch.sum(
                            last_hidden * norm_vector.to(t.device), dim=-1
                        )  # [batch]
                        self.projections.extend(projection.cpu().tolist())

            # Then apply steering
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
                else:
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

        # out may be tensor or tuple/list => normalise to tuple
        if torch.is_tensor(out):
            new_out = _add_and_project(out)
        elif isinstance(out, (tuple, list)):
            if not torch.is_tensor(out[0]):
                # unusual case – don't touch
                return out
            head = _add_and_project(out[0])
            new_out = (head, *out[1:])  # keep other entries
        else:
            return out  # unknown type – leave unchanged

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

    # ---------- context manager ----------
    def __enter__(self):
        """コンテキストマネージャ開始時にフックを登録"""
        layer = self._locate_layer()
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def __exit__(self, *exc):
        """コンテキストマネージャ終了時にフックを解除"""
        self.remove()  # always clean up

    def remove(self):
        """登録済みのフックを解除"""
        if self._handle:
            self._handle.remove()
            self._handle = None

    def get_projections(self):
        """投影値のリストを取得"""
        return self.projections.copy()

    def clear_projections(self):
        """投影値のリストをクリア"""
        self.projections.clear()


class ActivationSteererMultiple:
    """複数のステアリングベクトルを異なるレイヤーに同時適用する

    各指示は辞書形式（steering_vector, coeff, layer_idx, positions）で指定。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        instructions: Sequence[dict],
        *,
        debug: bool = False,
    ):
        """コンストラクタ

        Args:
            model (torch.nn.Module): 対象のモデル
            instructions (Sequence[dict]): ステアリング指示のリスト
            debug (bool): デバッグ出力を有効化
        """
        self.model = model
        self.instructions = instructions
        self.debug = debug
        self._handles = []
        self._steerers = []

        # Validate and create individual steerers
        for inst in self.instructions:
            steerer = ActivationSteerer(
                model,
                inst["steering_vector"],
                coeff=inst.get("coeff", 0.0),
                layer_idx=inst.get("layer_idx", -1),
                positions=inst.get("positions", "all"),
                renorm_to_original_norm=inst.get("renorm_to_original_norm", False),
                debug=debug,
            )
            self._steerers.append(steerer)

    def __enter__(self):
        """全ステアラーにフック登録し、コンテキストを開始"""
        for steerer in self._steerers:
            layer = steerer._locate_layer()
            handle = layer.register_forward_hook(steerer._hook_fn)
            steerer._handle = handle
            self._handles.append(handle)
        return self

    def __exit__(self, *exc):
        """コンテキスト終了時にすべてのフックを解除"""
        self.remove()

    def remove(self):
        """すべての登録済みフックを解除"""
        for steerer in self._steerers:
            steerer.remove()
        self._handles.clear()


class ActivationSteererBlock:
    """特定のblock入力またはlayer norm入力に (coeff * steering_vector) を加算する

    block入力やlayer norm入力にsteeringを適用するためのクラス。
    steering_typeは以下のいずれか:
    - "attn_input": attention blockへの入力（layer normの出力）
    - "mlp_input": MLP blockへの入力（layer normの出力）
    - "attn_layernorm_input": attention前のlayer normへの入力
    - "mlp_layernorm_input": MLP前のlayer normへの入力
    - "attn_output": attention blockの出力（residual加算前）
    - "mlp_output": MLP blockの出力（residual加算前）
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
        steering_vector: Union[torch.Tensor, Sequence[float]],
        *,
        coeff: float = 1.0,
        layer_idx: int = -1,
        positions: str = "all",
        steering_type: str = "attn_input",
        renorm_to_original_norm: bool = False,
        debug: bool = False,
        track_projections: bool = False,
    ):
        """コンストラクタ

        Args:
            model (torch.nn.Module): 対象のモデル
            steering_vector (Tensor|Sequence[float]): ステアリングベクトル（1次元）
            coeff (float): 係数（デフォルト: 1.0）
            layer_idx (int): 対象レイヤーのインデックス（0-based、0=1層目、デフォルト: -1）
            positions (str): 反映位置（"all"|"prompt"|"response"）
            steering_type (str): ステアリングタイプ（"attn_input"|"mlp_input"|"attn_layernorm_input"|"mlp_layernorm_input"|"attn_output"|"mlp_output"）
            renorm_to_original_norm (bool): 投影後のノルムを元のノルムにリノームするか（デフォルト: False）
            debug (bool): デバッグ出力を有効化（デフォルト: False）
            track_projections (bool): 投影値を追跡するか（デフォルト: False）
        """
        self.model, self.coeff, self.layer_idx = model, float(coeff), layer_idx
        self.positions = positions.lower()
        self.steering_type = steering_type.lower()
        self.debug = debug
        self.track_projections = track_projections
        self.renorm_to_original_norm = renorm_to_original_norm
        self._handle = None
        self.projections = []  # Store projection values for each token

        # --- build vector ---
        p = next(model.parameters())
        self.vector = torch.as_tensor(steering_vector, dtype=p.dtype, device=p.device)
        if self.vector.ndim != 1:
            raise ValueError("steering_vector must be 1‑D")
        hidden = getattr(model.config, "hidden_size", None)
        if hidden and self.vector.numel() != hidden:
            raise ValueError(
                f"Vector length {self.vector.numel()} ≠ model hidden_size {hidden}"
            )
        # Check if positions is valid
        valid_positions = {"all", "prompt", "response"}
        if self.positions not in valid_positions:
            raise ValueError("positions must be 'all', 'prompt', 'response'")
        # Check if steering_type is valid
        valid_steering_types = {
            "attn",
            "mlp",
            "attn_layernorm",
            "mlp_layernorm",
            "attn_output",
            "mlp_output",
        }
        if self.steering_type not in valid_steering_types:
            raise ValueError(f"steering_type must be one of {valid_steering_types}")

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
        """ステアリングタイプに応じて対象モジュールを特定する"""
        max_layer = self._get_max_layer()
        if not (-max_layer <= self.layer_idx < max_layer):
            raise IndexError(
                f"layer_idx {self.layer_idx} out of range [{-max_layer}, {max_layer})"
            )

        layer_list = self._locate_layer_list()
        layer = layer_list[self.layer_idx]

        # ステアリングタイプに応じて対象モジュールを探す
        if self.steering_type == "attn":
            # Attention前のlayer normの出力（attention blockへの入力）
            attn_ln_attrs = [
                "input_layernorm",
                "ln_1",
                "layer_norm",
                "pre_attention_layernorm",
            ]
            for attr in attn_ln_attrs:
                if hasattr(layer, attr):
                    return getattr(layer, attr), "output"
            # Layer normが見つからない場合、attention blockへの入力として直接フック
            attn_attrs = ["self_attn", "attention", "attn"]
            for attr in attn_attrs:
                if hasattr(layer, attr):
                    return getattr(layer, attr), "input"
        elif self.steering_type == "mlp":
            # MLP前のlayer normの出力（MLP blockへの入力）
            mlp_ln_attrs = ["post_attention_layernorm", "ln_2", "mlp_layernorm"]
            for attr in mlp_ln_attrs:
                if hasattr(layer, attr):
                    return getattr(layer, attr), "output"
            # Layer normが見つからない場合、MLP blockへの入力として直接フック
            mlp_attrs = ["mlp", "feed_forward", "ffn"]
            for attr in mlp_attrs:
                if hasattr(layer, attr):
                    return getattr(layer, attr), "input"
        elif self.steering_type == "attn_layernorm":
            # Attention前のlayer normへの入力
            attn_ln_attrs = [
                "input_layernorm",
                "ln_1",
                "layer_norm",
                "pre_attention_layernorm",
            ]
            for attr in attn_ln_attrs:
                if hasattr(layer, attr):
                    return getattr(layer, attr), "input"
        elif self.steering_type == "mlp_layernorm":
            # MLP前のlayer normへの入力
            mlp_ln_attrs = ["post_attention_layernorm", "ln_2", "mlp_layernorm"]
            for attr in mlp_ln_attrs:
                if hasattr(layer, attr):
                    return getattr(layer, attr), "input"
        elif self.steering_type == "attn_output":
            # Attention blockの出力（residual加算前）
            attn_attrs = ["self_attn", "attention", "attn"]
            for attr in attn_attrs:
                if hasattr(layer, attr):
                    return getattr(layer, attr), "output"
        elif self.steering_type == "mlp_output":
            # MLP blockの出力（residual加算前）
            mlp_attrs = ["mlp", "feed_forward", "ffn"]
            for attr in mlp_attrs:
                if hasattr(layer, attr):
                    return getattr(layer, attr), "output"

        raise ValueError(
            f"Could not find target module for steering_type={self.steering_type}"
        )

    def _hook_fn(self, module, ins, out):
        """forwardフック：入力または出力テンソルへステアリングを適用

        Args:
            module: 対象サブモジュール
            ins: 入力（pre_hookの場合）
            out: 出力（forward_hookの場合）

        Returns:
            出力と同型のテンソル/タプル（ステアリング適用後）
        """
        steer = self.coeff * self.vector  # (hidden,)

        def _add_and_project(t):
            # First compute projection if tracking is enabled
            if self.track_projections:
                with torch.no_grad():
                    # Normalize vector for projection computation
                    norm_vector = self.vector / (self.vector.norm() + 1e-8)

                    if self.positions == "all":
                        # Project all positions
                        projections = torch.sum(
                            t * norm_vector.to(t.device), dim=-1
                        )  # [batch, seq_len]
                        self.projections.extend(projections.cpu().flatten().tolist())
                    elif self.positions == "prompt":
                        if t.shape[1] > 1:  # Only during prompt processing
                            projections = torch.sum(
                                t * norm_vector.to(t.device), dim=-1
                            )  # [batch, seq_len]
                            self.projections.extend(
                                projections.cpu().flatten().tolist()
                            )
                    elif self.positions == "response":
                        # Project only the last position (current token being generated)
                        last_hidden = t[:, -1, :]  # [batch, hidden]
                        projection = torch.sum(
                            last_hidden * norm_vector.to(t.device), dim=-1
                        )  # [batch]
                        self.projections.extend(projection.cpu().tolist())

            # Then apply steering
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
                else:
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

        # insまたはoutを処理
        if ins is not None:
            # pre_hookの場合（入力にsteeringを適用）
            if isinstance(ins, tuple):
                if torch.is_tensor(ins[0]):
                    new_ins = (_add_and_project(ins[0]), *ins[1:])
                    return new_ins
                return ins
            elif torch.is_tensor(ins):
                return _add_and_project(ins)
            return ins
        else:
            # forward_hookの場合（出力にsteeringを適用）
            if torch.is_tensor(out):
                new_out = _add_and_project(out)
            elif isinstance(out, (tuple, list)):
                if not torch.is_tensor(out[0]):
                    return out
                head = _add_and_project(out[0])
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

    # ---------- context manager ----------
    def __enter__(self):
        """コンテキストマネージャ開始時にフックを登録"""
        target_module, hook_type = self._locate_target_module()
        if hook_type == "input":
            # pre_hookを登録（入力にsteeringを適用）
            self._handle = target_module.register_forward_pre_hook(
                lambda module, ins: self._hook_fn(module, ins, None)
            )
        else:
            # forward_hookを登録（出力にsteeringを適用）
            self._handle = target_module.register_forward_hook(
                lambda module, ins, out: self._hook_fn(module, None, out)
            )
        return self

    def __exit__(self, *exc):
        """コンテキストマネージャ終了時にフックを解除"""
        self.remove()  # always clean up

    def remove(self):
        """登録済みのフックを解除"""
        if self._handle:
            self._handle.remove()
            self._handle = None

    def get_projections(self):
        """投影値のリストを取得"""
        return self.projections.copy()

    def clear_projections(self):
        """投影値のリストをクリア"""
        self.projections.clear()
