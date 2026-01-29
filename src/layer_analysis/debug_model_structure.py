"""
モデル構造とフック登録順序のデバッグスクリプト

このスクリプトは：
1. モデルの構造を詳細に出力
2. 各レイヤーのサブモジュールを確認
3. フックの登録と呼び出し順序を検証

実際の推論やベクトル抽出は行いません。
"""

import argparse
import os
import sys

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from src.config import setup_credentials

# Set up credentials
config = setup_credentials()


def _locate_layer_list(model):
    """モデルからレイヤーリストを特定する（generate_vec_block.pyと同じ）"""
    possible_attrs = [
        "transformer.h",  # GPT-2/Neo, Bloom, etc.
        "encoder.layer",  # BERT/RoBERTa
        "model.layers",  # Llama/Mistral/Qwen
        "gpt_neox.layers",  # GPT-NeoX
        "block",  # Flan-T5
        "language_model.layers",  # Multimodal Gemma-3
    ]

    for attr_path in possible_attrs:
        parts = attr_path.split(".")
        cur = model
        found = True

        for part in parts:
            if hasattr(cur, part):
                cur = getattr(cur, part)
            else:
                found = False
                break

        if found and hasattr(cur, "__getitem__"):
            print(f"✓ Found layer list at: {attr_path}")
            return cur, attr_path

    raise ValueError("Could not find layer list in model")


def _get_max_layer(model):
    """モデルから最大レイヤー数を取得する（generate_vec_block.pyと同じ）"""
    possible_layer_attrs = ["num_hidden_layers", "n_layers", "num_layers", "n_layer"]
    max_layer = None

    for attr in possible_layer_attrs:
        if hasattr(model.config, attr):
            max_layer = getattr(model.config, attr)
            print(f"✓ Found layer count at config.{attr}: {max_layer}")
            return max_layer

    if max_layer is None and hasattr(model.config, "text_config"):
        text_config = model.config.text_config
        for attr in possible_layer_attrs:
            if hasattr(text_config, attr):
                max_layer = getattr(text_config, attr)
                print(f"✓ Found layer count at config.text_config.{attr}: {max_layer}")
                return max_layer

    if max_layer is None:
        raise AttributeError("Could not find layer count attribute in model config")

    return max_layer


def inspect_model_structure(model_name: str, inspect_layers: list[int] = None):
    """モデル構造を詳細に検査する

    Args:
        model_name: モデル名
        inspect_layers: 詳細に検査するレイヤー番号のリスト（Noneの場合は[0, 中間, 最後]）
    """
    print("=" * 80)
    print(f"モデル構造デバッグ: {model_name}")
    print("=" * 80)

    # モデルのロード
    print("\n[1] モデルをロード中...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cpu",  # CPUで十分
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=config.hf_token,
        )
        print(f"✓ AutoModelForCausalLM でロード成功")
    except ValueError as e:
        if "Unrecognized configuration class" in str(e):
            print(f"  AutoModelForCausalLM 失敗、AutoModel で再試行...")
            model = AutoModel.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=config.hf_token,
            )
            print(f"✓ AutoModel でロード成功")
        else:
            raise

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=config.hf_token
    )

    # モデルの基本情報
    print(f"\n[2] モデルの基本情報")
    print(f"  モデルクラス: {type(model).__name__}")
    print(f"  Config: {type(model.config).__name__}")

    # レイヤーリストの特定
    print(f"\n[3] レイヤーリストの特定")
    layer_list_module, layer_path = _locate_layer_list(model)
    max_layer = _get_max_layer(model)
    print(f"  総レイヤー数: {max_layer}")
    print(f"  レイヤーリスト型: {type(layer_list_module)}")

    # 検査するレイヤーを決定
    if inspect_layers is None:
        middle = max_layer // 2
        inspect_layers = [0, middle, max_layer - 1]

    print(f"\n[4] レイヤー詳細構造の検査（レイヤー {inspect_layers}）")

    for layer_idx in inspect_layers:
        print(f"\n{'=' * 60}")
        print(f"  レイヤー {layer_idx}")
        print(f"{'=' * 60}")

        layer = layer_list_module[layer_idx]
        print(f"  型: {type(layer).__name__}")
        print(f"  モジュール階層: {layer_path}[{layer_idx}]")

        # サブモジュールを探す
        print(f"\n  サブモジュール:")

        # Attention前のlayer norm
        attn_ln_attrs = [
            "input_layernorm",
            "ln_1",
            "layer_norm",
            "pre_attention_layernorm",
        ]
        attn_ln = None
        attn_ln_name = None
        for attr in attn_ln_attrs:
            if hasattr(layer, attr):
                attn_ln = getattr(layer, attr)
                attn_ln_name = attr
                print(
                    f"    ✓ Attention前LayerNorm: '{attr}' → {type(attn_ln).__name__}"
                )
                break
        if attn_ln is None:
            print(f"    ✗ Attention前LayerNorm: 見つかりません")

        # Attention block
        attn_attrs = ["self_attn", "attention", "attn"]
        attn_block = None
        attn_block_name = None
        for attr in attn_attrs:
            if hasattr(layer, attr):
                attn_block = getattr(layer, attr)
                attn_block_name = attr
                print(
                    f"    ✓ Attentionブロック: '{attr}' → {type(attn_block).__name__}"
                )
                break
        if attn_block is None:
            print(f"    ✗ Attentionブロック: 見つかりません")

        # MLP前のlayer norm
        mlp_ln_attrs = ["post_attention_layernorm", "ln_2", "mlp_layernorm"]
        mlp_ln = None
        mlp_ln_name = None
        for attr in mlp_ln_attrs:
            if hasattr(layer, attr):
                mlp_ln = getattr(layer, attr)
                mlp_ln_name = attr
                print(f"    ✓ MLP前LayerNorm: '{attr}' → {type(mlp_ln).__name__}")
                break
        if mlp_ln is None:
            print(f"    ✗ MLP前LayerNorm: 見つかりません")

        # MLP block
        mlp_attrs = ["mlp", "feed_forward", "ffn"]
        mlp_block = None
        mlp_block_name = None
        for attr in mlp_attrs:
            if hasattr(layer, attr):
                mlp_block = getattr(layer, attr)
                mlp_block_name = attr
                print(f"    ✓ MLPブロック: '{attr}' → {type(mlp_block).__name__}")
                break
        if mlp_block is None:
            print(f"    ✗ MLPブロック: 見つかりません")

        # レイヤーの全属性を表示（参考）
        print(f"\n  全属性（参考）:")
        all_attrs = [attr for attr in dir(layer) if not attr.startswith("_")]
        module_attrs = [
            attr
            for attr in all_attrs
            if hasattr(getattr(layer, attr), "__call__")
            or isinstance(getattr(layer, attr), torch.nn.Module)
        ]
        for attr in sorted(module_attrs[:20]):  # 最初の20個のみ
            obj = getattr(layer, attr)
            if isinstance(obj, torch.nn.Module):
                print(f"    - {attr}: {type(obj).__name__}")

    # フックの呼び出し順序テスト
    print(f"\n{'=' * 80}")
    print(f"[5] フック呼び出し順序のテスト（レイヤー0のみ）")
    print(f"{'=' * 80}")

    hook_call_order = []
    handles = []

    layer = layer_list_module[0]

    def make_test_hook(name, hook_type):
        """テスト用フック関数"""

        def hook_fn(module, input, output=None):
            input_shape = (
                input[0].shape if isinstance(input, tuple) and len(input) > 0 else "N/A"
            )
            output_shape = None
            if output is not None:
                if isinstance(output, tuple):
                    output_shape = output[0].shape if len(output) > 0 else "empty tuple"
                else:
                    output_shape = output.shape

            hook_call_order.append(
                {
                    "name": name,
                    "type": hook_type,
                    "module": type(module).__name__,
                    "input_shape": input_shape,
                    "output_shape": output_shape,
                }
            )

        return hook_fn

    # フックを登録
    attn_ln = None
    for attr in attn_ln_attrs:
        if hasattr(layer, attr):
            attn_ln = getattr(layer, attr)
            break

    mlp_ln = None
    for attr in mlp_ln_attrs:
        if hasattr(layer, attr):
            mlp_ln = getattr(layer, attr)
            break

    attn_block = None
    for attr in attn_attrs:
        if hasattr(layer, attr):
            attn_block = getattr(layer, attr)
            break

    mlp_block = None
    for attr in mlp_attrs:
        if hasattr(layer, attr):
            mlp_block = getattr(layer, attr)
            break

    if attn_ln:
        handles.append(
            attn_ln.register_forward_pre_hook(
                make_test_hook("attn_layernorm_input", "pre_hook")
            )
        )
        handles.append(
            attn_ln.register_forward_hook(make_test_hook("attn_input", "forward_hook"))
        )

    if attn_block:
        handles.append(
            attn_block.register_forward_hook(
                make_test_hook("attn_output", "forward_hook")
            )
        )

    if mlp_ln:
        handles.append(
            mlp_ln.register_forward_pre_hook(
                make_test_hook("mlp_layernorm_input", "pre_hook")
            )
        )
        handles.append(
            mlp_ln.register_forward_hook(make_test_hook("mlp_input", "forward_hook"))
        )

    if mlp_block:
        handles.append(
            mlp_block.register_forward_hook(
                make_test_hook("mlp_output", "forward_hook")
            )
        )

    # テスト用の短い入力で推論
    print("\n  テスト推論を実行中...")
    test_text = "Hello"
    inputs = tokenizer(test_text, return_tensors="pt")

    with torch.no_grad():
        try:
            outputs = model(**inputs, use_cache=False)
            print("  ✓ 推論完了")
        except Exception as e:
            print(f"  ✗ 推論エラー: {e}")

    # フックを削除
    for handle in handles:
        handle.remove()

    # 結果を表示
    print(f"\n  フック呼び出し順序（合計: {len(hook_call_order)}回）:")
    print(
        f"  {'順序':<6} {'名前':<25} {'タイプ':<15} {'モジュール':<20} {'入力形状':<20} {'出力形状':<20}"
    )
    print(f"  {'-'*6} {'-'*25} {'-'*15} {'-'*20} {'-'*20} {'-'*20}")

    for i, call in enumerate(hook_call_order):
        print(
            f"  {i+1:<6} {call['name']:<25} {call['type']:<15} {call['module']:<20} {str(call['input_shape']):<20} {str(call['output_shape']):<20}"
        )

    # 期待される順序をチェック
    print(f"\n[6] 順序の検証")
    expected_names = [
        "attn_layernorm_input",
        "attn_input",
        "attn_output",
        "mlp_layernorm_input",
        "mlp_input",
        "mlp_output",
    ]

    actual_names = [call["name"] for call in hook_call_order]

    if actual_names == expected_names:
        print(f"  ✓ フック呼び出し順序は期待通りです！")
        print(f"    1. attn_layernorm_input (Attention前LayerNormへの入力)")
        print(f"    2. attn_input (Attention前LayerNormの出力 = Attentionへの入力)")
        print(f"    3. attn_output (Attentionの出力)")
        print(f"    4. mlp_layernorm_input (MLP前LayerNormへの入力)")
        print(f"    5. mlp_input (MLP前LayerNormの出力 = MLPへの入力)")
        print(f"    6. mlp_output (MLPの出力)")
    else:
        print(f"  ✗ フック呼び出し順序が期待と異なります！")
        print(f"    期待: {expected_names}")
        print(f"    実際: {actual_names}")

    print(f"\n{'=' * 80}")
    print(f"デバッグ完了")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="モデル構造とフック登録順序をデバッグ")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="検査するモデル名（例: Qwen/Qwen2.5-7B-Instruct）",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="詳細に検査するレイヤー番号（例: 0 10 20）。指定しない場合は最初・中間・最後を自動選択",
    )

    args = parser.parse_args()

    inspect_model_structure(args.model_name, args.layers)
