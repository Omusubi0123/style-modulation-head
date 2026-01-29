"""
main.py - Residual Stream Cosine Similarity分析のエントリーポイント

使用例:
    # 単一traitの分析
    python -m src.layer_analysis.cosine_similarity.main analyze_trait \
        --model_name "Qwen/Qwen2.5-7B-Instruct" \
        --trait "evil"

    # 全traitの分析
    python -m src.layer_analysis.cosine_similarity.main analyze_all \
        --model_name "Qwen/Qwen2.5-7B-Instruct"
"""

from pathlib import Path
from typing import List, Optional

import fire

from src.layer_analysis.cosine_similarity.compute import (
    compute_residual_stream_similarity,
    load_vectors_for_positions,
)
from src.layer_analysis.cosine_similarity.visualize import (
    visualize_residual_stream_similarity,
)


# デフォルト設定
DEFAULT_PERSONA_VECTORS_DIR = "data/persona_vectors"
DEFAULT_OUTPUT_DIR = "data/layer_analysis"
DEFAULT_VECTOR_TYPE = "response_avg_diff"
LAYER_POSITIONS = ["attn_layernorm", "mlp_layernorm", "attn_output", "mlp_output"]


def analyze_trait(
    model_name: str,
    trait: str,
    persona_vectors_dir: str = DEFAULT_PERSONA_VECTORS_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    vector_type: str = DEFAULT_VECTOR_TYPE,
    stream_type: str = "input",  # "input" or "output"
) -> Optional[str]:
    """単一traitのResidual Stream類似度を分析

    Args:
        model_name: モデル名 (e.g., "Qwen/Qwen2.5-7B-Instruct")
        trait: 特性名 (e.g., "evil")
        persona_vectors_dir: ベクトルディレクトリ
        output_dir: 出力ディレクトリ
        vector_type: ベクトルタイプ
        stream_type: "input" (layernorm) or "output"

    Returns:
        保存されたファイルパス
    """
    print(f"Analyzing: {model_name} - {trait} ({stream_type})")

    # ベクトルを読み込む
    vectors_by_trait = load_vectors_for_positions(
        Path(persona_vectors_dir),
        model_name,
        vector_type,
        LAYER_POSITIONS,
    )

    if trait not in vectors_by_trait:
        print(f"Error: trait '{trait}' not found")
        return None

    vectors_dict = vectors_by_trait[trait]

    # 必要なベクトルがあるか確認
    if stream_type == "input":
        attn_key, mlp_key = "attn_layernorm", "mlp_layernorm"
    else:
        attn_key, mlp_key = "attn_output", "mlp_output"

    if attn_key not in vectors_dict or mlp_key not in vectors_dict:
        print(f"Error: required vectors not found for {stream_type}")
        return None

    # 類似度行列を計算
    similarity_matrix = compute_residual_stream_similarity(
        vectors_dict[attn_key],
        vectors_dict[mlp_key],
    )

    num_layers = vectors_dict[attn_key].shape[0]

    # 出力ディレクトリを作成
    safe_model_name = model_name.replace("/", "_")
    save_dir = Path(output_dir) / "residual_stream" / safe_model_name / trait
    save_dir.mkdir(parents=True, exist_ok=True)

    # 可視化
    save_path = save_dir / f"cosine_similarity_{stream_type}.png"
    result = visualize_residual_stream_similarity(
        similarity_matrix.numpy(),
        num_layers,
        save_path,
    )

    return result


def analyze_all(
    model_name: str,
    persona_vectors_dir: str = DEFAULT_PERSONA_VECTORS_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    vector_type: str = DEFAULT_VECTOR_TYPE,
    stream_types: List[str] = ["input", "output"],
) -> None:
    """全traitのResidual Stream類似度を分析

    Args:
        model_name: モデル名
        persona_vectors_dir: ベクトルディレクトリ
        output_dir: 出力ディレクトリ
        vector_type: ベクトルタイプ
        stream_types: 分析するstream type（"input", "output"）
    """
    print(f"Loading vectors for {model_name}...")

    vectors_by_trait = load_vectors_for_positions(
        Path(persona_vectors_dir),
        model_name,
        vector_type,
        LAYER_POSITIONS,
    )

    print(f"Found {len(vectors_by_trait)} traits")

    for trait in vectors_by_trait.keys():
        for stream_type in stream_types:
            try:
                analyze_trait(
                    model_name=model_name,
                    trait=trait,
                    persona_vectors_dir=persona_vectors_dir,
                    output_dir=output_dir,
                    vector_type=vector_type,
                    stream_type=stream_type,
                )
            except Exception as e:
                print(f"Error analyzing {trait} ({stream_type}): {e}")


if __name__ == "__main__":
    fire.Fire({
        "analyze_trait": analyze_trait,
        "analyze_all": analyze_all,
    })

