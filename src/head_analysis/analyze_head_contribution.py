"""
analyze_head_contribution.py - 各Attentionヘッドの貢献度を分析

実験設定:
1. OV計算前のVのPersona Vector（attn_pre_o_proj）を読み込む
2. 各ヘッドごとにO projectionを計算（他のヘッドは0固定）
3. attn_outputのPersona Vectorとの内積類似度を計算
4. 対数スケーリング (s' = sign(s) * log(1 + |s|)) を適用
5. 層ごとにZ-score正規化を適用してヒートマップを作成

使用するデータ:
- {trait}_prompt_avg_diff_attn_pre_o_proj.pt: OV計算前のPersona Vector
- {trait}_prompt_avg_diff_attn_output.pt: Attention出力のPersona Vector（generate_vec_block.pyで生成）
- {trait}_attn_config.pt: Attention設定（num_attention_heads, head_dimなど）

注意点:
- generate_vec_block.pyで保存したベクトルはリストの0が1層目を指す
- GQA対応：実際のヘッド数はnum_attention_headsで統一
- 内積値は対数スケーリングと層ごとのZ-score正規化により、層間で比較可能
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from transformers import AutoModel, AutoModelForCausalLM

from src.utils.persona_map import get_display_persona

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.config import setup_credentials

style_path = os.path.join(os.path.dirname(__file__), '..', 'style', 'paper_expansion_heatmap.mplstyle')
if os.path.exists(style_path):
    plt.style.use(style_path)
    print(f"Using style: {style_path}")
else:
    print(f"Style file not found: {style_path}")

# Set up credentials
config = setup_credentials()


def inner_product_similarity(vec1, vec2):
    """内積類似度を計算"""
    vec1 = vec1.float()
    vec2 = vec2.float()
    return torch.dot(vec1, vec2)


def load_o_proj_weights(model_name: str, device: str = "cpu"):
    """モデルからO projection weightsを読み込む

    Args:
        model_name: モデル名
        device: 計算デバイス

    Returns:
        dict: {layer_idx: o_proj_weight tensor}
    """
    print(f"Loading model for O projection weights: {model_name}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=config.hf_token,
        )
    except ValueError as e:
        if "Unrecognized configuration class" in str(e):
            model = AutoModel.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                token=config.hf_token,
            )
        else:
            raise

    # レイヤーリストを取得
    possible_attrs = [
        "transformer.h",
        "encoder.layer",
        "model.layers",
        "gpt_neox.layers",
        "block",
        "language_model.layers",
    ]

    layer_list = None
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
            layer_list = cur
            break

    if layer_list is None:
        raise ValueError("Could not find layer list in model")

    o_proj_weights = {}
    for layer_idx, layer in enumerate(layer_list):
        # Attention blockを探す
        attn_attrs = ["self_attn", "attention", "attn"]
        attn_block = None
        for attr in attn_attrs:
            if hasattr(layer, attr):
                attn_block = getattr(layer, attr)
                break

        if attn_block is None:
            continue

        # o_projを探す
        o_proj_attrs = ["o_proj", "out_proj", "dense"]
        o_proj = None
        for attr in o_proj_attrs:
            if hasattr(attn_block, attr):
                o_proj = getattr(attn_block, attr)
                break

        if o_proj is not None:
            o_proj_weights[layer_idx] = o_proj.weight.data.clone().float()

    # モデルを解放
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return o_proj_weights


def compute_head_contributions(
    attn_pre_o_proj_vec: torch.Tensor,
    attn_output_vec: torch.Tensor,
    o_proj_weight: torch.Tensor,
    num_heads: int,
    head_dim: int,
):
    """各ヘッドのO projection後の出力とattn_outputの内積類似度を計算

    Args:
        attn_pre_o_proj_vec: O projection前のPersona Vector [hidden_size]
        attn_output_vec: Attention出力のPersona Vector [hidden_size]
        o_proj_weight: O projection weight [hidden_size, hidden_size]
        num_heads: Attentionヘッド数
        head_dim: 各ヘッドの次元

    Returns:
        list: 各ヘッドの内積類似度
    """
    similarities = []
    hidden_size = num_heads * head_dim

    for head_idx in range(num_heads):
        # 該当ヘッド以外を0にしたベクトルを作成
        masked_vec = torch.zeros_like(attn_pre_o_proj_vec)
        start_idx = head_idx * head_dim
        end_idx = (head_idx + 1) * head_dim
        masked_vec[start_idx:end_idx] = attn_pre_o_proj_vec[start_idx:end_idx]

        # O projectionを適用
        # o_proj_weight: [out_features, in_features] = [hidden_size, hidden_size]
        projected_vec = torch.matmul(o_proj_weight, masked_vec)

        # 内積類似度を計算
        sim = inner_product_similarity(projected_vec, attn_output_vec)
        similarities.append(sim.item())

    return similarities


def analyze_head_contributions(
    model_name: str,
    vector_dir: str,
    trait: str,
    layer_list: list = None,
    output_dir: str = None,
    vector_type: str = "prompt_avg",
    use_zscore: bool = True,
):
    """各層の各ヘッドの貢献度を分析してヒートマップを作成

    内積類似度を計算し、対数スケーリングと層ごとのZ-score正規化を適用する。

    Args:
        model_name: モデル名（O projection weightを取得するため）
        vector_dir: Persona Vectorが保存されているディレクトリ
        trait: 特性名
        layer_list: 分析する層のリスト（Noneの場合は全層）
        output_dir: 出力ディレクトリ（Noneの場合はvector_dir）
        vector_type: 使用するベクトルタイプ（"prompt_avg", "response_avg", "prompt_last"）
        use_zscore: Z-score正規化を適用するかどうか（デフォルト: True）
    """
    if output_dir is None:
        output_dir = vector_dir

    os.makedirs(output_dir, exist_ok=True)

    # ベクトルを読み込む
    attn_pre_o_proj_path = os.path.join(
        vector_dir, f"{trait}_{vector_type}_diff_attn_pre_o_proj.pt"
    )
    attn_output_path = os.path.join(
        vector_dir, f"{trait}_{vector_type}_diff_attn_output.pt"
    )
    attn_config_path = os.path.join(vector_dir, f"{trait}_attn_config.pt")

    print(f"Loading vectors from {vector_dir}")

    if not os.path.exists(attn_pre_o_proj_path):
        raise FileNotFoundError(
            f"attn_pre_o_proj vector not found: {attn_pre_o_proj_path}\n"
            f"Please run generate_vec_attn.py first."
        )

    if not os.path.exists(attn_output_path):
        raise FileNotFoundError(
            f"attn_output vector not found: {attn_output_path}\n"
            f"Please run generate_vec_block.py first."
        )

    attn_pre_o_proj = torch.load(attn_pre_o_proj_path, map_location="cpu")
    attn_output = torch.load(attn_output_path, map_location="cpu")

    # Attention設定を読み込む（または推測）
    if os.path.exists(attn_config_path):
        attn_config = torch.load(attn_config_path, map_location="cpu")
        num_heads = attn_config["num_attention_heads"]
        head_dim = attn_config["head_dim"]
    else:
        # ベクトルの形状から推測
        hidden_size = attn_pre_o_proj.shape[-1]
        # 一般的なヘッド数を試す
        for n_heads in [32, 28, 40, 24, 16, 12, 8]:
            if hidden_size % n_heads == 0:
                num_heads = n_heads
                head_dim = hidden_size // n_heads
                print(
                    f"Warning: attn_config not found, guessing num_heads={num_heads}, head_dim={head_dim}"
                )
                break
        else:
            raise ValueError(
                "Could not determine num_heads from vector shape. "
                "Please ensure attn_config.pt exists."
            )

    print(f"Attention config: num_heads={num_heads}, head_dim={head_dim}")
    print(f"attn_pre_o_proj shape: {attn_pre_o_proj.shape}")
    print(f"attn_output shape: {attn_output.shape}")

    # 層数を取得
    num_layers = attn_pre_o_proj.shape[0]

    if layer_list is None:
        layer_list = list(range(num_layers))

    # O projection weightsを読み込む
    o_proj_weights = load_o_proj_weights(model_name)

    # 各層・各ヘッドのcosine類似度を計算
    similarity_matrix = np.zeros((len(layer_list), num_heads))

    for i, layer_idx in enumerate(layer_list):
        if layer_idx not in o_proj_weights:
            print(f"Warning: O proj weight not found for layer {layer_idx}")
            continue

        pre_o_proj_vec = attn_pre_o_proj[layer_idx]
        output_vec = attn_output[layer_idx]
        o_proj_weight = o_proj_weights[layer_idx]

        similarities = compute_head_contributions(
            pre_o_proj_vec, output_vec, o_proj_weight, num_heads, head_dim
        )

        similarity_matrix[i, :] = similarities

    # 対数スケーリング: s' = sign(s) * log(1 + |s|)
    log_scaled_matrix = np.sign(similarity_matrix) * np.log1p(np.abs(similarity_matrix))
    
    # 層ごとにZ-score正規化（オプション）
    if use_zscore:
        normalized_matrix = np.zeros_like(log_scaled_matrix)
        for i in range(len(layer_list)):
            layer_values = log_scaled_matrix[i, :]
            mean = np.mean(layer_values)
            std = np.std(layer_values)
            if std > 0:
                normalized_matrix[i, :] = (layer_values - mean) / std
            else:
                normalized_matrix[i, :] = layer_values - mean
    else:
        normalized_matrix = log_scaled_matrix

    # ヒートマップを作成（全体ができるだけ正方形に近くなるように調整）
    n_layers = len(layer_list)
    n_heads = num_heads
    base_size = max(8, max(n_heads, n_layers) * 0.35)
    fig_width = base_size
    fig_height = base_size
    plt.figure(figsize=(fig_width, fig_height))

    # カラーマップの設定（負の値も考慮）
    vmin = normalized_matrix.min()
    vmax = normalized_matrix.max()

    # 値の範囲に応じてカラーマップを選択
    if vmin < 0:
        # 負の値がある場合は発散型カラーマップ
        abs_max = max(abs(vmin), abs(vmax))
        cmap = "RdBu_r"
        vmin_plot, vmax_plot = -abs_max, abs_max
    else:
        # 正の値のみの場合
        cmap = "YlOrRd"
        vmin_plot, vmax_plot = vmin, vmax

    ax = sns.heatmap(
        normalized_matrix,
        xticklabels=[f"H{i+1}" for i in range(num_heads)],
        yticklabels=[f"L{l+1}" for l in layer_list],
        cmap=cmap,
        vmin=vmin_plot,
        vmax=vmax_plot,
        annot=False,
        fmt=".2f",
        cbar_kws={"label": "Normalized Inner Product (Z-score)" if use_zscore else "Inner Product (log-scaled)"},
    )

    # 軸ラベルと行名・列名はスタイルファイル側で大きめ・太字に設定しつつ、
    # ヘッド数が多くても読めるように x ラベルは縦方向に回転
    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold', rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
    plt.xlabel("Head Index")
    plt.ylabel("Layer Index")
    # normalization_text = "log-scaled + Z-score" if use_zscore else "log-scaled"
    # plt.title(
    #     f"Head Contribution to Persona Vector ({trait}, {vector_type})\n"
    #     f"Layer-wise Normalized Inner Product ({normalization_text}) between Head-wise O Projection and attn_output"
    # )
    plt.tight_layout()

    # 保存
    suffix = "_no_zscore" if not use_zscore else ""
    output_path = os.path.join(
        output_dir, f"{trait}_inner_product_head_contribution_{vector_type}{suffix}.png"
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved to {output_path}")

    # 類似度行列も保存（正規化後のものと生の内積値の両方）
    suffix = "_no_zscore" if not use_zscore else ""
    np.save(
        os.path.join(output_dir, f"{trait}_inner_product_head_contribution_{vector_type}{suffix}.npy"),
        normalized_matrix,
    )
    np.save(
        os.path.join(output_dir, f"{trait}_inner_product_head_contribution_{vector_type}_raw.npy"),
        similarity_matrix,
    )

    # 上位のヘッドを表示（正規化後の値で）
    print("\nTop 10 heads by contribution (normalized):")
    flat_idx = np.argsort(normalized_matrix.flatten())[::-1]
    for rank, idx in enumerate(flat_idx[:10]):
        layer_idx = layer_list[idx // num_heads]
        head_idx = idx % num_heads
        sim_norm = normalized_matrix[idx // num_heads, head_idx]
        sim_raw = similarity_matrix[idx // num_heads, head_idx]
        print(f"  {rank + 1}. Layer {layer_idx}, Head {head_idx}: {sim_norm:.4f} (raw: {sim_raw:.2e})")

    return normalized_matrix


def analyze_multiple_traits(
    model_name: str,
    vector_dir: str,
    traits: list[str],
    layer_list: list = None,
    output_dir: str = None,
    vector_type: str = "prompt_avg",
    use_zscore: bool = True,
):
    """複数の特性について分析を実行"""
    results = {}
    for trait in traits:
        print(f"\n{'=' * 60}")
        print(f"Analyzing trait: {trait}")
        print(f"{'=' * 60}")
        try:
            similarity_matrix = analyze_head_contributions(
                model_name=model_name,
                vector_dir=vector_dir,
                trait=trait,
                layer_list=layer_list,
                output_dir=output_dir,
                vector_type=vector_type,
                use_zscore=use_zscore,
            )
            results[trait] = similarity_matrix
        except FileNotFoundError as e:
            print(f"Skipping {trait}: {e}")
            continue

    return results


def analyze_traits_at_layer(
    model_name: str,
    vector_dir: str,
    traits: list[str],
    layer_idx: int,
    output_dir: str = None,
    vector_type: str = "response_avg",
    use_log: bool = True,
    use_zscore: bool = True,
):
    """複数のtraitについて特定の層でのヘッド貢献度を比較するヒートマップを作成

    内積類似度を計算し、対数スケーリングとtraitごとのZ-score正規化を適用する。
    
    横軸: ヘッド位置
    縦軸: trait名

    Args:
        model_name: モデル名（O projection weightを取得するため）
        vector_dir: Persona Vectorが保存されているディレクトリ
        traits: 特性名のリスト
        layer_idx: 分析する層のインデックス（0-based）
        output_dir: 出力ディレクトリ（Noneの場合はvector_dir）
        vector_type: 使用するベクトルタイプ（"prompt_avg", "response_avg", "prompt_last"）
        use_zscore: Z-score正規化を適用するかどうか（デフォルト: True）

    Returns:
        tuple: (正規化された類似度行列 [num_traits, num_heads], 有効なtrait名のリスト)
    """
    if output_dir is None:
        output_dir = vector_dir

    os.makedirs(output_dir, exist_ok=True)

    # 最初のtraitからAttention設定を読み込む
    first_trait = traits[0]
    attn_config_path = os.path.join(vector_dir, f"{first_trait}_attn_config.pt")
    attn_config_json_path = os.path.join(vector_dir, "attn_config.json")

    num_heads = None
    head_dim = None

    if os.path.exists(attn_config_path):
        attn_config = torch.load(attn_config_path, map_location="cpu")
        num_heads = attn_config["num_attention_heads"]
        head_dim = attn_config["head_dim"]
    elif os.path.exists(attn_config_json_path):
        import json

        with open(attn_config_json_path, "r") as f:
            attn_config = json.load(f)
        num_heads = attn_config["num_attention_heads"]
        head_dim = attn_config["head_dim"]
    else:
        # 最初のベクトルから推測
        attn_pre_o_proj_path = os.path.join(
            vector_dir, f"{first_trait}_{vector_type}_diff_attn_pre_o_proj.pt"
        )
        if os.path.exists(attn_pre_o_proj_path):
            attn_pre_o_proj = torch.load(attn_pre_o_proj_path, map_location="cpu")
            hidden_size = attn_pre_o_proj.shape[-1]
            for n_heads in [32, 28, 40, 24, 16, 12, 8]:
                if hidden_size % n_heads == 0:
                    num_heads = n_heads
                    head_dim = hidden_size // n_heads
                    print(
                        f"Warning: attn_config not found, guessing num_heads={num_heads}"
                    )
                    break

    if num_heads is None:
        raise ValueError(
            "Could not determine num_heads. Please ensure attn_config exists."
        )

    print(f"Attention config: num_heads={num_heads}, head_dim={head_dim}")
    print(f"Analyzing layer {layer_idx} for {len(traits)} traits")

    # O projection weightsを読み込む
    o_proj_weights = load_o_proj_weights(model_name)

    if layer_idx not in o_proj_weights:
        raise ValueError(f"O proj weight not found for layer {layer_idx}")

    o_proj_weight = o_proj_weights[layer_idx]

    # 各traitのヘッド貢献度を計算
    similarity_matrix = np.zeros((len(traits), num_heads))
    valid_traits = []

    for i, trait in enumerate(traits):
        attn_pre_o_proj_path = os.path.join(
            vector_dir, f"{trait}_{vector_type}_diff_attn_pre_o_proj.pt"
        )
        attn_output_path = os.path.join(
            vector_dir, f"{trait}_{vector_type}_diff_attn_output.pt"
        )

        if not os.path.exists(attn_pre_o_proj_path):
            print(f"Warning: Skipping {trait} - attn_pre_o_proj not found")
            continue

        if not os.path.exists(attn_output_path):
            print(f"Warning: Skipping {trait} - attn_output not found")
            continue

        attn_pre_o_proj = torch.load(attn_pre_o_proj_path, map_location="cpu")
        attn_output = torch.load(attn_output_path, map_location="cpu")

        pre_o_proj_vec = attn_pre_o_proj[layer_idx]
        output_vec = attn_output[layer_idx]

        similarities = compute_head_contributions(
            pre_o_proj_vec, output_vec, o_proj_weight, num_heads, head_dim
        )

        similarity_matrix[i, :] = similarities
        valid_traits.append(trait)

    if not valid_traits:
        raise ValueError("No valid traits found with required vector files")

    # 有効なtraitのみに絞る
    valid_indices = [i for i, t in enumerate(traits) if t in valid_traits]
    similarity_matrix = similarity_matrix[valid_indices, :]

    # 対数スケーリング: s' = sign(s) * log(1 + |s|)
    if use_log:
        log_scaled_matrix = np.sign(similarity_matrix) * np.log1p(np.abs(similarity_matrix))
    else:
        log_scaled_matrix = similarity_matrix

    # traitごとにZ-score正規化（オプション）
    if use_zscore:
        normalized_matrix = np.zeros_like(log_scaled_matrix)
        for i in range(len(valid_traits)):
            trait_values = log_scaled_matrix[i, :]
            mean = np.mean(trait_values)
            std = np.std(trait_values)
            if std > 0:
                normalized_matrix[i, :] = (trait_values - mean) / std
            else:
                normalized_matrix[i, :] = trait_values - mean
    else:
        normalized_matrix = log_scaled_matrix

    # ヒートマップを作成（全体ができるだけ正方形に近くなるように調整）
    n_traits = len(valid_traits)
    n_heads = num_heads
    base_size = max(8, max(n_heads, n_traits) * 0.35)
    fig_width = base_size
    fig_height = int(base_size / 5 * 3)
    plt.figure(figsize=(fig_width, fig_height))

    # カラーマップの設定
    vmin = normalized_matrix.min()
    vmax = normalized_matrix.max()

    if vmin < 0:
        abs_max = max(abs(vmin), abs(vmax))
        cmap = "RdBu_r"
        vmin_plot, vmax_plot = -abs_max, abs_max
    else:
        cmap = "YlOrRd"
        vmin_plot, vmax_plot = vmin, vmax

    ax = sns.heatmap(
        normalized_matrix,
        xticklabels=[f"H{i+1}" for i in range(num_heads)],
        yticklabels=[get_display_persona(trait) for trait in valid_traits],
        cmap=cmap,
        vmin=vmin_plot,
        vmax=vmax_plot,
        # center=0,
        # vmin=-20,
        # vmax=80,
        annot=False,
        cbar=True,
        cbar_kws={
            "label": "Head Contribution Score",
            "pad": 0.02,
        },
    )

    # 軸ラベルと行名・列名はスタイルファイル側で大きめ・太字に設定しつつ、
    # ヘッド数が多くても読めるように x ラベルは縦方向に回転
    ax.set_xticklabels(ax.get_xticklabels(), fontweight='bold', rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), fontweight='bold')
    plt.xlabel("Head Index")
    plt.ylabel("Trait")
    # normalization_text = "log-scaled + Z-score" if use_zscore else "log-scaled"
    # plt.title(
    #     f"Head Contribution Comparison Across Traits (Layer {layer_idx+1}, {vector_type})\n"
    #     f"Trait-wise Normalized Inner Product ({normalization_text}) between Head-wise O Projection and attn_output"
    # )
    plt.tight_layout()

    # 保存
    suffix = "_no_zscore" if not use_zscore else ""
    output_filename = f"traits_comparison_inner_product_layer{layer_idx+1}_{vector_type}{suffix}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.savefig(output_path.replace('.png', '.pdf'), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved to {output_path}")

    # 類似度行列も保存（正規化後のものと生の内積値の両方）
    suffix = "_no_zscore" if not use_zscore else ""
    np.save(
        os.path.join(
            output_dir, f"traits_comparison_inner_product_layer{layer_idx+1}_{vector_type}{suffix}.npy"
        ),
        normalized_matrix,
    )
    np.save(
        os.path.join(
            output_dir, f"traits_comparison_inner_product_layer{layer_idx+1}_{vector_type}_raw.npy"
        ),
        similarity_matrix,
    )

    # 各ヘッドについて、全trait間での一貫性を表示（正規化後の値で）
    print(f"\nHead consistency across traits (Layer {layer_idx+1}) [normalized]:")
    head_means = normalized_matrix.mean(axis=0)
    head_stds = normalized_matrix.std(axis=0)
    top_heads = np.argsort(head_means)[::-1][:10]

    print(f"Top 10 heads by average contribution (normalized):")
    for rank, head_idx in enumerate(top_heads):
        mean_val = head_means[head_idx]
        std_val = head_stds[head_idx]
        print(f"  {rank + 1}. Head {head_idx+1}: {mean_val:.4f} ± {std_val:.4f}")

    # trait間の相関を表示（正規化後の値で）
    if len(valid_traits) > 1:
        print(f"\nCorrelation between traits (based on normalized head contributions):")
        trait_corr = np.corrcoef(normalized_matrix)
        for i in range(len(valid_traits)):
            for j in range(i + 1, len(valid_traits)):
                print(
                    f"  {valid_traits[i]} vs {valid_traits[j]}: {trait_corr[i, j]:.4f}"
                )

    return normalized_matrix, valid_traits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze head contribution to Persona Vector"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (e.g., Qwen/Qwen2.5-7B-Instruct)",
    )
    parser.add_argument(
        "--vector_dir",
        type=str,
        required=True,
        help="Directory containing Persona Vectors",
    )
    parser.add_argument(
        "--trait",
        type=str,
        default=None,
        help="Trait name for single-trait analysis (e.g., agreeableness)",
    )
    parser.add_argument(
        "--traits",
        type=str,
        nargs="+",
        default=None,
        help="Multiple trait names for cross-trait comparison (e.g., evil apathetic humorous)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Layer indices to analyze (default: all layers for single-trait mode)",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Single layer index for cross-trait comparison mode",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as vector_dir)",
    )
    parser.add_argument(
        "--vector_type",
        type=str,
        default="response_avg",
        choices=["prompt_avg", "response_avg", "prompt_last"],
        help="Vector type to analyze (default: response_avg)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "compare"],
        help="Analysis mode: 'single' for single trait, 'compare' for cross-trait comparison",
    )
    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Skip log-scaling (use only Z-score normalization)",
    )
    parser.add_argument(
        "--no_zscore",
        action="store_true",
        help="Skip Z-score normalization (use only log-scaling)",
    )

    args = parser.parse_args()

    use_zscore = not args.no_zscore
    use_log = not args.no_log
    
    if args.mode == "compare":
        # Cross-trait comparison mode
        if args.traits is None:
            raise ValueError("--traits is required for compare mode")
        if args.layer is None:
            raise ValueError("--layer is required for compare mode")

        analyze_traits_at_layer(
            model_name=args.model_name,
            vector_dir=args.vector_dir,
            traits=args.traits,
            layer_idx=args.layer,
            output_dir=args.output_dir,
            vector_type=args.vector_type,
            use_log=use_log,
            use_zscore=use_zscore,
        )
    else:
        # Single trait mode
        if args.trait is None:
            raise ValueError("--trait is required for single mode")

        analyze_head_contributions(
            model_name=args.model_name,
            vector_dir=args.vector_dir,
            trait=args.trait,
            layer_list=args.layers,
            output_dir=args.output_dir,
            vector_type=args.vector_type,
            use_log=use_log,
            use_zscore=use_zscore,
        )
