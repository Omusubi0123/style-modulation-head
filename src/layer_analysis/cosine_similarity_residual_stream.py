"""
Layer-wise Persona Vector Analysis - Residual Stream Version
Analyzes relationships between persona vectors across different layers within models,
focusing on residual stream flow (combining attention and MLP blocks).
"""
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
import torch
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from tqdm import tqdm


style_path = os.path.join(os.path.dirname(__file__), '..', '..', 'style', 'paper_expansion_4figure.mplstyle')
if os.path.exists(style_path):
    plt.style.use(style_path)


class CosineSimilarityAnalyzer:
    """各層のPersona Vector間のcosine類似度を分析・可視化するクラス"""

    def __init__(
        self, model_name: str, persona_vectors_dir: str, output_dir: str, save_fig: bool
    ):
        """
        Args:
            model_name: モデル名
            persona_vectors_dir: persona vectorsが保存されているディレクトリ
            output_dir: 分析結果を保存するディレクトリ
        """
        self.model_name = model_name
        self.persona_vectors_dir = Path(persona_vectors_dir) / model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_fig = save_fig

        self.vectors: Dict[str, torch.Tensor] = {}

    def load_vectors_for_positions(
        self, vector_type: str, layer_positions: List[str]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """複数のlayer positionのベクトルを読み込む

        Args:
            vector_type: 'response_avg_diff', 'prompt_avg_diff', 'prompt_last_diff'
            layer_positions: 読み込むlayer positionのリスト

        Returns:
            {trait_name: {layer_position: vector}} の辞書
        """
        vectors_dict: Dict[str, Dict[str, torch.Tensor]] = {}

        for layer_position in layer_positions:
            for vector_file in self.persona_vectors_dir.glob(
                f"*_{vector_type}_{layer_position}.pt"
            ):
                trait_name = vector_file.stem.replace(
                    f"_{vector_type}_{layer_position}", ""
                )

                try:
                    vector = torch.load(
                        vector_file, weights_only=False, map_location="cpu"
                    )

                    if trait_name not in vectors_dict:
                        vectors_dict[trait_name] = {}
                    vectors_dict[trait_name][layer_position] = vector

                except Exception as e:
                    print(f"Warning: Failed to load {vector_file}: {e}")
                    continue

        return vectors_dict

    def analyze_residual_stream(
        self,
        model_name: str,
        trait_name: str,
        vectors_dict: Dict[str, torch.Tensor],
        save_dir: Optional[Path] = None,
        save_pdf: bool = False,
    ) -> Dict[str, str]:
        """Residual streamの分析を実行

        Args:
            model_name: モデル名
            trait_name: ペルソナ名
            vectors_dict: {layer_position: vector}の辞書
            save_dir: 保存ディレクトリ
            save_pdf: PDFでも保存するか

        Returns:
            各分析結果のファイルパス
        """
        if save_dir is None:
            safe_model_name = model_name.replace("/", "_")
            save_dir = (
                self.output_dir / "residual_stream" / safe_model_name / trait_name
            )
        save_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # 1. Residual stream input (attn_layernorm + mlp_layernorm)
        if "attn_layernorm" in vectors_dict and "mlp_layernorm" in vectors_dict:
            try:
                results["residual_stream_input"] = (
                    self._visualize_residual_stream_cosine_similarity(
                        vectors_dict["attn_layernorm"],
                        vectors_dict["mlp_layernorm"],
                        model_name,
                        trait_name,
                        save_dir / "residual_stream_cosine_similarity_input.png",
                        stream_type="input",
                        save_pdf=save_pdf,
                    )
                )
            except Exception as e:
                print(
                    f"  Warning: Failed to create residual stream input similarity: {e}"
                )
                results["residual_stream_input"] = None

        # 2. Residual stream output (attn_output + mlp_output)
        if "attn_output" in vectors_dict and "mlp_output" in vectors_dict:
            try:
                results["residual_stream_output"] = (
                    self._visualize_residual_stream_cosine_similarity(
                        vectors_dict["attn_output"],
                        vectors_dict["mlp_output"],
                        model_name,
                        trait_name,
                        save_dir / "residual_stream_cosine_similarity_output.png",
                        stream_type="output",
                        save_pdf=save_pdf,
                    )
                )
            except Exception as e:
                print(
                    f"  Warning: Failed to create residual stream output similarity: {e}"
                )
                results["residual_stream_output"] = None

        return results

    def _visualize_layer_cosine_similarity(
        self,
        vector: torch.Tensor,
        model_name: str,
        trait_name: str,
        layer_position: str,
        save_path: Path,
        save_pdf: bool = False,
    ) -> str:
        """層間のcosine類似度をヒートマップで可視化"""
        # 既に画像が存在する場合はスキップ
        # if Path(save_path).exists():
        #     print(f"  Skipping layer cosine similarity (already exists): {save_path}")
        #     return str(save_path)

        vectors_np = vector.numpy()
        similarity_matrix = cosine_similarity(vectors_np)
        num_layers = vector.shape[0]

        plt.figure(figsize=(12, 10))

        # 1始まりのラベルを作成
        layer_labels = [str(i + 1) for i in range(num_layers)]

        # ヒートマップを描画（cosine類似度の範囲は-1〜1、0を白に）
        ax = sns.heatmap(
            similarity_matrix,
            annot=False,  # テキストアノテーションを無効化
            cmap="RdBu_r",  # 0が白になるdivergingカラーマップ
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            # cbar_kws={"label": "Cosine Similarity"},
            xticklabels=layer_labels,
            yticklabels=layer_labels,
        )

        # 中間層を強調
        mid_layer = num_layers // 2
        ax.add_patch(
            plt.Rectangle(
                (mid_layer, 0), 1, num_layers, fill=False, edgecolor="red", lw=3
            )
        )
        ax.add_patch(
            plt.Rectangle(
                (0, mid_layer), num_layers, 1, fill=False, edgecolor="red", lw=3
            )
        )

        plt.xlabel("Layer")
        plt.ylabel("Layer")
        # plt.title(
        #     f"Layer-to-Layer Cosine Similarity\n{model_name} - {trait_name} - {layer_position}",
        #     fontsize=16,
        #     pad=15,
        # )
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        save_path = str(save_path).replace(".png", ".pdf")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.close()

        print(f"Saved layer cosine similarity to {save_path}")
        return str(save_path)

    def _visualize_residual_stream_cosine_similarity(
        self,
        attn_vectors: torch.Tensor,
        mlp_vectors: torch.Tensor,
        model_name: str,
        trait_name: str,
        save_path: Path,
        stream_type: str = "input",  # "input" or "output"
        save_pdf: bool = False,
        # highlight_layers: List[int] = [20],  # ハイライトする層番号のリスト
        highlight_layers: List[int] = [14],  # ハイライトする層番号のリスト
    ) -> str:
        """Residual streamのcosine類似度をヒートマップで可視化（論文用）

        Args:
            attn_vectors: Attention層のベクトル [num_layers, hidden_dim]
            mlp_vectors: MLP層のベクトル [num_layers, hidden_dim]
            model_name: モデル名
            trait_name: ペルソナ名
            save_path: 保存パス
            stream_type: "input" (layernorm) または "output"
            save_pdf: PDFでも保存するか
            highlight_layers: ハイライトする層番号のリスト（デフォルト: [15, 20]）
        """
        num_layers = attn_vectors.shape[0]

        # Residual streamの順序に従って結合: [attn_1, mlp_1, attn_2, mlp_2, ...]
        combined_vectors = []
        for layer_idx in range(num_layers):
            combined_vectors.append(attn_vectors[layer_idx])
            combined_vectors.append(mlp_vectors[layer_idx])

        # Tensorに変換
        combined_vectors = torch.stack(combined_vectors)  # [num_layers*2, hidden_dim]

        # Cosine類似度を計算
        vectors_np = combined_vectors.numpy()
        similarity_matrix = cosine_similarity(vectors_np)

        # プロット（コンパクトなサイズ）
        fig, ax = plt.subplots(figsize=(8, 8))

        # ヒートマップを描画（0が白になるdivergingカラーマップ）
        heatmap = sns.heatmap(
            similarity_matrix,
            annot=False,
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            cbar_kws={"shrink": 0.7, "pad": 0.02},  # カラーバーをヒートマップに近づける
            ax=ax,
            xticklabels=False,
            yticklabels=False,
        )

        # カラーバーの設定
        cbar = heatmap.collections[0].colorbar
        cbar.ax.set_ylabel("Cosine Similarity", fontsize=14, fontweight='bold', labelpad=10)
        cbar.ax.tick_params(labelsize=9)  # 数値は小さく

        total_positions = num_layers * 2

        # === X軸の装飾 ===
        arrow_y = total_positions + 2
        text_y = total_positions + 4

        # 1. Shallow側（左向き矢印 + テキスト）
        ax.annotate(
            '', xy=(total_positions * 0.02, arrow_y),
            xytext=(total_positions * 0.15, arrow_y),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            annotation_clip=False
        )
        ax.text(total_positions * 0.08, text_y, 'Shallow', color='gray',
                ha='center', va='top', fontsize=11, fontweight='bold')

        # 2. Deep側（右向き矢印 + テキスト）
        ax.annotate(
            '', xy=(total_positions * 0.98, arrow_y),
            xytext=(total_positions * 0.85, arrow_y),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            annotation_clip=False
        )
        ax.text(total_positions * 0.92, text_y, 'Deep', color='gray',
                ha='center', va='top', fontsize=11, fontweight='bold')

        # 3. ハイライト層のラベル（矢印と同じ高さに配置）
        highlight_colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e']
        for i, hl in enumerate(highlight_layers):
            if hl <= num_layers:
                color = highlight_colors[i % len(highlight_colors)]

                # L{hl}_Attn（index = (hl-1)*2）
                attn_idx = (hl - 1) * 2
                ax.annotate(
                    '', xy=(attn_idx + 0.5, total_positions + 0.5),
                    xytext=(attn_idx + 0.5, arrow_y - 0.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    annotation_clip=False
                )
                # Attnラベル（上にずらして重ならないように）
                ax.text(
                    attn_idx - 4.5, text_y + 0.5,
                    f'L{hl}_Attn',
                    ha='center', va='top', fontsize=10, fontweight='bold', color='red'
                )

                # L{hl}_MLP（index = (hl-1)*2 + 1、Attnの隣）
                mlp_idx = attn_idx + 1
                ax.annotate(
                    '', xy=(mlp_idx + 0.5, total_positions + 0.5),
                    xytext=(mlp_idx + 0.5, arrow_y - 0.5),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                    annotation_clip=False
                )
                # MLPラベル（下にずらして重ならないように）
                ax.text(
                    mlp_idx + 4.5, text_y + 0.5,
                    f'L{hl}_MLP',
                    ha='center', va='top', fontsize=10, fontweight='bold', color='blue'
                )

        # 4. 軸タイトル
        ax.set_xlabel("Position in Residual Stream", fontsize=13, fontweight='bold', labelpad=40)

        # === Y軸の装飾 ===
        arrow_x = -2
        text_x = -4

        # 1. Shallow側（上向き矢印 + テキスト）
        ax.annotate(
            '', xy=(arrow_x, total_positions * 0.02),
            xytext=(arrow_x, total_positions * 0.15),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            annotation_clip=False
        )
        ax.text(text_x, total_positions * 0.08, 'Shallow', color='gray',
                ha='right', va='center', fontsize=11, fontweight='bold', rotation=90)

        # 2. Deep側（下向き矢印 + テキスト）
        ax.annotate(
            '', xy=(arrow_x, total_positions * 0.98),
            xytext=(arrow_x, total_positions * 0.85),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            annotation_clip=False
        )
        ax.text(text_x, total_positions * 0.92, 'Deep', color='gray',
                ha='right', va='center', fontsize=11, fontweight='bold', rotation=90)

        # 3. ハイライト層のラベル（Y軸側、矢印と同じ位置に配置）
        for i, hl in enumerate(highlight_layers):
            if hl <= num_layers:
                color = highlight_colors[i % len(highlight_colors)]

                # L{hl}_Attn
                attn_idx = (hl - 1) * 2
                ax.annotate(
                    '', xy=(-0.5, attn_idx + 0.5),
                    xytext=(arrow_x + 0.5, attn_idx + 0.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    annotation_clip=False
                )
                ax.text(
                    text_x - 1, attn_idx - 4.5,
                    f'L{hl}_Attn',
                    ha='right', va='center', fontsize=10, fontweight='bold', color='red', rotation=90
                )

                # L{hl}_MLP
                mlp_idx = attn_idx + 1
                ax.annotate(
                    '', xy=(-0.5, mlp_idx + 0.5),
                    xytext=(arrow_x + 0.5, mlp_idx + 0.5),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                    annotation_clip=False
                )
                ax.text(
                    text_x - 1, mlp_idx + 4.5,
                    f'L{hl}_MLP',
                    ha='right', va='center', fontsize=10, fontweight='bold', color='blue', rotation=90
                )

        # 4. 軸タイトル
        ax.set_ylabel("Position in Residual Stream", fontsize=13, fontweight='bold', labelpad=50)

        # 余白を調整
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        pdf_path = str(save_path).replace(".png", ".pdf")
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close()

        print(f"Saved residual stream cosine similarity to {save_path}")
        return str(save_path)

    def compute_layerwise_cosine_similarity(
        self, vector: torch.Tensor
    ) -> torch.Tensor:
        """各層ベクトル間のcosine類似度を計算

        Args:
            vector: [num_layers, hidden_dim] のベクトル

        Returns:
            similarity_matrix: [num_layers, num_layers] のcosine類似度行列
        """
        vectors_np = vector.numpy()
        similarity_matrix = cosine_similarity(vectors_np)
        return torch.tensor(similarity_matrix)

    def compute_adjacent_layer_similarity(
        self, vector: torch.Tensor
    ) -> torch.Tensor:
        """隣接層間のcosine類似度を計算

        Args:
            vector: [num_layers, hidden_dim] のベクトル

        Returns:
            adjacent_similarities: [num_layers-1] の隣接層間cosine類似度
            sim[i] = cos_sim(L_i, L_{i+1})
        """
        vectors_np = vector.numpy()
        num_layers = vector.shape[0]

        adjacent_similarities = []
        for i in range(num_layers - 1):
            sim = cosine_similarity(
                vectors_np[i : i + 1], vectors_np[i + 1 : i + 2]
            )[0, 0]
            adjacent_similarities.append(sim)

        return torch.tensor(adjacent_similarities)

    def compute_adjacent_layer_difference(
        self, vector: torch.Tensor
    ) -> torch.Tensor:
        """隣接層間類似度の差分を計算

        層iについて: sim(L_{i-1}, L_i) - sim(L_i, L_{i+1})
        例: 層20の場合、(L19-L20間の類似度) - (L20-L21間の類似度)

        Args:
            vector: [num_layers, hidden_dim] のベクトル

        Returns:
            differences: [num_layers-2] の差分
            diff[i] は層 i+1 における前後の類似度差分
        """
        adjacent_sims = self.compute_adjacent_layer_similarity(vector)
        # diff[i] = adjacent_sims[i] - adjacent_sims[i+1]
        # これは層 i+1 について: sim(L_i, L_{i+1}) - sim(L_{i+1}, L_{i+2})
        differences = adjacent_sims[:-1] - adjacent_sims[1:]
        return differences

    def visualize_adjacent_layer_difference_single_row(
        self,
        vector: torch.Tensor,
        model_name: str,
        trait_name: str,
        layer_position: str,
        save_path: Path,
        save_pdf: bool = False,
    ) -> str:
        """隣接層間類似度の差分を横長1行のヒートマップで可視化

        層iについて: sim(L_{i-1}, L_i) - sim(L_i, L_{i+1})

        Args:
            vector: [num_layers, hidden_dim] のベクトル
            model_name: モデル名
            trait_name: ペルソナ名
            layer_position: 層の位置
            save_path: 保存パス
            save_pdf: PDFでも保存するか

        Returns:
            保存パス
        """
        differences = self.compute_adjacent_layer_difference(vector)
        num_layers = len(differences)

        # 横長1行の図を作成
        fig, ax = plt.subplots(figsize=(max(20, num_layers * 0.5), 2))

        # 1行のヒートマップ
        diff_matrix = differences.numpy().reshape(1, -1)

        # ラベル作成（層2から始まる: L2, L3, ...）
        # diff[i] は層 i+2 における差分 (sim(L_{i+1}, L_{i+2}) - sim(L_{i+2}, L_{i+3}))
        labels = [f"L{i+2}" for i in range(num_layers)]

        # 差分の範囲を計算（-2〜2が理論上の最大だが、実際はもっと小さい）
        max_abs = max(abs(differences.min().item()), abs(differences.max().item()), 0.5)
        vmin, vmax = -max_abs, max_abs

        sns.heatmap(
            diff_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",  # 0が白
            vmin=vmin,
            vmax=vmax,
            center=0,
            cbar_kws={"label": "Similarity Difference", "shrink": 0.8},
            ax=ax,
            xticklabels=labels,
            yticklabels=False,
            annot_kws={"size": 8},
        )

        ax.set_xlabel("Layer (sim(L_{i-1}→L_i) - sim(L_i→L_{i+1}))")
        ax.set_title(
            f"Adjacent Layer Similarity Difference\n{model_name} - {trait_name} - {layer_position}",
            fontsize=12,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_pdf:
            pdf_path = str(save_path).replace(".png", ".pdf")
            plt.savefig(pdf_path, dpi=300, bbox_inches="tight")

        plt.close()

        print(f"Saved adjacent layer difference (single row) to {save_path}")
        return str(save_path)

    def visualize_adjacent_layer_difference_wrapped(
        self,
        vector: torch.Tensor,
        model_name: str,
        trait_name: str,
        layer_position: str,
        save_path: Path,
        wrap_every: int = 10,
        save_pdf: bool = False,
    ) -> str:
        """隣接層間類似度の差分を指定層数ごとに折り返したヒートマップで可視化

        層iについて: sim(L_{i-1}, L_i) - sim(L_i, L_{i+1})

        Args:
            vector: [num_layers, hidden_dim] のベクトル
            model_name: モデル名
            trait_name: ペルソナ名
            layer_position: 層の位置
            save_path: 保存パス
            wrap_every: 何層ごとに折り返すか（default: 10）
            save_pdf: PDFでも保存するか

        Returns:
            保存パス
        """
        differences = self.compute_adjacent_layer_difference(vector)
        num_layers = len(differences)

        # 折り返し用に行数を計算
        num_rows = (num_layers + wrap_every - 1) // wrap_every

        # パディングして整形
        padded_length = num_rows * wrap_every
        padded_diffs = torch.full((padded_length,), float("nan"))
        padded_diffs[:num_layers] = differences

        # 2D行列に整形
        diff_matrix = padded_diffs.numpy().reshape(num_rows, wrap_every)

        # ラベル作成（層2から始まる）
        col_labels = []
        for row in range(num_rows):
            row_labels = []
            for col in range(wrap_every):
                idx = row * wrap_every + col
                if idx < num_layers:
                    row_labels.append(f"L{idx+2}")
                else:
                    row_labels.append("")
            if row == 0:
                col_labels = row_labels

        row_labels_y = [f"L{row * wrap_every + 2}-L{min((row + 1) * wrap_every + 1, num_layers + 1)}" for row in range(num_rows)]

        # 図のサイズを調整
        fig, ax = plt.subplots(figsize=(max(12, wrap_every * 1.2), num_rows * 1.5 + 1))

        # カスタムカラーマップ（NaNを灰色に）
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color="lightgray")

        # 差分の範囲を計算
        max_abs = max(abs(differences.min().item()), abs(differences.max().item()), 0.5)
        vmin, vmax = -max_abs, max_abs

        sns.heatmap(
            diff_matrix,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0,
            cbar_kws={"label": "Similarity Difference", "shrink": 0.6},
            ax=ax,
            xticklabels=col_labels,
            yticklabels=row_labels_y,
            annot_kws={"size": 9},
            mask=torch.isnan(padded_diffs.reshape(num_rows, wrap_every)).numpy(),
        )

        ax.set_xlabel("Layer (sim(L_{i-1}→L_i) - sim(L_i→L_{i+1}))")
        ax.set_ylabel("Layer Range")
        ax.set_title(
            f"Adjacent Layer Similarity Difference (Wrapped every {wrap_every} layers)\n{model_name} - {trait_name} - {layer_position}",
            fontsize=12,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_pdf:
            pdf_path = str(save_path).replace(".png", ".pdf")
            plt.savefig(pdf_path, dpi=300, bbox_inches="tight")

        plt.close()

        print(f"Saved adjacent layer difference (wrapped) to {save_path}")
        return str(save_path)

    def visualize_adjacent_layer_difference_lineplot(
        self,
        vector: torch.Tensor,
        model_name: str,
        trait_name: str,
        layer_position: str,
        save_path: Path,
        save_pdf: bool = False,
    ) -> str:
        """隣接層間類似度の差分を折れ線グラフで可視化

        層iについて: sim(L_{i-1}, L_i) - sim(L_i, L_{i+1})

        Args:
            vector: [num_layers, hidden_dim] のベクトル
            model_name: モデル名
            trait_name: ペルソナ名
            layer_position: 層の位置
            save_path: 保存パス
            save_pdf: PDFでも保存するか

        Returns:
            保存パス
        """
        differences = self.compute_adjacent_layer_difference(vector)
        num_layers = len(differences)
        # x軸は層番号（2から始まる）
        x = list(range(2, num_layers + 2))
        y = differences.numpy()

        fig, ax = plt.subplots(figsize=(14, 5))

        # 折れ線グラフ
        ax.plot(x, y, marker='o', linewidth=2, markersize=4, color='#2563eb', label='Similarity Difference')

        # 塗りつぶし（正の値は青、負の値は赤）
        ax.fill_between(x, y, 0, where=(y >= 0), alpha=0.3, color='#3b82f6', interpolate=True)
        ax.fill_between(x, y, 0, where=(y < 0), alpha=0.3, color='#ef4444', interpolate=True)

        # 0のライン
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

        # y軸の範囲（差分なので-2〜2が理論最大だが、実際はもっと小さい）
        max_abs = max(abs(y.min()), abs(y.max()), 0.5)
        ax.set_ylim(-max_abs * 1.2, max_abs * 1.2)

        # グリッド
        ax.grid(True, alpha=0.3, linestyle='-')
        ax.set_axisbelow(True)

        # ラベル
        ax.set_xlabel("Layer i (sim(L_{i-1}→L_i) - sim(L_i→L_{i+1}))", fontsize=11)
        ax.set_ylabel("Similarity Difference", fontsize=11)
        ax.set_title(
            f"Adjacent Layer Similarity Difference (Line Plot)\n{model_name} - {trait_name} - {layer_position}",
            fontsize=12,
        )

        # x軸のティック（5層ごとにラベル表示）
        tick_positions = list(range(2, num_layers + 2, 5))
        if num_layers + 1 not in tick_positions:
            tick_positions.append(num_layers + 1)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"L{i}" for i in tick_positions])

        # 統計情報を追加
        mean_diff = y.mean()
        std_diff = y.std()
        min_diff = y.min()
        max_diff = y.max()
        min_idx = y.argmin() + 2  # 層番号は2から始まる
        max_idx = y.argmax() + 2

        stats_text = f"Mean: {mean_diff:.3f} ± {std_diff:.3f}\nMin: {min_diff:.3f} (L{min_idx})\nMax: {max_diff:.3f} (L{max_idx})"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_pdf:
            pdf_path = str(save_path).replace(".png", ".pdf")
            plt.savefig(pdf_path, dpi=300, bbox_inches="tight")

        plt.close()

        print(f"Saved adjacent layer difference (line plot) to {save_path}")
        return str(save_path)

    def visualize_adjacent_layer_difference_enhanced_heatmap(
        self,
        vector: torch.Tensor,
        model_name: str,
        trait_name: str,
        layer_position: str,
        save_path: Path,
        wrap_every: int = 10,
        save_pdf: bool = False,
    ) -> str:
        """隣接層間類似度の差分を色強調したヒートマップで可視化

        層iについて: sim(L_{i-1}, L_i) - sim(L_i, L_{i+1})
        データの範囲に合わせてvmin/vmaxを調整し、コントラストを上げる

        Args:
            vector: [num_layers, hidden_dim] のベクトル
            model_name: モデル名
            trait_name: ペルソナ名
            layer_position: 層の位置
            save_path: 保存パス
            wrap_every: 何層ごとに折り返すか
            save_pdf: PDFでも保存するか

        Returns:
            保存パス
        """
        differences = self.compute_adjacent_layer_difference(vector)
        num_layers = len(differences)

        # 折り返し用に行数を計算
        num_rows = (num_layers + wrap_every - 1) // wrap_every

        # パディングして整形
        padded_length = num_rows * wrap_every
        padded_diffs = torch.full((padded_length,), float("nan"))
        padded_diffs[:num_layers] = differences

        # 2D行列に整形
        diff_matrix = padded_diffs.numpy().reshape(num_rows, wrap_every)

        # データに基づいて色範囲を調整（コントラスト強調）
        valid_diffs = differences.numpy()
        data_min = valid_diffs.min()
        data_max = valid_diffs.max()
        data_range = data_max - data_min

        # 範囲を少し広げてマージンを持たせる（10%のマージン）
        margin = max(data_range * 0.1, 0.05)
        vmin = data_min - margin
        vmax = data_max + margin

        # centerを0に固定（差分なので0が基準）
        center = 0

        # ラベル作成（層2から始まる）
        col_labels = []
        for row in range(num_rows):
            row_labels = []
            for col in range(wrap_every):
                idx = row * wrap_every + col
                if idx < num_layers:
                    row_labels.append(f"L{idx+2}")
                else:
                    row_labels.append("")
            if row == 0:
                col_labels = row_labels

        row_labels_y = [f"L{row * wrap_every + 2}-L{min((row + 1) * wrap_every + 1, num_layers + 1)}" for row in range(num_rows)]

        # 図のサイズを調整
        fig, ax = plt.subplots(figsize=(max(12, wrap_every * 1.2), num_rows * 1.5 + 1))

        # カスタムカラーマップ（NaNを灰色に）
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad(color="lightgray")

        sns.heatmap(
            diff_matrix,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=center,
            cbar_kws={"label": "Similarity Difference", "shrink": 0.6},
            ax=ax,
            xticklabels=col_labels,
            yticklabels=row_labels_y,
            annot_kws={"size": 9},
            mask=torch.isnan(padded_diffs.reshape(num_rows, wrap_every)).numpy(),
        )

        ax.set_xlabel("Layer (sim(L_{i-1}→L_i) - sim(L_i→L_{i+1}))")
        ax.set_ylabel("Layer Range")
        ax.set_title(
            f"Adjacent Layer Similarity Difference (Enhanced Contrast)\n{model_name} - {trait_name} - {layer_position}\nRange: [{vmin:.3f}, {vmax:.3f}]",
            fontsize=12,
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_pdf:
            pdf_path = str(save_path).replace(".png", ".pdf")
            plt.savefig(pdf_path, dpi=300, bbox_inches="tight")

        plt.close()

        print(f"Saved adjacent layer difference (enhanced heatmap) to {save_path}")
        return str(save_path)

    def analyze_adjacent_layer_difference(
        self,
        model_name: str,
        trait_name: str,
        vectors_dict: Dict[str, torch.Tensor],
        save_dir: Optional[Path] = None,
        wrap_every: int = 10,
        save_pdf: bool = False,
    ) -> Dict[str, Dict[str, str]]:
        """隣接層間の差分分析を実行

        Args:
            model_name: モデル名
            trait_name: ペルソナ名
            vectors_dict: {layer_position: vector}の辞書
            save_dir: 保存ディレクトリ
            wrap_every: 折り返し層数
            save_pdf: PDFでも保存するか

        Returns:
            各分析結果のファイルパス
        """
        if save_dir is None:
            safe_model_name = model_name.replace("/", "_")
            save_dir = (
                self.output_dir / "adjacent_layer_diff" / safe_model_name / trait_name
            )
        save_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        for layer_position, vector in vectors_dict.items():
            try:
                results[layer_position] = {}

                # 横長1行バージョン
                # single_row_path = save_dir / f"adjacent_diff_{layer_position}_single_row.png"
                # results[layer_position]["single_row"] = (
                #     self.visualize_adjacent_layer_difference_single_row(
                #         vector,
                #         model_name,
                #         trait_name,
                #         layer_position,
                #         single_row_path,
                #         save_pdf=save_pdf,
                #     )
                # )

                # 折り返しバージョン
                # wrapped_path = save_dir / f"adjacent_diff_{layer_position}_wrapped.png"
                # results[layer_position]["wrapped"] = (
                #     self.visualize_adjacent_layer_difference_wrapped(
                #         vector,
                #         model_name,
                #         trait_name,
                #         layer_position,
                #         wrapped_path,
                #         wrap_every=wrap_every,
                #         save_pdf=save_pdf,
                #     )
                # )

                # 折れ線グラフバージョン
                # lineplot_path = save_dir / f"adjacent_diff_{layer_position}_lineplot.png"
                # results[layer_position]["lineplot"] = (
                #     self.visualize_adjacent_layer_difference_lineplot(
                #         vector,
                #         model_name,
                #         trait_name,
                #         layer_position,
                #         lineplot_path,
                #         save_pdf=save_pdf,
                #     )
                # )

                # 色強調ヒートマップバージョン
                # enhanced_path = save_dir / f"adjacent_diff_{layer_position}_enhanced.png"
                # results[layer_position]["enhanced"] = (
                #     self.visualize_adjacent_layer_difference_enhanced_heatmap(
                #         vector,
                #         model_name,
                #         trait_name,
                #         layer_position,
                #         enhanced_path,
                #         wrap_every=wrap_every,
                #         save_pdf=save_pdf,
                #     )
                # )

            except Exception as e:
                print(f"  Warning: Failed to analyze {layer_position}: {e}")
                results[layer_position] = None

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze residual stream relationships of persona vectors"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # default="Qwen/Qwen2.5-7B-Instruct",
        default="meta-llama/Llama-3.1-8B-Instruct",
        # default="Qwen/Qwen3-30B-A3B-Instruct-2507",
        # default="google/gemma-3-27b-it",
        help="Model name. One of: Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument(
        "--persona_vectors_dir",
        type=str,
        default="data/persona_vectors",
        help="Directory containing persona vectors",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/layer_analysis",
        # default="data/layer_analysis_prompt_avg_diff",
        # default="data/layer_analysis_prompt_last_diff",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--vector_type",
        type=str,
        default="response_avg_diff",
        # default="prompt_avg_diff",
        # default="prompt_last_diff",
        choices=["response_avg_diff", "prompt_avg_diff", "prompt_last_diff"],
        help="Type of persona vector to analyze",
    )
    parser.add_argument(
        "--save_fig",
        type=bool,
        default=True,
        help="Save figures",
    )

    args = parser.parse_args()

    analyzer = CosineSimilarityAnalyzer(
        args.model_name, args.persona_vectors_dir, args.output_dir, args.save_fig
    )

    # Residual streamに必要なlayer positions
    layer_positions = [
        "attn_layernorm",  # Input to attention
        "mlp_layernorm",  # Input to MLP
        "attn_output",  # Output from attention
        "mlp_output",  # Output from MLP
    ]

    print(f"Loading vectors for residual stream analysis...")
    vectors_by_trait = analyzer.load_vectors_for_positions(
        vector_type=args.vector_type, layer_positions=layer_positions
    )

    print(f"\nAnalyzing residual stream for {len(vectors_by_trait)} traits...")
    for trait_name, vectors_dict in tqdm(vectors_by_trait.items()):
        # 必要なベクトルが全て揃っているかチェック
        if all(pos in vectors_dict for pos in layer_positions):
            # Residual stream分析
            analyzer.analyze_residual_stream(
                model_name=args.model_name,
                trait_name=trait_name,
                vectors_dict=vectors_dict,
                save_pdf=False,
            )

            # 隣接層間差分の分析
            analyzer.analyze_adjacent_layer_difference(
                model_name=args.model_name,
                trait_name=trait_name,
                vectors_dict=vectors_dict,
                wrap_every=10,
                save_pdf=False,
            )
        else:
            missing = [pos for pos in layer_positions if pos not in vectors_dict]
            print(f"  Skipping {trait_name}: missing {missing}")


if __name__ == "__main__":
    main()
