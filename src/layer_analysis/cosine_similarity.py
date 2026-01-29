"""
Layer-wise Persona Vector Analysis
Analyzes relationships between persona vectors across different layers within models.
"""

import argparse
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
import torch
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from tqdm import tqdm


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

    def load_all_vectors(
        self, vector_type: str = "response_avg_diff", layer_position: str = ""
    ):
        """全モデル・全traitのpersona vectorを読み込む

        Args:
            vector_type: 'response_avg_diff', 'prompt_avg_diff', 'prompt_last_diff'
        """
        print(f"Loading layer-wise persona vectors from {self.persona_vectors_dir}...")

        # ディレクトリ構造: persona_vectors/{organization}/{model}/{trait}_{vector_type}.pt
        for vector_file in self.persona_vectors_dir.glob(
            f"*_{vector_type}_{layer_position}.pt"
        ):
            trait_name = vector_file.stem.replace(
                f"_{vector_type}_{layer_position}", ""
            )

            try:
                # Load vector: shape = [num_layers, hidden_dim]
                vector = torch.load(vector_file, weights_only=False, map_location="cpu")

                self.vectors[trait_name] = vector

            except Exception as e:
                print(f"Warning: Failed to load {vector_file}: {e}")
                continue

        print(f"Loaded {len(self.vectors)} vectors")
        print(f"Vectors: {self.vectors.keys()}")

    def analyze_single_vector_layers(
        self,
        model_name: str,
        trait_name: str,
        layer_position: str,
        save_dir: Optional[Path] = None,
        save_pdf: bool = False,
    ) -> Dict[str, str]:
        """1つのpersona vector（1つのモデル・1つのペルソナ）における各層の関係性を分析

        Args:
            model_name: モデル名
            trait_name: ペルソナ名
            layer_position: 層の位置
            save_dir: 保存ディレクトリ

        Returns:
            各分析結果のファイルパス
        """
        vector = self.vectors[trait_name]  # [num_layers, hidden_dim]

        if save_dir is None:
            safe_model_name = model_name.replace("/", "_")
            save_dir = self.output_dir / "single_vector" / safe_model_name / trait_name
        save_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # 1. Layer-to-layer cosine similarity
        try:
            results["cosine_similarity"] = self._visualize_layer_cosine_similarity(
                vector,
                model_name,
                trait_name,
                layer_position,
                save_dir / f"layer_cosine_similarity_{layer_position}.png",
                save_pdf,
            )
        except Exception as e:
            print(f"  Warning: Failed to create cosine similarity: {e}")
            results["cosine_similarity"] = None

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
        if Path(save_path).exists():
            print(f"  Skipping layer cosine similarity (already exists): {save_path}")
            return str(save_path)

        vectors_np = vector.numpy()
        similarity_matrix = cosine_similarity(vectors_np)
        num_layers = vector.shape[0]

        plt.figure(figsize=(12, 10))

        # 層番号を1-basedに変換（index 0 = layer 1）
        layer_labels = [str(i + 1) for i in range(num_layers)]

        # ヒートマップを描画（cosine類似度の範囲は-1〜1）
        ax = sns.heatmap(
            similarity_matrix,
            annot=False,  # テキストアノテーションを無効化
            cmap="RdYlBu_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            cbar_kws={"label": "Cosine Similarity"},
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

        plt.xlabel("Layer", fontsize=14)
        plt.ylabel("Layer", fontsize=14)
        plt.title(
            f"Layer-to-Layer Cosine Similarity\n{model_name} - {trait_name} - {layer_position}",
            fontsize=16,
            pad=15,
        )
        plt.tight_layout()
        if self.save_fig:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_pdf:
            save_path = str(save_path).replace(".png", ".pdf")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved layer cosine similarity to {save_path}")
        return str(save_path)

    def analyze_all(self):
        """全ての分析を実行"""
        for model_name in self.model_names:
            for trait_name in self.vectors[model_name].keys():
                print(f"\nAnalyzing: {model_name} - {trait_name}")
                self.analyze_single_vector_layers(model_name, trait_name, save_pdf=True)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze layer-wise relationships of persona vectors"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        # default="Qwen/Qwen2.5-7B-Instruct",
        # default="meta-llama/Llama-3.1-8B-Instruct",
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
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
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--vector_type",
        type=str,
        default="response_avg_diff",
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

    layer_positions = [
        "attn_layernorm",
        "attn",
        "attn_output",
        "mlp_layernorm",
        "mlp",
        "mlp_output",
    ]

    for layer_position in tqdm(layer_positions):
        analyzer.load_all_vectors(
            vector_type=args.vector_type, layer_position=layer_position
        )
        for trait_name in tqdm(analyzer.vectors.keys()):
            analyzer.analyze_single_vector_layers(
                model_name=args.model_name,
                trait_name=trait_name,
                layer_position=layer_position,
                save_pdf=False,
            )


if __name__ == "__main__":
    main()
