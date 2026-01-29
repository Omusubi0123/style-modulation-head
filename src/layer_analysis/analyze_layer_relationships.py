"""
Layer-wise Persona Vector Analysis
Analyzes relationships between persona vectors across different layers within models.
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # type: ignore
import torch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA  # type: ignore
from sklearn.manifold import TSNE  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore


class LayerRelationshipAnalyzer:
    """各層のPersona Vector間の関係性を分析・可視化するクラス"""

    def __init__(self, persona_vectors_dir: str, output_dir: str, save_fig: bool):
        """
        Args:
            persona_vectors_dir: persona vectorsが保存されているディレクトリ
            output_dir: 分析結果を保存するディレクトリ
        """
        self.persona_vectors_dir = Path(persona_vectors_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_fig = save_fig

        # モデルとtraitごとのvectorを格納（全層）
        self.layer_vectors: Dict[str, Dict[str, torch.Tensor]] = {}
        self.model_names: List[str] = []
        self.trait_names: List[str] = []

    def load_all_vectors(self, vector_type: str = "response_avg_diff"):
        """全モデル・全traitのpersona vectorを読み込む（全層）

        Args:
            vector_type: 'response_avg_diff', 'prompt_avg_diff', 'prompt_last_diff'
        """
        print(f"Loading layer-wise persona vectors from {self.persona_vectors_dir}...")

        # ディレクトリ構造: persona_vectors/{organization}/{model}/{trait}_{vector_type}.pt
        for org_dir in self.persona_vectors_dir.iterdir():
            if not org_dir.is_dir():
                continue

            for model_dir in org_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                model_name = f"{org_dir.name}/{model_dir.name}"
                self.layer_vectors[model_name] = {}

                for vector_file in model_dir.glob(f"*_{vector_type}.pt"):
                    trait_name = vector_file.stem.replace(f"_{vector_type}", "")

                    try:
                        # Load vector: shape = [num_layers, hidden_dim]
                        vector = torch.load(
                            vector_file, weights_only=False, map_location="cpu"
                        )

                        self.layer_vectors[model_name][trait_name] = vector

                        if trait_name not in self.trait_names:
                            self.trait_names.append(trait_name)

                    except Exception as e:
                        print(f"Warning: Failed to load {vector_file}: {e}")
                        continue

                if self.layer_vectors[model_name]:
                    self.model_names.append(model_name)

        self.trait_names.sort()
        print(
            f"\nLoaded {len(self.model_names)} models with {len(self.trait_names)} traits"
        )
        print(f"Models: {self.model_names}")
        print(f"Traits: {self.trait_names}")

    def analyze_single_vector_layers(
        self,
        model_name: str,
        trait_name: str,
        save_dir: Optional[Path] = None,
        save_pdf: bool = False,
    ) -> Dict[str, str]:
        """1つのpersona vector（1つのモデル・1つのペルソナ）における各層の関係性を分析

        Args:
            model_name: モデル名
            trait_name: ペルソナ名
            save_dir: 保存ディレクトリ

        Returns:
            各分析結果のファイルパス
        """
        if model_name not in self.layer_vectors:
            raise ValueError(f"Model {model_name} not found")
        if trait_name not in self.layer_vectors[model_name]:
            raise ValueError(f"Trait {trait_name} not found in {model_name}")

        vector = self.layer_vectors[model_name][trait_name]  # [num_layers, hidden_dim]
        num_layers = vector.shape[0]

        if save_dir is None:
            safe_model_name = model_name.replace("/", "_")
            save_dir = self.output_dir / "single_vector" / safe_model_name / trait_name
        save_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # 1. Layer-wise norm
        try:
            results["norm"] = self._visualize_layer_norms(
                vector, model_name, trait_name, save_dir / "layer_norms.png"
            )
        except Exception as e:
            print(f"  Warning: Failed to create layer norms: {e}")
            results["norm"] = None

        # 2. Layer-to-layer cosine similarity
        try:
            results["cosine_similarity"] = self._visualize_layer_cosine_similarity(
                vector,
                model_name,
                trait_name,
                save_dir / "layer_cosine_similarity.png",
                save_pdf,
            )
        except Exception as e:
            print(f"  Warning: Failed to create cosine similarity: {e}")
            results["cosine_similarity"] = None

        # 3. PCA of all layers
        try:
            results["pca"] = self._visualize_layer_pca(
                vector, model_name, trait_name, save_dir / "layer_pca.png"
            )
        except Exception as e:
            print(f"  Warning: Failed to create PCA: {e}")
            results["pca"] = None

        # 4. t-SNE of all layers
        try:
            results["tsne"] = self._visualize_layer_tsne(
                vector, model_name, trait_name, save_dir / "layer_tsne.png"
            )
        except Exception as e:
            print(f"  Warning: Failed to create t-SNE: {e}")
            results["tsne"] = None

        return results

    def _visualize_layer_norms(
        self, vector: torch.Tensor, model_name: str, trait_name: str, save_path: Path
    ) -> str:
        """各層のベクトルのノルムを可視化"""
        # 既に画像が存在する場合はスキップ
        if Path(save_path).exists():
            print(f"  Skipping layer norms (already exists): {save_path}")
            return str(save_path)

        norms = torch.norm(vector, dim=1).numpy()
        num_layers = len(norms)
        layers = np.arange(num_layers)

        # 中間層を強調表示するための色
        mid_layer = num_layers // 2
        colors = ["#1f77b4"] * num_layers
        colors[mid_layer] = "#ff7f0e"  # 中間層をオレンジ色に

        plt.figure(figsize=(14, 8))
        bars = plt.bar(
            layers, norms, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5
        )

        # 中間層にラベルを追加
        bars[mid_layer].set_edgecolor("red")
        bars[mid_layer].set_linewidth(3)

        plt.xlabel("Layer", fontsize=16)
        plt.ylabel("L2 Norm", fontsize=16)
        plt.title(
            f"Layer-wise Vector Norms\n{model_name} - {trait_name}", fontsize=18, pad=15
        )
        plt.grid(True, alpha=0.3, axis="y")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # 中間層に注釈
        plt.annotate(
            f"Middle Layer ({mid_layer})",
            xy=(mid_layer, norms[mid_layer]),
            xytext=(mid_layer, norms[mid_layer] * 1.15),
            fontsize=14,
            ha="center",
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="red", lw=2),
        )

        plt.tight_layout()
        if self.save_fig:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved layer norms to {save_path}")
        return str(save_path)

    def _visualize_layer_cosine_similarity(
        self,
        vector: torch.Tensor,
        model_name: str,
        trait_name: str,
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
            f"Layer-to-Layer Cosine Similarity\n{model_name} - {trait_name}",
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

    def _visualize_layer_pca(
        self, vector: torch.Tensor, model_name: str, trait_name: str, save_path: Path
    ) -> str:
        """各層のベクトルをPCAで2次元にマッピング"""
        # 既に画像が存在する場合はスキップ
        if Path(save_path).exists():
            print(f"  Skipping layer PCA (already exists): {save_path}")
            return str(save_path)

        vectors_np = vector.numpy()
        num_layers = vectors_np.shape[0]

        # Check if vectors have sufficient variance
        var = np.var(vectors_np, axis=0)
        if np.all(var < 1e-10):
            print(
                f"  Warning: All layers have identical or near-identical vectors, skipping PCA"
            )
            raise ValueError("Insufficient variance for PCA")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pca = PCA(n_components=2)
            vectors_2d = pca.fit_transform(vectors_np)

        # 層の深さに応じて色の濃さを変える（浅い層=薄い、深い層=濃い）
        colors = plt.cm.viridis(np.linspace(0, 1, num_layers))

        plt.figure(figsize=(14, 10))

        # 各層をプロット
        for i in range(num_layers):
            size = 200 if i == num_layers // 2 else 100  # 中間層を大きく
            marker = "s" if i == num_layers // 2 else "o"  # 中間層を四角に
            alpha = 0.9 if i == num_layers // 2 else 0.6

            plt.scatter(
                vectors_2d[i, 0],
                vectors_2d[i, 1],
                s=size,
                alpha=alpha,
                color=colors[i],
                marker=marker,
                edgecolors="red" if i == num_layers // 2 else "black",
                linewidths=3 if i == num_layers // 2 else 1,
            )

            # 層番号のラベル
            plt.annotate(
                str(i),
                (vectors_2d[i, 0], vectors_2d[i, 1]),
                fontsize=10 if i == num_layers // 2 else 8,
                ha="center",
                va="center",
                fontweight="bold" if i == num_layers // 2 else "normal",
                color="white",
            )

        # カラーバーを追加
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=num_layers - 1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label("Layer Depth", fontsize=14)

        plt.xlabel(
            f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=16
        )
        plt.ylabel(
            f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=16
        )
        plt.title(
            f"Layer-wise PCA Mapping\n{model_name} - {trait_name}", fontsize=18, pad=15
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if self.save_fig:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved layer PCA to {save_path}")
        return str(save_path)

    def _visualize_layer_tsne(
        self, vector: torch.Tensor, model_name: str, trait_name: str, save_path: Path
    ) -> str:
        """各層のベクトルをt-SNEで2次元にマッピング"""
        # 既に画像が存在する場合はスキップ
        if Path(save_path).exists():
            print(f"  Skipping layer t-SNE (already exists): {save_path}")
            return str(save_path)

        vectors_np = vector.numpy()
        num_layers = vectors_np.shape[0]

        # Check if vectors have sufficient variance
        var = np.var(vectors_np, axis=0)
        if np.all(var < 1e-10):
            print(
                f"  Warning: All layers have identical or near-identical vectors, skipping t-SNE"
            )
            raise ValueError("Insufficient variance for t-SNE")

        # t-SNEのperplexityを調整
        perplexity = min(30, num_layers - 1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            vectors_2d = tsne.fit_transform(vectors_np)

        # 層の深さに応じて色の濃さを変える
        colors = plt.cm.plasma(np.linspace(0, 1, num_layers))

        plt.figure(figsize=(14, 10))

        # 各層をプロット
        for i in range(num_layers):
            size = 200 if i == num_layers // 2 else 100  # 中間層を大きく
            marker = "s" if i == num_layers // 2 else "o"  # 中間層を四角に
            alpha = 0.9 if i == num_layers // 2 else 0.6

            plt.scatter(
                vectors_2d[i, 0],
                vectors_2d[i, 1],
                s=size,
                alpha=alpha,
                color=colors[i],
                marker=marker,
                edgecolors="red" if i == num_layers // 2 else "black",
                linewidths=3 if i == num_layers // 2 else 1,
            )

            # 層番号のラベル
            plt.annotate(
                str(i),
                (vectors_2d[i, 0], vectors_2d[i, 1]),
                fontsize=10 if i == num_layers // 2 else 8,
                ha="center",
                va="center",
                fontweight="bold" if i == num_layers // 2 else "normal",
                color="white",
            )

        # カラーバーを追加
        sm = plt.cm.ScalarMappable(
            cmap=plt.cm.plasma, norm=plt.Normalize(vmin=0, vmax=num_layers - 1)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label("Layer Depth", fontsize=14)

        plt.xlabel("t-SNE Dimension 1", fontsize=16)
        plt.ylabel("t-SNE Dimension 2", fontsize=16)
        plt.title(
            f"Layer-wise t-SNE Mapping\n{model_name} - {trait_name}",
            fontsize=18,
            pad=15,
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if self.save_fig:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved layer t-SNE to {save_path}")
        return str(save_path)

    def analyze_model_all_personas_layers(
        self, model_name: str, save_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """同じモデルの全ペルソナの全層を1つのグラフにマッピング

        Args:
            model_name: モデル名
            save_dir: 保存ディレクトリ

        Returns:
            各分析結果のファイルパス
        """
        if model_name not in self.layer_vectors:
            raise ValueError(f"Model {model_name} not found")

        if save_dir is None:
            safe_model_name = model_name.replace("/", "_")
            save_dir = self.output_dir / "model_all_personas" / safe_model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # t-SNE: 全ペルソナ・全層を1つのグラフに
        try:
            results["tsne"] = self._visualize_model_all_personas_tsne(
                model_name, save_dir / "all_personas_layers_tsne.png"
            )
        except Exception as e:
            print(f"  Warning: Failed to create all personas t-SNE: {e}")
            results["tsne"] = None

        # Norm comparison: 全ペルソナの層ごとのノルム
        try:
            results["norms"] = self._visualize_model_all_personas_norms(
                model_name, save_dir / "all_personas_layer_norms.png"
            )
        except Exception as e:
            print(f"  Warning: Failed to create all personas norms: {e}")
            results["norms"] = None

        return results

    def _visualize_model_all_personas_tsne(
        self, model_name: str, save_path: Path
    ) -> str:
        """同じモデルの全ペルソナの全層をt-SNEで1つのグラフにマッピング"""
        # 既に画像が存在する場合はスキップ
        if Path(save_path).exists():
            print(f"  Skipping all personas t-SNE (already exists): {save_path}")
            return str(save_path)

        trait_vectors = self.layer_vectors[model_name]
        traits = sorted(trait_vectors.keys())

        # 全ペルソナ・全層のベクトルを収集
        all_vectors = []
        labels = []

        for trait in traits:
            vector = trait_vectors[trait]  # [num_layers, hidden_dim]
            num_layers = vector.shape[0]

            for layer_idx in range(num_layers):
                all_vectors.append(vector[layer_idx].numpy())
                labels.append((trait, layer_idx, num_layers))

        all_vectors = np.array(all_vectors)

        # Check if vectors have sufficient variance
        var = np.var(all_vectors, axis=0)
        if np.all(var < 1e-10):
            print(f"  Warning: All vectors have identical values, skipping t-SNE")
            raise ValueError("Insufficient variance for t-SNE")

        # t-SNE
        # perplexityはサンプル数より小さくする必要がある
        n_samples = len(all_vectors)
        perplexity = min(30, n_samples - 1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            vectors_2d = tsne.fit_transform(all_vectors)

        # ペルソナごとに異なる色を割り当て
        trait_colors = plt.cm.tab10(np.linspace(0, 1, len(traits)))
        if len(traits) > 10:
            trait_colors = plt.cm.tab20(np.linspace(0, 1, len(traits)))

        plt.figure(figsize=(18, 14))

        # 各ペルソナ・各層をプロット
        for i, (trait, layer_idx, num_layers) in enumerate(labels):
            trait_idx = traits.index(trait)
            color = trait_colors[trait_idx]

            # 層の深さに応じて透明度を変える
            alpha = 0.3 + 0.7 * (layer_idx / num_layers)

            # 中間層を強調
            mid_layer = num_layers // 2
            size = 150 if layer_idx == mid_layer else 50
            marker = "s" if layer_idx == mid_layer else "o"
            edgecolor = "red" if layer_idx == mid_layer else color
            linewidth = 2 if layer_idx == mid_layer else 0.5

            plt.scatter(
                vectors_2d[i, 0],
                vectors_2d[i, 1],
                s=size,
                alpha=alpha,
                color=color,
                marker=marker,
                edgecolors=edgecolor,
                linewidths=linewidth,
                label=trait if layer_idx == 0 else "",
            )

        plt.xlabel("t-SNE Dimension 1", fontsize=20)
        plt.ylabel("t-SNE Dimension 2", fontsize=20)
        plt.title(
            f"All Personas & Layers t-SNE Mapping\n{model_name}\n(Square markers = middle layers)",
            fontsize=24,
            pad=20,
        )
        plt.grid(True, alpha=0.3)
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16, framealpha=0.9
        )
        plt.tick_params(axis="both", which="major", labelsize=16)
        plt.tight_layout()
        if self.save_fig:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved all personas/layers t-SNE to {save_path}")
        return str(save_path)

    def _visualize_model_all_personas_norms(
        self, model_name: str, save_path: Path
    ) -> str:
        """同じモデルの全ペルソナの層ごとのノルムを可視化"""
        # 既に画像が存在する場合はスキップ
        if Path(save_path).exists():
            print(f"  Skipping all personas norms (already exists): {save_path}")
            return str(save_path)

        trait_vectors = self.layer_vectors[model_name]
        traits = sorted(trait_vectors.keys())

        plt.figure(figsize=(16, 10))

        # ペルソナごとに異なる色
        colors = plt.cm.tab10(np.linspace(0, 1, len(traits)))
        if len(traits) > 10:
            colors = plt.cm.tab20(np.linspace(0, 1, len(traits)))

        for i, trait in enumerate(traits):
            vector = trait_vectors[trait]
            norms = torch.norm(vector, dim=1).numpy()
            layers = np.arange(len(norms))

            plt.plot(
                layers,
                norms,
                label=trait,
                color=colors[i],
                linewidth=2,
                marker="o",
                markersize=6,
                alpha=0.8,
            )

        plt.xlabel("Layer", fontsize=18)
        plt.ylabel("L2 Norm", fontsize=18)
        plt.title(
            f"Layer-wise Norms Across All Personas\n{model_name}", fontsize=22, pad=15
        )
        plt.grid(True, alpha=0.3)
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=14, framealpha=0.9
        )
        plt.tick_params(axis="both", which="major", labelsize=14)
        plt.tight_layout()
        if self.save_fig:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved all personas norms to {save_path}")
        return str(save_path)

    def analyze_cross_model_persona(
        self, trait_name: str, save_dir: Optional[Path] = None
    ) -> Dict[str, str]:
        """異なるモデルの同じペルソナにおける層間関係を可視化

        Args:
            trait_name: ペルソナ名
            save_dir: 保存ディレクトリ

        Returns:
            各分析結果のファイルパス
        """
        if save_dir is None:
            save_dir = self.output_dir / "cross_model" / trait_name
        save_dir.mkdir(parents=True, exist_ok=True)

        results = {}

        # t-SNE: 異なるモデルの同じペルソナの全層を比較
        try:
            results["tsne"] = self._visualize_cross_model_tsne(
                trait_name, save_dir / "cross_model_layers_tsne.png"
            )
        except Exception as e:
            print(f"  Warning: Failed to create cross-model t-SNE: {e}")
            results["tsne"] = None

        # Norm comparison
        try:
            results["norms"] = self._visualize_cross_model_norms(
                trait_name, save_dir / "cross_model_layer_norms.png"
            )
        except Exception as e:
            print(f"  Warning: Failed to create cross-model norms: {e}")
            results["norms"] = None

        return results

    def _visualize_cross_model_tsne(self, trait_name: str, save_path: Path) -> str:
        """異なるモデルの同じペルソナの全層をt-SNEで可視化"""
        # 既に画像が存在する場合はスキップ
        if Path(save_path).exists():
            print(f"  Skipping cross-model t-SNE (already exists): {save_path}")
            return str(save_path)

        # 対象のペルソナを持つモデルを収集
        models_with_trait = [
            model
            for model in self.model_names
            if trait_name in self.layer_vectors[model]
        ]

        if len(models_with_trait) == 0:
            raise ValueError(f"No models found with trait {trait_name}")

        # 全モデル・全層のベクトルを収集
        all_vectors = []
        labels = []

        for model in models_with_trait:
            vector = self.layer_vectors[model][trait_name]
            num_layers = vector.shape[0]

            for layer_idx in range(num_layers):
                all_vectors.append(vector[layer_idx].numpy())
                labels.append((model, layer_idx, num_layers))

        all_vectors = np.array(all_vectors)

        # Check if vectors have sufficient variance
        var = np.var(all_vectors, axis=0)
        if np.all(var < 1e-10):
            print(f"  Warning: All vectors have identical values, skipping t-SNE")
            raise ValueError("Insufficient variance for t-SNE")

        # t-SNE
        # perplexityはサンプル数より小さくする必要がある
        n_samples = len(all_vectors)
        perplexity = min(30, n_samples - 1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            vectors_2d = tsne.fit_transform(all_vectors)

        # モデルごとに異なる色
        model_colors = plt.cm.tab10(np.linspace(0, 1, len(models_with_trait)))
        if len(models_with_trait) > 10:
            model_colors = plt.cm.tab20(np.linspace(0, 1, len(models_with_trait)))

        plt.figure(figsize=(18, 14))

        # 各モデル・各層をプロット
        for i, (model, layer_idx, num_layers) in enumerate(labels):
            model_idx = models_with_trait.index(model)
            color = model_colors[model_idx]

            # 層の深さに応じて透明度を変える
            alpha = 0.3 + 0.7 * (layer_idx / num_layers)

            # 中間層を強調
            mid_layer = num_layers // 2
            size = 150 if layer_idx == mid_layer else 50
            marker = "s" if layer_idx == mid_layer else "o"
            edgecolor = "red" if layer_idx == mid_layer else color
            linewidth = 2 if layer_idx == mid_layer else 0.5

            plt.scatter(
                vectors_2d[i, 0],
                vectors_2d[i, 1],
                s=size,
                alpha=alpha,
                color=color,
                marker=marker,
                edgecolors=edgecolor,
                linewidths=linewidth,
                label=model if layer_idx == 0 else "",
            )

        plt.xlabel("t-SNE Dimension 1", fontsize=20)
        plt.ylabel("t-SNE Dimension 2", fontsize=20)
        plt.title(
            f"Cross-Model Layer Comparison: {trait_name}\n(Square markers = middle layers)",
            fontsize=24,
            pad=20,
        )
        plt.grid(True, alpha=0.3)
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=14, framealpha=0.9
        )
        plt.tick_params(axis="both", which="major", labelsize=16)
        plt.tight_layout()
        if self.save_fig:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved cross-model t-SNE to {save_path}")
        return str(save_path)

    def _visualize_cross_model_norms(self, trait_name: str, save_path: Path) -> str:
        """異なるモデルの同じペルソナの層ごとのノルムを可視化"""
        # 既に画像が存在する場合はスキップ
        if Path(save_path).exists():
            print(f"  Skipping cross-model norms (already exists): {save_path}")
            return str(save_path)

        models_with_trait = [
            model
            for model in self.model_names
            if trait_name in self.layer_vectors[model]
        ]

        plt.figure(figsize=(16, 10))

        # モデルごとに異なる色
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_with_trait)))
        if len(models_with_trait) > 10:
            colors = plt.cm.tab20(np.linspace(0, 1, len(models_with_trait)))

        for i, model in enumerate(models_with_trait):
            vector = self.layer_vectors[model][trait_name]
            norms = torch.norm(vector, dim=1).numpy()
            layers = np.arange(len(norms))

            plt.plot(
                layers,
                norms,
                label=model,
                color=colors[i],
                linewidth=2,
                marker="o",
                markersize=6,
                alpha=0.8,
            )

        plt.xlabel("Layer", fontsize=18)
        plt.ylabel("L2 Norm", fontsize=18)
        plt.title(f"Cross-Model Layer-wise Norms: {trait_name}", fontsize=22, pad=15)
        plt.grid(True, alpha=0.3)
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=14, framealpha=0.9
        )
        plt.tick_params(axis="both", which="major", labelsize=14)
        plt.tight_layout()
        if self.save_fig:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved cross-model norms to {save_path}")
        return str(save_path)

    def analyze_all(self) -> Dict:
        """全ての分析を実行

        Returns:
            分析結果のサマリ
        """
        results = {"single_vector": {}, "model_all_personas": {}, "cross_model": {}}

        # 1. Single vector analysis (各モデル・各ペルソナ)
        print("\n=== Analyzing individual persona vectors (layer relationships) ===")
        for model_name in self.model_names:
            results["single_vector"][model_name] = {}
            for trait_name in self.layer_vectors[model_name].keys():
                print(f"\nAnalyzing: {model_name} - {trait_name}")
                try:
                    analysis_results = self.analyze_single_vector_layers(
                        model_name, trait_name, save_pdf=True
                    )
                    results["single_vector"][model_name][trait_name] = analysis_results
                except Exception as e:
                    print(f"  Failed: {e}")
                    results["single_vector"][model_name][trait_name] = None

        # 2. Model all personas analysis
        print("\n=== Analyzing all personas within each model ===")
        for model_name in self.model_names:
            print(f"\nAnalyzing model: {model_name}")
            try:
                analysis_results = self.analyze_model_all_personas_layers(model_name)
                results["model_all_personas"][model_name] = analysis_results
            except Exception as e:
                print(f"  Failed: {e}")
                results["model_all_personas"][model_name] = None

        # 3. Cross-model analysis (各ペルソナ)
        print("\n=== Analyzing same persona across different models ===")
        for trait_name in self.trait_names:
            print(f"\nAnalyzing trait: {trait_name}")
            try:
                analysis_results = self.analyze_cross_model_persona(trait_name)
                results["cross_model"][trait_name] = analysis_results
            except Exception as e:
                print(f"  Failed: {e}")
                results["cross_model"][trait_name] = None

        return results

    def save_analysis_summary(
        self, results: Dict, filename: str = "layer_analysis_summary.json"
    ):
        """分析結果のサマリをJSONファイルに保存

        Args:
            results: analyze_allの結果
            filename: 出力ファイル名
        """
        summary_path = self.output_dir / filename

        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, (str, Path)):
                return str(obj)
            elif obj is None:
                return None
            else:
                return obj

        json_results = convert_paths(results)

        with open(summary_path, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"\nLayer analysis summary saved to {summary_path}")
        return str(summary_path)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze layer-wise relationships of persona vectors"
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

    # Create analyzer
    analyzer = LayerRelationshipAnalyzer(
        args.persona_vectors_dir, args.output_dir, args.save_fig
    )

    # Load vectors
    analyzer.load_all_vectors(vector_type=args.vector_type)

    # Run all analyses
    results = analyzer.analyze_all()

    # Save summary
    analyzer.save_analysis_summary(results)

    print("\n=== Layer Analysis Complete ===")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
