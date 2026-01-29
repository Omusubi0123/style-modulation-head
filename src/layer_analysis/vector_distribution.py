import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def load_all_vectors(
    persona_vectors_dir: str,
    trait_names: list[str],
    vector_type: str = "response_avg_diff",
    layer_position: str = "",
):
    vectors: dict[str, torch.Tensor] = {}
    for trait_name in trait_names:
        vector_file = (
            persona_vectors_dir / f"{trait_name}_{vector_type}_{layer_position}.pt"
        )
        trait_name = vector_file.stem.replace(f"_{vector_type}_{layer_position}", "")
        vector = torch.load(vector_file, weights_only=False, map_location="cpu")
        vectors[trait_name] = vector

    return vectors


def visualize_vector_distribution(
    vectors: dict[str, torch.Tensor],
    save_dir: str,
    trait_names: list[str],
    layer_position: str,
    plot_layer_numbers: list[int] = [0],
):
    save_dir.mkdir(parents=True, exist_ok=True)
    for plot_layer_number in plot_layer_numbers:
        plt.figure(figsize=(12, 10))
        for trait_name in trait_names:
            plt.plot(vectors[trait_name][plot_layer_number], label=trait_name)
        plt.legend()
        plt.title(f"Vector Distribution {layer_position}")
        plt.savefig(
            f"{save_dir}/vector_distribution_{layer_position}_{plot_layer_number}.png"
        )
        plt.close()


def plot_vector_difference_distribution(
    vectors: dict[str, torch.Tensor],
    save_dir: str,
    trait_names: list[str],
    layer_position: str,
    plot_layer_numbers: list[int] = [0],
):
    """plot_layer_numbersの隣接する層のベクトルの差をプロットする

    Args:
        vectors (dict[str, torch.Tensor]): ベクトルの辞書
        save_dir (str): 保存先ディレクトリ
        trait_names (list[str]): プロットするtrait名のリスト
        layer_position (str): 層の位置
        plot_layer_numbers (list[int], optional): プロットする層の番号のリスト. Defaults to [0].
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    for trait_name in trait_names:
        for idx in range(len(plot_layer_numbers)):
            plot_layer_number = plot_layer_numbers[idx]
            if idx >= len(plot_layer_numbers) - 1:
                continue
            plot_layer_number_next = plot_layer_numbers[idx + 1]

            vector_difference = (
                vectors[trait_name][plot_layer_number]
                - vectors[trait_name][plot_layer_number_next]
            )
            plt.figure(figsize=(12, 10))
            plt.plot(vector_difference, label=trait_name)
            plt.legend()
            plt.title(f"Vector Difference {layer_position}")
            plt.savefig(
                f"{save_dir}/vector_difference_{layer_position}_{plot_layer_number}-{plot_layer_number_next}.png"
            )


def main():
    parser = argparse.ArgumentParser(description="Analyze vector distribution")
    parser.add_argument(
        "--persona_vectors_dir",
        type=str,
        default="data/persona_vectors",
        help="Directory containing persona vectors",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/layer_analysis/vector_distribution",
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
    parser.add_argument(
        "--layer_position",
        type=str,
        default="attn_layernorm",
        help="Layer position to analyze",
    )
    parser.add_argument(
        "--plot_layer_numbers",
        type=list[int],
        default=[2, 11, 18, 19, 20],
        help="Layer numbers to plot",
    )
    parser.add_argument(
        "--trait_names",
        type=list[str],
        default=["evil"],
        help="Trait names to plot",
    )
    args = parser.parse_args()

    persona_vectors_dir = Path(args.persona_vectors_dir) / args.model_name
    output_dir = Path(args.output_dir) / args.model_name

    vectors = load_all_vectors(
        persona_vectors_dir=persona_vectors_dir,
        trait_names=args.trait_names,
        vector_type=args.vector_type,
        layer_position=args.layer_position,
    )
    # visualize_vector_distribution(
    #     vectors=vectors,
    #     save_dir=output_dir,
    #     trait_names=args.trait_names,
    #     layer_position=args.layer_position,
    #     plot_layer_numbers=args.plot_layer_numbers,
    # )
    plot_vector_difference_distribution(
        vectors=vectors,
        save_dir=output_dir,
        trait_names=args.trait_names,
        layer_position=args.layer_position,
        plot_layer_numbers=args.plot_layer_numbers,
    )


if __name__ == "__main__":
    main()
