"""
utils.py - 評価用ユーティリティ関数
"""

import pandas as pd


def a_or_an(word: str) -> str:
    """英語の不定冠詞を決定するヘルパー関数"""
    return "an" if word[0].lower() in "aeiou" else "a"


def print_results(outputs: pd.DataFrame, trait: str) -> None:
    """評価結果を表示する

    Args:
        outputs: 評価結果のDataFrame
        trait: 評価した特性名
    """
    for t in [trait, "coherence"]:
        print(f"{t}:  {outputs[t].mean():.2f} +- {outputs[t].std():.2f}")

