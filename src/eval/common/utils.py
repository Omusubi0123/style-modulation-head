"""
utils.py - Evaluation utility functions
"""

import pandas as pd


def a_or_an(word: str) -> str:
    """Helper function to determine English indefinite article"""
    return "an" if word[0].lower() in "aeiou" else "a"


def print_results(outputs: pd.DataFrame, trait: str) -> None:
    """Display evaluation results

    Args:
        outputs: DataFrame containing evaluation results
        trait: Evaluated trait name
    """
    for t in [trait, "coherence"]:
        print(f"{t}:  {outputs[t].mean():.2f} +- {outputs[t].std():.2f}")
