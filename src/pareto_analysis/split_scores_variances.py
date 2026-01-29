"""
split_scores_variances.py

Transform wide-format CSV to long-format with separate columns for:
- trait score (value) and its standard deviation
- coherency score and its standard deviation
"""

import os
import re

import numpy as np
import pandas as pd


def split_scores_variances(
    input_file: str = 'data/steering_position_plot/Qwen2.5-7B-Instruct/steering_position_comparison_qwen.csv',
    output_file: str = 'data/steering_position_plot/Qwen2.5-7B-Instruct/steering_position_comparison_qwen_formatted.csv',
):
    """
    Transform wide-format CSV to long-format with extracted scores.
    
    Args:
        input_file: Path to input CSV file (wide format)
        output_file: Path to output CSV file (long format)
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # === 1) Load CSV ===
    df = pd.read_csv(input_file)

    # === 2) Preprocessing: Fill empty cells and rename columns ===
    # First two columns may have omitted values (like Excel merged cells), 
    # so forward-fill from above
    df['Unnamed: 0'] = df['Unnamed: 0'].ffill()
    df['Unnamed: 1'] = df['Unnamed: 1'].ffill()

    # Rename to descriptive column names
    df = df.rename(columns={
        'Unnamed: 0': 'trait',
        'Unnamed: 1': 'module',
        'Unnamed: 2': 'steering_method'
    })

    # === 3) Convert to long format ===
    # Columns to keep as ID
    id_vars = ['trait', 'module', 'steering_method']
    # Value columns (0.5, 1, 1.5, ... etc.)
    value_vars = [c for c in df.columns if c not in id_vars]

    # Melt to long format. Variable name is 'multiplier'
    df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="multiplier", value_name="text")

    # === 4) Value extraction function ===
    num_re = r"([-+]?\d+(?:\.\d+)?)"
    # Regex pattern: "key: value +- std"
    general_pattern = re.compile(
        r"(\w+)\s*:\s*" + num_re + r"\s*\+\-\s*" + num_re, re.IGNORECASE
    )

    def parse_cell(cell):
        """Extract trait score and coherency score from cell text."""
        if pd.isna(cell):
            return pd.Series([np.nan, np.nan, np.nan, np.nan])
        s = str(cell).strip()
        if s == "":
            return pd.Series([np.nan, np.nan, np.nan, np.nan])

        matches = list(general_pattern.finditer(s))
        
        value = value_std = coherence = coherence_std = np.nan
        
        try:
            for match in matches:
                key = match.group(1).lower()
                val = float(match.group(2))
                std = float(match.group(3))
                
                if key == "coherence":
                    coherence = val
                    coherence_std = std
                else:
                    # Non-coherence values are treated as trait scores
                    value = val
                    value_std = std
        except Exception:
            pass
            
        return pd.Series([value, value_std, coherence, coherence_std])

    # === 5) Apply extraction ===
    df_melted[["value", "value_std", "coherence", "coherence_std"]] = df_melted["text"].apply(parse_cell)

    # === 6) Format and save ===
    # Remove original text column
    df_final = df_melted.drop(columns=["text"])

    # Drop rows where all parsed values are NaN (cells with no data in original CSV)
    df_final = df_final.dropna(subset=["value", "coherence"], how='all')

    # Convert multiplier to numeric for sorting
    df_final['multiplier'] = pd.to_numeric(df_final['multiplier'], errors='coerce')

    # Sort for readability
    df_final = df_final.sort_values(by=['trait', 'module', 'steering_method', 'multiplier'])

    # Output CSV
    df_final.to_csv(output_file, index=False)

    print(f"Done. Saved to {output_file}")
    print(df_final.head())


if __name__ == "__main__":
    from fire import Fire
    Fire(split_scores_variances)
