"""
transform_log_to_csv.py

Parse experiment log files and transform to structured CSV format.
Extracts trait scores and coherency scores from steering position comparison logs.
"""

import os
import re

import pandas as pd

# ==========================================
# Configuration
# ==========================================

# CSV column headers (coefficients)
COLUMN_HEADERS = [
    "0.5",
    "1",
    "1.5",
    "2",
    "2.5",
    "3",
    "4",
    "5",
    "6",
    "8",
    "10",
    "12",
    "14",
    "16",
    "18",
    "20",
    "22",
    "24",
]

# Output order and display labels
# (internal ID, CSV display label)
CATEGORY_DEFINITIONS = [
    ("attn_residual", "attn_residual"),
    ("mlp_residual", "mlp_residual"),
    ("attn_output", "attn_output"),
    ("head_cor_normal", "head_cor"),
    ("head_cor_mul_h_div_s", "head_cor_mul_h_div_s"),
    ("head_cor_anti_normal", "head_cor_anti"),
    ("head_cor_anti_mul_h_div_s", "head_cor_anti_mul_h_div_s"),
]

SUB_TYPES_ORDER = ["neg_add", "pos_add", "pos_subtract"]

# ==========================================
# Parsing Logic
# ==========================================


def get_category_id(path):
    """Determine category ID from file path."""
    if "post_attention_residual" in path:
        return "attn_residual"
    elif "mlp_residual" in path:
        return "mlp_residual"
    elif "attention_output" in path:
        return "attn_output"

    # Head-related determination
    is_anti = "correlated_anti_heads" in path
    is_cor = "correlated_heads" in path

    # Check for mul_h_div_s or normal
    is_mul_h_div_s = "mul_h_div_s" in path
    is_normal = "normal" in path

    if is_anti:
        if is_mul_h_div_s:
            return "head_cor_anti_mul_h_div_s"
        elif is_normal:
            return "head_cor_anti_normal"
    elif is_cor:
        if is_mul_h_div_s:
            return "head_cor_mul_h_div_s"
        elif is_normal:
            return "head_cor_normal"

    return None


def parse_log_file(
    input_path: str,
    output_path: str,
    extract_trait: str = "evil",
):
    """
    Parse log file and transform to structured CSV.

    Args:
        input_path: Path to input log file
        output_path: Path to output CSV file
        extract_trait: Trait name to extract (e.g., "evil", "sycophantic")
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        log_text = f.read()

    # Initialize data storage dictionary
    # Key: (category_id, sub_type), Value: {col: text}
    data_store = {}
    for cat_id, _ in CATEGORY_DEFINITIONS:
        for sub in SUB_TYPES_ORDER:
            data_store[(cat_id, sub)] = {col: None for col in COLUMN_HEADERS}

    blocks = log_text.strip().split("\n\n")
    print(f"--- Starting parsing ({len(blocks)} blocks) ---")

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        path_line = lines[0]
        extract_trait_line = lines[1]
        coh_line = lines[2]

        # 1. Identify category
        cat_id = get_category_id(path_line)
        if not cat_id:
            continue

        # 2. Extract direction and coefficient
        match = re.search(f"{extract_trait}_(neg|pos)_coef\\s*(-?[\\d\\.]+)", path_line)
        if not match:
            continue

        direction = match.group(1)  # neg or pos
        coef_str = match.group(2)  # string "-3.0", "10.0" etc.
        try:
            coef_val = float(coef_str)
        except ValueError:
            continue

        # 3. Determine sub-type
        sub_type = None
        if direction == "neg":
            sub_type = "neg_add"
        elif direction == "pos":
            if coef_val < 0:
                sub_type = "pos_subtract"
            else:
                sub_type = "pos_add"

        # Determine column name (use absolute value)
        abs_coef = abs(coef_val)
        if abs_coef.is_integer():
            target_col = str(int(abs_coef))  # 10.0 -> "10"
        else:
            target_col = str(abs_coef)

        if target_col not in COLUMN_HEADERS:
            continue

        # 4. Store data
        cell_content = f"{extract_trait_line}\n{coh_line}"

        # Check for duplicates and store
        current_val = data_store[(cat_id, sub_type)][target_col]
        if current_val is None:
            data_store[(cat_id, sub_type)][target_col] = cell_content
        else:
            # If multiple data for the same cell, append
            data_store[(cat_id, sub_type)][target_col] = (
                current_val + "\n\n" + cell_content
            )

    # ==========================================
    # CSV Output
    # ==========================================
    rows = []
    is_very_first = True

    for cat_id, cat_label in CATEGORY_DEFINITIONS:
        is_comp_first = True
        for sub in SUB_TYPES_ORDER:
            row_data = {}

            # Level 0: trait (only for the very first row)
            row_data["Level0"] = extract_trait if is_very_first else None

            # Level 1: Category Label (only for first row of each category)
            row_data["Level1"] = cat_label if is_comp_first else None

            # Level 2: Sub Type
            row_data["Level2"] = sub

            # Data Columns
            for col in COLUMN_HEADERS:
                row_data[col] = data_store[(cat_id, sub)][col]

            rows.append(row_data)

            is_very_first = False
            is_comp_first = False

    df = pd.DataFrame(rows)

    # Set column headers to empty strings for visual alignment
    final_columns = ["", "", ""] + COLUMN_HEADERS
    df.columns = final_columns

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"--- Transformation complete: Created {output_path} ---")


if __name__ == "__main__":
    from fire import Fire

    Fire(parse_log_file)
