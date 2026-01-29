import pandas as pd
import re
import os

# ==========================================
# 設定
# ==========================================

# CSVの列（係数）の定義
COLUMN_HEADERS = ['0.5', '1', '1.5', '2', '2.5', '3', '4', '5', '6', '8', '10', '12', '14', '16', '18', '20', '22', '24']

# 出力順序と表示ラベルの定義
# (内部識別ID, CSV上の表示ラベル)
# ※ CSV上のラベルは添付ファイルに基づきますが、最後の項目は文脈に合わせて _anti を付与しています
CATEGORY_DEFINITIONS = [
    ('attn_residual', 'attn_residual'),
    ('mlp_residual', 'mlp_residual'),
    ('attn_output', 'attn_output'),
    ('head_cor_normal', 'head_cor'),
    ('head_cor_mul_h_div_s', 'head_cor_mul_h_div_s'),
    ('head_cor_anti_normal', 'head_cor_anti'),
    ('head_cor_anti_mul_h_div_s', 'head_cor_anti_mul_h_div_s') 
]

SUB_TYPES_ORDER = ['neg_add', 'pos_add', 'pos_subtract']

# ==========================================
# 解析ロジック
# ==========================================

def get_category_id(path):
    """ファイルパスからカテゴリIDを判定する"""
    if 'post_attention_residual' in path:
        return 'attn_residual'
    elif 'mlp_residual' in path:
        return 'mlp_residual'
    elif 'attention_output' in path:
        return 'attn_output'
    
    # Head関連の判定
    is_anti = 'correlated_anti_heads' in path
    is_cor = 'correlated_heads' in path
    
    # mul_s_div_h か div_h かの判定
    # mul_s_div_h は div_h を含むため、先に mul_s をチェックする
    is_mul_h_div_s = 'mul_h_div_s' in path
    is_normal = 'normal' in path
    
    if is_anti:
        if is_mul_h_div_s:
            return 'head_cor_anti_mul_h_div_s'
        elif is_normal:
            return 'head_cor_anti_normal'
    elif is_cor:
        if is_mul_h_div_s:
            return 'head_cor_mul_h_div_s'
        elif is_normal:
            return 'head_cor_normal'
            
    return None

def parse_log_file(
    input_path: str,
    output_path: str,
    extract_trait: str = "evil",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(input_path):
        print(f"エラー: 入力ファイル '{input_path}' が見つかりません。")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        log_text = f.read()

    # データ格納用辞書の初期化
    # キー: (category_id, sub_type), 値: {col: text}
    data_store = {}
    for cat_id, _ in CATEGORY_DEFINITIONS:
        for sub in SUB_TYPES_ORDER:
            data_store[(cat_id, sub)] = {col: None for col in COLUMN_HEADERS}

    blocks = log_text.strip().split('\n\n')
    print(f"--- 解析開始 ({len(blocks)} ブロック) ---")

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        path_line = lines[0]
        extract_trait_line = lines[1]
        coh_line = lines[2]

        # 1. カテゴリの特定
        cat_id = get_category_id(path_line)
        if not cat_id:
            # print(f"[スキップ] カテゴリ不明: {path_line}")
            continue

        # 2. 方向と係数の抽出
        match = re.search(f'{extract_trait}_(neg|pos)_coef\s*(-?[\d\.]+)', path_line)
        if not match:
            # print(f"[スキップ] 係数パターン不一致: {path_line}")
            continue

        direction = match.group(1)   # neg or pos
        coef_str = match.group(2)    # 文字列 "-3.0", "10.0" 等
        try:
            coef_val = float(coef_str)
        except ValueError:
            continue

        # 3. サブタイプの決定
        sub_type = None
        if direction == 'neg':
            sub_type = 'neg_add'
        elif direction == 'pos':
            if coef_val < 0:
                sub_type = 'pos_subtract'
            else:
                sub_type = 'pos_add'

        # 列名の決定（絶対値を使用）
        abs_coef = abs(coef_val)
        if abs_coef.is_integer():
            target_col = str(int(abs_coef)) # 10.0 -> "10"
        else:
            target_col = str(abs_coef)

        if target_col not in COLUMN_HEADERS:
            continue

        # 4. データの格納
        cell_content = f"{extract_trait_line}\n{coh_line}"
        
        # 重複チェックと格納
        current_val = data_store[(cat_id, sub_type)][target_col]
        if current_val is None:
            data_store[(cat_id, sub_type)][target_col] = cell_content
        else:
            # 万が一同じセルに複数のデータが来る場合は追記
            data_store[(cat_id, sub_type)][target_col] = current_val + "\n\n" + cell_content

    # ==========================================
    # CSV出力
    # ==========================================
    rows = []
    is_very_first = True

    for cat_id, cat_label in CATEGORY_DEFINITIONS:
        is_comp_first = True
        for sub in SUB_TYPES_ORDER:
            row_data = {}
            
            # Level 0: evil (全体の最初だけ)
            row_data['Level0'] = extract_trait if is_very_first else None
            
            # Level 1: Category Label (各カテゴリの最初だけ)
            row_data['Level1'] = cat_label if is_comp_first else None
            
            # Level 2: Sub Type
            row_data['Level2'] = sub
            
            # Data Columns
            for col in COLUMN_HEADERS:
                row_data[col] = data_store[(cat_id, sub)][col]
            
            rows.append(row_data)
            
            is_very_first = False
            is_comp_first = False

    df = pd.DataFrame(rows)
    
    # ヘッダーを空文字にしてCSVの見た目を調整
    final_columns = ['', '', ''] + COLUMN_HEADERS
    df.columns = final_columns

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"--- 変換完了: {output_path} を作成しました ---")


if __name__ == '__main__':
    from fire import Fire
    Fire(parse_log_file)
