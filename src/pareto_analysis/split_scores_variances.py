import re
import numpy as np
import pandas as pd
import os


def split_scores_variances(
    input_file = '/work/gc64/c64096/persona_vectors/backend/data/steering_position_plot/Qwen2.5-7B-Instruct/steering_position_comparison_qwen.csv',
    output_file = '/work/gc64/c64096/persona_vectors/backend/data/steering_position_plot/Qwen2.5-7B-Instruct/steering_position_comparison_qwen_formatted.csv',
):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # === 1) CSVを読み込む ===
    # ヘッダー行をそのまま読み込みます
    df = pd.read_csv(input_file)

    # === 2) 前処理: 空白セルの穴埋めと列名変更 ===
    # 1列目と2列目は値が省略されている（Excelの結合セルのような）場合があるため、上の行から値をコピー(ffill)します
    df['Unnamed: 0'] = df['Unnamed: 0'].ffill()
    df['Unnamed: 1'] = df['Unnamed: 1'].ffill()

    # わかりやすい列名に変更
    df = df.rename(columns={
        'Unnamed: 0': 'trait',
        'Unnamed: 1': 'module',
        'Unnamed: 2': 'steering_method'
    })

    # === 3) 縦長形式(Long format)に変換 ===
    # IDとして残す列
    id_vars = ['trait', 'module', 'steering_method']
    # 値が入っている列（0.5, 1, 1.5, ... など）
    value_vars = [c for c in df.columns if c not in id_vars]

    # meltで縦持ちに変換。変数名は 'multiplier' とします
    df_melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="multiplier", value_name="text")

    # === 4) 値抽出用の関数 ===
    num_re = r"([-+]?\d+(?:\.\d+)?)"
    # 正規表現: "key: value +- std" のパターン
    general_pattern = re.compile(
        r"(\w+)\s*:\s*" + num_re + r"\s*\+\-\s*" + num_re, re.IGNORECASE
    )

    def parse_cell(cell):
        """セル内の文字列から traitスコア と coherenceスコア を抽出する"""
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
                    # coherence 以外は trait のスコアとして扱う（evilなど）
                    value = val
                    value_std = std
        except Exception:
            pass
            
        return pd.Series([value, value_std, coherence, coherence_std])

    # === 5) 抽出結果を追加 ===
    df_melted[["value", "value_std", "coherence", "coherence_std"]] = df_melted["text"].apply(parse_cell)

    # === 6) 整形と保存 ===
    # 元のテキスト列を削除
    df_final = df_melted.drop(columns=["text"])

    # 解析結果がすべてNaNの行（元のCSVでデータがなかったセル）を削除
    df_final = df_final.dropna(subset=["value", "coherence"], how='all')

    # multiplier（0.5, 1, ...）を数値型に変換してソートしやすくする
    df_final['multiplier'] = pd.to_numeric(df_final['multiplier'], errors='coerce')

    # 見やすくソート
    df_final = df_final.sort_values(by=['trait', 'module', 'steering_method', 'multiplier'])

    # CSV出力
    df_final.to_csv(output_file, index=False)

    print(f"✅ Done. Saved to {output_file}")
    print(df_final.head())


if __name__ == "__main__":
    from fire import Fire
    Fire(split_scores_variances)
