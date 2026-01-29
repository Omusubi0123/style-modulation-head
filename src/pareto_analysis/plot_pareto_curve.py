"""
plot_pareto_curve.py

Steering位置比較実験の結果をPareto-Frontier風にプロットする

可視化:
- 横軸: trait スコア
- 縦軸: coherency スコア
- 各Steering位置を異なる色で表示
- 係数が小さい方から大きい方へ矢印で繋ぐ

オプション:
- 線形軸プロット（通常）
- 対数軸プロット（右上の差を顕著に表示）
"""

import os
from typing import List, Optional

import fire
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# 統一カラーマップ（他のプロットと統一）
MODULE_ORDER = ['mlp_residual', 'attn_residual', 'attn_output', 'head_cor', 'head_cor_anti']
MODULE_COLORS = {
    'mlp_residual': '#27ae60',      # 緑（より鮮やか）
    'attn_residual': '#2980b9',     # 青（より鮮やか）
    'attn_output': '#c0392b',       # 赤（より鮮やか）
    'head_cor': '#8e44ad',          # 紫（より鮮やか）
    'head_cor_anti': '#d35400',     # オレンジ（より鮮やか）
}

MODULE_LABELS = {
    'mlp_residual': 'MLP Residual',
    'attn_residual': 'Attn Residual',
    'attn_output': 'Attn Output',
    'head_cor': 'Head (Correlated)',
    'head_cor_anti': 'Head (Cor+Anti)',
}

METHOD_LABELS = {
    'neg_add': 'Negative + Add (Enhance)',
    'pos_add': 'Positive + Add (Enhance)',
    'pos_subtract': 'Positive + Subtract (Suppress)',
}


def draw_arrow_curve(
    ax,
    points: List[tuple],
    color: str,
    alpha: float = 0.7,
    use_log: bool = False,
):
    """
    点を結ぶ曲線を描画し、矢印を追加する
    
    Args:
        ax: matplotlib axes
        points: [(x1, y1), (x2, y2), ...] 順番に並んだ点のリスト
        color: 線の色
        alpha: 透明度
        use_log: 対数軸を使用しているか
    """
    if len(points) < 2:
        return
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    # 曲線を描画
    ax.plot(xs, ys, color=color, alpha=alpha, linestyle='-', zorder=1)
    
    # 矢印を追加（中間点ごとに）
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        
        if use_log:
            # 対数軸の場合は、対数スケールで中間点を計算
            if x1 > 0 and x2 > 0:
                mid_x = np.sqrt(x1 * x2)
            else:
                mid_x = (x1 + x2) / 2
            if y1 > 0 and y2 > 0:
                mid_y = np.sqrt(y1 * y2)
            else:
                mid_y = (y1 + y2) / 2
        else:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
        
        # 方向ベクトル
        dx = x2 - x1
        dy = y2 - y1
        
        # 長さが0でない場合のみ矢印を描画
        if dx != 0 or dy != 0:
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                if use_log:
                    # 対数軸の場合、矢印のスケールを調整
                    scale_factor = 0.15
                    dx_norm = dx * scale_factor
                    dy_norm = dy * scale_factor
                else:
                    # 正規化
                    dx_norm = dx / length * 2
                    dy_norm = dy / length * 2
                
                ax.annotate('', 
                    xy=(mid_x + dx_norm, mid_y + dy_norm),
                    xytext=(mid_x - dx_norm, mid_y - dy_norm),
                    arrowprops=dict(
                        arrowstyle='->',
                        color=color,
                        alpha=alpha,
                        lw=8.5,
                        mutation_scale=22
                    ),
                    zorder=2
                )


def plot_pareto_curve(
    input_file: str,
    output_dir: str,
    trait: str = None,
    steering_method: str = None,
    filter_modules: str = 'mlp_residual,attn_residual,attn_output,head_cor,head_cor_anti',
    figsize: tuple = (10, 8),
    show_coef_labels: bool = False,
    use_log_scale: bool = False,
    save_pdf: bool = True,
):
    """
    Pareto-Frontier風のプロットを作成
    
    Args:
        input_file: CSVファイルのパス（value, coherence列が必要）
        output_dir: 出力ディレクトリ
        trait: 対象trait（Noneの場合はCSVから取得）
        steering_method: 対象steering method (neg_add, pos_add, pos_subtract)
        filter_modules: カンマ区切りのモジュールリスト
        figsize: 図のサイズ
        show_coef_labels: 各点に係数ラベルを表示するか
        use_log_scale: 対数軸を使用するか（右上の差を顕著に表示）
        save_pdf: PDFも保存するか
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # データ読み込み
    df = pd.read_csv(input_file)
    
    # trait取得
    if trait is None:
        trait = df['trait'].iloc[0] if 'trait' in df.columns else 'unknown'
    
    # モジュールフィルタ
    if filter_modules:
        modules_filter = [m.strip() for m in filter_modules.split(',')]
        df = df[df['module'].isin(modules_filter)]
    
    # mul_h_div_s系を除外
    drop_modules = ['head_cor_mul_h_div_s', 'head_cor_anti_mul_h_div_s']
    df = df[~df['module'].isin(drop_modules)]
    
    if df.empty:
        print(f"No data to plot for trait={trait}")
        return
    
    # steering_methodリスト
    if steering_method:
        methods = [steering_method]
    else:
        methods = ['neg_add', 'pos_add', 'pos_subtract']
    
    for method in methods:
        method_df = df[df['steering_method'] == method].copy()
        
        if method_df.empty:
            print(f"No data for trait={trait}, method={method}")
            continue
        
        # プロット作成
        fig, ax = plt.subplots(figsize=figsize)
        
        # 背景色
        ax.set_facecolor('#f8f9fa')
        
        legend_handles = []
        
        # モジュール順でプロット
        ordered_modules = [m for m in MODULE_ORDER if m in method_df['module'].unique()]
        
        for module in ordered_modules:
            module_df = method_df[method_df['module'] == module].copy()
            
            if module_df.empty:
                continue
            
            # 係数でソート
            module_df = module_df.sort_values('multiplier')
            
            color = MODULE_COLORS.get(module, '#333333')
            label = MODULE_LABELS.get(module, module)
            
            # 各点の座標を取得
            coherences = module_df['coherence'].values
            values = module_df['value'].values
            multipliers = module_df['multiplier'].values
            
            if use_log_scale:
                # 対数軸用に変換: 100からの差分 + 小さなオフセット（0を避ける）
                epsilon = 0.1
                plot_x = np.maximum(100.01 - values, epsilon)
                plot_y = np.maximum(100.01 - coherences, epsilon)
            else:
                # 横軸: trait, 縦軸: coherency
                plot_x = values
                plot_y = coherences
            
            # 曲線と矢印を描画
            points = list(zip(plot_x, plot_y))
            draw_arrow_curve(ax, points, color=color, alpha=0.7, use_log=use_log_scale)
            
            # 全点をプロット
            for i, (px, py, mult, coh, val) in enumerate(zip(plot_x, plot_y, multipliers, coherences, values)):
                ax.scatter(px, py, color=color, s=160, alpha=0.85, 
                          edgecolors='white', zorder=3)
                
                if show_coef_labels:
                    offset = (5, 5) if not use_log_scale else (8, 8)
                    ax.annotate(f'{mult:.1f}', (px, py), 
                               textcoords="offset points", xytext=offset,
                               fontsize=8, alpha=0.8, fontweight='medium')
            
            # 凡例用ハンドル
            legend_handles.append(
                Line2D([0], [0], marker='o', color=color, markerfacecolor=color,
                      markersize=10, label=f'{label}', linestyle='-', 
                      markeredgecolor='white')
            )
        
        # グラフ装飾
        method_label = METHOD_LABELS.get(method, method)
        
        if use_log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('100 - Trait Score (log scale)')
            ax.set_ylabel('100 - Coherency Score (log scale)')
            
            # 軸の範囲設定（対数軸）
            ax.set_xlim(0.05, 110)
            ax.set_ylim(0.05, 110)
            
            # グリッド設定
            ax.grid(True, which='major', alpha=0.5, linestyle='-')
            ax.grid(True, which='minor', alpha=0.3, linestyle='--')
        else:
            ax.set_xlabel('Trait Score')
            ax.set_ylabel('Coherency Score')
            ax.set_xlim(-5, 105)
            ax.set_ylim(-5, 105)
            ax.grid(True, alpha=0.4, linestyle='--')
        
        # 凡例の追加
        if legend_handles:
            legend = ax.legend(handles=legend_handles, loc='best',
                              framealpha=0.95, edgecolor='#bdc3c7', fancybox=True,
                              shadow=False)
            legend.get_frame().set_linewidth(1.5)
        
        plt.tight_layout()
        
        # 保存
        if use_log_scale:
            output_path = os.path.join(output_dir, f'pareto_log_{trait}_{method}.png')
        else:
            output_path = os.path.join(output_dir, f'pareto_{trait}_{method}.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        if save_pdf:
            plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        print(f"Saved: {output_path}")


def plot_all_pareto(
    data_dir: str = 'data/steering_position_plot',
    output_base_dir: str = 'data/steering_position_plot',
    model: str = 'qwen',
    traits: str = 'evil,sycophantic,hallucinating',
    filter_modules: str = 'mlp_residual,attn_residual,attn_output,head_cor,head_cor_anti',
    show_coef_labels: bool = False,
    use_log_scale: bool = False,
    save_pdf: bool = True,
):
    """
    全ペルソナ・全steering methodのPareto曲線をプロット
    
    Args:
        data_dir: CSVデータのディレクトリ
        output_base_dir: 出力ベースディレクトリ
        model: モデル名 ('llama' or 'qwen')
        traits: カンマ区切りのtrait名リスト
        filter_modules: プロット対象のモジュール
        show_coef_labels: 係数ラベルを表示するか
        use_log_scale: 対数軸を使用するか
        save_pdf: PDFも保存するか
    """
    # モデル名のマッピング
    model_names = {
        'llama': 'Llama-3.1-8B-Instruct',
        'qwen': 'Qwen2.5-7B-Instruct',
    }
    
    model_prefixes = {
        'llama': 'llama',
        'qwen': 'qwen',
    }
    
    if model not in model_names:
        print(f"Error: Unknown model '{model}'. Use 'llama' or 'qwen'.")
        return
    
    model_name = model_names[model]
    model_prefix = model_prefixes[model]
    
    # traitsのパース
    if isinstance(traits, (list, tuple)):
        trait_list = list(traits)
    else:
        trait_list = [t.strip() for t in traits.split(',')]
    
    scale_label = "Log Scale" if use_log_scale else "Linear Scale"
    print(f"=== Generating Pareto Curves ({scale_label}) ===")
    print(f"Model: {model_name}")
    print(f"Traits: {trait_list}")
    print()
    
    for trait in trait_list:
        # 入力ファイルパス（F値なしのフォーマットCSV）
        input_file = os.path.join(
            data_dir, 
            model_name, 
            f'steering_position_comparison_{model_prefix}_{trait}_formatted.csv'
        )
        
        # F値付きファイルがある場合はそちらを優先（互換性のため）
        fvalue_file = os.path.join(
            data_dir, 
            model_name, 
            f'steering_position_comparison_{model_prefix}_{trait}_fvalue.csv'
        )
        if os.path.exists(fvalue_file):
            input_file = fvalue_file
        
        if not os.path.exists(input_file):
            print(f"Warning: File not found: {input_file}")
            continue
        
        # 出力ディレクトリ（対数軸の場合は別フォルダ）
        if use_log_scale:
            output_dir = os.path.join(output_base_dir, model_name, 'pareto_plots_log')
        else:
            output_dir = os.path.join(output_base_dir, model_name, 'pareto_plots')
        
        print(f"Processing: {trait}")
        
        # 全methodをプロット
        plot_pareto_curve(
            input_file=input_file,
            output_dir=output_dir,
            trait=trait,
            steering_method=None,  # 全method
            filter_modules=filter_modules,
            show_coef_labels=show_coef_labels,
            use_log_scale=use_log_scale,
            save_pdf=save_pdf,
        )
    
    print()
    print(f"=== Completed ===")
    if use_log_scale:
        print(f"Output directory: {os.path.join(output_base_dir, model_name, 'pareto_plots_log')}")
    else:
        print(f"Output directory: {os.path.join(output_base_dir, model_name, 'pareto_plots')}")


def plot_single(
    input_file: str,
    output_dir: str,
    trait: str = None,
    steering_method: str = None,
    filter_modules: str = 'mlp_residual,attn_residual,attn_output,head_cor,head_cor_anti',
    show_coef_labels: bool = False,
    use_log_scale: bool = False,
    save_pdf: bool = True,
):
    """
    単一ファイルからPareto曲線をプロット（コマンドラインから使いやすいラッパー）
    """
    plot_pareto_curve(
        input_file=input_file,
        output_dir=output_dir,
        trait=trait,
        steering_method=steering_method,
        filter_modules=filter_modules,
        show_coef_labels=show_coef_labels,
        use_log_scale=use_log_scale,
        save_pdf=save_pdf,
    )


def plot_both_scales(
    data_dir: str = 'data/steering_position_plot',
    output_base_dir: str = 'data/steering_position_plot',
    model: str = 'qwen',
    traits: str = 'evil,sycophantic,hallucinating',
    filter_modules: str = 'mlp_residual,attn_residual,attn_output,head_cor,head_cor_anti',
    show_coef_labels: bool = False,
    save_pdf: bool = True,
):
    """
    線形軸と対数軸の両方でプロットを生成
    """
    print("=" * 60)
    print("Generating Linear Scale Plots")
    print("=" * 60)
    plot_all_pareto(
        data_dir=data_dir,
        output_base_dir=output_base_dir,
        model=model,
        traits=traits,
        filter_modules=filter_modules,
        show_coef_labels=show_coef_labels,
        use_log_scale=False,
        save_pdf=save_pdf,
    )
    
    print()
    print("=" * 60)
    print("Generating Log Scale Plots")
    print("=" * 60)
    plot_all_pareto(
        data_dir=data_dir,
        output_base_dir=output_base_dir,
        model=model,
        traits=traits,
        filter_modules=filter_modules,
        show_coef_labels=show_coef_labels,
        use_log_scale=True,
        save_pdf=save_pdf,
    )


if __name__ == '__main__':
    fire.Fire({
        'single': plot_single,
        'all': plot_all_pareto,
        'both': plot_both_scales,
    })
