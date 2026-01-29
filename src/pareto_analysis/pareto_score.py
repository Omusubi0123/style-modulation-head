"""
pareto_score.py

Pareto フロントの評価指標: Right-normalized Constrained Envelope Area

多目的最適化において、離散点の個数に依存しない Pareto フロント評価を行う。

評価指標:
    Score_τ(P) = (1 / (x_max_common - τ)) * ∫_τ^{x_max_common} y_P(x) dx

ここで y_P(x) は Pareto フロントの envelope であり、
    upper: y_P(x) = max { y | (x', y) ∈ P, x' ≥ x }
    lower: y_P(x) = min { y | (x', y) ∈ P, x' ≤ x }
と定義される。

この指標は x 軸（coherency）に対する制約 τ を課した下で、
y 軸（trait score）の平均的な達成度を測定する。
"""

from typing import List, Tuple

import numpy as np


def build_upper_envelope(
    pareto_points: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """
    Pareto フロントの upper envelope を構築する。
    
    Upper envelope の定義:
        y_P(x) = max { y | (x', y) ∈ P, x' ≥ x }
    
    これは x を降順にソートして右から左へ走査し、
    y の累積最大値を取ることで構築される。
    
    結果は区分定数関数（step function）を表現する点群となる。
    x_i から x_{i+1} の間は y_i で一定。
    
    Args:
        pareto_points: Pareto フロントの点群 [(x1, y1), (x2, y2), ...]
                      x: coherency, y: trait score
    
    Returns:
        Upper envelope を構成する点群（x 降順でソート済み）
        各点 (x_i, y_i) は x_i 以上の全ての x で達成可能な最大 y を表す
        y 値が変化する点のみを返す（step function の角点）
    
    Note:
        - 入力点が空の場合は空リストを返す
        - 同一 x 値に複数の点がある場合は最大 y を採用
    """
    if not pareto_points:
        return []
    
    # x で降順にソート（右から左へ走査するため）
    sorted_points = sorted(pareto_points, key=lambda p: -p[0])
    
    # 累積最大値を計算
    # upper envelope は step function なので、y が変化する点のみを記録
    envelope = []
    current_max_y = float('-inf')
    
    for x, y in sorted_points:
        if y > current_max_y:
            # y が新しい最大値になった場合のみ点を追加
            # これにより step function の角点を記録
            envelope.append((x, y))
            current_max_y = y
    
    # envelope は x 降順のまま
    # 最初の点が最大 x、最後の点が最小 x（ただし y が増加した点のみ）
    
    return envelope


def integrate_envelope_step_function(
    envelope: List[Tuple[float, float]],
    tau: float,
    x_max: float,
    data_x_min: float = None,
    data_x_max: float = None,
    extrapolation_mode: str = "boundary",
    envelope_type: str = "lower",
) -> Tuple[float, float]:
    """
    Upper envelope を step function として [τ, x_max] の範囲で積分する。
    
    Upper envelope の定義:
        y_P(x) = max { y | (x', y) ∈ P, x' ≥ x }
    
    重要: envelope は y が更新された点のみを含むが、upper envelope y_P(x) は
    元のデータ点の全範囲で well-defined。
    
    - x ≤ data_x_max: 常に x' ≥ x を満たす点が存在するため y_P(x) は定義される
    - x > data_x_max: x' ≥ x を満たす点がないため未定義（外挿が必要）
    - x < data_x_min: x' ≥ x を満たす点は全て存在するため y_P(x) = y_max
    
    Args:
        envelope: Upper envelope の点群（x 降順、y 増加点のみ）
        tau: 積分下限（coherency の制約）
        x_max: 積分上限
        data_x_min: 元のデータ点の最小 x（有効範囲の判定用）
        data_x_max: 元のデータ点の最大 x（有効範囲の判定用）
        extrapolation_mode: 境界外の処理方法
            - "boundary": 境界値で外挿
            - "zero": 外挿領域は 0
            - "none": 外挿領域を積分に含めない
        envelope_type: エンベロープの種類
            - "upper": 各セグメントで左端（大きい y）の値を使用
            - "lower": 各セグメントで右端（小さい y）の値を使用
    
    Returns:
        (積分値, 有効積分範囲の幅) のタプル
    """
    if not envelope or tau >= x_max:
        return 0.0, 0.0
    
    # envelope を x 降順で取得
    env_desc = sorted(envelope, key=lambda p: -p[0])
    
    # envelope の範囲
    x_max_env = env_desc[0][0]   # envelope の最大 x
    x_min_env = env_desc[-1][0]  # envelope の最小 x（y 最大の点）
    
    # データ範囲（指定されなければ envelope の範囲を使用）
    if data_x_min is None:
        data_x_min = x_min_env
    if data_x_max is None:
        data_x_max = x_max_env
    
    # 各境界での y 値
    y_at_max_x = env_desc[0][1]   # 最大 x での y（最小の y）
    y_at_min_x = env_desc[-1][1]  # envelope の最大 y
    
    integral = 0.0
    effective_width = 0.0
    
    # 積分範囲
    x_start = tau
    x_end = x_max
    
    # ===== 領域2: 左側外挿 [x_start, data_x_min] =====
    # x < data_x_min ではデータ点がないため真の外挿
    if x_start < data_x_min:
        left_extrap_end = min(x_end, data_x_min)
        left_extrap_width = left_extrap_end - x_start
        
        if left_extrap_width > 0:
            if extrapolation_mode == "boundary":
                integral += y_at_min_x * left_extrap_width
            # "zero" や "none" では何も加算しない
            
            x_start = data_x_min
    
    # ===== 領域3: データ範囲内だが envelope の角点より左 =====
    # [data_x_min, x_min_env] で data_x_min ≤ x < x_min_env
    # これは外挿ではない：x' ≥ x を満たす点が存在し、y_P(x) = y_max
    if x_start < x_min_env and x_end > x_start:
        left_end = min(x_end, x_min_env)
        left_width = left_end - x_start
        
        if left_width > 0:
            # データ範囲内なので常に積分に含める
            integral += y_at_min_x * left_width
            effective_width += left_width
            
            x_start = x_min_env
    
    # ===== 領域4: envelope の角点間 =====
    if x_start < x_end:
        # Step function として積分
        for i in range(len(env_desc) - 1):
            seg_x_right = env_desc[i][0]
            seg_x_left = env_desc[i + 1][0]
            
            # envelope_type に基づいてセグメントの y 値を選択
            if envelope_type == "upper":
                seg_y = env_desc[i + 1][1]  # 左端の点の y 値（Upper envelope）
            else:  # "lower"
                seg_y = env_desc[i][1]  # 右端の点の y 値（Lower envelope）
            
            overlap_start = max(x_start, seg_x_left)
            overlap_end = min(x_end, seg_x_right)
            
            if overlap_end > overlap_start:
                integral += seg_y * (overlap_end - overlap_start)
                effective_width += overlap_end - overlap_start
    
    # ===== 領域5: envelope の最大 x から data_x_max まで =====
    # [x_max_env, data_x_max] では y = y_at_max_x（データ範囲内）
    if x_end > x_max_env:
        seg_start = max(x_start, x_max_env)
        seg_end = x_end
        seg_width = seg_end - seg_start
        
        if seg_width > 0:
            integral += y_at_max_x * seg_width
            effective_width += seg_width
    
    return integral, effective_width


def compute_score_tau(
    pareto_points: List[Tuple[float, float]],
    tau: float,
    x_max_common: float,
    invert_y: bool = False,
    extrapolation_mode: str = "none",
    envelope_type: str = "lower",
) -> float:
    """
    Right-normalized Constrained Envelope Area (Pareto スコア) を計算する。
    
    数式:
        Score_τ(P) = (1 / (x_max_common - τ)) * ∫_τ^{x_max_common} y_P(x) dx
    
    ここで y_P(x) は envelope:
        upper: y_P(x) = max { y | (x', y) ∈ P, x' ≥ x }
        lower: y_P(x) = min { y | (x', y) ∈ P, x' ≤ x }
    
    この指標は:
        - coherency ≥ τ の制約下での trait score の平均達成度を測定
        - 離散点の個数に依存しない連続曲線評価
        - x_max_common で正規化することで異なる Pareto フロント間の公平な比較が可能
    
    Args:
        pareto_points: Pareto フロントの点群 [(x, y), ...]
                      x: coherency (0-100), y: trait score (0-100)
        tau: coherency の下限制約 (例: 50.0)
        x_max_common: 比較対象全体で共通に達成可能な最大 coherency
        invert_y: True の場合、y を 100 - y に変換してから計算
                  (pos_subtract など、低 trait が良い場合に使用)
        extrapolation_mode: 境界外の処理方法
            - "boundary": 境界値で外挿（最大 y を使用）← 楽観的
            - "zero": 外挿領域は 0 として扱う ← 保守的
            - "none": 外挿領域を含めず、有効範囲で正規化 ← 公平（デフォルト）
        envelope_type: エンベロープの種類
            - "upper": 各セグメントで左端（大きい y）の値を使用（楽観的）
            - "lower": 各セグメントで右端（小さい y）の値を使用（保守的、デフォルト）
    
    Returns:
        Score_τ(P): 0-100 の範囲のスコア
                   高いほど良い Pareto フロント
    
    Raises:
        ValueError: tau >= x_max_common の場合
    
    Examples:
        >>> points = [(90, 80), (85, 85), (80, 88), (70, 90)]
        >>> score = compute_score_tau(points, tau=50.0, x_max_common=90.0)
        >>> print(f"Score: {score:.2f}")
        
        # Upper envelope を使用する場合
        >>> score = compute_score_tau(points, tau=50.0, x_max_common=90.0, envelope_type="upper")
    
    Note:
        - 点群が空の場合は 0.0 を返す
        - x_max_common より右側の点は envelope 構築時に使用されるが、
          積分範囲には含まれない
        - extrapolation_mode="none" の場合、実際にデータがある範囲のみで
          積分・正規化を行う。これにより、データ範囲が狭いモジュールが
          不当に有利/不利にならない。
    """
    if not pareto_points:
        return 0.0
    
    if tau >= x_max_common:
        raise ValueError(
            f"tau ({tau}) must be less than x_max_common ({x_max_common})"
        )
    
    # y を反転（pos_subtract 用）
    if invert_y:
        pareto_points = [(x, 100.0 - y) for x, y in pareto_points]
    
    # Upper envelope を構築
    envelope = build_upper_envelope(pareto_points)
    
    if not envelope:
        return 0.0
    
    # 元のデータ点の x 範囲を取得
    data_x_min = min(p[0] for p in pareto_points)
    data_x_max = max(p[0] for p in pareto_points)
    
    # step function として積分
    integral, effective_width = integrate_envelope_step_function(
        envelope, tau, x_max_common, 
        data_x_min=data_x_min,
        data_x_max=data_x_max,
        extrapolation_mode=extrapolation_mode,
        envelope_type=envelope_type,
    )
    
    # 正規化
    if extrapolation_mode == "none":
        # 有効範囲のみで正規化
        normalization_factor = effective_width
    else:
        # 全範囲で正規化
        normalization_factor = x_max_common - tau
    
    if normalization_factor <= 0:
        return 0.0
    
    score = integral / normalization_factor
    
    return score


def find_common_x_max(
    pareto_fronts: List[List[Tuple[float, float]]]
) -> float:
    """
    複数の Pareto フロントで共通に達成可能な最大 x を見つける。
    
    各 Pareto フロントの最大 x の最小値を返す。
    これにより、全てのフロントで到達可能な x 範囲で比較できる。
    
    Args:
        pareto_fronts: 複数の Pareto フロントのリスト
    
    Returns:
        共通最大 x
    
    Raises:
        ValueError: 空の入力の場合
    """
    if not pareto_fronts:
        raise ValueError("Empty pareto_fronts")
    
    max_x_values = []
    for pf in pareto_fronts:
        if pf:
            max_x = max(p[0] for p in pf)
            max_x_values.append(max_x)
    
    if not max_x_values:
        raise ValueError("All pareto fronts are empty")
    
    return min(max_x_values)


def find_common_x_min(
    pareto_fronts: List[List[Tuple[float, float]]]
) -> float:
    """
    複数の Pareto フロントで共通に存在する最小 x を見つける。
    
    Args:
        pareto_fronts: 複数の Pareto フロントのリスト
    
    Returns:
        共通最小 x（各フロントの最小 x の最大値）
    """
    if not pareto_fronts:
        raise ValueError("Empty pareto_fronts")
    
    min_x_values = []
    for pf in pareto_fronts:
        if pf:
            min_x = min(p[0] for p in pf)
            min_x_values.append(min_x)
    
    if not min_x_values:
        raise ValueError("All pareto fronts are empty")
    
    return max(min_x_values)


# =============================================================================
# CLI Entry Point
# =============================================================================

def main(
    pareto_points: str = None,
    tau: float = 50.0,
    x_max_common: float = 90.0,
    invert_y: bool = False,
    extrapolation_mode: str = "none",
    envelope_type: str = "lower",
):
    """
    Pareto スコアを計算する CLI エントリポイント。
    
    Args:
        pareto_points: カンマ区切りの点群文字列 (例: "90,80;85,85;80,88")
        tau: coherency の下限制約
        x_max_common: 共通最大 coherency
        invert_y: y を反転するか
        extrapolation_mode: 外挿モード ("boundary", "zero", "none")
        envelope_type: エンベロープタイプ ("upper", "lower")
    
    Example:
        python pareto_score.py --pareto_points "90,80;85,85;80,88;70,90" --tau 50 --x_max_common 90
    """
    if pareto_points is None:
        # テストデータ
        print("No pareto_points provided. Running test with sample data.")
        points = [(90, 60), (80, 70), (70, 80), (60, 85), (50, 90)]
    else:
        # パース: "90,80;85,85;80,88;70,90" → [(90, 80), (85, 85), ...]
        points = []
        for pair in pareto_points.split(";"):
            x, y = pair.split(",")
            points.append((float(x), float(y)))
    
    print(f"Input points: {points}")
    print(f"tau: {tau}, x_max_common: {x_max_common}")
    print(f"envelope_type: {envelope_type}")
    print(f"extrapolation_mode: {extrapolation_mode}")
    
    score = compute_score_tau(
        points, tau, x_max_common,
        invert_y=invert_y,
        extrapolation_mode=extrapolation_mode,
        envelope_type=envelope_type,
    )
    
    print(f"Pareto Score: {score:.4f}")
    return score


if __name__ == "__main__":
    import fire
    fire.Fire(main)
