"""
pareto_score.py

Pareto frontier evaluation metric: Right-normalized Constrained Envelope Area

Performs Pareto frontier evaluation independent of the number of discrete points
in multi-objective optimization.

Evaluation metric:
    Score_τ(P) = (1 / (x_max_common - τ)) * ∫_τ^{x_max_common} y_P(x) dx

Where y_P(x) is the Pareto frontier envelope:
    upper: y_P(x) = max { y | (x', y) ∈ P, x' ≥ x }
    lower: y_P(x) = min { y | (x', y) ∈ P, x' ≤ x }

This metric measures the average achievable trait score under coherency
constraint τ on the x-axis.
"""

from typing import Tuple


def build_upper_envelope(
    pareto_points: list[Tuple[float, float]],
) -> list[Tuple[float, float]]:
    """
    Build the upper envelope of a Pareto frontier.

    Upper envelope definition:
        y_P(x) = max { y | (x', y) ∈ P, x' ≥ x }

    This is constructed by sorting x in descending order and scanning from
    right to left, taking the cumulative maximum of y values.

    The result represents a step function as a set of points.
    Between x_i and x_{i+1}, y remains constant at y_i.

    Args:
        pareto_points: Pareto frontier points [(x1, y1), (x2, y2), ...]
                      x: coherency, y: trait score

    Returns:
        Points constituting the upper envelope (sorted by x descending)
        Each point (x_i, y_i) represents the maximum achievable y for all x ≥ x_i
        Only points where y changes are returned (step function corners)

    Note:
        - Returns empty list if input is empty
        - If multiple points have the same x, maximum y is used
    """
    if not pareto_points:
        return []

    # Sort by x descending (to scan from right to left)
    sorted_points = sorted(pareto_points, key=lambda p: -p[0])

    # Calculate cumulative maximum
    # Upper envelope is a step function, so only record points where y changes
    envelope = []
    current_max_y = float("-inf")

    for x, y in sorted_points:
        if y > current_max_y:
            # Add point only when y becomes a new maximum
            # This records the step function corners
            envelope.append((x, y))
            current_max_y = y

    # Envelope remains sorted by x descending
    # First point is maximum x, last point is minimum x (only points where y increased)

    return envelope


def integrate_envelope_step_function(
    envelope: list[Tuple[float, float]],
    tau: float,
    x_max: float,
    data_x_min: float = None,
    data_x_max: float = None,
    extrapolation_mode: str = "boundary",
    envelope_type: str = "lower",
) -> Tuple[float, float]:
    """
    Integrate the upper envelope as a step function over [τ, x_max].

    Upper envelope definition:
        y_P(x) = max { y | (x', y) ∈ P, x' ≥ x }

    Important: envelope contains only points where y updated, but upper envelope
    y_P(x) is well-defined over the entire range of original data points.

    - x ≤ data_x_max: Points satisfying x' ≥ x exist, so y_P(x) is defined
    - x > data_x_max: No points satisfy x' ≥ x, so undefined (extrapolation needed)
    - x < data_x_min: All points satisfy x' ≥ x, so y_P(x) = y_max

    Args:
        envelope: Upper envelope points (x descending, only y-increase points)
        tau: Lower integration bound (coherency constraint)
        x_max: Upper integration bound
        data_x_min: Minimum x of original data points (for valid range check)
        data_x_max: Maximum x of original data points (for valid range check)
        extrapolation_mode: Handling of out-of-bounds:
            - "boundary": Extrapolate with boundary values
            - "zero": Treat extrapolation regions as 0
            - "none": Exclude extrapolation regions from integration
        envelope_type: Type of envelope:
            - "upper": Use left-end y value (larger y) for each segment
            - "lower": Use right-end y value (smaller y) for each segment

    Returns:
        Tuple of (integral value, effective integration width)
    """
    if not envelope or tau >= x_max:
        return 0.0, 0.0

    # Get envelope sorted by x descending
    env_desc = sorted(envelope, key=lambda p: -p[0])

    # Envelope range
    x_max_env = env_desc[0][0]  # Maximum x of envelope
    x_min_env = env_desc[-1][0]  # Minimum x of envelope (point with max y)

    # Data range (use envelope range if not specified)
    if data_x_min is None:
        data_x_min = x_min_env
    if data_x_max is None:
        data_x_max = x_max_env

    # y values at boundaries
    y_at_max_x = env_desc[0][1]  # y at maximum x (minimum y)
    y_at_min_x = env_desc[-1][1]  # Maximum y of envelope

    integral = 0.0
    effective_width = 0.0

    # Integration range
    x_start = tau
    x_end = x_max

    # ===== Region 2: Left extrapolation [x_start, data_x_min] =====
    # x < data_x_min: No data points, true extrapolation
    if x_start < data_x_min:
        left_extrap_end = min(x_end, data_x_min)
        left_extrap_width = left_extrap_end - x_start

        if left_extrap_width > 0:
            if extrapolation_mode == "boundary":
                integral += y_at_min_x * left_extrap_width
            # "zero" or "none": add nothing

            x_start = data_x_min

    # ===== Region 3: Within data range but left of envelope corners =====
    # [data_x_min, x_min_env] where data_x_min ≤ x < x_min_env
    # Not extrapolation: points satisfying x' ≥ x exist, y_P(x) = y_max
    if x_start < x_min_env and x_end > x_start:
        left_end = min(x_end, x_min_env)
        left_width = left_end - x_start

        if left_width > 0:
            # Within data range, always include in integration
            integral += y_at_min_x * left_width
            effective_width += left_width

            x_start = x_min_env

    # ===== Region 4: Between envelope corner points =====
    if x_start < x_end:
        # Integrate as step function
        for i in range(len(env_desc) - 1):
            seg_x_right = env_desc[i][0]
            seg_x_left = env_desc[i + 1][0]

            # Select segment y value based on envelope_type
            if envelope_type == "upper":
                seg_y = env_desc[i + 1][1]  # Left-end y value (upper envelope)
            else:  # "lower"
                seg_y = env_desc[i][1]  # Right-end y value (lower envelope)

            overlap_start = max(x_start, seg_x_left)
            overlap_end = min(x_end, seg_x_right)

            if overlap_end > overlap_start:
                integral += seg_y * (overlap_end - overlap_start)
                effective_width += overlap_end - overlap_start

    # ===== Region 5: From max envelope x to data_x_max =====
    # [x_max_env, data_x_max]: y = y_at_max_x (within data range)
    if x_end > x_max_env:
        seg_start = max(x_start, x_max_env)
        seg_end = x_end
        seg_width = seg_end - seg_start

        if seg_width > 0:
            integral += y_at_max_x * seg_width
            effective_width += seg_width

    return integral, effective_width


def compute_score_tau(
    pareto_points: list[Tuple[float, float]],
    tau: float,
    x_max_common: float,
    invert_y: bool = False,
    extrapolation_mode: str = "none",
    envelope_type: str = "lower",
) -> float:
    """
    Calculate Right-normalized Constrained Envelope Area (Pareto score).

    Formula:
        Score_τ(P) = (1 / (x_max_common - τ)) * ∫_τ^{x_max_common} y_P(x) dx

    Where y_P(x) is the envelope:
        upper: y_P(x) = max { y | (x', y) ∈ P, x' ≥ x }
        lower: y_P(x) = min { y | (x', y) ∈ P, x' ≤ x }

    This metric:
        - Measures average achievable trait score under coherency ≥ τ constraint
        - Evaluates continuous curve independent of number of discrete points
        - Enables fair comparison between Pareto frontiers via x_max_common normalization

    Args:
        pareto_points: Pareto frontier points [(x, y), ...]
                      x: coherency (0-100), y: trait score (0-100)
        tau: Lower coherency constraint (e.g., 50.0)
        x_max_common: Maximum coherency achievable by all compared methods
        invert_y: If True, transform y to 100 - y before calculation
                  (used for pos_subtract where lower trait is better)
        extrapolation_mode: Handling of out-of-bounds:
            - "boundary": Extrapolate with boundary values (optimistic)
            - "zero": Treat extrapolation as 0 (conservative)
            - "none": Exclude and normalize by effective range only (fair, default)
        envelope_type: Type of envelope:
            - "upper": Use left-end y value (larger y) for each segment (optimistic)
            - "lower": Use right-end y value (smaller y) for each segment (conservative, default)

    Returns:
        Score_τ(P): Score in range 0-100
                   Higher is better Pareto frontier

    Raises:
        ValueError: If tau >= x_max_common

    Examples:
        >>> points = [(90, 80), (85, 85), (80, 88), (70, 90)]
        >>> score = compute_score_tau(points, tau=50.0, x_max_common=90.0)
        >>> print(f"Score: {score:.2f}")

        # Using upper envelope
        >>> score = compute_score_tau(points, tau=50.0, x_max_common=90.0, envelope_type="upper")

    Note:
        - Returns 0.0 if points is empty
        - Points to the right of x_max_common are used for envelope construction
          but not included in integration range
        - With extrapolation_mode="none", integration/normalization uses only
          ranges with actual data, preventing unfair advantage/disadvantage
          for methods with narrow data ranges.
    """
    if not pareto_points:
        return 0.0

    if tau >= x_max_common:
        raise ValueError(f"tau ({tau}) must be less than x_max_common ({x_max_common})")

    # Invert y (for pos_subtract)
    if invert_y:
        pareto_points = [(x, 100.0 - y) for x, y in pareto_points]

    # Build upper envelope
    envelope = build_upper_envelope(pareto_points)

    if not envelope:
        return 0.0

    # Get x range of original data points
    data_x_min = min(p[0] for p in pareto_points)
    data_x_max = max(p[0] for p in pareto_points)

    # Integrate as step function
    integral, effective_width = integrate_envelope_step_function(
        envelope,
        tau,
        x_max_common,
        data_x_min=data_x_min,
        data_x_max=data_x_max,
        extrapolation_mode=extrapolation_mode,
        envelope_type=envelope_type,
    )

    # Normalization
    if extrapolation_mode == "none":
        # Normalize by effective range only
        normalization_factor = effective_width
    else:
        # Normalize by full range
        normalization_factor = x_max_common - tau

    if normalization_factor <= 0:
        return 0.0

    score = integral / normalization_factor

    return score


def find_common_x_max(pareto_fronts: list[list[Tuple[float, float]]]) -> float:
    """
    Find the maximum x commonly achievable across multiple Pareto frontiers.

    Returns the minimum of maximum x values across all frontiers.
    This enables fair comparison within x ranges reachable by all frontiers.

    Args:
        pareto_fronts: list of multiple Pareto frontiers

    Returns:
        Common maximum x

    Raises:
        ValueError: If input is empty
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


def find_common_x_min(pareto_fronts: list[list[Tuple[float, float]]]) -> float:
    """
    Find the minimum x commonly present across multiple Pareto frontiers.

    Args:
        pareto_fronts: list of multiple Pareto frontiers

    Returns:
        Common minimum x (maximum of minimum x values across frontiers)
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
    CLI entry point for Pareto score calculation.

    Args:
        pareto_points: Semicolon-separated point pairs (e.g., "90,80;85,85;80,88")
        tau: Lower coherency constraint
        x_max_common: Common maximum coherency
        invert_y: Whether to invert y values
        extrapolation_mode: Extrapolation mode ("boundary", "zero", "none")
        envelope_type: Envelope type ("upper", "lower")

    Example:
        python pareto_score.py --pareto_points "90,80;85,85;80,88;70,90" --tau 50 --x_max_common 90
    """
    if pareto_points is None:
        # Test data
        print("No pareto_points provided. Running test with sample data.")
        points = [(90, 60), (80, 70), (70, 80), (60, 85), (50, 90)]
    else:
        # Parse: "90,80;85,85;80,88;70,90" → [(90, 80), (85, 85), ...]
        points = []
        for pair in pareto_points.split(";"):
            x, y = pair.split(",")
            points.append((float(x), float(y)))

    print(f"Input points: {points}")
    print(f"tau: {tau}, x_max_common: {x_max_common}")
    print(f"envelope_type: {envelope_type}")
    print(f"extrapolation_mode: {extrapolation_mode}")

    score = compute_score_tau(
        points,
        tau,
        x_max_common,
        invert_y=invert_y,
        extrapolation_mode=extrapolation_mode,
        envelope_type=envelope_type,
    )

    print(f"Pareto Score: {score:.4f}")
    return score


if __name__ == "__main__":
    import fire

    fire.Fire(main)
