import numpy as np
from typing import List


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def aggregate_frame_scores(frame_scores: List[float]) -> float:
    """
    Robust temporal aggregation:
    - Uses median + p90 to reduce outlier influence
    - Rewards consistently high suspicious scores across frames
    """
    if not frame_scores:
        return 0.0

    sorted_scores = sorted(clamp01(x) for x in frame_scores)
    n = len(sorted_scores)
    median = sorted_scores[n // 2]
    p90 = sorted_scores[min(n - 1, int(round((n - 1) * 0.9)))]
    mean = sum(sorted_scores) / n

    # Consistency metric: proportion of frames above moderate suspicion threshold.
    suspicious_ratio = sum(1 for x in sorted_scores if x >= 0.7) / n  # Raise threshold to 0.7

    # Bias towards lower scores: use geometric mean for more conservative aggregation
    if all(x > 0 for x in sorted_scores):
        geo_mean = np.exp(np.mean(np.log(sorted_scores)))
    else:
        geo_mean = mean  # Fallback

    agg = 0.4 * geo_mean + 0.3 * median + 0.2 * p90 + 0.1 * suspicious_ratio
    return clamp01(agg)

