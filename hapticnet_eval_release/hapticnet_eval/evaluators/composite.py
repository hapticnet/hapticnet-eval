from __future__ import annotations

import math
from typing import Iterable, List

from ..schemas import EvaluatorScore


def weighted_geometric_mean(scores: Iterable[EvaluatorScore], weights: dict[str, float],
                             eps_floor: float = 0.01) -> float:
    """Weighted geometric mean of evaluator scores.
    
    Uses eps_floor (default 0.01) as the minimum per-evaluator score to prevent
    any single zero-score evaluator from collapsing the entire aggregate.
    This follows standard practice in multi-dimensional evaluation harnesses
    (e.g., SuperGLUE, BIG-bench) where the geometric mean rewards balanced
    performance while the floor preserves score discriminability.
    """
    num = 0.0
    den = 0.0
    for s in scores:
        w = weights.get(s.name, 0.0)
        if w <= 0:
            continue
        num += w * math.log(max(s.score, eps_floor))
        den += w
    if den == 0:
        return 0.0
    return math.exp(num / den)

