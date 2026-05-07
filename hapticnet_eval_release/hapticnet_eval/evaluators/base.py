from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..schemas import CanonicalClaim, EvaluatorScore, MatchResult
from ..regimes.base import Regime

@dataclass
class ClaimIndex:
    claims: Tuple[CanonicalClaim, ...]
    _map: Dict[str, CanonicalClaim] = field(init=False)

    def __post_init__(self):
        self._map = {c.claim_id: c for c in self.claims}

    def get(self, claim_id: str | None) -> CanonicalClaim | None:
        if claim_id is None:
            return None
        return self._map.get(claim_id)


class BaseEvaluator(abc.ABC):
    """Base class for all evaluators in the unified HapticNetEval framework."""

    NAME: str = "base"
    REGIMES: Tuple[Regime, ...] = (Regime.FIXED_DOCS, Regime.URL_ONLY, Regime.OPEN_WEB)
    ACADEMIC_SOURCES: Tuple[str, ...] = ()
    DESCRIPTION: str = ""

    @abc.abstractmethod
    def evaluate(
        self,
        gt_index: ClaimIndex,
        pred_index: ClaimIndex,
        matches: List[MatchResult],
        context: Dict[str, Any] | None = None
    ) -> EvaluatorScore:
        """Evaluate the provided canonical claims and their matches.
        
        Args:
            gt_index: Indexed canonical claims from the ground truth.
            pred_index: Indexed canonical claims from the prediction.
            matches: The mapping between GT and prediction claims.
            context: The raw full objects (gt_obj, pred_obj, task, etc.) 
                     for evaluators that need file-level or schema-level details.
                     
        Returns:
            An EvaluatorScore summarizing the result.
        """
        raise NotImplementedError
