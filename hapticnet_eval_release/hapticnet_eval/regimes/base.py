from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum

class Regime(str, Enum):
    FIXED_DOCS = "closed_docs"
    URL_ONLY = "url_only"
    OPEN_WEB = "open_web"



@dataclass
class EvaluationTask:
    task_id: str
    regime: str
    gt_path: str
    pred_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class RegimeAdapter:
    name = "base"

    def preprocess(self, gt_obj: dict, pred_obj: dict) -> tuple[dict, dict, dict]:
        """Return preprocessed (gt_obj, pred_obj, metadata).

        The first implementation is intentionally light. It leaves room for
        future additions such as URL-only fetch traces, open-web source audits,
        or no-answer annotations.
        """
        return gt_obj, pred_obj, {}
