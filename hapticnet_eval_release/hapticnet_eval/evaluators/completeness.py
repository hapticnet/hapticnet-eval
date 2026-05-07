from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseEvaluator, ClaimIndex
from ..schemas import EvaluatorScore, MatchResult
from ..regimes.base import Regime

def value_kind(value_obj: Dict[str, Any]) -> str:
    if not isinstance(value_obj, dict):
        return "unknown"
    if value_obj.get("value") is not None:
        return "scalar"
    if value_obj.get("min") is not None and value_obj.get("max") is not None:
        return "range"
    if value_obj.get("data_points") is not None:
        return "series"
    return "unknown"

def flatten_scalar_values(payload: Dict[str, Any]) -> List[float]:
    kind = value_kind(payload)
    if kind == "scalar":
        return [float(payload["value"])]
    if kind == "range":
        return [float(payload["min"]), float(payload["max"])]
    if kind == "series":
        return [float(x) for x in payload.get("data_points", [])]
    return []

def greedy_numeric_match(pred: List[float], gt: List[float], rel_tol: float = 1e-3) -> tuple[int, int, int, list]:
    tp = 0
    used_gt = set()
    detail_rows = []
    for p in pred:
        matched = False
        for gi, g in enumerate(gt):
            if gi in used_gt:
                continue
            denom = max(abs(g), 1e-12)
            if abs(g - p) / denom <= rel_tol:
                tp += 1
                used_gt.add(gi)
                matched = True
                detail_rows.append({"category": "TP", "value": p, "matched_to": g})
                break
        if not matched:
            detail_rows.append({"category": "FP", "value": p, "matched_to": None})
    for gi, g in enumerate(gt):
        if gi not in used_gt:
            detail_rows.append({"category": "FN", "value": g, "matched_to": None})
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn, detail_rows

class ValueListCoverageEvaluator(BaseEvaluator):
    """File-level value inventory coverage evaluator.
    
    For indirect (formula-computed) queries, uses a wider tolerance (default 5%)
    since computed values inherently differ from GT due to rounding of component
    parameters. This is consistent with engineering calculation tolerances.
    """

    NAME = "value_list_coverage"
    REGIMES = (Regime.FIXED_DOCS, Regime.URL_ONLY, Regime.OPEN_WEB)

    def __init__(self, rel_tol: float = 1e-3, indirect_rel_tol: float = 0.05) -> None:
        self.rel_tol = rel_tol
        self.indirect_rel_tol = indirect_rel_tol

    @staticmethod
    def _is_indirect_run(pred_obj: dict) -> bool:
        """Detect if the prediction was produced by an indirect extraction pipeline."""
        for vcm in pred_obj.get("value_condition_mapping", [])[:10]:
            for sg in vcm.get("successful_groundings", []):
                if sg.get("indirect_parameter_contributions"):
                    return True
            # Also check measurement conditions for formula indicators
            for cond in vcm.get("measurement_conditions", []):
                label = (cond.get("label", "") or "").lower()
                val = (cond.get("val", "") or "").lower()
                if any(kw in label or kw in val for kw in ("equation", "formula", "sqrt", "computed", "e =")):
                    return True
        return False

    def _flatten_from_value_list(self, obj: Dict, key: str) -> List[float]:
        values = []
        value_list = obj.get("value_list")
        if value_list is None:
            return values
        # Handle nested dict format: {"all_values": {"values": [...]}}
        if isinstance(value_list, dict):
            nested = (value_list.get(key) or {}).get("values") or []
            for v in nested:
                values.extend(flatten_scalar_values(v))
        # value_list is a plain list — skip (use _flatten_from_records instead)
        return values

    def _flatten_from_records(self, obj: Dict) -> List[float]:
        values: List[float] = []
        for entry in obj.get("value_condition_mapping", []) or []:
            if "value" in entry:
                values.extend(flatten_scalar_values(entry["value"]))
        return values

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        context = context or {}
        gt = context.get("gt_obj", {})
        pred = context.get("pred_obj", {})

        # Use wider tolerance for indirect/computed queries
        is_indirect = self._is_indirect_run(pred)
        effective_tol = self.indirect_rel_tol if is_indirect else self.rel_tol

        gt_vals = self._flatten_from_value_list(gt, "all_values") or self._flatten_from_records(gt)
        pr_vals = self._flatten_from_value_list(pred, "all_values") or self._flatten_from_records(pred)
        tp, fp, fn, detail_rows = greedy_numeric_match(pr_vals, gt_vals, effective_tol)
        
        precision = tp / (tp + fp) if (tp + fp) else 1.0 if not pr_vals and not gt_vals else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0 if not pr_vals and not gt_vals else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        
        return EvaluatorScore(name=self.NAME, score=f1, details={
            "gt_scalar_value_count": len(gt_vals),
            "pred_scalar_value_count": len(pr_vals),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "is_indirect": is_indirect,
            "effective_rel_tol": effective_tol,
            "rows": detail_rows,
        })

