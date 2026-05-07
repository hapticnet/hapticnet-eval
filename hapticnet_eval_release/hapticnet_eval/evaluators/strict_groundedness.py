from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseEvaluator, ClaimIndex
from ..schemas import EvaluatorScore, MatchResult
from ..regimes.base import Regime
from ..utils.normalization import normalize_units, relative_closeness

class StrictGroundednessEvaluator(BaseEvaluator):
    """FEVER-style evidence-gated correctness. All dimensions must pass."""
    
    NAME = "strict_groundedness"
    REGIMES = (Regime.FIXED_DOCS, Regime.URL_ONLY, Regime.OPEN_WEB)

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        vals = []
        rows = []
        for m in matches:
            if m.pred_only or m.gt_only:
                vals.append(0.0)
                continue
                
            g = gt_index.get(m.gt_claim_id)
            p = pred_index.get(m.pred_claim_id)
            assert g and p

            value_score = 0.0
            if g.value_type == p.value_type:
                if g.value_type == "scalar":
                    value_score = float(relative_closeness(g.normalized_value, p.normalized_value) >= 0.999)
                elif g.value_type == "range":
                    value_score = float(
                        relative_closeness(g.range_min, p.range_min) >= 0.999 and 
                        relative_closeness(g.range_max, p.range_max) >= 0.999
                    )
                elif g.value_type == "series":
                    if g.data_points and p.data_points and len(g.data_points) == len(p.data_points):
                        value_score = float(all(relative_closeness(gv, pv) >= 0.999 for gv, pv in zip(g.data_points, p.data_points)))
            
            units_ok = normalize_units(g.units) == normalize_units(p.units)
            
            
            
            gt_urls = {ge.source_url for ge in g.provenance if ge.source_url}
            pred_urls = {pe.source_url for pe in p.provenance if pe.source_url}
            citation_ok = float(bool(gt_urls & pred_urls)) if gt_urls else 1.0
            
            # Only gate on value, units, and citation — conditions are not
            # treated as fact-checked GT and should not block correctness.
            passed = value_score > 0 and units_ok and citation_ok > 0
            
            vals.append(float(passed))
            rows.append({
                "gt": g.claim_id,
                "pred": p.claim_id,
                "passed": passed,
                "value_score": value_score,
                "units_ok": units_ok,
                "citation_ok": citation_ok,
            })
            
        score = sum(vals) / len(vals) if vals else 1.0
        return EvaluatorScore(name=self.NAME, score=score, details={"rows": rows})
