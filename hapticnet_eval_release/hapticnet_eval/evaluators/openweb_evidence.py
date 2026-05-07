from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseEvaluator, ClaimIndex
from ..schemas import EvaluatorScore, MatchResult
from ..regimes.base import Regime
from urllib.parse import urlparse

class OpenWebApproxEvidenceEvaluator(BaseEvaluator):
    """Approximate Open Web Evidence Evaluator."""
    NAME = "open_web_approx_evidence"
    REGIMES = (Regime.OPEN_WEB,)

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        vals = []
        rows = []

        for m in matches:
            if m.pred_only or m.gt_only:
                vals.append(0.0)
                rows.append({"gt": getattr(m, 'gt_claim_id', None), "pred": getattr(m, 'pred_claim_id', None), "score": 0.0})
                continue

            g = gt_index.get(m.gt_claim_id)
            p = pred_index.get(m.pred_claim_id)
            assert g and p

            gt_domains = {urlparse(e.source_url).netloc for e in g.provenance if e.source_url}
            
            best = 0.0
            best_detail = None
            
            for pe in p.provenance:
                txt = " ".join(filter(None, [pe.citation_snippet, pe.matched_snippet] + list(pe.matched_snippet_pieces)))
                
                domain_match = float(bool(pe.source_url and urlparse(pe.source_url).netloc in gt_domains))
                
                # Approximate text token support
                txt_lower = txt.lower()
                toks = [k for k, v in g.measurement_conditions] + [v for k, v in g.measurement_conditions]
                toks = [t for t in toks if t]
                token_ratio = sum(1 for t in toks if t and t in txt_lower) / len(toks) if toks else 1.0
                
                value_match = 0.0
                if g.value_type == "scalar" and g.normalized_value is not None:
                    value_match = float(str(round(g.normalized_value, 2)) in txt_lower)
                
                score = max(domain_match, 0.55 * value_match + 0.45 * token_ratio)
                if score > best:
                    best = score
                    best_detail = {
                        "source_url": pe.source_url,
                        "domain_match": domain_match,
                        "value_match": value_match,
                        "condition_token_ratio": token_ratio
                    }
            
            vals.append(best)
            rows.append({"gt": g.claim_id, "pred": p.claim_id, "score": best, "detail": best_detail})

        score = sum(vals) / len(vals) if vals else 1.0
        return EvaluatorScore(name=self.NAME, score=score, details={"rows": rows})
