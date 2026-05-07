from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from .base import BaseEvaluator, ClaimIndex
from ..schemas import EvaluatorScore, MatchResult
from ..regimes.base import Regime

def _extract_numbers(text: str) -> Set[str]:
    # Extract numeric-looking strings
    nums = re.findall(r"(?<!\d)(?:\d{1,3}(?:[ ,]\d{3})*(?:[\.,]\d+)?|\d+(?:[\.,]\d+)?)(?!\d)", text)
    return set(nums)

class StatementSupportEvaluator(BaseEvaluator):
    """Deterministic statement-level citation support evaluator. (ALCE / AIS style)"""
    
    NAME = "statement_support"
    REGIMES = (Regime.FIXED_DOCS, Regime.URL_ONLY, Regime.OPEN_WEB)

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        vals = []
        rows = []
        
        # ALCE style atomic claim matching calculates F1 of cited claims
        gt_ids = {c.claim_id for c in gt_index.claims}
        pr_ids = {c.claim_id for c in pred_index.claims}
        
        supported_pred_ids = set()
        for p in pred_index.claims:
            support_text = " ".join(
                filter(None, [e.citation_snippet for e in p.provenance] + 
                             [e.matched_snippet for e in p.provenance] + 
                             [piece for e in p.provenance for piece in (e.matched_snippet_pieces or [])])
            ).lower()
            
            # Condition tokens
            cond_tokens = [k.lower() for k, v in p.measurement_conditions] + [v.lower() for k, v in p.measurement_conditions]
            cond_tokens = [t for t in cond_tokens if t]
            token_ratio = sum(1 for t in cond_tokens if t in support_text) / len(cond_tokens) if cond_tokens else 1.0
            
            # Value tokens
            txt_nums = _extract_numbers(support_text)
            val_supported = False
            if p.value_type == "scalar" and p.normalized_value is not None:
                val_supported = any(str(round(p.normalized_value, 2)) in tn for tn in txt_nums) or (str(round(p.normalized_value, 2)) in support_text)
            elif p.value_type == "range":
                val_supported = True # simplify range check
            elif p.value_type == "series":
                val_supported = True # simplify series check
                
            if val_supported and token_ratio >= 0.5:
                supported_pred_ids.add(p.claim_id)

        # Mapping between GT and Supported Pred
        matched_gt_ids = {m.gt_claim_id for m in matches if m.pred_claim_id in supported_pred_ids}
        
        tp = len(matched_gt_ids)
        fp = len(supported_pred_ids) - tp
        fn = len(gt_ids) - tp
        
        precision = tp / (tp + fp) if (tp + fp) else 1.0 if not supported_pred_ids and not gt_ids else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 1.0 if not gt_ids and not supported_pred_ids else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        
        return EvaluatorScore(name=self.NAME, score=f1, details={
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "supported_pred_ids": list(supported_pred_ids),
            "gt_claim_count": len(gt_ids)
        })
