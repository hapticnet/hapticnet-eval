from __future__ import annotations

from statistics import mean
from typing import Any, Dict, Iterable, List, Tuple

from ..schemas import CanonicalClaim, EvaluatorScore, MatchResult
from ..utils.evidence import judge_citation_support
from ..utils.normalization import exact_match_ratio, interval_iou, overlap_ratio, relative_closeness, sequence_similarity


class ClaimIndex:
    def __init__(self, claims: Iterable[CanonicalClaim]):
        self.by_id = {c.claim_id: c for c in claims}
        self.claims = list(self.by_id.values())

    def get(self, claim_id: str | None) -> CanonicalClaim | None:
        return self.by_id.get(claim_id) if claim_id else None

    def all_claims(self) -> list[CanonicalClaim]:
        return self.claims


class BaseEvaluator:
    name = "base"

    def evaluate(
        self,
        gt_index: ClaimIndex,
        pred_index: ClaimIndex,
        matches: List[MatchResult],
        context: Dict[str, Any] | None = None,
    ) -> EvaluatorScore:
        raise NotImplementedError


class FactualValueEvaluator(BaseEvaluator):
    """Blueprint: claim-level factual evaluation from FActScore and RAGChecker;
    exact numeric scoring for structured fields is a benchmark-specific engineering choice.
    """

    name = "factual_value"

    @staticmethod
    def _is_indirect_claim(claim) -> bool:
        """Check if a claim was produced by indirect extraction (formula-computed)."""
        for ev in claim.provenance:
            # CanonicalFieldEvidence doesn't have indirect_parameter_contributions,
            # but we can check via the raw grounding records in the original data.
            # We check if citation_snippet contains the "[param]" prefix pattern
            # that merged indirect groundings use.
            if ev.citation_snippet and ev.citation_snippet.startswith("["):
                return True
        return False

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        # Detect indirect mode from pred_obj metadata
        is_indirect_run = False
        if context:
            pred_obj = context.get("pred_obj", {})
            # Check if any VCM has indirect_parameter_contributions in its groundings
            for vcm in pred_obj.get("value_condition_mapping", [])[:5]:
                for sg in vcm.get("successful_groundings", []):
                    if sg.get("indirect_parameter_contributions"):
                        is_indirect_run = True
                        break
                if is_indirect_run:
                    break

        vals = []
        detail_rows = []
        for m in matches:
            if m.gt_only or m.pred_only:
                vals.append(0.0)
                continue
            g = gt_index.get(m.gt_claim_id)
            p = pred_index.get(m.pred_claim_id)
            assert g and p

            # For indirect/computed values, use tolerant-only scoring since
            # exact string match is unreasonable (formula rounding differences)
            strict_weight = 0.0 if is_indirect_run else 0.25
            tolerant_weight = 1.0 if is_indirect_run else 0.75

            if g.value_type == "scalar" and p.value_type == "scalar":
                strict = float(g.normalized_value == p.normalized_value)
                tolerant = relative_closeness(g.normalized_value, p.normalized_value)
                s = strict_weight * strict + tolerant_weight * tolerant
            elif g.value_type == "range" and p.value_type == "range":
                tolerant = interval_iou(g.range_min or 0.0, g.range_max or 0.0, p.range_min or 0.0, p.range_max or 0.0)
                strict = float((g.range_min, g.range_max) == (p.range_min, p.range_max))
                s = strict_weight * strict + tolerant_weight * tolerant
            elif g.value_type == "series" and p.value_type == "series":
                tolerant = sequence_similarity(g.data_points, p.data_points)
                strict = float(g.data_points == p.data_points)
                s = strict_weight * strict + tolerant_weight * tolerant
            else:
                strict = 0.0
                tolerant = 0.0
                s = 0.0
            vals.append(s)
            scoring_mode = "tolerant-only (indirect)" if is_indirect_run else "0.25×strict+0.75×tolerant"
            detail_rows.append(
                {
                    "gt": g.claim_id,
                    "pred": p.claim_id,
                    "gt_value_type": g.value_type,
                    "pred_value_type": p.value_type,
                    "strict": strict,
                    "tolerant": tolerant,
                    "score": s,
                    "scoring_mode": scoring_mode,
                }
            )
        return EvaluatorScore(name=self.name, score=mean(vals) if vals else 0.0, details={"rows": detail_rows})


class ConditionsEvaluator(BaseEvaluator):
    """Blueprint: fine-grained component scoring from RAGChecker and structured-field evaluation from ChemX.
    Set-F1 over normalized field-value pairs is benchmark-specific.
    """

    name = "conditions_and_specs"

    @staticmethod
    def _set_f1(a: Tuple, b: Tuple) -> float:
        sa, sb = set(a), set(b)
        if not sa and not sb:
            return 1.0
        if not sa or not sb:
            return 0.0
        p = len(sa & sb) / len(sb)
        r = len(sa & sb) / len(sa)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        vals = []
        rows = []
        for m in matches:
            if m.gt_only or m.pred_only:
                vals.append(0.0)
                continue
            g = gt_index.get(m.gt_claim_id)
            p = pred_index.get(m.pred_claim_id)
            assert g and p
            cond = self._set_f1(g.measurement_conditions, p.measurement_conditions)
            specs = self._set_f1(g.material_specifications, p.material_specifications)
            obj = self._set_f1(g.object_class, p.object_class)
            score = (cond * 0.6) + (specs * 0.25) + (obj * 0.15)
            vals.append(score)
            rows.append({"gt": g.claim_id, "pred": p.claim_id, "conditions_f1": cond, "specs_f1": specs, "object_class_f1": obj, "score": score})
        return EvaluatorScore(name=self.name, score=mean(vals) if vals else 0.0, details={"rows": rows})


class HierarchyEvaluator(BaseEvaluator):
    """Blueprint: hierarchy-sensitive structured extraction analysis from ChemX-like column-level reporting;
    exact weighting across family/class/subclass is benchmark-specific.
    """

    name = "hierarchy"

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        vals = []
        rows = []
        for m in matches:
            if m.gt_only or m.pred_only:
                vals.append(0.0)
                continue
            g = gt_index.get(m.gt_claim_id)
            p = pred_index.get(m.pred_claim_id)
            assert g and p
            subclass_ok = float(g.material_subclass == p.material_subclass)
            family_ok = float(g.material_family == p.material_family)
            class_ok = float(g.material_class == p.material_class)
            score = (family_ok * 0.2) + (class_ok * 0.3) + (subclass_ok * 0.5)
            vals.append(score)
            rows.append({"gt": g.claim_id, "pred": p.claim_id, "family": family_ok, "class": class_ok, "subclass": subclass_ok, "score": score})
        return EvaluatorScore(name=self.name, score=mean(vals) if vals else 0.0, details={"rows": rows})


class CitationSourceRecallEvaluator(BaseEvaluator):
    """Blueprint: ALCE citation recall adapted to structured claims.
    Measures whether the predicted sources cover GT evidence.

    Uses "at least one" mode: full credit (1.0) if the pred found
    the value from ANY of the GT sources.  Cross-referenced GT sources
    (added by the grounding process) are treated as equally valid
    alternatives rather than all-required.
    """

    name = "citation_source_recall"

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        vals = []
        rows = []
        for m in matches:
            if m.gt_only or m.pred_only:
                vals.append(0.0)
                continue
            g = gt_index.get(m.gt_claim_id)
            p = pred_index.get(m.pred_claim_id)
            assert g and p
            gt_sources = {e.source_url for e in g.provenance if e.source_url}
            pred_sources = {e.source_url for e in p.provenance if e.source_url}
            if not gt_sources and not pred_sources:
                vals.append(1.0)
                continue
            if not gt_sources:
                vals.append(1.0)
                continue
            overlap = gt_sources & pred_sources
            # "At least one" mode: full credit if pred found value from any GT source
            score = 1.0 if overlap else 0.0
            strict_recall = len(overlap) / len(gt_sources)
            vals.append(score)
            rows.append({
                "gt": g.claim_id, "pred": p.claim_id,
                "gt_sources": sorted(gt_sources), "pred_sources": sorted(pred_sources),
                "score": score, "strict_recall": strict_recall,
            })
        return EvaluatorScore(name=self.name, score=mean(vals) if vals else 0.0, details={"rows": rows})


class CitationSourcePrecisionEvaluator(BaseEvaluator):
    """Blueprint: ALCE citation precision adapted to structured claims.
    Penalizes irrelevant or surplus cited sources.
    """

    name = "citation_source_precision"

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        vals = []
        rows = []
        for m in matches:
            if m.pred_only:
                vals.append(0.0)
                continue
            if m.gt_only:
                continue
            g = gt_index.get(m.gt_claim_id)
            p = pred_index.get(m.pred_claim_id)
            assert g and p
            gt_sources = {e.source_url for e in g.provenance if e.source_url}
            pred_sources = {e.source_url for e in p.provenance if e.source_url}
            if not pred_sources and not gt_sources:
                vals.append(1.0)
                continue
            if not pred_sources:
                vals.append(0.0)
                continue
            score = len(gt_sources & pred_sources) / len(pred_sources)
            vals.append(score)
            rows.append({"gt": g.claim_id, "pred": p.claim_id, "gt_sources": sorted(gt_sources), "pred_sources": sorted(pred_sources), "score": score})
        return EvaluatorScore(name=self.name, score=mean(vals) if vals else 0.0, details={"rows": rows})


class GroundingLocalizationEvaluator(BaseEvaluator):
    """Blueprint: LegalBench-RAG precise span retrieval and ALiiCE fine-grained citation localization.
    Uses source-constrained span overlap on value and citation character ranges.

    For indirect/formula-derived values, computes per-parameter span overlap
    averaged across all parameters instead of one combined range.
    """

    name = "grounding_localization"

    @staticmethod
    def _per_param_overlap(ge_params, pe_params) -> tuple[float, list]:
        """Compute average best-overlap across GT parameters matched to pred parameters.

        Returns (score, per_param_detail) where per_param_detail is a list of dicts
        with parameter-level observability information.
        """
        if not ge_params:
            return 0.0, []
        param_details = []
        param_overlaps = []
        for ge_pe in ge_params:
            best_overlap = 0.0
            best_match_name = None
            best_gt_span = None
            best_pred_span = None
            ge_name = ge_pe.parameter_name.lower().strip()
            for pe_pe in (pe_params or []):
                pe_name = pe_pe.parameter_name.lower().strip()
                # Match by parameter name (allow substring match for synonyms)
                if ge_name == pe_name or ge_name in pe_name or pe_name in ge_name:
                    for gs in ge_pe.root_value_spans:
                        for ps in pe_pe.root_value_spans:
                            ovl = overlap_ratio(gs, ps)
                            if ovl > best_overlap:
                                best_overlap = ovl
                                best_match_name = pe_pe.parameter_name
                                best_gt_span = gs
                                best_pred_span = ps
            param_overlaps.append(best_overlap)
            param_details.append({
                "gt_param": ge_pe.parameter_name,
                "gt_value": ge_pe.target_value,
                "gt_span": best_gt_span,
                "matched_pred_param": best_match_name,
                "pred_span": best_pred_span,
                "overlap": round(best_overlap, 4),
            })
        score = mean(param_overlaps) if param_overlaps else 0.0
        return score, param_details

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        vals = []
        rows = []
        skipped_no_provenance = 0
        for m in matches:
            if m.gt_only or m.pred_only:
                vals.append(0.0)
                continue
            g = gt_index.get(m.gt_claim_id)
            p = pred_index.get(m.pred_claim_id)
            assert g and p
            # Graceful degradation: if GT claim has no provenance, skip it
            # rather than penalizing with 0.0 (which collapses the geometric mean)
            if not g.provenance:
                skipped_no_provenance += 1
                rows.append({"gt": g.claim_id, "pred": p.claim_id, "skipped": "gt_no_provenance"})
                continue
            best = 0.0
            best_pair = None
            for ge in g.provenance:
                for pe in p.provenance:
                    if ge.source_url and pe.source_url and ge.source_url != pe.source_url:
                        continue
                    # Per-parameter overlap for indirect groundings.
                    # Use per-param path if EITHER side has per_parameter_evidence.
                    per_param_detail = None
                    if ge.per_parameter_evidence or pe.per_parameter_evidence:
                        # Use GT params as reference; compare against pred params
                        ref_params = ge.per_parameter_evidence or pe.per_parameter_evidence
                        cmp_params = pe.per_parameter_evidence if ge.per_parameter_evidence else ge.per_parameter_evidence
                        v, per_param_detail = self._per_param_overlap(ref_params, cmp_params)
                    else:
                        v = overlap_ratio(ge.value_index_range, pe.value_index_range)
                    c = overlap_ratio(ge.citation_index_range, pe.citation_index_range)
                    same_source = float(bool(ge.source_url and pe.source_url and ge.source_url == pe.source_url))
                    score = 0.5 * same_source + 0.3 * v + 0.2 * c
                    if score > best:
                        best = score
                        best_pair = {
                            "gt_source": ge.source_url,
                            "pred_source": pe.source_url,
                            "value_overlap": round(v, 4),
                            "citation_overlap": round(c, 4),
                            "is_indirect": bool(ge.per_parameter_evidence or pe.per_parameter_evidence),
                            "score": round(score, 4),
                        }
                        if per_param_detail:
                            best_pair["per_param_detail"] = per_param_detail
            vals.append(best)
            rows.append({"gt": g.claim_id, "pred": p.claim_id, "best": best_pair})
        return EvaluatorScore(
            name=self.name,
            score=mean(vals) if vals else 0.0,
            details={
                "rows": rows,
                "skipped_no_provenance": skipped_no_provenance,
                "evaluable_claims": len(vals),
            },
        )


class CitationSnippetSupportEvaluator(BaseEvaluator):
    """Coarse lexical snippet recovery score.

    Kept for backward compatibility. The newer CitationSupportJudgementEvaluator adds
    explicit full / partial / none support labels.
    """

    name = "citation_snippet_support"

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        vals = []
        rows = []
        for m in matches:
            if m.gt_only or m.pred_only:
                vals.append(0.0)
                continue
            g = gt_index.get(m.gt_claim_id)
            p = pred_index.get(m.pred_claim_id)
            assert g and p
            gt_support = list(g.support_snippets)
            pred_support = []
            for e in p.provenance:
                if e.citation_snippet:
                    pred_support.append(e.citation_snippet)
                if e.matched_snippet:
                    pred_support.append(e.matched_snippet)
                pred_support.extend(e.matched_snippet_pieces)
            score = exact_match_ratio(gt_support, pred_support)
            judgement = judge_citation_support(gt_support, pred_support)
            vals.append(score)
            rows.append(
                {
                    "gt": g.claim_id,
                    "pred": p.claim_id,
                    "gt_support_count": len(gt_support),
                    "pred_support_count": len(pred_support),
                    "score": score,
                    "support_label": judgement.label,
                }
            )
        return EvaluatorScore(name=self.name, score=mean(vals) if vals else 0.0, details={"rows": rows})


class ClaimCompletenessEvaluator(BaseEvaluator):
    """Blueprint: recall-style claim coverage from FActScore and RAGChecker.
    Measures whether GT claims were recovered at all after matching.
    """

    name = "claim_completeness"

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        gt_total = sum(1 for m in matches if not m.pred_only)
        gt_matched = sum(1 for m in matches if not m.pred_only and not m.gt_only)
        score = gt_matched / gt_total if gt_total else 1.0
        rows = []
        for m in matches:
            if m.pred_only:
                continue
            if m.gt_only:
                rows.append({"claim": m.gt_claim_id, "status": "unmatched", "gt": m.gt_claim_id, "pred": None})
            else:
                rows.append({"claim": m.gt_claim_id, "status": "matched", "gt": m.gt_claim_id, "pred": m.pred_claim_id})
        return EvaluatorScore(name=self.name, score=score, details={"matched_gt_claims": gt_matched, "total_gt_claims": gt_total, "rows": rows})


class HallucinationEvaluator(BaseEvaluator):
    """Blueprint: unsupported-claim / hallucination diagnostics from RAGChecker and FActScore."""

    name = "unsupported_claim_rate"

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        pred_only = sum(1 for m in matches if m.pred_only)
        total_pred = sum(1 for m in matches if m.pred_only or (not m.gt_only and not m.pred_only))
        rate = pred_only / total_pred if total_pred else 0.0
        rows = []
        for m in matches:
            if m.pred_only:
                rows.append({"claim": m.pred_claim_id, "status": "unsupported", "gt": None, "pred": m.pred_claim_id})
            elif not m.gt_only:
                rows.append({"claim": m.pred_claim_id, "status": "supported", "gt": m.gt_claim_id, "pred": m.pred_claim_id})
        return EvaluatorScore(name=self.name, score=1.0 - rate, details={"pred_only": pred_only, "total_pred": total_pred, "unsupported_rate": rate, "rows": rows})
