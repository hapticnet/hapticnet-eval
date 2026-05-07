from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List

from .core import BaseEvaluator, ClaimIndex, FactualValueEvaluator
from ..schemas import EvaluatorScore, MatchResult
from ..utils.evidence import (
    _tokenize,
    assess_citation_faithfulness,
    build_claim_anchors,
    canonicalize_url,
    efficiency_score,
    judge_citation_support,
    number_variants_from_string,
    source_equivalence,
)
from ..utils.normalization import normalize_text


class SourceEquivalenceEvaluator(BaseEvaluator):
    """Open-web oriented source comparison.

    Blueprint inspiration: citation-aware evaluation in ALCE/ALiiCE, but adapted to the
    benchmark reality that equivalent sources may appear under mirrored URLs or URL variants.
    This is still a conservative equivalence check, not a full "alternative valid source" judge.
    """

    name = "source_equivalence"

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        context = context or {}
        gt_obj = context.get("gt_obj", {})
        pred_obj = context.get("pred_obj", {})
        gt_sources = {s.get("url"): s for s in gt_obj.get("sources", []) if s.get("url")}
        pred_sources = {s.get("url"): s for s in pred_obj.get("sources", []) if s.get("url")}

        vals = []
        rows = []
        for m in matches:
            if m.gt_only or m.pred_only:
                continue
            g = gt_index.get(m.gt_claim_id)
            p = pred_index.get(m.pred_claim_id)
            assert g and p
            gt_evidence = [(e.source_url, e.source_uid, gt_sources.get(e.source_url, {}).get("title")) for e in g.provenance if e.source_url]
            pred_evidence = [(e.source_url, e.source_uid, pred_sources.get(e.source_url, {}).get("title")) for e in p.provenance if e.source_url]
            if not gt_evidence and not pred_evidence:
                vals.append(1.0)
                continue
            if not gt_evidence or not pred_evidence:
                vals.append(0.0)
                continue
            best_scores = []
            pair_rows = []
            for pu, puid, ptitle in pred_evidence:
                best = 0.0
                best_method = None
                best_gt = None
                for gu, guid, gtitle in gt_evidence:
                    assess = source_equivalence(pu, ptitle, gu, gtitle, puid, guid)
                    if assess.score > best:
                        best = assess.score
                        best_method = assess.method
                        best_gt = gu
                best_scores.append(best)
                pair_rows.append({"pred_source": pu, "best_gt_source": best_gt, "score": best, "method": best_method})
            score = mean(best_scores) if best_scores else 0.0
            vals.append(score)
            rows.append({"gt": g.claim_id, "pred": p.claim_id, "pairs": pair_rows, "score": score})
        return EvaluatorScore(name=self.name, score=mean(vals) if vals else 0.0, details={"rows": rows})


class CitationSupportJudgementEvaluator(BaseEvaluator):
    """Full / partial / no-support citation judgement.

    Blueprint inspiration: ALiiCE and fine-grained citation evaluation work that separates
    degrees of support instead of binary citation correctness.
    """

    name = "citation_support_judgement"

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        vals = []
        rows = []
        label_counts = {"full": 0, "partial": 0, "none": 0}
        for m in matches:
            if m.gt_only or m.pred_only:
                vals.append(0.0)
                label_counts["none"] += 1
                continue
            g = gt_index.get(m.gt_claim_id)
            p = pred_index.get(m.pred_claim_id)
            assert g and p
            pred_evidence = []
            for e in p.provenance:
                if e.citation_snippet:
                    pred_evidence.append(e.citation_snippet)
                if e.matched_snippet:
                    pred_evidence.append(e.matched_snippet)
                pred_evidence.extend(e.matched_snippet_pieces)
            judgement = judge_citation_support(g.support_snippets, pred_evidence)
            vals.append(judgement.score)
            label_counts[judgement.label] += 1
            rows.append({
                "gt": g.claim_id,
                "pred": p.claim_id,
                "label": judgement.label,
                "score": judgement.score,
                "lexical_overlap": judgement.lexical_overlap,
                "numeric_overlap": judgement.numeric_overlap,
                "best_gt_snippet": judgement.best_gt_snippet,
            })
        return EvaluatorScore(name=self.name, score=mean(vals) if vals else 0.0, details={"rows": rows, "label_counts": label_counts})


class CitationFaithfulnessEvaluator(BaseEvaluator):
    """Approximate anti-post-rationalization check.

    Blueprint inspiration: work showing that citation correctness is not the same as faithfulness.
    This implementation is a pragmatic approximation based on anchor coverage and support strength.
    
    For indirect/formula-derived claims, anchors are built from the component parameter values
    (e.g., thermal conductivity, density, specific heat) rather than the final computed value.
    If a parameter was derived via logical jump, the root values are also checked.
    """

    name = "citation_faithfulness"

    @staticmethod
    def _build_indirect_anchors(claim, evidence_list) -> tuple[set, set]:
        """Build anchors from per-parameter evidence for indirect claims.
        
        Returns (text_anchors, numeric_anchors) where numeric_anchors contains
        the component parameter values AND root values (if the parameter was
        derived via logical_jump or source_note_derived).
        """
        text_anchors = set()
        numeric_anchors = set()
        
        if claim.material_subclass and claim.material_subclass != "unspecified":
            text_anchors |= _tokenize(claim.material_subclass)
        else:
            text_anchors |= _tokenize(claim.material_class)
        
        for prov in claim.provenance:
            if not prov.per_parameter_evidence:
                continue
            for pe in prov.per_parameter_evidence:
                # Add parameter name as text anchor
                text_anchors |= _tokenize(pe.parameter_name)
                # Add target value as numeric anchor
                numeric_anchors |= number_variants_from_string(pe.target_value)
                # For derived params (logical_jump, source_note_derived),
                # also check root values — these are what actually appear
                # in the citation text
                if pe.grounding_type in ("logical_jump", "source_note_derived"):
                    # Root values are stored as spans, but we need to find
                    # the actual numeric values. Check if they're available
                    # via the original grounding data.
                    pass  # Root value text is not stored in ParameterEvidence
                    # but we still check target_value which is the useful signal
            break  # Use first provenance with PPE
        
        return text_anchors, numeric_anchors

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        value_eval = FactualValueEvaluator()
        value_report = value_eval.evaluate(gt_index, pred_index, matches, context)
        value_by_pair = {(r["gt"], r["pred"]): r["score"] for r in value_report.details.get("rows", [])}

        vals = []
        rows = []
        suspicious = 0
        considered = 0
        for m in matches:
            if m.gt_only or m.pred_only:
                vals.append(0.0)
                continue
            g = gt_index.get(m.gt_claim_id)
            p = pred_index.get(m.pred_claim_id)
            assert g and p
            evidence_texts = []
            for e in p.provenance:
                if e.citation_snippet:
                    evidence_texts.append(e.citation_snippet)
                if e.matched_snippet:
                    evidence_texts.append(e.matched_snippet)
                evidence_texts.extend(e.matched_snippet_pieces)
            support = judge_citation_support(g.support_snippets, evidence_texts)
            
            # Check if this is an indirect claim with per_parameter_evidence
            has_ppe = any(prov.per_parameter_evidence for prov in p.provenance)
            is_indirect = False
            
            if has_ppe:
                is_indirect = True
                text_anchors, numeric_anchors = self._build_indirect_anchors(p, evidence_texts)
            else:
                text_anchors, numeric_anchors = build_claim_anchors(
                    material_subclass=g.material_subclass,
                    material_class=g.material_class,
                    measurement_conditions=g.measurement_conditions,
                    material_specifications=g.material_specifications,
                    normalized_value=g.normalized_value,
                    range_min=g.range_min,
                    range_max=g.range_max,
                    data_points=g.data_points,
                )
            
            hint = value_by_pair.get((g.claim_id, p.claim_id), 0.0)
            assessment = assess_citation_faithfulness(evidence_texts, text_anchors, numeric_anchors, support, hint)
            considered += 1
            if assessment.likely_post_rationalized:
                suspicious += 1
            vals.append(assessment.score)
            rows.append({
                "gt": g.claim_id,
                "pred": p.claim_id,
                "score": assessment.score,
                "support_label": assessment.support_label,
                "numeric_anchor_coverage": assessment.numeric_anchor_coverage,
                "entity_anchor_coverage": assessment.entity_anchor_coverage,
                "likely_post_rationalized": assessment.likely_post_rationalized,
                "is_indirect": is_indirect,
                "anchor_type": "per_parameter" if is_indirect else "final_value",
            })
        return EvaluatorScore(
            name=self.name,
            score=mean(vals) if vals else 0.0,
            details={"rows": rows, "likely_post_rationalized_count": suspicious, "considered": considered},
        )


class NoAnswerEvaluator(BaseEvaluator):
    """Abstention evaluator for explicit or implicit no-answer tasks."""

    name = "no_answer"

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        context = context or {}
        gt_obj = context.get("gt_obj", {})
        pred_obj = context.get("pred_obj", {})

        gt_answer_exists_flags = [s.get("answer_exists") for s in gt_obj.get("sources", []) if isinstance(s, dict) and s.get("answer_exists") is not None]
        gt_no_answer = (len(gt_index.claims) == 0) or (gt_answer_exists_flags and not any(gt_answer_exists_flags))
        pred_no_answer = bool(pred_obj.get("no_answer") or pred_obj.get("abstain") or pred_obj.get("insufficient_evidence")) or len(pred_index.claims) == 0
        score = float(gt_no_answer == pred_no_answer)
        return EvaluatorScore(name=self.name, score=score, details={"gt_no_answer": gt_no_answer, "pred_no_answer": pred_no_answer})


class ConflictingEvidenceEvaluator(BaseEvaluator):
    """Checks whether systems keep conflicting / condition-dependent claims separated.

    The evaluator searches for GT groups that contain multiple distinct condition/value signatures
    under the same material family/class/property. It then checks whether the prediction preserved
    that multiplicity with distinct matched prediction claims.
    """

    name = "conflicting_evidence"

    @staticmethod
    def _group_key(claim) -> tuple:
        return (claim.material_family, claim.material_class, claim.haptic_property, claim.units)

    @staticmethod
    def _condition_signature(claim) -> tuple:
        return (claim.material_subclass, claim.measurement_conditions, claim.material_specifications, claim.object_class)

    @staticmethod
    def _value_signature(claim) -> tuple:
        if claim.value_type == "scalar":
            return ("scalar", round(claim.normalized_value or 0.0, 6))
        if claim.value_type == "range":
            return ("range", round(claim.range_min or 0.0, 6), round(claim.range_max or 0.0, 6))
        return ("series", tuple(round(x, 6) for x in (claim.data_points or ())))

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        groups: Dict[tuple, list] = {}
        for g in gt_index.claims:
            groups.setdefault(self._group_key(g), []).append(g)

        match_map = {m.gt_claim_id: m.pred_claim_id for m in matches if not m.gt_only and not m.pred_only}
        cluster_rows = []
        cluster_scores = []
        for key, claims in groups.items():
            cond_sigs = {self._condition_signature(c) for c in claims}
            value_sigs = {self._value_signature(c) for c in claims}
            if len(claims) < 2 or len(cond_sigs) < 2 or len(value_sigs) < 2:
                continue
            matched_pred_ids = {match_map[c.claim_id] for c in claims if c.claim_id in match_map}
            score = len(matched_pred_ids) / len(claims)
            cluster_scores.append(score)
            cluster_rows.append({
                "group": key,
                "gt_claim_ids": [c.claim_id for c in claims],
                "distinct_condition_signatures": len(cond_sigs),
                "distinct_value_signatures": len(value_sigs),
                "matched_pred_ids": sorted(pid for pid in matched_pred_ids if pid),
                "score": score,
            })

        score = mean(cluster_scores) if cluster_scores else 1.0
        return EvaluatorScore(name=self.name, score=score, details={"clusters": cluster_rows, "cluster_count": len(cluster_rows)})


class _BaseResourceEvaluator(BaseEvaluator):
    name = "resource_base"
    metric_keys: tuple[str, ...] = ()
    budget_key: str = ""
    default_budget: float = 1.0

    def _extract_metric(self, pred_obj: Dict[str, Any]) -> float | None:
        candidate_roots = [
            pred_obj.get("system_metrics", {}),
            pred_obj.get("run_metadata", {}),
            pred_obj.get("resource_usage", {}),
            pred_obj.get("execution_stats", {}),
            pred_obj.get("metrics", {}),
            pred_obj,
        ]
        for root in candidate_roots:
            if not isinstance(root, dict):
                continue
            for key in self.metric_keys:
                if key in root and root[key] is not None:
                    try:
                        return float(root[key])
                    except (TypeError, ValueError):
                        continue
        return None

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        context = context or {}
        pred_obj = context.get("pred_obj", {})
        task = context.get("task")
        budget = self.default_budget
        if task is not None and getattr(task, "metadata", None):
            budget = float(task.metadata.get(self.budget_key, budget))
        value = self._extract_metric(pred_obj)
        score = efficiency_score(value, budget, floor=0.0)
        return EvaluatorScore(name=self.name, score=score, details={"value": value, "budget": budget})


class CostEfficiencyEvaluator(_BaseResourceEvaluator):
    name = "cost_efficiency"
    metric_keys = ("total_cost_usd", "cost_usd", "usd_cost", "cost")
    budget_key = "cost_budget_usd"
    default_budget = 0.10


class LatencyEfficiencyEvaluator(_BaseResourceEvaluator):
    name = "latency_efficiency"
    metric_keys = ("latency_seconds", "wall_time_seconds", "runtime_seconds", "elapsed_seconds")
    budget_key = "latency_budget_seconds"
    default_budget = 30.0


class ToolUsageEfficiencyEvaluator(BaseEvaluator):
    name = "tool_usage_efficiency"

    def evaluate(self, gt_index: ClaimIndex, pred_index: ClaimIndex, matches: List[MatchResult], context: Dict[str, Any] | None = None) -> EvaluatorScore:
        context = context or {}
        pred_obj = context.get("pred_obj", {})
        task = context.get("task")
        candidate_roots = [
            pred_obj.get("system_metrics", {}),
            pred_obj.get("run_metadata", {}),
            pred_obj.get("resource_usage", {}),
            pred_obj.get("execution_stats", {}),
            pred_obj.get("metrics", {}),
            pred_obj,
        ]
        vals = {"tool_calls": None, "model_calls": None, "pages_fetched": None}
        alias_map = {
            "tool_calls": ("tool_calls", "num_tool_calls"),
            "model_calls": ("model_calls", "num_model_calls"),
            "pages_fetched": ("pages_fetched", "source_fetches", "documents_read"),
        }
        for root in candidate_roots:
            if not isinstance(root, dict):
                continue
            for out_key, aliases in alias_map.items():
                if vals[out_key] is not None:
                    continue
                for alias in aliases:
                    if alias in root and root[alias] is not None:
                        try:
                            vals[out_key] = float(root[alias])
                            break
                        except (TypeError, ValueError):
                            pass
        budgets = {"tool_calls": 10.0, "model_calls": 8.0, "pages_fetched": 20.0}
        if task is not None and getattr(task, "metadata", None):
            budgets["tool_calls"] = float(task.metadata.get("tool_calls_budget", budgets["tool_calls"]))
            budgets["model_calls"] = float(task.metadata.get("model_calls_budget", budgets["model_calls"]))
            budgets["pages_fetched"] = float(task.metadata.get("pages_fetched_budget", budgets["pages_fetched"]))
        components = {
            k: efficiency_score(vals[k], budgets[k], floor=0.0) for k in vals
        }
        available = [v for v in components.values() if v is not None]
        score = mean(available) if available else 0.0
        return EvaluatorScore(name=self.name, score=score, details={"metrics": vals, "budgets": budgets, "components": components})
