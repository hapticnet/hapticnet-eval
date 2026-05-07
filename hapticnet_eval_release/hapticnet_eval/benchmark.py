from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from .canonicalize import gt_to_claims, prediction_to_claims
from .evaluators.advanced import (
    CitationFaithfulnessEvaluator,
    CitationSupportJudgementEvaluator,
    ConflictingEvidenceEvaluator,
    CostEfficiencyEvaluator,
    LatencyEfficiencyEvaluator,
    NoAnswerEvaluator,
    SourceEquivalenceEvaluator,
    ToolUsageEfficiencyEvaluator,
)
from .evaluators.completeness import ValueListCoverageEvaluator
from .evaluators.openweb_evidence import OpenWebApproxEvidenceEvaluator
from .evaluators.statement_support import StatementSupportEvaluator
from .evaluators.strict_groundedness import StrictGroundednessEvaluator
from .evaluators.composite import weighted_geometric_mean
from .evaluators.llm_evaluators import CalibratedClaimJudgeEvaluator, SemanticConditionsEvaluator, VerifiedClaimsF1Evaluator, UnverifiedClaimRateEvaluator
from .evaluators.core import (
    BaseEvaluator,
    CitationSnippetSupportEvaluator,
    CitationSourcePrecisionEvaluator,
    CitationSourceRecallEvaluator,
    ClaimCompletenessEvaluator,
    ClaimIndex,
    ConditionsEvaluator,
    FactualValueEvaluator,
    GroundingLocalizationEvaluator,
    HallucinationEvaluator,
    HierarchyEvaluator,
)
from .evaluators.matching import match_claims
from .regimes.base import EvaluationTask
from .regimes.closed_docs import ClosedDocsRegime
from .regimes.open_web import OpenWebRegime
from .regimes.url_only import UrlOnlyRegime
from .schemas import EvaluationReport, GTFile
from .utils.io import load_json


_REGIMES = {
    "closed_docs": ClosedDocsRegime(),
    "url_only": UrlOnlyRegime(),
    "open_web": OpenWebRegime(),
}


DEFAULT_EVALUATORS: List[BaseEvaluator] = [
    FactualValueEvaluator(),
    ConditionsEvaluator(),
    CitationSourceRecallEvaluator(),
    CitationSourcePrecisionEvaluator(),
    CitationSnippetSupportEvaluator(),
    CitationSupportJudgementEvaluator(),
    GroundingLocalizationEvaluator(),
    HierarchyEvaluator(),
    ClaimCompletenessEvaluator(),
    HallucinationEvaluator(),
    SourceEquivalenceEvaluator(),
    CitationFaithfulnessEvaluator(),
    NoAnswerEvaluator(),
    ConflictingEvidenceEvaluator(),
    CostEfficiencyEvaluator(),
    LatencyEfficiencyEvaluator(),
    ToolUsageEfficiencyEvaluator(),
    StrictGroundednessEvaluator(),
    StatementSupportEvaluator(),
    ValueListCoverageEvaluator(),
    OpenWebApproxEvidenceEvaluator(),
    # NovelClaimJudgeEvaluator() removed — superseded by calibrated version below
    SemanticConditionsEvaluator(),
    VerifiedClaimsF1Evaluator(),
    UnverifiedClaimRateEvaluator(),
    CalibratedClaimJudgeEvaluator(),  # outputs name='verified_novel_claim_rate'
]

# Direct-query weights (default profile).
# semantic_conditions added at 0.12 (redistributed from strict_groundedness -0.05,
# citation_source_recall -0.02, grounding_localization -0.05).
DEFAULT_WEIGHTS = {
    "factual_value": 0.25,
    "conditions_and_specs": 0.0,              # DEPRECATED v2 — replaced by semantic_conditions
    "citation_source_recall": 0.10,           # was 0.12, gave 0.02 to semantic_conditions
    "citation_source_precision": 0.02,
    "citation_snippet_support": 0.0,          # DEPRECATED v2 — near-zero for both direct & indirect
    "grounding_localization": 0.15,           # was 0.20, gave 0.05 to semantic_conditions
    "hierarchy": 0.0,
    "claim_completeness": 0.03,
    "unsupported_claim_rate": 0.01,
    # Advanced / regime-specific evaluators. These are implemented and reported,
    # but kept out of the aggregate by default until the benchmark owners decide
    # per-track weighting policy.
    "citation_support_judgement": 0.08,
    "source_equivalence": 0.0,
    "citation_faithfulness": 0.0,
    "no_answer": 0.04,
    "conflicting_evidence": 0.0,
    "cost_efficiency": 0.0,
    "latency_efficiency": 0.0,
    "tool_usage_efficiency": 0.0,
    "strict_groundedness": 0.05,              # was 0.10, gave 0.05 to semantic_conditions
    "statement_support": 0.0,                 # DEPRECATED v2 — near-zero for indirect
    "value_list_coverage": 0.15,
    "open_web_approx_evidence": 0.0,
    "verified_novel_claim_rate": 0.0,          # Calibrated judge (E5_E1_E3 prompt)
    "semantic_conditions": 0.12,              # NEW — LLM judge for conditions
    "verified_claims_f1": 0.15,
    "unverified_claim_rate": 0.0,
}

# Indirect-query weights — optimized for derived/computed property extraction.
# Key difference from direct: strict_groundedness is too strict for derived values,
# so its remaining 0.05 weight is transferred to citation_support_judgement (LLM-based).
INDIRECT_WEIGHTS = {
    "factual_value": 0.25,
    "conditions_and_specs": 0.0,
    "citation_source_recall": 0.10,
    "citation_source_precision": 0.02,
    "citation_snippet_support": 0.0,
    "grounding_localization": 0.15,
    "hierarchy": 0.0,
    "claim_completeness": 0.03,
    "unsupported_claim_rate": 0.01,
    "citation_support_judgement": 0.13,       # was 0.08, +0.05 from strict_groundedness
    "source_equivalence": 0.0,
    "citation_faithfulness": 0.0,
    "no_answer": 0.04,
    "conflicting_evidence": 0.0,
    "cost_efficiency": 0.0,
    "latency_efficiency": 0.0,
    "tool_usage_efficiency": 0.0,
    "strict_groundedness": 0.0,               # zeroed — too strict for derived values
    "statement_support": 0.0,
    "value_list_coverage": 0.15,
    "open_web_approx_evidence": 0.0,
    "verified_novel_claim_rate": 0.0,
    "semantic_conditions": 0.12,
    "verified_claims_f1": 0.15,
    "unverified_claim_rate": 0.0,
    # Note: the evaluator class name is CalibratedClaimJudgeEvaluator but it
    # outputs scores under the key 'verified_novel_claim_rate' (see llm_evaluators.py:1069).
}

# Open-web / Deep Research weights — GUIDED variant.
# Used when the DR agent receives GT source URLs as guidance, so we expect
# it to find similar values and the factual_value/coverage evaluators are fair.
OPEN_WEB_GUIDED_WEIGHTS = {
    "factual_value": 0.25,                    # fair: guided URLs → similar values expected
    "conditions_and_specs": 0.0,
    "citation_source_recall": 0.0,            # zeroed — different source universe
    "citation_source_precision": 0.0,         # zeroed — same reason
    "citation_snippet_support": 0.0,
    "grounding_localization": 0.0,            # zeroed — no shared documents
    "hierarchy": 0.0,
    "claim_completeness": 0.05,
    "unsupported_claim_rate": 0.0,
    "citation_support_judgement": 0.0,        # zeroed — needs shared docs
    "source_equivalence": 0.0,
    "citation_faithfulness": 0.0,
    "no_answer": 0.05,
    "conflicting_evidence": 0.0,
    "cost_efficiency": 0.05,                  # track DR API costs
    "latency_efficiency": 0.05,               # track DR API latency
    "tool_usage_efficiency": 0.0,
    "strict_groundedness": 0.0,               # zeroed — no shared docs
    "statement_support": 0.0,
    "value_list_coverage": 0.15,              # fair: guided → expect same values
    "open_web_approx_evidence": 0.0,
    "verified_novel_claim_rate": 0.0,
    "semantic_conditions": 0.0,               # zeroed — needs grounded conditions
    "verified_claims_f1": 0.0,                # zeroed — needs calibrated judge on shared docs
    "unverified_claim_rate": 0.0,
    "verified_novel_claim_rate": 0.40,         # primary quality signal for open-web (CalibratedClaimJudgeEvaluator)
}

# Open-web / Deep Research weights — UNGUIDED variant.
# Used when the DR agent searches freely without GT hints.
# Cannot fairly compare factual values (different source universe → may find
# correct values not in GT), so 100% weight goes to calibrated judge verification.
OPEN_WEB_UNGUIDED_WEIGHTS = {
    "factual_value": 0.0,                     # zeroed — unfair when source universe unknown
    "conditions_and_specs": 0.0,
    "citation_source_recall": 0.0,
    "citation_source_precision": 0.0,
    "citation_snippet_support": 0.0,
    "grounding_localization": 0.0,
    "hierarchy": 0.0,
    "claim_completeness": 0.0,
    "unsupported_claim_rate": 0.0,
    "citation_support_judgement": 0.0,
    "source_equivalence": 0.0,
    "citation_faithfulness": 0.0,
    "no_answer": 0.00,
    "conflicting_evidence": 0.0,
    "cost_efficiency": 0.05,
    "latency_efficiency": 0.05,
    "tool_usage_efficiency": 0.0,
    "strict_groundedness": 0.0,
    "statement_support": 0.0,
    "value_list_coverage": 0.0,               # zeroed — different source universe
    "open_web_approx_evidence": 0.0,
    "verified_novel_claim_rate": 0.0,
    "semantic_conditions": 0.0,
    "verified_claims_f1": 0.0,
    "unverified_claim_rate": 0.0,
    "verified_novel_claim_rate": 0.90,         # 100% of quality signal from judge verification (CalibratedClaimJudgeEvaluator)
}

# Backward-compatible alias — defaults to guided variant.
OPEN_WEB_WEIGHTS = OPEN_WEB_GUIDED_WEIGHTS


@dataclass
class BenchmarkRunner:
    """Orchestrates evaluation tasks with optional source fetching.

    Attributes:
        evaluators: List of evaluator instances to run.
        weights: Weight profile for aggregate scoring.
        fetch_missing_sources: If True, fetch missing source documents
            via Tavily Extract before running evaluators.
        source_cache_dir: Directory for cached fetched content.
            Defaults to ~/.hapticnet_source_cache.
    """
    evaluators: List[BaseEvaluator] = None
    weights: Dict[str, float] = None
    fetch_missing_sources: bool = False
    source_cache_dir: str = ""

    def __post_init__(self) -> None:
        if self.evaluators is None:
            self.evaluators = DEFAULT_EVALUATORS
        if self.weights is None:
            self.weights = DEFAULT_WEIGHTS
        if not self.source_cache_dir:
            import os as _os
            self.source_cache_dir = _os.path.expanduser("~/.hapticnet_source_cache")

    def evaluate_task(self, task: EvaluationTask) -> EvaluationReport:
        import os as _os
        import logging

        gt_obj = load_json(task.gt_path)
        pred_obj = load_json(task.pred_path)
        # Merge run_metadata into pred_obj so resource evaluators can find
        # cost, latency, and tool-call metrics from simple LLM runs.
        run_meta_path = task.pred_path.replace(".json", "_run_metadata.json")
        if _os.path.isfile(run_meta_path):
            try:
                run_meta = load_json(run_meta_path)
                pred_obj.setdefault("run_metadata", {}).update(run_meta)
            except Exception:
                pass
        regime = _REGIMES.get(task.regime)
        if regime is None:
            raise ValueError(f"Unknown regime: {task.regime}")
        gt_obj, pred_obj, regime_meta = regime.preprocess(gt_obj, pred_obj)

        gt_model = GTFile.model_validate(gt_obj)
        gt_claims = gt_to_claims(gt_model)
        pred_claims = prediction_to_claims(pred_obj)

        source_index_base = _os.environ.get(
            "HAPTICNET_SOURCE_INDEX", "/mnt/cgm-atlas/ofri/HapticNet/SourceIndex"
        )

        # ── Evidence Fetcher: scrape-on-demand for missing sources ────────
        source_fetch_stats = None
        if self.fetch_missing_sources:
            from .evidence_fetcher import EvidenceFetcher
            logger = logging.getLogger(__name__)
            logger.info(
                "EvidenceFetcher: Scanning %d pred claims for missing sources...",
                len(pred_claims),
            )
            fetcher = EvidenceFetcher(cache_dir=self.source_cache_dir)
            source_fetch_stats = fetcher.ensure_sources(
                pred_claims, source_index_base=source_index_base,
            )
            logger.info(
                "EvidenceFetcher: %s",
                {k: v for k, v in source_fetch_stats.items() if k != "records"},
            )

        matches = match_claims(gt_claims, pred_claims)

        gt_index = ClaimIndex(gt_claims)
        pred_index = ClaimIndex(pred_claims)
        context = {
            "gt_obj": gt_obj,
            "pred_obj": pred_obj,
            "gt_model": gt_model,
            "gt_claims": gt_claims,
            "pred_claims": pred_claims,
            "task": task,
            "regime": task.regime,
            "haptic_property": gt_model.haptic_property,
            "material_class": gt_model.material_class,
            "source_index_base": source_index_base,
            "source_cache_dir": self.source_cache_dir,
        }
        if source_fetch_stats:
            context["source_fetch_stats"] = source_fetch_stats

        # Run evaluators, passing prior scores in context for cross-evaluator dependencies
        scores = []
        for ev in self.evaluators:
            # Make prior scores available to downstream evaluators (e.g. VerifiedClaimsF1)
            context["_prior_scores"] = {s.name: s.details for s in scores}
            scores.append(ev.evaluate(gt_index, pred_index, matches, context=context))
        aggregate = weighted_geometric_mean(scores, self.weights)

        metadata = {
            **task.metadata,
            **regime_meta,
            "num_gt_claims": len(gt_claims),
            "num_pred_claims": len(pred_claims),
        }
        if source_fetch_stats:
            metadata["source_fetch_stats"] = source_fetch_stats

        return EvaluationReport(
            task_id=task.task_id,
            regime=task.regime,
            aggregate_score=aggregate,
            scores=scores,
            matches=matches,
            metadata=metadata,
        )

    def evaluate_manifest(self, manifest: Iterable[EvaluationTask]) -> List[EvaluationReport]:
        return [self.evaluate_task(task) for task in manifest]
