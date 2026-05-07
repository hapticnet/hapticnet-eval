import os
import re
from statistics import mean
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .core import BaseEvaluator, ClaimIndex
from ..schemas import EvaluatorScore, MatchResult, CanonicalClaim
from .llm_judge import LLMJudge, get_resource_tracker
from ..utils.normalization import relative_closeness
from ..evidence_fetcher import resolve_content_path

# Indirect property detection for novel claim pre-check
_INDIRECT_PROPS = {"thermal effusivity"}

_PARAM_SYNONYMS_NCJ = {
    "thermal conductivity": {"thermal conductivity", "tc", "k", "lambda", "λ"},
    "density": {"density", "rho", "ρ", "bulk density"},
    "specific heat capacity": {"specific heat capacity", "specific heat", "cp", "c_p", "c", "heat capacity"},
    "thermal diffusivity": {"thermal diffusivity", "alpha", "α", "diffusivity"},
}

_NUM_RE_NCJ = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _canonicalize_param_ncj(name: str) -> str | None:
    nl = name.lower().strip()
    nl = re.sub(r"\s*\([^)]*\)", "", nl).strip()
    for canon, synonyms in _PARAM_SYNONYMS_NCJ.items():
        if nl in synonyms or any(syn in nl for syn in synonyms if len(syn) > 2):
            return canon
    return None


def _extract_pred_params(claim: CanonicalClaim) -> dict[str, float]:
    """Extract component parameter values from a pred claim."""
    params: dict[str, float] = {}
    for prov in claim.provenance:
        if prov.per_parameter_evidence:
            for pe in prov.per_parameter_evidence:
                canon = _canonicalize_param_ncj(pe.parameter_name)
                if canon and canon not in params:
                    nums = _NUM_RE_NCJ.findall(pe.target_value)
                    if nums:
                        try:
                            params[canon] = float(nums[0])
                        except ValueError:
                            pass
            break
    if not params:
        for k, v in claim.measurement_conditions:
            kl = k.lower().strip()
            if kl in ("equation", "formula", "method", "unspecified"):
                continue
            canon = _canonicalize_param_ncj(kl)
            if canon and canon not in params:
                nums = _NUM_RE_NCJ.findall(v)
                if nums:
                    try:
                        params[canon] = float(nums[0])
                    except ValueError:
                        pass
    return params


def _extract_gt_params(claim: CanonicalClaim) -> dict[str, float]:
    """Extract component parameter values from a GT claim."""
    return _extract_pred_params(claim)  # Same logic for GT

class NovelClaimJudgement(BaseModel):
    is_valid: bool = Field(description="True if the extracted claim is strictly supported by the evidence text.")
    confidence: float = Field(description="Confidence score from 0.0 to 1.0 that the evidence explicitly supports the claim.")
    reasoning: str = Field(description="Detailed step-by-step reasoning explaining why the evidence supports or does not support the extracted claim.")

class SemanticConditionJudgement(BaseModel):
    similarity_score: float = Field(description="Confidence score from 0.0 to 1.0 representing how semantically similar the predicted conditions are to the ground truth conditions.")
    reasoning: str = Field(description="Detailed explanation of the semantic differences or equivalencies between both sets of conditions.")

# ── UNCALIBRATED PROMPT (superseded by CalibratedClaimJudgeEvaluator below) ──
# Kept for reference; the calibrated version achieved 0.730 AUROC vs 0.648 baseline.
class _UncalibratedNovelClaimJudgeEvaluator(BaseEvaluator):
    """
    [UNCALIBRATED - DO NOT USE DIRECTLY]
    Evaluates 'pred_only' claims (predictions not matched to any GT claim).
    Uses an LLM-as-a-judge to verify if the claim is factual based on its own evidence snippets.
    
    STRICT EVIDENCE GATING: Only calls the LLM if the claim has grounded provenance
    (at least one provenance record with a valid source_uid). Claims without grounding
    are scored 0.0 immediately — ungrounded claims cannot be reliably verified and
    are prime candidates for hallucination.
    """
    name = "_uncalibrated_novel_claim_rate"

    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        try:
            self.judge = LLMJudge()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("LLM Judge unavailable: %s", e)
            self.judge = None

    @staticmethod
    def _merge_windows(windows: List[tuple]) -> List[tuple]:
        """Merge overlapping (start, end, labels) windows into consolidated ranges."""
        if not windows:
            return []
        # Sort by start position
        sorted_w = sorted(windows, key=lambda x: x[0])
        merged = [list(sorted_w[0])]  # [start, end, labels_set]
        for start, end, labels in sorted_w[1:]:
            if start <= merged[-1][1]:
                # Overlapping — extend the end and merge labels
                merged[-1][1] = max(merged[-1][1], end)
                merged[-1][2] = merged[-1][2] | labels
            else:
                merged.append([start, end, labels])
        return [(s, e, lbl) for s, e, lbl in merged]

    def _gather_evidence(self, p: CanonicalClaim, context: Dict[str, Any] | None) -> str:
        """
        Gather evidence text for a pred_only claim from GROUNDED sources only.
        
        For each unique source_uid in provenance:
        1. Citation/matched snippets from provenance records
        2. For INDIRECT groundings: per-parameter text windows (3000 chars/side),
           merged when overlapping. Each window labeled with parameter name.
        3. For DIRECT groundings: single text window around value_index_range.
        
        Each source's evidence is labeled with its UID so the judge can
        see which component parameter came from which source.
        """
        # Group provenance by source_uid to ensure multi-source coverage
        source_evidence: Dict[str, List[str]] = {}  # uid → list of text chunks
        seen_uids = []

        for prov in p.provenance:
            uid = prov.source_uid or "_unknown"
            if uid not in source_evidence:
                source_evidence[uid] = []
                seen_uids.append(uid)
            if prov.citation_snippet and prov.citation_snippet not in source_evidence[uid]:
                source_evidence[uid].append(prov.citation_snippet)
            if prov.matched_snippet and prov.matched_snippet not in source_evidence[uid]:
                source_evidence[uid].append(prov.matched_snippet)

        # Add source document text for each unique source
        if context and "source_index_base" in context:
            for uid in seen_uids:
                if uid == "_unknown":
                    continue
                content_path = resolve_content_path(
                    uid,
                    context.get("source_index_base", ""),
                    context.get("source_cache_dir", ""),
                )
                if not content_path:
                    continue
                try:
                    with open(content_path, "r", errors="ignore") as f:
                        doc_text = f.read()
                except Exception:
                    continue

                # Check if any provenance for this UID has per-parameter evidence
                has_param_evidence = False
                for prov in p.provenance:
                    if prov.source_uid == uid and prov.per_parameter_evidence:
                        has_param_evidence = True
                        break

                if has_param_evidence:
                    # INDIRECT: Collect per-parameter windows, then merge overlapping ones
                    window_size = 3000
                    raw_windows: List[tuple] = []  # (start, end, {label_set})
                    for prov in p.provenance:
                        if prov.source_uid != uid or not prov.per_parameter_evidence:
                            continue
                        for pe in prov.per_parameter_evidence:
                            if pe.source_uid and pe.source_uid != uid:
                                continue
                            for span_start, span_end in pe.root_value_spans:
                                w_start = max(0, span_start - window_size)
                                w_end = min(len(doc_text), span_end + window_size)
                                label = f"{pe.parameter_name}={pe.target_value} ({pe.grounding_type})"
                                raw_windows.append((w_start, w_end, {label}))

                    if raw_windows:
                        merged = self._merge_windows(raw_windows)
                        for w_start, w_end, labels in merged:
                            label_str = ", ".join(sorted(labels))
                            window_text = doc_text[w_start:w_end]
                            source_evidence[uid].append(
                                f"[Source {uid[:12]} — parameters: {label_str}, "
                                f"chars {w_start}-{w_end}]:\n{window_text}"
                            )
                    elif len(doc_text) <= 20000:
                        source_evidence[uid].append(
                            f"[Source {uid[:12]} — full document ({len(doc_text)} chars)]:\n{doc_text}"
                        )
                    else:
                        source_evidence[uid].append(
                            f"[Source {uid[:12]} — first 20000 chars]:\n{doc_text[:20000]}"
                        )
                else:
                    # DIRECT: single window around value_index_range
                    best_prov = None
                    for prov in p.provenance:
                        if prov.source_uid == uid and prov.value_index_range and len(prov.value_index_range) >= 1:
                            best_prov = prov
                            break

                    if best_prov:
                        center = best_prov.value_index_range[0]
                        window_size = 3000
                        start = max(0, center - window_size)
                        end = min(len(doc_text), center + window_size)
                        window_text = doc_text[start:end]
                        source_evidence[uid].append(
                            f"[Source {uid[:12]} — document window chars {start}-{end}]:\n{window_text}"
                        )
                    elif len(doc_text) <= 20000:
                        source_evidence[uid].append(
                            f"[Source {uid[:12]} — full document ({len(doc_text)} chars)]:\n{doc_text}"
                        )
                    else:
                        source_evidence[uid].append(
                            f"[Source {uid[:12]} — first 20000 chars]:\n{doc_text[:20000]}"
                        )

        # Join all sources, labeled
        all_parts = []
        for uid in seen_uids:
            parts = source_evidence.get(uid, [])
            if parts:
                all_parts.append("\n".join(parts))
        return "\n\n--- SOURCE BOUNDARY ---\n\n".join(all_parts)

    def evaluate(
        self,
        gt_index: ClaimIndex,
        pred_index: ClaimIndex,
        matches: List[MatchResult],
        context: Dict[str, Any] | None = None,
    ) -> EvaluatorScore:
        if not self.judge:
            return EvaluatorScore(name=self.name, score=0.0, details={"error": "LLMJudge not initialized"})

        rows = []
        valid_novel_count = 0
        pred_only_count = 0
        call_index = 0  # Track which LLM call corresponds to which row

        query_property = ""
        if context:
            query_property = context.get("query_property", "") or context.get("haptic_property", "")

        for m in matches:
            if not m.pred_only:
                continue
            
            p = pred_index.get(m.pred_claim_id)
            assert p is not None
            pred_only_count += 1

            # --- Param-level pre-check for indirect claims ---
            # If this pred_only claim's component parameters all match a GT claim,
            # it's a matching failure (not genuine novelty). Skip the LLM call.
            param_precheck_result = None
            has_ppe = any(prov.per_parameter_evidence for prov in p.provenance)
            if has_ppe and p.haptic_property.lower().strip() in _INDIRECT_PROPS:
                pred_params = _extract_pred_params(p)
                if pred_params:
                    for gt_claim in gt_index.all_claims():
                        gt_params = _extract_gt_params(gt_claim)
                        if not gt_params:
                            continue
                        # Check if all pred params match this GT claim
                        matched_count = 0
                        total_count = len(pred_params)
                        for canon, pred_val in pred_params.items():
                            if canon in gt_params:
                                if relative_closeness(pred_val, gt_params[canon]) >= 0.85:
                                    matched_count += 1
                        if total_count > 0 and matched_count == total_count:
                            param_precheck_result = {
                                "type": "param_match",
                                "matched_gt": gt_claim.claim_id,
                                "matched_params": {k: v for k, v in pred_params.items()},
                            }
                            break

            if param_precheck_result:
                # All params matched a GT claim — this is a matching failure, not novelty
                score = 0.75
                reasoning = (f"PARAM-LEVEL PRE-CHECK: All component parameters match GT claim "
                            f"'{param_precheck_result['matched_gt']}'. This claim is NOT novel — "
                            f"it's the same extraction with a slightly different computed value. "
                            f"Scored 0.75 (matched but unrecognized by Hungarian matching).")
                evidence_str = ""
                evidence_for_display = ""
                call_resource = None
                has_grounded_provenance = True  # treat as grounded for display
            # STRICT EVIDENCE GATING: only call LLM if claim has grounded provenance
            elif not any(prov.source_uid for prov in p.provenance):
                has_grounded_provenance = False
                # No grounding → automatic 0.0, save the LLM call
                score = 0.0
                reasoning = ("SKIPPED — No grounded provenance. This claim has no source_uid in its "
                            "provenance records, meaning it was never grounded to a source document. "
                            "Ungrounded claims cannot be reliably verified and are not worth an LLM call.")
                evidence_str = ""
                evidence_for_display = ""
                call_resource = None
            else:
                has_grounded_provenance = True
                evidence_str = self._gather_evidence(p, context)
                evidence_for_display = evidence_str[:2000] if evidence_str else ""
                
                if not evidence_str.strip():
                    score = 0.0
                    reasoning = "Claim has grounded provenance but no evidence text could be extracted from the source document."
                    call_resource = None
                else:
                    # Count calls before this one to index the per-call resource record
                    tracker = get_resource_tracker()
                    calls_before = len([r for r in tracker.per_call_records if r["evaluator"] == self.name])

                    # Detect indirect/derived claims
                    # Check multiple signals: condition labels OR presence of per_parameter_evidence
                    _indirect_labels = {"equation", "formula"}
                    is_indirect_claim = any(
                        k.lower() in _indirect_labels for k, v in p.measurement_conditions
                    )
                    # Also detect via "method" label containing computation keywords
                    if not is_indirect_claim:
                        is_indirect_claim = any(
                            k.lower() == "method" and any(
                                kw in v.lower() for kw in ["computed", "derived", "calculated", "√", "sqrt"]
                            )
                            for k, v in p.measurement_conditions
                        )
                    # Final fallback: presence of per_parameter_evidence on any provenance
                    if not is_indirect_claim:
                        is_indirect_claim = any(
                            prov.per_parameter_evidence for prov in p.provenance
                        )
                    component_params = {k: v for k, v in p.measurement_conditions 
                                       if k.lower() not in ("equation", "formula", "method", "unspecified")}
                    
                    indirect_rule = ""
                    if is_indirect_claim:
                        indirect_rule = f"""
5. INDIRECT/DERIVED PROPERTY (CRITICAL — this claim uses formula computation):
   This claim is for an INDIRECTLY DERIVED property (e.g., thermal effusivity = √(λ·ρ·c)).
   The claimed value ({p.normalized_value or p.range_min or 'N/A'} {p.units}) was COMPUTED using a formula 
   from component parameters. It will NOT appear literally in the source evidence.
   
   Component parameters used: {', '.join(f'{k}={v}' for k,v in component_params.items())}
   
   The evidence below provides LABELED WINDOWS around each parameter's grounding location
   in the source document. Each window is tagged with the parameter name, value, and grounding type.
   
   For indirect claims, verify EACH of these instead:
   a) The COMPONENT PARAMETER VALUES (e.g., thermal conductivity, density, specific heat,
      or thermal diffusivity) must appear in or be derivable from the evidence.
      NOTE: There are MULTIPLE valid formulas. The primary is b = √(λ·ρ·c) using thermal
      conductivity + density + specific heat. An alternative is b = λ/√α using thermal
      conductivity + thermal diffusivity. BOTH are valid — accept whichever was used.
   b) Do NOT reject just because the final computed value doesn't appear in the evidence.
      The pipeline computed it from the evidenced components — that's the expected workflow.
   
   MATERIAL-LEVEL VALIDATION FOR COMPONENT PARAMETERS (CRITICAL):
   Each component parameter must be logically tied to the target material — not a proxy
   value from a different material. Note that the QUERY itself may be at CLASS level 
   (e.g., "cedar") or SUBCLASS level (e.g., "Western Red Cedar"). Both are valid targets.
   
   For each parameter's evidence window, assess whether it belongs to the target:
   
   c) CLASS-LEVEL QUERY with ALL class-level params → highest confidence (0.90-1.0)
      Example: Query "cedar thermal effusivity", all params from generic "cedar" data → ACCEPT
   d) SUBCLASS-LEVEL QUERY with ALL exact-subclass params → highest confidence (0.90-1.0)
      Example: Query "Western Red Cedar effusivity", all params from "Western Red Cedar" → ACCEPT
   e) MIXED: some params at subclass-level, others at class-level → acceptable (0.70-0.85)
      Example: Density from "Western Red Cedar" but specific heat from generic "wood" → PARTIAL
   f) Params from DIFFERENT material entirely → REJECT (0.0-0.3) 
      Example: Using aluminum density for a cedar calculation → REJECT
   g) Subclass query using OTHER subclass's params → REJECT (0.0-0.3)
      Example: Query "Western Red Cedar" but using "Eastern White Cedar" specific heat → REJECT
   
   When multiple evidence windows are provided, check EACH to verify the surrounding text
   mentions the target material (the class name OR the subclass name as appropriate).
"""

                    prompt = f"""You are a strict fact-checker for a materials science knowledge base. Determine if the following extracted material property claim is genuinely supported by the provided source evidence.

QUERY OF INTEREST: {query_property or p.haptic_property}

CLAIM DETAILS:
- Material: {p.material_class} (Subclass: {p.material_subclass})
- Property: {p.haptic_property}
- Value: {p.normalized_value or p.range_min or 'N/A'} {p.units}
- Conditions: {', '.join(f'{k}={v}' for k, v in p.measurement_conditions) or 'None'}
- Specifications: {', '.join(f'{k}={v}' for k, v in p.material_specifications) or 'None'}

EVIDENCE:
{evidence_str}

VERIFICATION CRITERIA — evaluate ALL of these step by step:

1. MATERIAL MATCH:
   - EXACT subclass match (e.g., "6061-T6") → accept.
   - CLASS-LEVEL match (e.g., evidence says "aluminum" for a claim about "6061 aluminum") → ACCEPT. This is a class-level answer and is valid.
   - DIFFERENT subclass (e.g., evidence says "7075-T6" but claim says "6061-T6") → REJECT unless the evidence clearly provides class-level (generic) data, not subclass-specific data.

2. PROPERTY TYPE MATCH (CRITICAL for friction, modulus, conductivity queries):
    ⚠️ EXCEPTION: If this claim has an `equation=` condition (i.e., it's an indirect/derived property),
    SKIP this rule entirely — Rule 5 applies instead. Seeing component properties like
    thermal conductivity in evidence for a thermal effusivity claim is EXPECTED and correct.
    
    For FRICTION properties specifically:
    - If the query is about KINETIC friction: evidence must either (a) explicitly say kinetic/dynamic/sliding friction, OR (b) say "friction coefficient" without specifying type (acceptable for both).
    - If the evidence EXPLICITLY states "STATIC friction" and the query is KINETIC friction → REJECT (confidence 0.0). These are different measurements.
    - EXAMPLE: A table row showing value 1.05 under a "Static" column when the query asks for kinetic friction → REJECT.
    - EXAMPLE: A graph title "Friction Coefficient vs Time" with annotation "μk = 0.435" → ACCEPT for kinetic friction.
    For OTHER property pairs (only when NOT an indirect/derived claim):
    - Young's modulus ≠ shear modulus ≠ bulk modulus. Thermal conductivity ≠ thermal effusivity.

3. COUNTER-MATERIAL / MATERIAL B MATCH (CRITICAL for friction):
   - The claim's counter-material (material sliding against) must match what the evidence describes.
   - If claim says "sliding against steel" but evidence data is for "aluminum on aluminum" → REJECT.
   - If the counter-material is not mentioned in the claim but is in the evidence, that's acceptable.

4. VALUE VERIFICATION:
   - The specific numeric value must appear in or be reasonably derivable from the evidence.
   - Small rounding differences are acceptable (0.38 vs 0.379).
{indirect_rule}
SCORING:
- 0.0: Evidence contradicts claim (wrong property type, wrong material, wrong counter-material)
- 0.3-0.5: Evidence is ambiguous but somewhat supportive
- 0.75-1.0: Evidence clearly and explicitly supports the claim
"""
                    judgement = self.judge.generate_structured(prompt, NovelClaimJudgement, evaluator_name=self.name)
                    
                    # Grab the per-call resource record for this specific call
                    tracker = get_resource_tracker()
                    all_records = [r for r in tracker.per_call_records if r["evaluator"] == self.name]
                    call_resource = all_records[calls_before] if len(all_records) > calls_before else None
                    
                    if judgement:
                        score = judgement.confidence
                        reasoning = judgement.reasoning
                    else:
                        score = 0.0
                        reasoning = "LLM Judge failed to produce a valid rating."

            is_verified = score >= self.threshold
            if is_verified:
                valid_novel_count += 1

            row = {
                "pred": p.claim_id,
                "value": str(p.normalized_value or p.range_min or "N/A"),
                "units": p.units,
                "material": f"{p.material_class} ({p.material_subclass})",
                "conditions": ", ".join(f"{k}={v}" for k, v in p.measurement_conditions) or "None",
                "specs": ", ".join(f"{k}={v}" for k, v in p.material_specifications) or "None",
                "has_grounded_provenance": has_grounded_provenance,
                "evidence_available": bool(evidence_str.strip()) if has_grounded_provenance else False,
                "evidence_sources": [prov.source_uid for prov in p.provenance if prov.source_uid],
                "evidence_text": evidence_for_display if has_grounded_provenance else "",
                "confidence": score,
                "is_verified": is_verified,
                "reasoning": reasoning,
            }
            if call_resource:
                row["call_resource"] = call_resource
            rows.append(row)

        final_score = (valid_novel_count / pred_only_count) if pred_only_count > 0 else 1.0
        
        tracker = get_resource_tracker()
        resource_summary = tracker.evaluator_summary(self.name)

        return EvaluatorScore(
            name=self.name,
            score=final_score,
            details={
                "pred_only_count": pred_only_count,
                "valid_novel_count": valid_novel_count,
                "rows": rows,
                "llm_resource_usage": resource_summary,
            }
        )

class SemanticConditionsEvaluator(BaseEvaluator):
    """
    Evaluates the semantic similarity of measurement conditions, material specifications,
    and object class between a matched prediction and GT claim using an LLM-as-a-judge.
    """
    name = "semantic_conditions"

    def __init__(self):
        try:
            self.judge = LLMJudge()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("LLM Judge unavailable: %s", e)
            self.judge = None

    def evaluate(
        self,
        gt_index: ClaimIndex,
        pred_index: ClaimIndex,
        matches: List[MatchResult],
        context: Dict[str, Any] | None = None,
    ) -> EvaluatorScore:
        if not self.judge:
            return EvaluatorScore(name=self.name, score=0.0, details={"error": "LLMJudge not initialized"})

        vals = []
        rows = []

        query_property = ""
        if context:
            query_property = context.get("query_property", "") or context.get("haptic_property", "")

        for m in matches:
            if m.gt_only or m.pred_only:
                vals.append(0.0)
                continue
            
            g = gt_index.get(m.gt_claim_id)
            p = pred_index.get(m.pred_claim_id)
            assert g and p

            gt_cond_str = ", ".join(f"{k}={v}" for k, v in g.measurement_conditions)
            gt_spec_str = ", ".join(f"{k}={v}" for k, v in g.material_specifications)
            gt_obj_str = ", ".join(g.object_class)
            pred_cond_str = ", ".join(f"{k}={v}" for k, v in p.measurement_conditions)
            pred_spec_str = ", ".join(f"{k}={v}" for k, v in p.material_specifications)
            pred_obj_str = ", ".join(p.object_class)

            all_gt_empty = not gt_cond_str and not gt_spec_str and not gt_obj_str
            all_pred_empty = not pred_cond_str and not pred_spec_str and not pred_obj_str
            if all_gt_empty and all_pred_empty:
                score = 1.0
                reasoning = "Both GT and prediction have completely empty conditions, specifications, and object class."
                call_resource = None
            elif gt_cond_str == pred_cond_str and gt_spec_str == pred_spec_str and gt_obj_str == pred_obj_str:
                # Exact string match — skip the LLM call (common in GT vs GT baseline)
                score = 1.0
                reasoning = "Exact match — GT and prediction conditions/specs/object_class are identical strings."
                call_resource = None
            else:
                tracker = get_resource_tracker()
                calls_before = len([r for r in tracker.per_call_records if r["evaluator"] == self.name])
                
                prompt = f"""You are a materials science domain expert performing a pseudo-F1 evaluation of measurement conditions and specifications.

TASK: Compare PREDICTED conditions/specs/object_class against GROUND TRUTH for semantic equivalence. Break each side into individual claims, then assess which GT claims are captured by the prediction and whether any prediction claims conflict with the GT.

QUERY CONTEXT: This is a measurement of "{query_property or g.haptic_property}" for "{g.material_class}".

GROUND TRUTH:
  Measurement Conditions: {gt_cond_str or 'None'}
  Material Specifications: {gt_spec_str or 'None'}
  Object Class: {gt_obj_str or 'None'}

PREDICTED:
  Measurement Conditions: {pred_cond_str or 'None'}
  Material Specifications: {pred_spec_str or 'None'}
  Object Class: {pred_obj_str or 'None'}

═══════════════════════════════════════════════════
SCORING RULES — apply these strictly in order:
═══════════════════════════════════════════════════

RULE 0: UNSPECIFIED / PLACEHOLDER HANDLING (HIGHEST PRIORITY — apply BEFORE all other rules)
  Treat the following as semantically EMPTY (i.e., no useful information):
    - "unspecified", "N/A", "not specified", "unknown", "None"
  If ALL GT conditions/specs/object_class contain only such placeholders AND
  ALL Pred conditions/specs/object_class are also empty or placeholders → score 1.0
  If GT has REAL conditions plus some unspecified placeholders → IGNORE the placeholders,
  evaluate only the real conditions against the prediction.
  Example: GT="temperature=unspecified, purity=unspecified" and Pred="None" → 1.0 (both uninformative)
  Example: GT="temperature=25°C, purity=unspecified" and Pred="None" → 0.0 (real content missing)
  Example: GT="temperature=unspecified" and Pred="temperature=25°C" → 0.70 (Pred has useful non-contradictory info)

PROPERTY-SPECIFIC CONTEXT (determines which factors matter most):
  - For FRICTION properties: counter-material (material B) is the highest priority factor;
    surface condition (dry/lubricated) is also important.
  - For ROUGHNESS properties: measurement instrument and method are the highest priority factors.
  - For THERMAL properties: measurement method (e.g., guarded hot plate, laser flash) is most important.
  - For MECHANICAL properties (Young's modulus, Poisson's ratio): test standard (ASTM, ISO) and
    direction/axis of test (longitudinal, transverse, through-thickness) are most important.
  GENERAL EMPHASIS across ALL properties: temperature, relative humidity, and other test environment
  conditions are always valuable when present.

GENERAL SCORING PRINCIPLE:
  Unless a property has a special rule that shifts the logic (e.g., counter-material for friction),
  split credit roughly evenly across the three categories:
    - Measurement Conditions: ~0.33
    - Material Specifications: ~0.33
    - Object Class: ~0.33
  Score each category based on match quality, then combine.

RULE 1: COUNTER-MATERIAL / MATERIAL B (HIGHEST PRIORITY for FRICTION properties only)
  1a. SAME counter-material → base score 0.85
      Example: GT="material b=steel" and Pred="counterface=steel" → 0.85
      Example: GT="material b=aluminum (flat surface)" and Pred="sliding against=al 7075-t6" → 0.85 (7075-t6 IS aluminum)
  1b. DIFFERENT counter-material → score 0.0 REGARDLESS of other matches
  1c. UNCERTAIN whether same material → score 0.5-0.85

RULE 2: TEST METHOD / GEOMETRY (adds incremental value)
  2a. Counter-material matches + test method matches → 0.90-1.0
  2b. Counter-material matches + test method MISSING from Pred → keep 0.85
  2c. Counter-material matches + test method CONFLICTS → 0.70-0.80

RULE 3: MATERIAL SPECIFICATIONS (temper, grade, heat treatment)
  3a. Counter-material + test match + specs match → 1.0
  3b. Missing specs from Pred → minor or no penalty
  3c. Conflicting specs → reduce by 0.05-0.10
  3d. Pred has EXTRA specs not in GT → NO penalty if they don't contradict GT

RULE 4: EMPTY PREDICTION vs NON-EMPTY GT
  Score 0.0 ONLY if prediction has ZERO useful information across ALL three categories
  (conditions, specs, object_class). If prediction has at least ONE matching or relevant
  piece of information in ANY category, the minimum score should be 0.10-0.20.

RULE 5: NOVEL PREDICTION CONDITIONS — acceptable if non-contradictory

RULE 6: OBJECT CLASS (minor factor, ~0.05 impact unless it carries condition-like info)

RULE 7: INDIRECT / DERIVED PROPERTIES (applies to Thermal Effusivity, and any property computed from a formula)
  For properties computed from component parameters using a formula:
  PRIMARY formula: thermal effusivity = √(λ·ρ·c)  [thermal conductivity × density × specific heat]
  ALTERNATIVE formula: thermal effusivity = λ/√α  [thermal conductivity / √thermal diffusivity]
  BOTH formulas are equally valid — the pipeline may use either depending on available data.
  7a. FORMULA MATCH: If both GT and Pred specify the SAME formula (or equivalent notation), this is a
      strong positive signal. Award 0.30-0.40 just for formula match.
      "equation=b = sqrt(lambda * rho * c)" ≡ "formula=e = √(k · ρ · c_p)" ≡ "equation=b = (λ·ρ·c)^{{1/2}}"
      "equation=b = lambda / sqrt(alpha)" ≡ "formula=e = λ/√α" ≡ "equation=b = k/√a"
  7b. COMPONENT PARAMETER MATCH: Compare the component parameter VALUES used in each claim.
      - Same parameters with same values → 0.90-1.0
      - Same parameters with similar values (within ~20%) → 0.70-0.85
      - Same parameters with different values → 0.40-0.60 (formula matches but different sources)
      - Different parameter names that are semantically equivalent → still count as match
        (e.g., "thermal conductivity" = "λ" = "k"; "density" = "ρ" = "bulk density")
  7c. MIXED CONDITIONS: If one side has formula+params and the other has measurement conditions
      (e.g., temperature, sample method), these are complementary — score 0.40-0.60 for partial coverage.
  7d. DIRECT vs INDIRECT: If one claim was derived from a formula and the other was directly reported,
      formula conditions (equation, param values) and measurement conditions (temperature, method) are
      both valid. Score based on whatever overlap exists.
  7e. GT UNSPECIFIED + Pred has formula+params: If GT has unspecified/empty conditions but Pred provides
      useful non-contradictory derivation info (formula, parameter values), score 0.50-0.70.
      The Pred is providing useful info where GT offers nothing — reward this.

═══════════════════════════════════════════════════
LABELING CONVENTION EQUIVALENCES (treat as identical):
═══════════════════════════════════════════════════
- "material b=steel" = "sliding on=steel" = "counterface=steel" = "counter-surface=steel" = "sliding against=steel"
- "fof" = "flat-on-flat" = "flat surface on flat surface" = "flat surface sliding against flat surface"
- "dry" = "dry contact" = "unlubricated"
- "temper=t6" = "grade=t6" = "T6 temper" = "T6 (precipitation-hardened)"
- "coating=no" = "coating=none" = "uncoated"
- "thermal conductivity" = "λ" = "k" = "TC"
- "density" = "ρ" = "bulk density" = "rho"
- "specific heat capacity" = "c" = "c_p" = "Cp"
- "thermal diffusivity" = "α" = "alpha" = "a" (in m²/s)
- "equation" = "formula" = "derivation"

═══════════════════════════════════════════════════
CALIBRATION EXAMPLES (use these as reference):
═══════════════════════════════════════════════════
| GT Conditions | Pred Conditions | Target Score | Why |
|---|---|---|---|
| material b=steel, test=fof, temper=t6 | counterface=steel | 0.85 | Counter-material matches; missing test/specs not penalized heavily |
| material b=steel, test=fof | sliding against=steel, grade=t6 | 0.90 | Counter-material + spec captured |
| material b=steel, test=fof | sliding against=steel, fof, grade=t6 | 1.00 | Everything matches |
| material b=steel | sliding against=titanium | 0.00 | Counter-material mismatch |
| material b=steel, test=fof | test=fof, material2=titanium | 0.00 | Wrong counter-material despite correct test method |
| material b=aluminum, test=fof | counterface=aluminum | 0.85 | Counter-material matches |
| material b=aluminum, test=fof | sliding against=al 7075-t6 | 0.85 | 7075-T6 IS aluminum |
| material b=aluminum, test=fof | sliding against=aluminum, fof, temper=t6 | 0.95 | Nearly perfect match |
| ANY conditions present | None (all empty) | 0.00 | Pred provides no useful context |
| None (all empty) | None (all empty) | 1.00 | Both equally uninformative |
| temperature=unspecified, purity=unspecified | None | 1.00 | Both sides uninformative — Rule 0 |
| instrument=profiler, mode=vsi, area=370um | mode=vsi, area=370um | 0.90 | Core method captured |
| instrument=profiler, mode=vsi, area=370um | None | 0.00 | Relevant info exists but entirely missing |
| fiber_content=100% cotton (obj=woven fabric) | None (obj=woven fabric) | 0.50 | Conds OK (both empty), specs missing, obj matches |
| temperature=unspecified (specs=none) | temperature=25°C, surface=polished | 0.70 | Pred has useful non-contradictory conds, GT was unspec |
| fiber_content=100% cotton | fiber_content=cotton blend | 0.70 | Partial match, different blend |
| test_standard=ASTM D638, direction=longitudinal | test_standard=ASTM D638 | 0.80 | Core test standard captured, direction missing |
| equation=b=(λ·ρ·c)^0.5, λ=0.016, ρ=457.3, c=2147 | equation=b=sqrt(lambda*rho*c), TC=0.016, density=457.3, Cp=2147 | 0.95 | Same formula + same param values (Rule 7a+7b) |
| equation=b=(λ·ρ·c)^0.5, λ=0.016, ρ=457.3, c=2147 | equation=b=sqrt(lambda*rho*c), TC=0.019, density=457.3, Cp=2147 | 0.75 | Same formula, 2/3 params match, TC differs (Rule 7b) |
| equation=b=(λ·ρ·c)^0.5, λ=0.016, ρ=457.3, c=2147 | temperature=25°C, method=laser flash | 0.50 | Formula vs measurement — complementary info (Rule 7c) |
| unspecified=unspecified | equation=b=sqrt(lambda*rho*c), TC=0.09, density=818, Cp=1.26 | 0.60 | Pred has useful derivation info, GT uninformative — Rule 7e |
| equation=b=(λ·ρ·c)^0.5, λ=315, ρ=19.32, c=129 | equation=e=√(k·ρ·c_p), k=315, ρ=19.32, c_p=129 | 1.00 | Identical formula and params despite notation differences |
| equation=b=(λ·ρ·c)^0.5, λ=0.016, ρ=457.3, c=2147 | equation=b=λ/√α, λ=0.016, α=1.6e-7 | 0.40 | Different but equivalent formulas, partial param overlap (Rule 7a+7d) |
| equation=b=λ/√α, λ=237, α=9.7e-5 | equation=b=k/√a, k=237, alpha=9.7e-5 | 1.00 | Identical alternative formula + same param values |

Return a similarity_score reflecting the above rules. Use the examples to calibrate your score.
"""
                judgement = self.judge.generate_structured(prompt, SemanticConditionJudgement, evaluator_name=self.name)
                
                tracker = get_resource_tracker()
                all_records = [r for r in tracker.per_call_records if r["evaluator"] == self.name]
                call_resource = all_records[calls_before] if len(all_records) > calls_before else None
                
                if judgement:
                    score = judgement.similarity_score
                    reasoning = judgement.reasoning
                else:
                    score = 0.0
                    reasoning = "LLM Judge failed to produce a valid rating."

            vals.append(score)
            row = {
                "gt": g.claim_id,
                "pred": p.claim_id,
                "gt_conditions": gt_cond_str or "None",
                "gt_specs": gt_spec_str or "None",
                "gt_object_class": gt_obj_str or "None",
                "pred_conditions": pred_cond_str or "None",
                "pred_specs": pred_spec_str or "None",
                "pred_object_class": pred_obj_str or "None",
                "score": score,
                "reasoning": reasoning,
            }
            if call_resource:
                row["call_resource"] = call_resource
            rows.append(row)

        tracker = get_resource_tracker()
        resource_summary = tracker.evaluator_summary(self.name)

        return EvaluatorScore(
            name=self.name,
            score=mean(vals) if vals else 0.0,
            details={"rows": rows, "llm_resource_usage": resource_summary}
        )


class VerifiedClaimsF1Evaluator(BaseEvaluator):
    """
    Computes F1 over verified claims — combining match quality with novel claim verification.
    
    A pred claim is "verified" if:
      1. It matched a GT claim AND factual_value score > value_threshold (human-verified via GT), OR
      2. It's pred_only AND the NovelClaimJudge verified it (confidence >= novel_threshold).
    
    A pred claim is "falsified" if:
      1. It's pred_only AND the judge rejected it (ungrounded or low confidence), OR
      2. It matched BUT factual_value = 0.0 (wrong value despite matching material/property).
    
    Precision = verified_pred_claims / total_pred_claims
    Recall = matched_gt_with_good_value / total_gt_claims
    F1 = 2 * P * R / (P + R)
    
    This evaluator runs AFTER NovelClaimJudgeEvaluator and FactualValueEvaluator
    and reads their results from the context.
    """
    name = "verified_claims_f1"
    
    def __init__(self, value_threshold: float = 0.3, novel_threshold: float = 0.75):
        self.value_threshold = value_threshold
        self.novel_threshold = novel_threshold
    
    def evaluate(
        self,
        gt_index: ClaimIndex,
        pred_index: ClaimIndex,
        matches: List[MatchResult],
        context: Dict[str, Any] | None = None,
    ) -> EvaluatorScore:
        verified_pred = []
        falsified_pred = []
        matched_gt_with_value = 0
        total_gt = 0
        total_pred = 0
        
        # Gather factual_value scores from prior evaluator output
        fv_scores = {}  # (gt_claim_id, pred_claim_id) -> score
        novel_verdicts = {}  # pred_claim_id -> confidence
        if context:
            prior_scores = context.get("_prior_scores", {})
            # Factual value per-match scores
            fv_details = prior_scores.get("factual_value", {})
            for row in fv_details.get("rows", []):
                fv_scores[(row.get("gt"), row.get("pred"))] = row.get("score", 0.0)
            # Novel claim verdicts
            nc_details = prior_scores.get("verified_novel_claim_rate", {})
            for row in nc_details.get("rows", []):
                novel_verdicts[row["pred"]] = row["confidence"]
        
        detail_rows = []
        
        for m in matches:
            if m.gt_only:
                total_gt += 1
                detail_rows.append({
                    "claim": m.gt_claim_id,
                    "type": "gt_only",
                    "status": "missed",
                    "factual_score": 0.0,
                })
                continue
            
            if m.pred_only:
                total_pred += 1
                p = pred_index.get(m.pred_claim_id)
                
                # Check if this pred_only claim was verified by the novel claim judge
                nc_confidence = novel_verdicts.get(m.pred_claim_id, 0.0)
                
                if nc_confidence >= self.novel_threshold:
                    verified_pred.append(m.pred_claim_id)
                    detail_rows.append({
                        "claim": m.pred_claim_id,
                        "type": "pred_only",
                        "status": "verified_novel",
                        "novel_confidence": nc_confidence,
                    })
                else:
                    has_grounding = any(prov.source_uid for prov in p.provenance) if p else False
                    falsified_pred.append(m.pred_claim_id)
                    detail_rows.append({
                        "claim": m.pred_claim_id,
                        "type": "pred_only",
                        "status": "falsified" if has_grounding else "ungrounded",
                        "novel_confidence": nc_confidence,
                    })
                continue
            
            # Matched claim — both GT and Pred exist
            total_gt += 1
            total_pred += 1
            
            # Get factual value score from prior evaluator
            fv_score = fv_scores.get((m.gt_claim_id, m.pred_claim_id), 0.0)
            
            if fv_score >= self.value_threshold:
                verified_pred.append(m.pred_claim_id)
                matched_gt_with_value += 1
                detail_rows.append({
                    "claim": f"{m.gt_claim_id} ↔ {m.pred_claim_id}",
                    "type": "matched",
                    "status": "verified_match",
                    "factual_score": round(fv_score, 4),
                })
            else:
                falsified_pred.append(m.pred_claim_id)
                detail_rows.append({
                    "claim": f"{m.gt_claim_id} ↔ {m.pred_claim_id}",
                    "type": "matched",
                    "status": "falsified_match" if fv_score == 0.0 else "weak_match",
                    "factual_score": round(fv_score, 4),
                })
        
        # Compute F1
        precision = len(verified_pred) / max(total_pred, 1)
        recall = matched_gt_with_value / max(total_gt, 1)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
        return EvaluatorScore(
            name=self.name,
            score=f1,
            details={
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "verified_pred_count": len(verified_pred),
                "falsified_pred_count": len(falsified_pred),
                "matched_gt_with_value": matched_gt_with_value,
                "total_gt": total_gt,
                "total_pred": total_pred,
                "value_threshold": self.value_threshold,
                "novel_threshold": self.novel_threshold,
                "rows": detail_rows,
            }
        )


class UnverifiedClaimRateEvaluator(BaseEvaluator):
    """
    Like unsupported_claim_rate but gives credit to judge-verified novel claims.
    
    Counts non_verified_pred_only / total_pred as hallucination rate.
    A pred_only claim is "non-verified" if it was NOT verified by the novel claim
    judge (confidence < novel_threshold). Verified novel claims are NOT hallucinations.
    
    Score = 1.0 - unverified_rate
    
    The existing unsupported_claim_rate is kept unchanged as baseline.
    """
    name = "unverified_claim_rate"
    
    def __init__(self, novel_threshold: float = 0.75):
        self.novel_threshold = novel_threshold
    
    def evaluate(
        self,
        gt_index: ClaimIndex,
        pred_index: ClaimIndex,
        matches: List[MatchResult],
        context: Dict[str, Any] | None = None,
    ) -> EvaluatorScore:
        novel_verdicts = {}
        if context:
            prior_scores = context.get("_prior_scores", {})
            nc_details = prior_scores.get("verified_novel_claim_rate", {})
            for row in nc_details.get("rows", []):
                novel_verdicts[row["pred"]] = row["confidence"]
        
        unverified_pred_only = 0
        verified_pred_only = 0
        total_pred = 0
        
        for m in matches:
            if m.gt_only:
                continue
            
            if m.pred_only:
                total_pred += 1
                nc_conf = novel_verdicts.get(m.pred_claim_id, 0.0)
                if nc_conf >= self.novel_threshold:
                    verified_pred_only += 1
                else:
                    unverified_pred_only += 1
            else:
                # Matched claim
                total_pred += 1
        
        rate = unverified_pred_only / total_pred if total_pred else 0.0
        
        return EvaluatorScore(
            name=self.name,
            score=1.0 - rate,
            details={
                "unverified_pred_only": unverified_pred_only,
                "verified_pred_only": verified_pred_only,
                "total_pred": total_pred,
                "unverified_rate": round(rate, 4),
                "novel_threshold": self.novel_threshold,
            }
        )


# ═══════════════════════════════════════════════════════════════════════════
# E5_E1_E2_E3_E4 Calibrated Prompt — Aligned with Judge Calibration Package
# (All 4 features: Scoring + Permissive Conditions + Exemplars + Property Notes
#  → 0.735 AUROC with gemini-3.1-flash-lite thinking=low)
# ═══════════════════════════════════════════════════════════════════════════

# E1: Calibrated confidence scoring block (verbatim from run_prompt_experiment.py)
_E1_SCORING = """SCORING — CALIBRATE YOUR CONFIDENCE CAREFULLY (use the full 0-1 range!):
- is_valid = true only when verdict == "supported"
- confidence (0-1): Your certainty in the verdict. YOU MUST USE THE FULL RANGE:
  0.95-1.0: Value explicitly found verbatim, exact material match, all conditions verified
  0.80-0.90: Value found but needs unit conversion, or material is generic/class-level
  0.60-0.75: Ambiguous — value approximately matches, conditions slightly differ, or evidence is indirect
  0.40-0.55: Uncertain — evidence is tangential, value not directly stated, or unclear material match
  0.10-0.30: Evidence appears to contradict or is clearly for wrong material/property
  
  For "unsupported" verdicts, confidence means how certain you are it is unsupported:
  0.95-1.0: Clear mismatch (wrong material, wrong property, contradictory value)
  0.60-0.80: Evidence doesn't contain the value but might in unshown portions
  0.40-0.55: Borderline — could go either way"""

# E3: Few-shot exemplars (verbatim from run_prompt_experiment.py)
_E3_EXEMPLARS = """
═══════════════════════════════════════════════════════════════
WORKED EXAMPLES — Study these before judging.
═══════════════════════════════════════════════════════════════

EXAMPLE 1 — SUPPORTED (direct value match):
Claim: Portland Cement Concrete has static friction coefficient = 0.85 (counter: rubber tire, dry)
Evidence: Table states "Rubber tire on concrete (dry): static = 0.60-0.85"
Verdict: supported (confidence: 0.95). The value 0.85 is the max of the reported range.
The material and property match exactly. Conditions (dry, rubber tire) are explicit.

EXAMPLE 2 — SUPPORTED (conditions differ but core fact matches):
Claim: Butyl rubber (IIR) has static friction = 0.86 (counter: stainless steel, ASTM D1894)
Evidence: "Bare butyl rubber: static CoF = 0.86 against stainless steel" (different test standard)
Verdict: supported (confidence: 0.85). Same material, same property, same value, same counter-material.
The test standard difference is a minor condition mismatch — flag it but do NOT reject.

EXAMPLE 3 — UNSUPPORTED (wrong material subclass and different value):
Claim: Xerox Transit copy paper (80 g/m²) has Young's modulus = 5468 MPa
Evidence: "Generic copy paper: Elastic Modulus = 6.52 GPa (MD), 2.23 GPa (CD)"
Verdict: unsupported (confidence: 0.90). The value 5468 MPa (5.468 GPa) does not match
either reported value. The material is a different subclass (generic vs Xerox Transit).

EXAMPLE 4 — UNSUPPORTED (different material, different value):
Claim: MB78 + 15 vol.% SiC composite has Poisson's ratio = 0.26
Evidence: "Al6061-4vol.%SiC: Poisson's ratio = 0.284"
Verdict: unsupported (confidence: 0.95). Completely different material (MB78 vs Al6061,
15vol% vs 4vol%) and different value (0.26 vs 0.284). Clear rejection.
"""

# E2: Permissive condition step (verbatim from run_prompt_experiment.py)
_E2_CONDITION_STEP = """STEP 4 — CONDITION CHECK (LEAN PERMISSIVE)
- If the evidence mentions the SAME MATERIAL and SAME PROPERTY and the SAME VALUE as the claim,
  this is SUPPORTED even if conditions (counter-material, temperature, test method) differ.
- You MAY flag condition differences in failure_tags and reduce confidence slightly,
  but condition differences alone should NOT cause an "unsupported" verdict.
- Generic counter-material (e.g., "steel") COVERS all steel subtypes (304, 316, etc.)
- If claim specifies a condition but evidence is generic/unspecified → ACCEPT
- Dry/ambient conditions are DEFAULT unless explicitly stated otherwise
- Lab temperature (20-25°C) is assumed if no temperature is given
- "Room temperature" ≈ "23°C" ≈ "ambient" — these are compatible
- OCR artifacts in units (e.g., "Ω" instead of "·") are formatting errors, not evidence problems
- Minor rounding differences in values (within 5%) are ACCEPTABLE
- Tags: wrong_counter_material, condition_mismatch (use as informational flags, not rejection triggers)"""

# E4: Property-specific notes (verbatim from run_prompt_experiment.py)
_E4_PROPERTY_NOTES_YM = """
YOUNG'S MODULUS VERIFICATION NOTES:
- Class-level or family-level material matches ARE acceptable if the evidence explicitly
  states measurements for that class/family (e.g., "Aluminum" data is valid for a 6061 query
  if the source measured generic Aluminum, not a proxy).
- However, different subclasses within the same class are NOT interchangeable
  (e.g., evidence for 7075-T6 does NOT support a claim about 6061-T6).
- Watch for unit confusion: MPa vs GPa (factor of 1000).
"""

_E4_PROPERTY_NOTES_TE = """
THERMAL EFFUSIVITY VERIFICATION NOTES:
- Thermal effusivity (b) = sqrt(k × ρ × c_p), units: J/(m²·K·s^½) or W·s^½/(m²·K)
- "Thermal inertia" is often used synonymously with thermal effusivity in the literature.
- 1 W·s^½/(m²·K) = 1 J/(m²·K·s^½) — these are equivalent units.
- For indirect/derived claims: focus on verifying the component parameters (k, ρ, c_p),
  not the final computed value.
"""


class CalibratedJudgeDecision(BaseModel):
    """Output schema matching JudgeDecisionV1 from the judge calibration package.

    This is the EXACT same schema the LLM was trained to produce during calibration.
    We map its fields back to NovelClaimJudgement for downstream compatibility.
    """
    is_valid: bool = Field(description="True only when verdict is supported.")
    confidence: float = Field(description="Confidence 0.0-1.0 that the verdict is correct.")
    verdict: str = Field(description="One of: supported, unsupported, insufficient_evidence")
    support_mode: str = Field(default="not_applicable", description="direct, derived, mixed, or not_applicable")
    failure_tags: List[str] = Field(default_factory=list, description="List of failure tags if unsupported.")
    evidence_quality: str = Field(default="weak", description="strong, medium, or weak")
    reasoning: str = Field(description="Step-by-step explanation grounded in supplied evidence.")


class CalibratedClaimJudgeEvaluator(_UncalibratedNovelClaimJudgeEvaluator):
    """
    Calibrated claim judge evaluator using the E5_E1_E2_E3_E4 prompt — ALIGNED
    with the exact calibration process.

    This is the production evaluator for novel claim verification.

    Alignment guarantees (matching run_prompt_experiment.py + run_judge_calibration.py):
    - Evidence formatted as [EVIDENCE N] blocks (render_evidence_bundle format)
    - Claim presented with all JudgeClaimPackage fields
    - Output schema matches JudgeDecisionV1 (7 fields)
    - Indirect section matches render_indirect_section() exactly
    - E1 scoring + E2 permissive conditions + E3 exemplars + E4 property notes

    Validated at 0.735 AUROC on 176-item calibration set with gemini-3.1-flash-lite thinking=low.
    """
    name = "verified_novel_claim_rate"

    def __init__(
        self,
        threshold: float = 0.95,
        use_calibrated: bool = True,
        model_name: str = "gemini-3.1-flash-lite-preview",
        thinking_budget: int = 1024,
        thinking_level: str | None = "low",
        allow_ungrounded_fallback: bool = False,
    ):
        self.threshold = threshold
        self.use_calibrated = use_calibrated
        self._model_name = model_name
        self._thinking_budget = thinking_budget
        self._thinking_level = thinking_level
        self.allow_ungrounded_fallback = allow_ungrounded_fallback
        try:
            self.judge = LLMJudge(
                model_name=model_name,
                thinking_budget=thinking_budget,
                thinking_level=thinking_level,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("LLM Judge unavailable: %s", e)
            self.judge = None

    # ── Evidence bundle construction (matches render_evidence_bundle) ──

    def _build_evidence_bundles(
        self, p: CanonicalClaim, context: Dict[str, Any] | None
    ) -> List[Dict[str, Any]]:
        """Build structured evidence bundles matching JudgeEvidenceBundle format.

        Each bundle corresponds to one provenance record and includes:
        - source metadata (uid, url, title, confidence)
        - citation/matched snippets
        - evidence window text extracted from the source document
        - indirect parameter contributions (for derived claims)
        """
        bundles = []
        source_index_base = None
        if context:
            source_index_base = context.get("source_index_base")

        for prov in p.provenance:
            if not prov.source_uid:
                continue

            bundle: Dict[str, Any] = {
                "source_uid": prov.source_uid,
                "source_url": prov.source_url or None,
                "source_title": None,
                "grounding_confidence": prov.confidence,
                "citation_snippet": prov.citation_snippet,
                "matched_snippet": prov.matched_snippet,
                "evidence_window_text": None,
                "indirect_parameter_contributions": None,
                "per_parameter_grounding": None,
            }

            # Read source document for evidence window
            doc_text = None
            if source_index_base or (context and context.get("source_cache_dir")):
                cache_dir = context.get("source_cache_dir", "") if context else ""
                content_path = resolve_content_path(
                    prov.source_uid,
                    source_index_base or "",
                    cache_dir,
                )
                if content_path:
                    try:
                        with open(content_path, "r", errors="ignore") as f:
                            doc_text = f.read()
                    except Exception:
                        pass

            # Extract evidence window
            if doc_text and prov.per_parameter_evidence:
                # INDIRECT: merge per-parameter windows
                window_size = 3000
                raw_windows = []
                for pe in prov.per_parameter_evidence:
                    if pe.source_uid and pe.source_uid != prov.source_uid:
                        continue
                    for span_start, span_end in pe.root_value_spans:
                        w_start = max(0, span_start - window_size)
                        w_end = min(len(doc_text), span_end + window_size)
                        raw_windows.append((w_start, w_end))
                if raw_windows:
                    raw_windows.sort()
                    merged = [list(raw_windows[0])]
                    for s, e in raw_windows[1:]:
                        if s <= merged[-1][1]:
                            merged[-1][1] = max(merged[-1][1], e)
                        else:
                            merged.append([s, e])
                    parts = [doc_text[s:e] for s, e in merged]
                    bundle["evidence_window_text"] = "\n[...]\n".join(parts)
                elif self.allow_ungrounded_fallback:
                    # Fallback: dump full doc (NOT part of calibration protocol)
                    bundle["evidence_window_text"] = doc_text[:20000] if len(doc_text) > 20000 else doc_text
                    bundle["_fallback_used"] = "indirect_no_spans"
                # else: leave evidence_window_text = None (skip ungrounded)
            elif doc_text and prov.value_index_range and len(prov.value_index_range) >= 1:
                center = prov.value_index_range[0]
                window_size = 3000
                start = max(0, center - window_size)
                end = min(len(doc_text), center + window_size)
                bundle["evidence_window_text"] = doc_text[start:end]
            elif doc_text and self.allow_ungrounded_fallback:
                # Fallback: dump full doc (NOT part of calibration protocol)
                bundle["evidence_window_text"] = doc_text[:20000] if len(doc_text) > 20000 else doc_text
                bundle["_fallback_used"] = "direct_no_spans"

            # Populate indirect parameter contributions
            if prov.per_parameter_evidence:
                ipc = {}
                ppg = {}
                for pe in prov.per_parameter_evidence:
                    ipc[pe.parameter_name] = pe.target_value
                    ppg[pe.parameter_name] = {
                        "grounding_type": pe.grounding_type,
                        "root_values": [
                            {"value": pe.target_value, "source_uid": pe.source_uid}
                        ],
                    }
                bundle["indirect_parameter_contributions"] = ipc
                bundle["per_parameter_grounding"] = ppg

            bundles.append(bundle)

        return bundles

    # ── Rendering (exact match of render_evidence_bundle from calibration) ──

    @staticmethod
    def _render_evidence_text(bundles: List[Dict[str, Any]]) -> str:
        """Render evidence bundles — character-identical to render_evidence_bundle()
        in run_judge_calibration.py."""
        parts = []
        for i, eb in enumerate(bundles):
            lines = [
                f"[EVIDENCE {i+1}]",
                f"Source UID: {eb.get('source_uid') or '—'}",
                f"Source URL: {eb.get('source_url') or '—'}",
                f"Source Title: {eb.get('source_title') or '—'}",
                f"Grounding Confidence: {eb.get('grounding_confidence') or '—'}",
            ]
            if eb.get("citation_snippet"):
                lines.extend(["", "Citation Snippet:", eb["citation_snippet"]])
            ms = eb.get("matched_snippet")
            if ms and isinstance(ms, str):
                snippet = ms[:1500] if len(ms) > 1500 else ms
                lines.extend(["", "Matched Snippet:", snippet])
            ewt = eb.get("evidence_window_text")
            if ewt:
                window = ewt
                if len(window) > 5000:
                    window = window[:2500] + "\n[... truncated ...]\n" + window[-2500:]
                lines.extend(["", "Evidence Window:", window])
            ipc = eb.get("indirect_parameter_contributions")
            if ipc:
                lines.append("")
                lines.append("Indirect Parameter Contributions:")
                ppg = eb.get("per_parameter_grounding") or {}
                for param, val in ipc.items():
                    pg = ppg.get(param, {})
                    gtype = pg.get("grounding_type", "unknown")
                    lines.append(f"  - {param}: {val}  [grounding: {gtype}]")
                    root_vals = pg.get("root_values", [])
                    if root_vals:
                        for rv in root_vals:
                            rv_text = rv.get("matched_text", str(rv.get("value", "")))
                            rv_uid = (rv.get("source_uid") or "")[:16]
                            lines.append(f"    root value: {rv_text} (source: {rv_uid})")
                    jump = pg.get("jump_explanation", "")
                    if jump:
                        lines.append(f"    derivation reasoning: {jump}")
            parts.append("\n".join(lines))
        return "\n\n---\n\n".join(parts)

    # ── Indirect section (exact match of render_indirect_section) ──

    @staticmethod
    def _render_indirect_section(
        is_indirect: bool, value_text: str, units: str,
        bundles: List[Dict[str, Any]],
    ) -> str:
        """Render indirect section — character-identical to render_indirect_section()
        in run_judge_calibration.py."""
        if not is_indirect:
            return ""

        component_params = {}
        for eb in bundles:
            ipc = eb.get("indirect_parameter_contributions")
            if ipc:
                component_params.update(ipc)

        params_text = "\n".join(
            f"  - {k}: {v}" for k, v in component_params.items()
        ) if component_params else "  (none specified)"

        return f"""INDIRECT/DERIVED PROPERTY (CRITICAL — claim uses formula computation):
This claim is for an INDIRECTLY DERIVED thermal effusivity (b = √(λ·ρ·c_p)).
The claimed value ({value_text} {units}) was COMPUTED from:

Component parameters:
{params_text}

For indirect claims, verify EACH component parameter instead of the final value:
a) Each component parameter value must appear in the evidence.
b) All component values must be for THE SAME MATERIAL.
c) If all evidenced and matching → ACCEPT (0.85–1.0).
d) If most but not all → partial (0.5–0.7).
e) Do NOT reject because the computed value doesn't appear literally.

IMPORTANT — DERIVED PARAMETERS:
Some component parameters may be marked with grounding type "logical_jump" or
"source_note_derived". This means the parameter value was COMPUTED from formulas or
unit conversions found in the source text, not stated as a literal number.
A "derivation reasoning" field explains how the value was obtained.
Evaluate whether this derivation is SOUND:
- If the formula, input values, and reasoning are correct → treat as grounded.
- If the derivation has errors or unsupported assumptions → flag as param_mismatch.
- Root values show the actual numbers found in the source that anchor the derivation."""

    # ── Prompt (exact match of render_prompt_holistic with E5_E1_E3 features) ──

    def _build_calibrated_prompt(
        self, p: CanonicalClaim, bundles: List[Dict[str, Any]],
        query_property: str, is_indirect: bool,
    ) -> str:
        """Build prompt — character-identical to render_prompt_holistic() from
        run_prompt_experiment.py with get_prompt_features('E5_E1_E2_E3_E4')."""
        evidence_text = self._render_evidence_text(bundles)
        indirect_section = self._render_indirect_section(
            is_indirect,
            str(p.normalized_value or p.range_min or "N/A"),
            p.units, bundles,
        )

        # Map CanonicalClaim fields → JudgeClaimPackage fields
        value_text = str(p.normalized_value or p.range_min or "N/A")
        condition_sig = ", ".join(f"{k}={v}" for k, v in p.measurement_conditions) or None
        object_class_sig = ", ".join(p.object_class) or None
        claim_text = (
            f"{p.material_class} ({p.material_subclass}) has "
            f"{p.haptic_property} = {value_text} {p.units}"
        )

        # E4: Inject property-specific notes for YM and TE
        prop_lower = p.haptic_property.lower()
        property_notes = "None"
        if "young" in prop_lower or "elastic" in prop_lower or "tensile modulus" in prop_lower:
            property_notes = _E4_PROPERTY_NOTES_YM
        elif "effusiv" in prop_lower:
            property_notes = _E4_PROPERTY_NOTES_TE

        return f"""You are an evidence-aware scientific judge for the HapticNet materials database.

Your job is to decide whether a haptic-material-property claim is supported by
the supplied evidence. You must be strict, local, and evidence-bound.

- You are NOT grading writing quality.
- You are NOT guessing what is probably true from background knowledge.
- You are ONLY deciding whether the provided evidence supports the claim as written.

TASK MODE: novel_claim
QUERY: {query_property or p.haptic_property}
HAPTIC PROPERTY OF INTEREST: {p.haptic_property}

CLAIM PACKAGE:
- Claim text: {claim_text}
- Material family: {p.material_family}
- Material class: {p.material_class}
- Material subclass: {p.material_subclass}
- Property name: {p.haptic_property}
- Value: {value_text} {p.units}
- Original value type: {p.value_type}
- Canonical suffix: none
  (If "min" or "max", this value is one bound of a reported range.
   If "mean" or "std", it is a statistical summary component.)
- Condition signature: {condition_sig or 'None'}
- Object class: {object_class_sig or 'None'}
- Claim source type: pred_only

PROPERTY-SPECIFIC NOTES:
{property_notes}

{indirect_section}

EVIDENCE BUNDLE:
{evidence_text}

═══════════════════════════════════════════════════════════════
EVALUATION RUBRIC — Follow these steps IN ORDER.
═══════════════════════════════════════════════════════════════

STEP 1 — MATERIAL / HIERARCHY CHECK
- EXACT subclass match → accept.
- CLASS-LEVEL match (evidence says generic name for a specific query) → ACCEPT.
- DIFFERENT subclass (e.g., "7075-T6" vs "6061-T6") → REJECT unless generic.
- If wrong, tag: hierarchy_specificity_error

STEP 2 — PROPERTY TYPE CHECK
For FRICTION: kinetic/dynamic/sliding ≠ static. Generic "friction coefficient" acceptable.
For MODULUS: Young's ≠ shear ≠ bulk ≠ flexural.
For THERMAL: conductivity ≠ effusivity.
- If wrong, tag: wrong_property_type

STEP 3 — VALUE / UNIT CHECK
- Value must appear in evidence. Small rounding OK.
- If canonical_suffix is "min" or "max", verify the ORIGINAL range was reported.
- Units must match or be clearly convertible.
- Tags: value_mismatch, unit_mismatch

{_E2_CONDITION_STEP}

STEP 5 — EVIDENCE SUPPORT CHECK
- Citation/matched snippet must explicitly support the claim.
- Tags: overclaiming, citation_not_supporting, source_problem

STEP 6 — THERMAL EFFUSIVITY (only if indirect)
- Direct: value in evidence. Derived: component params evidenced. Mixed: partially.
- Set support_mode accordingly.

STEP 7 — FINAL VERDICT
Return one: supported, unsupported, insufficient_evidence
{_E3_EXEMPLARS}
{_E1_SCORING}

OUTPUT: JSON object matching:
{{"is_valid": true, "confidence": 0.85, "verdict": "supported", "support_mode": "direct", "failure_tags": [], "evidence_quality": "strong", "reasoning": "..."}}"""

    # ── Main evaluate method ──

    def evaluate(
        self,
        gt_index: ClaimIndex,
        pred_index: ClaimIndex,
        matches: List[MatchResult],
        context: Dict[str, Any] | None = None,
    ) -> EvaluatorScore:
        """Evaluate pred_only claims using the calibrated E5_E1_E2_E3_E4 prompt.

        If use_calibrated is False, falls back to the parent NovelClaimJudgeEvaluator
        prompt (for A/B testing).
        """
        if not self.use_calibrated:
            result = super().evaluate(gt_index, pred_index, matches, context)
            return EvaluatorScore(
                name=self.name, score=result.score,
                details={**result.details, "prompt_variant": "uncalibrated_fallback"},
            )

        if not self.judge:
            return EvaluatorScore(name=self.name, score=0.0, details={"error": "LLMJudge not initialized"})

        rows = []
        valid_novel_count = 0
        pred_only_count = 0

        query_property = ""
        if context:
            query_property = context.get("query_property", "") or context.get("haptic_property", "")

        for m in matches:
            if not m.pred_only:
                continue

            p = pred_index.get(m.pred_claim_id)
            assert p is not None
            pred_only_count += 1

            # --- Param-level pre-check for indirect claims (same as parent) ---
            param_precheck_result = None
            has_ppe = any(prov.per_parameter_evidence for prov in p.provenance)
            if has_ppe and p.haptic_property.lower().strip() in _INDIRECT_PROPS:
                pred_params = _extract_pred_params(p)
                if pred_params:
                    for gt_claim in gt_index.all_claims():
                        gt_params = _extract_gt_params(gt_claim)
                        if not gt_params:
                            continue
                        matched_count = 0
                        total_count = len(pred_params)
                        for canon, pred_val in pred_params.items():
                            if canon in gt_params:
                                if relative_closeness(pred_val, gt_params[canon]) >= 0.85:
                                    matched_count += 1
                        if total_count > 0 and matched_count == total_count:
                            param_precheck_result = {
                                "type": "param_match",
                                "matched_gt": gt_claim.claim_id,
                                "matched_params": {k: v for k, v in pred_params.items()},
                            }
                            break

            verdict = "unsupported"
            failure_tags = []

            if param_precheck_result:
                score = 0.75
                reasoning = (f"PARAM-LEVEL PRE-CHECK: All component parameters match GT claim "
                            f"'{param_precheck_result['matched_gt']}'. Scored 0.75.")
                evidence_for_display = ""
                call_resource = None
                has_grounded_provenance = True
                verdict = "supported"
            elif not any(prov.source_uid for prov in p.provenance):
                has_grounded_provenance = False
                score = 0.0
                reasoning = "SKIPPED — No grounded provenance."
                evidence_for_display = ""
                call_resource = None
                failure_tags = ["source_problem"]
            else:
                has_grounded_provenance = True

                # Build structured evidence bundles (calibration format)
                bundles = self._build_evidence_bundles(p, context)
                evidence_text = self._render_evidence_text(bundles)
                evidence_for_display = evidence_text[:2000] if evidence_text else ""

                if not evidence_text.strip():
                    score = 0.0
                    reasoning = "No evidence text could be extracted."
                    call_resource = None
                    failure_tags = ["source_problem"]
                else:
                    tracker = get_resource_tracker()
                    calls_before = len([r for r in tracker.per_call_records if r["evaluator"] == self.name])

                    # Detect indirect claims
                    is_indirect = any(prov.per_parameter_evidence for prov in p.provenance)
                    if not is_indirect:
                        is_indirect = any(
                            k.lower() in ("equation", "formula") for k, v in p.measurement_conditions
                        )

                    # Build calibrated prompt (exact calibration format)
                    prompt = self._build_calibrated_prompt(
                        p, bundles, query_property, is_indirect,
                    )

                    # Use CalibratedJudgeDecision (matches JudgeDecisionV1)
                    judgement = self.judge.generate_structured(
                        prompt, CalibratedJudgeDecision, evaluator_name=self.name,
                    )

                    tracker = get_resource_tracker()
                    all_records = [r for r in tracker.per_call_records if r["evaluator"] == self.name]
                    call_resource = all_records[calls_before] if len(all_records) > calls_before else None

                    if judgement:
                        score = judgement.confidence
                        reasoning = judgement.reasoning
                        verdict = judgement.verdict
                        failure_tags = judgement.failure_tags
                    else:
                        score = 0.0
                        reasoning = "LLM Judge failed to produce a valid rating."

            is_verified = score >= self.threshold
            if is_verified:
                valid_novel_count += 1

            row = {
                "pred": p.claim_id,
                "value": str(p.normalized_value or p.range_min or "N/A"),
                "units": p.units,
                "material": f"{p.material_class} ({p.material_subclass})",
                "conditions": ", ".join(f"{k}={v}" for k, v in p.measurement_conditions) or "None",
                "specs": ", ".join(f"{k}={v}" for k, v in p.material_specifications) or "None",
                "has_grounded_provenance": has_grounded_provenance,
                "evidence_available": bool(evidence_for_display.strip()),
                "evidence_sources": [prov.source_uid for prov in p.provenance if prov.source_uid],
                "evidence_text": evidence_for_display,
                "confidence": score,
                "is_verified": is_verified,
                "verdict": verdict,
                "failure_tags": failure_tags,
                "reasoning": reasoning,
            }
            if call_resource:
                row["call_resource"] = call_resource
            rows.append(row)

        final_score = (valid_novel_count / pred_only_count) if pred_only_count > 0 else 1.0

        tracker = get_resource_tracker()
        resource_summary = tracker.evaluator_summary(self.name)

        return EvaluatorScore(
            name=self.name,
            score=final_score,
            details={
                "pred_only_count": pred_only_count,
                "valid_novel_count": valid_novel_count,
                "rows": rows,
                "llm_resource_usage": resource_summary,
                "prompt_variant": "E5_E1_E3_calibrated",
                "judge_model": self._model_name,
                "thinking_budget": self._thinking_budget,
            }
        )


