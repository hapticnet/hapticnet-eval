from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rapidfuzz.fuzz import ratio
from scipy.optimize import linear_sum_assignment

from ..schemas import CanonicalClaim, MatchResult
from ..utils.normalization import interval_iou, relative_closeness, sequence_similarity

import re

# Known indirectly-computed properties (use formula-derived values)
_INDIRECT_PROPERTIES = {"thermal effusivity"}

# Synonyms for component parameter names used in indirect formulas
_PARAM_SYNONYMS = {
    "thermal conductivity": {"thermal conductivity", "tc", "k", "lambda", "λ"},
    "density": {"density", "rho", "ρ", "bulk density"},
    "specific heat capacity": {"specific heat capacity", "specific heat", "cp", "c_p", "c", "heat capacity"},
    "thermal diffusivity": {"thermal diffusivity", "alpha", "α", "a", "diffusivity"},
}

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


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


def _claim_origin(claim_id: str) -> str:
    """Extract structural origin from claim_id suffix."""
    if claim_id.endswith("::min"):
        return "min"
    if claim_id.endswith("::max"):
        return "max"
    if claim_id.endswith("::mean"):
        return "mean"
    if claim_id.endswith("::std"):
        return "std"
    if "::dp" in claim_id:
        return "datapoint"
    return "scalar"


def _source_urls(claim: 'CanonicalClaim') -> set:
    """Extract source URLs from claim provenance."""
    urls = set()
    for p in claim.provenance:
        if p.source_url:
            urls.add(p.source_url)
    return urls


def _is_indirect_claim(claim: 'CanonicalClaim') -> bool:
    """Check if a claim is from indirect/formula-based extraction.
    
    Three signals checked (any one suffices):
    1. Has 'equation' or 'formula' label in measurement_conditions
    2. Has per_parameter_evidence in any provenance record
    3. Has 'method' condition containing computation keywords
    
    Also requires haptic_property to be in _INDIRECT_PROPERTIES to avoid
    false positives on direct claims that happen to mention equations.
    """
    if claim.haptic_property.lower().strip() not in _INDIRECT_PROPERTIES:
        return False
    for k, v in claim.measurement_conditions:
        kl = k.lower().strip()
        if kl in ("equation", "formula"):
            return True
        if kl == "method" and any(kw in v.lower() for kw in ("computed", "derived", "calculated", "sqrt", "√")):
            return True
    for prov in claim.provenance:
        if prov.per_parameter_evidence:
            return True
    return False


def _extract_param_values(claim: 'CanonicalClaim') -> dict[str, float]:
    """Extract component parameter numeric values from a claim.
    
    Sources (in priority order):
    1. per_parameter_evidence target_value from provenance
    2. measurement_conditions labels that match known parameter names
    """
    params: dict[str, float] = {}  # canonical_name -> value
    
    # Source 1: per_parameter_evidence
    for prov in claim.provenance:
        if prov.per_parameter_evidence:
            for pe in prov.per_parameter_evidence:
                canon = _canonicalize_param_name(pe.parameter_name)
                if canon and canon not in params:
                    nums = _NUM_RE.findall(pe.target_value)
                    if nums:
                        try:
                            params[canon] = float(nums[0])
                        except ValueError:
                            pass
            break  # Use first provenance with PPE
    
    # Source 2: measurement_conditions
    for k, v in claim.measurement_conditions:
        kl = k.lower().strip()
        if kl in ("equation", "formula", "method", "unspecified"):
            continue
        canon = _canonicalize_param_name(kl)
        if canon and canon not in params:
            nums = _NUM_RE.findall(v)
            if nums:
                try:
                    params[canon] = float(nums[0])
                except ValueError:
                    pass
    return params


def _canonicalize_param_name(name: str) -> str | None:
    """Map a parameter name to its canonical form using synonym tables."""
    nl = name.lower().strip()
    # Remove parenthetical suffixes like "(k)" or "(ρ)"
    nl = re.sub(r"\s*\([^)]*\)", "", nl).strip()
    for canon, synonyms in _PARAM_SYNONYMS.items():
        if nl in synonyms or any(syn in nl for syn in synonyms if len(syn) > 2):
            return canon
    return None


def _indirect_param_closeness(gt: 'CanonicalClaim', pred: 'CanonicalClaim') -> float:
    """Compute parameter-level similarity for indirect claims.
    
    Extracts component parameters from both claims and computes
    relative_closeness for each matched parameter. Returns the
    average closeness across all GT parameters.
    """
    gt_params = _extract_param_values(gt)
    pred_params = _extract_param_values(pred)
    if not gt_params and not pred_params:
        return 1.0  # both have no params, equally uninformative
    if not gt_params or not pred_params:
        return 0.0
    scores = []
    for canon, gt_val in gt_params.items():
        if canon in pred_params:
            scores.append(relative_closeness(gt_val, pred_params[canon]))
        else:
            scores.append(0.0)  # unmatched param
    return sum(scores) / len(scores) if scores else 0.0


def claim_similarity(gt: 'CanonicalClaim', pred: 'CanonicalClaim') -> tuple[float, dict]:
    family = ratio(gt.material_family, pred.material_family) / 100.0
    cls = ratio(gt.material_class, pred.material_class) / 100.0
    subclass = ratio(gt.material_subclass, pred.material_subclass) / 100.0
    prop = ratio(gt.haptic_property, pred.haptic_property) / 100.0
    cond = _set_f1(gt.measurement_conditions, pred.measurement_conditions)
    specs = _set_f1(gt.material_specifications, pred.material_specifications)
    obj = _set_f1(gt.object_class, pred.object_class)
    units = 1.0 if gt.units == pred.units else 0.0
    value_type = 1.0 if gt.value_type == pred.value_type else 0.0

    # Detect indirect claims and use parameter-level matching
    is_indirect = _is_indirect_claim(gt) or _is_indirect_claim(pred)

    if gt.value_type == "scalar" and pred.value_type == "scalar":
        value_score = relative_closeness(gt.normalized_value, pred.normalized_value)
        # For indirect claims, use wider tolerance (10% instead of ~5% default)
        if is_indirect and value_score < 1.0:
            if gt.normalized_value and pred.normalized_value:
                diff = abs(gt.normalized_value - pred.normalized_value)
                ref = max(abs(gt.normalized_value), abs(pred.normalized_value), 1e-12)
                if diff / ref <= 0.10:
                    value_score = max(value_score, 1.0 - (diff / ref))  # linear falloff
    elif gt.value_type == "range" and pred.value_type == "range":
        value_score = interval_iou(gt.range_min or 0.0, gt.range_max or 0.0, pred.range_min or 0.0, pred.range_max or 0.0)
    elif gt.value_type == "series" and pred.value_type == "series":
        value_score = sequence_similarity(gt.data_points, pred.data_points)
    else:
        value_score = relative_closeness(gt.normalized_value, pred.normalized_value) * 0.5

    # For indirect claims, replace raw set-F1 with parameter-level similarity
    if is_indirect:
        cond = _indirect_param_closeness(gt, pred)

    # Structural origin: prefer matching scalar↔scalar, min↔min, max↔max, etc.
    gt_origin = _claim_origin(gt.claim_id)
    pred_origin = _claim_origin(pred.claim_id)
    origin_match = 1.0 if gt_origin == pred_origin else 0.0

    # Source overlap: prefer matching claims that share the same source URLs
    gt_urls = _source_urls(gt)
    pred_urls = _source_urls(pred)
    if gt_urls and pred_urls:
        intersection = len(gt_urls & pred_urls)
        union = len(gt_urls | pred_urls)
        source_overlap = intersection / union if union > 0 else 0.0
    elif not gt_urls and not pred_urls:
        source_overlap = 1.0  # both ungrounded — neutral
    else:
        source_overlap = 0.0  # one grounded, one not

    weights = {
        "family": 0.05,
        "class": 0.08,
        "subclass": 0.12,
        "property": 0.15,
        "conditions": 0.20,
        "specs": 0.07,
        "object_class": 0.05,
        "units": 0.05,
        "value_type": 0.03,
        "value": 0.07,
        "origin_match": 0.05,
        "source_overlap": 0.08,
    }
    components = {
        "family": family,
        "class": cls,
        "subclass": subclass,
        "property": prop,
        "conditions": cond,
        "specs": specs,
        "object_class": obj,
        "units": units,
        "value_type": value_type,
        "value": value_score,
        "origin_match": origin_match,
        "source_overlap": source_overlap,
    }
    sim = sum(weights[k] * components[k] for k in weights)
    return sim, components


def match_claims(gt_claims: List[CanonicalClaim], pred_claims: List[CanonicalClaim], min_similarity: float = 0.45) -> List[MatchResult]:
    if not gt_claims and not pred_claims:
        return []
    if not gt_claims:
        return [MatchResult(pred_claim_id=p.claim_id, pred_only=True, similarity=0.0) for p in pred_claims]
    if not pred_claims:
        return [MatchResult(gt_claim_id=g.claim_id, gt_only=True, similarity=0.0) for g in gt_claims]

    matrix = np.zeros((len(gt_claims), len(pred_claims)), dtype=float)
    debug_grid = {}
    for i, g in enumerate(gt_claims):
        for j, p in enumerate(pred_claims):
            sim, comps = claim_similarity(g, p)
            matrix[i, j] = sim
            debug_grid[(i, j)] = comps

    cost = 1.0 - matrix
    rows, cols = linear_sum_assignment(cost)

    matched_gt = set()
    matched_pred = set()
    results: List[MatchResult] = []
    for i, j in zip(rows, cols):
        sim = float(matrix[i, j])
        if sim >= min_similarity:
            matched_gt.add(i)
            matched_pred.add(j)
            results.append(
                MatchResult(
                    gt_claim_id=gt_claims[i].claim_id,
                    pred_claim_id=pred_claims[j].claim_id,
                    similarity=sim,
                    debug=debug_grid[(i, j)],
                )
            )

    for i, g in enumerate(gt_claims):
        if i not in matched_gt:
            results.append(MatchResult(gt_claim_id=g.claim_id, gt_only=True, similarity=0.0))
    for j, p in enumerate(pred_claims):
        if j not in matched_pred:
            results.append(MatchResult(pred_claim_id=p.claim_id, pred_only=True, similarity=0.0))
    return results
