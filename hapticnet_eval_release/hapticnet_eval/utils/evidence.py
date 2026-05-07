from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence
from urllib.parse import urlparse, urlunparse

from rapidfuzz.fuzz import ratio, token_set_ratio

from .normalization import normalize_text

_GENERIC_TOKENS = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "with", "at", "by",
    "dry", "wet", "bulk", "specimen", "sample", "value", "values", "coefficient", "friction",
    "kinetic", "sliding", "unitless", "dimensionless",
}

_NUMBER_RE = re.compile(r"[-+]?\d+(?:[\.,]\d+)?")


def canonicalize_url(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip())
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/")
    # Strip obvious view/query noise.
    return urlunparse(("https", netloc, path, "", "", ""))


def source_domain(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def url_path_slug(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    path = normalize_text(parsed.path)
    return re.sub(r"[^a-z0-9]+", " ", path).strip()


@dataclass(frozen=True)
class CitationSupportJudgement:
    label: str  # full / partial / none
    score: float
    lexical_overlap: float
    numeric_overlap: float
    best_gt_snippet: str | None


@dataclass(frozen=True)
class FaithfulnessAssessment:
    score: float
    numeric_anchor_coverage: float
    entity_anchor_coverage: float
    support_label: str
    likely_post_rationalized: bool


@dataclass(frozen=True)
class SourceEquivalenceAssessment:
    score: float
    method: str


def _tokenize(text: str) -> set[str]:
    text = normalize_text(text)
    toks = re.findall(r"[a-z0-9\.\-/]+", text)
    return {t for t in toks if t and t not in _GENERIC_TOKENS}


def extract_numbers(text: str | None) -> list[str]:
    if not text:
        return []
    nums = []
    for m in _NUMBER_RE.finditer(text):
        s = m.group(0)
        nums.append(s.replace(",", "."))
    return nums


def number_variants(x: float | int | None) -> set[str]:
    if x is None:
        return set()
    n = float(x)
    out = {str(n), str(x)}
    out.add(f"{n:.6g}")
    out.add(f"{n:.4f}".rstrip("0").rstrip("."))
    if n.is_integer():
        out.add(str(int(n)))
    else:
        s = f"{n}"
        out.add(s.replace(".", ","))
        out.add(f"{n:.2f}".rstrip("0").rstrip(".").replace(".", ","))
    return {v for v in out if v}


def number_variants_from_string(text: str) -> set[str]:
    """Extract number variants from a string that may contain numeric values.
    
    Handles values like '0.0904', '818', '1.26', '370-385', '370–385'.
    Returns the union of number_variants for all numbers found.
    """
    if not text:
        return set()
    nums = extract_numbers(text)
    variants = set()
    for n in nums:
        try:
            variants |= number_variants(float(n))
        except ValueError:
            pass
    return variants


def source_equivalence(
    pred_url: str | None,
    pred_title: str | None,
    gt_url: str | None,
    gt_title: str | None,
    pred_uid: str | None = None,
    gt_uid: str | None = None,
) -> SourceEquivalenceAssessment:
    if pred_uid and gt_uid and pred_uid == gt_uid:
        return SourceEquivalenceAssessment(1.0, "uid")

    p_can = canonicalize_url(pred_url)
    g_can = canonicalize_url(gt_url)
    if p_can and g_can and p_can == g_can:
        return SourceEquivalenceAssessment(1.0, "canonical_url")

    p_dom = source_domain(pred_url)
    g_dom = source_domain(gt_url)
    p_path = url_path_slug(pred_url)
    g_path = url_path_slug(gt_url)
    path_sim = ratio(p_path, g_path) / 100.0 if p_path and g_path else 0.0
    title_sim = token_set_ratio(normalize_text(pred_title), normalize_text(gt_title)) / 100.0 if pred_title and gt_title else 0.0

    if p_dom and g_dom and p_dom == g_dom:
        if path_sim >= 0.95:
            return SourceEquivalenceAssessment(0.95, "same_domain_near_identical_path")
        if title_sim >= 0.92:
            return SourceEquivalenceAssessment(0.9, "same_domain_title")
        if path_sim >= 0.8:
            return SourceEquivalenceAssessment(0.82, "same_domain_path")

    # Cross-domain mirrored copy or archive-like duplication.
    if title_sim >= 0.96:
        return SourceEquivalenceAssessment(0.8, "title_match")
    if p_dom and g_dom and (p_dom.endswith("nih.gov") and g_dom.endswith("nih.gov")) and path_sim >= 0.7:
        return SourceEquivalenceAssessment(0.78, "nih_family_path")

    return SourceEquivalenceAssessment(max(path_sim * 0.5, title_sim * 0.5), "weak")


def judge_citation_support(gt_support_snippets: Sequence[str], predicted_evidence_texts: Sequence[str]) -> CitationSupportJudgement:
    gt_support_snippets = [s for s in gt_support_snippets if s]
    predicted_evidence_texts = [s for s in predicted_evidence_texts if s]
    if not gt_support_snippets and not predicted_evidence_texts:
        return CitationSupportJudgement("full", 1.0, 1.0, 1.0, None)
    if not gt_support_snippets or not predicted_evidence_texts:
        return CitationSupportJudgement("none", 0.0, 0.0, 0.0, None)

    best = (0.0, 0.0, None)
    for gt in gt_support_snippets:
        gt_toks = _tokenize(gt)
        gt_nums = set(extract_numbers(gt))
        for pred in predicted_evidence_texts:
            pred_toks = _tokenize(pred)
            pred_nums = set(extract_numbers(pred))
            tok_j = len(gt_toks & pred_toks) / max(len(gt_toks | pred_toks), 1)
            num_ov = len(gt_nums & pred_nums) / max(len(gt_nums), 1) if gt_nums else 1.0
            lexical = max(tok_j, ratio(normalize_text(gt), normalize_text(pred)) / 100.0)
            if lexical + num_ov > best[0] + best[1]:
                best = (lexical, num_ov, gt)

    lexical, numeric, best_gt = best
    if lexical >= 0.72 and numeric >= 0.8:
        return CitationSupportJudgement("full", 1.0, lexical, numeric, best_gt)
    if lexical >= 0.35 or numeric >= 0.5:
        return CitationSupportJudgement("partial", 0.5, lexical, numeric, best_gt)
    return CitationSupportJudgement("none", 0.0, lexical, numeric, best_gt)


def build_claim_anchors(
    material_subclass: str,
    material_class: str,
    measurement_conditions: Iterable[tuple[str, str]],
    material_specifications: Iterable[tuple[str, str]],
    normalized_value: float | None,
    range_min: float | None,
    range_max: float | None,
    data_points: Sequence[float] | None,
) -> tuple[set[str], set[str]]:
    text_anchors = set()
    if material_subclass and material_subclass != "unspecified":
        text_anchors |= _tokenize(material_subclass)
    else:
        text_anchors |= _tokenize(material_class)
    for _, val in list(measurement_conditions)[:4]:
        text_anchors |= _tokenize(val)
    for _, val in list(material_specifications)[:3]:
        text_anchors |= _tokenize(val)

    numeric_anchors = set()
    if data_points:
        for x in data_points[: min(len(data_points), 5)]:
            numeric_anchors |= number_variants(x)
    elif range_min is not None or range_max is not None:
        numeric_anchors |= number_variants(range_min)
        numeric_anchors |= number_variants(range_max)
    else:
        numeric_anchors |= number_variants(normalized_value)
    return text_anchors, numeric_anchors


def assess_citation_faithfulness(
    evidence_texts: Sequence[str],
    text_anchors: set[str],
    numeric_anchors: set[str],
    support_judgement: CitationSupportJudgement,
    claim_correctness_hint: float,
) -> FaithfulnessAssessment:
    evidence_text = normalize_text(" ".join(evidence_texts))
    evidence_tokens = _tokenize(evidence_text)
    entity_cov = len(text_anchors & evidence_tokens) / max(len(text_anchors), 1) if text_anchors else 1.0
    num_cov = sum(1 for n in numeric_anchors if normalize_text(n) in evidence_text) / max(len(numeric_anchors), 1) if numeric_anchors else 1.0
    support_component = support_judgement.score
    score = 0.45 * num_cov + 0.35 * entity_cov + 0.20 * support_component
    likely_post = claim_correctness_hint >= 0.8 and score < 0.5
    return FaithfulnessAssessment(
        score=score,
        numeric_anchor_coverage=num_cov,
        entity_anchor_coverage=entity_cov,
        support_label=support_judgement.label,
        likely_post_rationalized=likely_post,
    )


def efficiency_score(value: float | None, budget: float, floor: float = 0.0) -> float:
    if value is None:
        return floor
    if value < 0:
        value = 0.0
    return 1.0 / (1.0 + (value / max(budget, 1e-9)))
