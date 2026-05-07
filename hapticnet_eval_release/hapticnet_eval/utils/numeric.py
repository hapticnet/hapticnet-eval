from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

UNIT_ALIASES = {
    'unitless': 'dimensionless',
    'dimensionless': 'dimensionless',
    'no unit': 'dimensionless',
    'none': 'dimensionless',
    '': '',
}


def canonical_text(text: Any) -> str:
    if text is None:
        return ''
    s = str(text).strip().lower()
    s = s.replace('μ', 'u').replace('µ', 'u').replace('–', '-').replace('—', '-')
    s = re.sub(r'\s+', ' ', s)
    return s


def canonical_unit(unit: Any) -> str:
    u = canonical_text(unit)
    return UNIT_ALIASES.get(u, u)


def extract_numeric_variants(x: Any) -> List[str]:
    out = set()
    try:
        n = float(x)
    except Exception:
        return [canonical_text(x)] if x is not None else []
    out.add(str(x))
    out.add(str(n))
    if float(n).is_integer():
        i = int(n)
        out.add(str(i))
        out.add(f"{i:,}")
    out.add(f"{n:.1e}")
    out.add(str(n).replace('.', ','))
    return [canonical_text(v) for v in out]


def normalize_pairs(pairs: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    out = []
    for p in pairs or []:
        label = canonical_text(p.get('label'))
        val = canonical_text(p.get('val'))
        out.append((label, val))
    return sorted(set(out))


def normalize_object_classes(values: Sequence[Any]) -> List[str]:
    return sorted(set(canonical_text(v) for v in (values or [])))


def record_signature(entry: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        canonical_text(entry.get('material_subclass')),
        tuple(normalize_pairs(entry.get('measurement_conditions', []))),
        tuple(normalize_pairs(entry.get('material_specifications', []))),
        tuple(normalize_object_classes(entry.get('object_class', []))),
    )


def pair_f1(gt_pairs: Sequence[Tuple[str, str]], pred_pairs: Sequence[Tuple[str, str]]) -> float:
    gt_set = set(gt_pairs)
    pred_set = set(pred_pairs)
    if not gt_set and not pred_set:
        return 1.0
    tp = len(gt_set & pred_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gt_set) if gt_set else 0.0
    return 2 * precision * recall / (precision + recall)


def contradiction_count(gt_pairs: Sequence[Tuple[str, str]], pred_pairs: Sequence[Tuple[str, str]]) -> int:
    pred_by_label = {}
    for label, val in pred_pairs:
        pred_by_label.setdefault(label, set()).add(val)
    count = 0
    for label, val in gt_pairs:
        if label in pred_by_label and val not in pred_by_label[label]:
            count += 1
    return count


@dataclass
class Match:
    gt_index: int
    pred_index: Optional[int]
    score: float
    reason: str


def entry_match_score(gt_entry: Dict[str, Any], pred_entry: Dict[str, Any]) -> float:
    gt_sub = canonical_text(gt_entry.get('material_subclass'))
    pr_sub = canonical_text(pred_entry.get('material_subclass'))
    subclass_score = 1.0 if gt_sub == pr_sub else 0.0
    cond_score = pair_f1(
        normalize_pairs(gt_entry.get('measurement_conditions', [])),
        normalize_pairs(pred_entry.get('measurement_conditions', [])),
    )
    spec_score = pair_f1(
        normalize_pairs(gt_entry.get('material_specifications', [])),
        normalize_pairs(pred_entry.get('material_specifications', [])),
    )
    obj_score = pair_f1(
        [(x, x) for x in normalize_object_classes(gt_entry.get('object_class', []))],
        [(x, x) for x in normalize_object_classes(pred_entry.get('object_class', []))],
    )
    return 0.45 * subclass_score + 0.35 * cond_score + 0.1 * spec_score + 0.1 * obj_score


def greedy_match_entries(gt_entries: Sequence[Dict[str, Any]], pred_entries: Sequence[Dict[str, Any]], threshold: float = 0.55) -> List[Match]:
    used_pred = set()
    matches: List[Match] = []
    for gi, gt_entry in enumerate(gt_entries):
        best = None
        best_score = -1.0
        for pi, pred_entry in enumerate(pred_entries):
            if pi in used_pred:
                continue
            score = entry_match_score(gt_entry, pred_entry)
            if score > best_score:
                best_score = score
                best = pi
        if best is not None and best_score >= threshold:
            used_pred.add(best)
            matches.append(Match(gi, best, best_score, 'matched'))
        else:
            matches.append(Match(gi, None, 0.0, 'unmatched'))
    return matches


def value_kind(value_obj: Dict[str, Any]) -> str:
    if not isinstance(value_obj, dict):
        return 'unknown'
    if 'value' in value_obj:
        return 'scalar'
    if 'min' in value_obj and 'max' in value_obj:
        return 'range'
    if 'data_points' in value_obj:
        return 'series'
    return 'unknown'


def scalar_close(a: float, b: float, rel_tol: float = 1e-3) -> float:
    denom = max(abs(a), 1e-12)
    rel_err = abs(a - b) / denom
    if rel_err <= rel_tol:
        return 1.0
    return max(0.0, 1.0 - rel_err)


def compare_value_objects(gt_obj: Dict[str, Any], pred_obj: Dict[str, Any], rel_tol: float = 1e-3):
    gt_kind = value_kind(gt_obj)
    pred_kind = value_kind(pred_obj)
    details = {'gt_kind': gt_kind, 'pred_kind': pred_kind}
    if gt_kind != pred_kind:
        return 0.0, {**details, 'reason': 'different value shapes'}
    if gt_kind == 'scalar':
        gv = float(gt_obj['value'])
        pv = float(pred_obj['value'])
        score = scalar_close(gv, pv, rel_tol)
        return score, {**details, 'gt_value': gv, 'pred_value': pv}
    if gt_kind == 'range':
        gmin, gmax = float(gt_obj['min']), float(gt_obj['max'])
        pmin, pmax = float(pred_obj['min']), float(pred_obj['max'])
        inter = max(0.0, min(gmax, pmax) - max(gmin, pmin))
        union = max(gmax, pmax) - min(gmin, pmin)
        overlap = inter / union if union > 0 else 1.0
        endpoint_score = 0.5 * scalar_close(gmin, pmin, rel_tol) + 0.5 * scalar_close(gmax, pmax, rel_tol)
        score = max(overlap, endpoint_score)
        return score, {**details, 'gt_range': [gmin, gmax], 'pred_range': [pmin, pmax], 'overlap': overlap}
    if gt_kind == 'series':
        gs = [float(x) for x in gt_obj.get('data_points', [])]
        ps = [float(x) for x in pred_obj.get('data_points', [])]
        if not gs and not ps:
            return 1.0, {**details, 'gt_len': 0, 'pred_len': 0}
        if not gs or not ps:
            return 0.0, {**details, 'gt_len': len(gs), 'pred_len': len(ps)}
        m = min(len(gs), len(ps))
        point_scores = [scalar_close(gs[i], ps[i], rel_tol) for i in range(m)]
        len_penalty = m / max(len(gs), len(ps))
        score = (sum(point_scores) / len(point_scores)) * len_penalty
        return score, {**details, 'gt_len': len(gs), 'pred_len': len(ps), 'point_scores': point_scores}
    return 0.0, {**details, 'reason': 'unsupported value object'}


def grounding_urls(entry: Dict[str, Any]) -> List[str]:
    urls = []
    for g in entry.get('successful_groundings', []) or []:
        url = g.get('source_url')
        if url:
            urls.append(url)
    return sorted(set(urls))


def parse_range(r: Any) -> Optional[Tuple[int, int]]:
    if isinstance(r, list) and len(r) == 2:
        try:
            return int(r[0]), int(r[1])
        except Exception:
            return None
    if isinstance(r, str):
        nums = re.findall(r'\d+', r)
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
    return None


def range_overlap(a: Optional[Tuple[int, int]], b: Optional[Tuple[int, int]]) -> float:
    if a is None or b is None:
        return 0.0
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 1.0


def token_f1(a: str, b: str) -> float:
    ta = canonical_text(a).split()
    tb = canonical_text(b).split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    sa, sb = set(ta), set(tb)
    tp = len(sa & sb)
    if tp == 0:
        return 0.0
    p = tp / len(sb)
    r = tp / len(sa)
    return 2 * p * r / (p + r)


def value_supported_by_text(value_obj: Dict[str, Any], text: str) -> bool:
    txt = canonical_text(text)
    kind = value_kind(value_obj)
    if kind == 'scalar':
        return any(v in txt for v in extract_numeric_variants(value_obj['value']))
    if kind == 'range':
        return any(v in txt for v in extract_numeric_variants(value_obj['min'])) and any(v in txt for v in extract_numeric_variants(value_obj['max']))
    if kind == 'series':
        points = value_obj.get('data_points', [])
        hits = sum(any(v in txt for v in extract_numeric_variants(x)) for x in points)
        return hits >= max(1, math.ceil(0.5 * len(points)))
    return False


def condition_tokens(entry: Dict[str, Any]) -> List[str]:
    toks = []
    for label, val in normalize_pairs(entry.get('measurement_conditions', [])):
        toks.extend([label, val])
    return [t for t in toks if t]


def coerce_value_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, (int, float)):
        return {'value': float(value)}
    return {}
