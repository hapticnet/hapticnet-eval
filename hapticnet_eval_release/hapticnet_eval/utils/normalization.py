from __future__ import annotations

import math
import re
import unicodedata
from typing import Iterable, List, Sequence, Tuple


_UNIT_SYNONYMS = {
    "": "",
    "dimensionless": "dimensionless",
    "unitless": "dimensionless",
    "none": "dimensionless",
    "dimension-less": "dimensionless",
}


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("μ", "u").replace("µ", "u")
    text = text.replace("°C", " deg c").replace("°c", " deg c")
    text = text.replace("–", "-").replace("—", "-")
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_label_value_pairs(pairs: Iterable[Tuple[str, str]]) -> Tuple[Tuple[str, str], ...]:
    norm = sorted((normalize_text(k), normalize_text(v)) for k, v in pairs)
    return tuple(norm)


def normalize_string_list(values: Sequence[str]) -> Tuple[str, ...]:
    return tuple(sorted(normalize_text(v) for v in values))


def normalize_units(units: str | None) -> str:
    key = normalize_text(units)
    # After NFKC: ² → 2, but ^2 stays as ^2.  Canonicalize ^N → N
    # so "m^2" == "m²" == "m2" after normalization.
    key = re.sub(r"\^([0-9./-]+)", r"\1", key)
    # Normalize middle dots and multiplication signs to a single separator
    # Only replace dots NOT between digits (to preserve decimals like 0.5)
    key = key.replace("·", "*").replace("⋅", "*").replace("×", "*")
    key = re.sub(r"(?<!\d)\.(?!\d)", "*", key)  # dot as multiplication only
    # Collapse whitespace again in case substitutions introduced spaces
    key = re.sub(r"\s+", " ", key).strip()
    return _UNIT_SYNONYMS.get(key, key)


_NUMBER_RE = re.compile(r"(?<!\d)(\d{1,3}(?:[ ,]\d{3})*(?:[\.,]\d+)?|\d+(?:[\.,]\d+)?)(?!\d)")


def safe_float(x: float | int | str | None) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = x.strip()
    if not s:
        return None
    s = s.replace("−", "-")
    # If both separators exist, assume comma thousands and dot decimal.
    if "," in s and "." in s:
        s = s.replace(",", "")
    else:
        s = s.replace(",", ".")
    s = s.replace(" ", "")
    try:
        return float(s)
    except ValueError:
        m = _NUMBER_RE.search(s)
        if not m:
            return None
        candidate = m.group(1).replace(" ", "")
        if "," in candidate and "." not in candidate:
            candidate = candidate.replace(",", ".")
        else:
            candidate = candidate.replace(",", "")
        try:
            return float(candidate)
        except ValueError:
            return None


def interval_iou(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    if a_min > a_max or b_min > b_max:
        return 0.0
    inter = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    union = max(a_max, b_max) - min(a_min, b_min)
    if union <= 0:
        return 1.0
    return inter / union


def relative_closeness(a: float | None, b: float | None, rel_tol: float = 0.05, abs_tol: float = 1e-6) -> float:
    if a is None or b is None:
        return 0.0
    if math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return 1.0
    denom = max(abs(a), abs(b), abs_tol)
    err = abs(a - b) / denom
    return max(0.0, 1.0 - err / max(rel_tol, 1e-9))


def sequence_similarity(a: Sequence[float] | None, b: Sequence[float] | None, rel_tol: float = 0.05) -> float:
    if a is None or b is None or not a or not b:
        return 0.0
    if len(a) != len(b):
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
    vals = [relative_closeness(x, y, rel_tol=rel_tol) for x, y in zip(a, b)]
    return sum(vals) / len(vals) if vals else 0.0


def overlap_ratio(span_a: Tuple[int, int] | None, span_b: Tuple[int, int] | None) -> float:
    if not span_a or not span_b:
        return 0.0
    # Defensive: handle single-element spans (treat as point → point+10 chars)
    def _unpack(span):
        if len(span) == 2:
            return span[0], span[1]
        if len(span) == 1:
            return span[0], span[0] + 10  # approximate word length
        return None, None
    a0, a1 = _unpack(span_a)
    b0, b1 = _unpack(span_b)
    if a0 is None or b0 is None:
        return 0.0
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 1.0
    return inter / union


def exact_match_ratio(a: Sequence[str] | None, b: Sequence[str] | None) -> float:
    sa = set(a or [])
    sb = set(b or [])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
