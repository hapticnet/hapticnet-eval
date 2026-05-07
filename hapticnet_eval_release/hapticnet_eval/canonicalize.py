from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

from .schemas import CanonicalClaim, CanonicalFieldEvidence, GTFile, ParameterEvidence, ValueConditionMappingEntry, ValuePayload
from .utils.normalization import (
    normalize_label_value_pairs,
    normalize_string_list,
    normalize_text,
    normalize_units,
)

_SUPPORTS_FIELD_RE = re.compile(r"value_condition_mapping\[(\d+)\]")


def _build_claim_id(gt: GTFile, idx: int, entry: ValueConditionMappingEntry) -> str:
    subclass = normalize_text(entry.material_subclass or "unspecified")
    return f"{normalize_text(gt.material_family)}::{normalize_text(gt.material_class)}::{normalize_text(gt.haptic_property)}::{subclass}::{idx}"


def _support_snippets_by_entry(gt: GTFile) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for citation in gt.citations:
        for field in citation.supports_fields:
            m = _SUPPORTS_FIELD_RE.search(field)
            if not m:
                continue
            idx = int(m.group(1))
            out.setdefault(idx, []).append(citation.verbatim_snippet)
    return out


def _normalized_payload(entry: ValueConditionMappingEntry) -> ValuePayload:
    # IMPORTANT: For judge evaluation, prefer raw entry.value over entry.normalized_value.
    # The normalized_value may be SI-converted (e.g., GPa→MPa, μm→mm) but the evidence
    # text contains the original source values. Using the converted value causes the
    # judge to see a different number than what appears in evidence → spurious
    # value_mismatch tags. This matches the calibration set builder (build_calibration_set.py)
    # which uses entry["value"] (the raw value) for the same reason.
    return entry.value


def _payload_to_fields(payload: ValuePayload) -> tuple[str, float | None, float | None, float | None, tuple[float, ...] | None]:
    value_type = payload.shape()
    if value_type == "scalar":
        return value_type, payload.normalized_scalar(), None, None, None
    if value_type == "range":
        return value_type, payload.normalized_scalar(), payload.min, payload.max, None
    if value_type == "mean_std":
        # Pass mean and std directly (not as bounds)
        return value_type, payload.normalized_scalar(), payload.mean, payload.std, None
    return value_type, payload.normalized_scalar(), None, None, tuple(payload.data_points or [])


def _entry_to_claims(gt: GTFile, idx: int, entry: ValueConditionMappingEntry, support_snippets: Iterable[str]) -> List[CanonicalClaim]:
    """Convert a single ValueConditionMappingEntry into one or more CanonicalClaims.

    Scalar  → 1 claim (unchanged)
    Range   → 2 claims: one for min (::min), one for max (::max)
    Series  → N claims: one per data point (::dp0, ::dp1, …)
    """
    value_payload = _normalized_payload(entry)
    value_type, normalized_value, range_min, range_max, data_points = _payload_to_fields(value_payload)

    provenance: List[CanonicalFieldEvidence] = []
    for g in entry.successful_groundings:
        # Convert per_parameter_grounding dict to structured ParameterEvidence tuples.
        # Two formats exist:
        #   GT format:   per_parameter_grounding  → dict of {param: {target_value, grounding_type, root_values: [{span, source_uid, ...}]}}
        #   Pred format: indirect_parameter_groundings → dict of {param: [grounding_records with value_index_range]}
        #                + indirect_parameter_contributions → dict of {param: target_value_str}
        param_evidence = None

        # --- Path 1: GT-format per_parameter_grounding (preferred, more structured) ---
        ppg = getattr(g, 'per_parameter_grounding', None) or (g.per_parameter_grounding if hasattr(g, 'per_parameter_grounding') else None)
        if ppg and isinstance(ppg, dict):
            pe_list = []
            for pname, pdata in ppg.items():
                if not isinstance(pdata, dict):
                    continue
                spans = tuple(
                    (rv["span"][0], rv["span"][1])
                    for rv in pdata.get("root_values", [])
                    if rv.get("span") and len(rv["span"]) == 2
                )
                # Get source_uid from first root value that has one
                rv_uid = ""
                for rv in pdata.get("root_values", []):
                    if rv.get("source_uid"):
                        rv_uid = rv["source_uid"]
                        break
                pe_list.append(ParameterEvidence(
                    parameter_name=pname,
                    target_value=str(pdata.get("target_value", "")),
                    grounding_type=pdata.get("grounding_type", "unknown"),
                    root_value_spans=tuple(spans),
                    source_uid=rv_uid or g.source_uid or "",
                ))
            if pe_list:
                param_evidence = tuple(pe_list)

        # --- Path 2: Pred-format indirect_parameter_groundings (fallback) ---
        # Pred extraction pipelines store per-param grounding as a dict of
        # {param_name: [grounding_records]} where each record has value_index_range.
        # Target values come from the sibling indirect_parameter_contributions dict.
        if param_evidence is None:
            ipg = getattr(g, 'indirect_parameter_groundings', None)
            ipc = getattr(g, 'indirect_parameter_contributions', None) or {}
            if ipg and isinstance(ipg, dict):
                pe_list = []
                for pname, records in ipg.items():
                    if not isinstance(records, list) or not records:
                        continue
                    # Collect all value_index_range spans across grounding records
                    spans = []
                    rec_uid = ""
                    for rec in records:
                        if not isinstance(rec, dict):
                            continue
                        vir = rec.get("value_index_range")
                        if vir and isinstance(vir, (list, tuple)) and len(vir) == 2:
                            spans.append((int(vir[0]), int(vir[1])))
                        if not rec_uid and rec.get("source_uid"):
                            rec_uid = rec["source_uid"]
                    pe_list.append(ParameterEvidence(
                        parameter_name=pname,
                        target_value=str(ipc.get(pname, ipc.get(pname.lower(), ""))),
                        grounding_type="direct",  # pred groundings are always direct spans
                        root_value_spans=tuple(spans),
                        source_uid=rec_uid or g.source_uid or "",
                    ))
                if pe_list:
                    param_evidence = tuple(pe_list)

        provenance.append(
            CanonicalFieldEvidence(
                source_url=g.source_url,
                source_uid=g.source_uid,
                citation_snippet=g.citation_snippet,
                matched_snippet=g.matched_snippet,
                matched_snippet_pieces=tuple(g.matched_snippet_pieces or []),
                citation_index_range=tuple(g.citation_index_range) if g.citation_index_range else None,
                value_index_range=tuple(g.value_index_range) if g.value_index_range else None,
                confidence=g.confidence,
                per_parameter_evidence=param_evidence,
            )
        )

    base_id = _build_claim_id(gt, idx, entry)
    shared = dict(
        entry_index=idx,
        material_family=normalize_text(gt.material_family),
        material_class=normalize_text(gt.material_class),
        material_subclass=normalize_text(entry.material_subclass or "unspecified"),
        haptic_property=normalize_text(gt.haptic_property),
        measurement_conditions=normalize_label_value_pairs((p.label, p.val) for p in entry.measurement_conditions),
        material_specifications=normalize_label_value_pairs((p.label, p.val) for p in entry.material_specifications),
        object_class=normalize_string_list(entry.object_class),
        units=normalize_units(entry.units or entry.normalized_units),
        grounded=entry.is_grounded,
        provenance=tuple(provenance),
        support_snippets=tuple(dict.fromkeys(support_snippets)),
    )

    claims: List[CanonicalClaim] = []

    if value_type == "scalar":
        claims.append(CanonicalClaim(
            claim_id=base_id,
            value_type="scalar",
            normalized_value=normalized_value,
            range_min=None,
            range_max=None,
            data_points=None,
            **shared,
        ))

    elif value_type == "range":
        # Emit one claim per range bound
        claims.append(CanonicalClaim(
            claim_id=f"{base_id}::min",
            value_type="scalar",
            normalized_value=float(range_min) if range_min is not None else None,
            range_min=None,
            range_max=None,
            data_points=None,
            **shared,
        ))
        claims.append(CanonicalClaim(
            claim_id=f"{base_id}::max",
            value_type="scalar",
            normalized_value=float(range_max) if range_max is not None else None,
            range_min=None,
            range_max=None,
            data_points=None,
            **shared,
        ))

    elif value_type == "mean_std":
        # Emit two claims: mean (matchable against scalar GT) and std
        claims.append(CanonicalClaim(
            claim_id=f"{base_id}::mean",
            value_type="scalar",
            normalized_value=float(range_min) if range_min is not None else None,  # range_min holds mean here
            range_min=None,
            range_max=None,
            data_points=None,
            **shared,
        ))
        claims.append(CanonicalClaim(
            claim_id=f"{base_id}::std",
            value_type="scalar",
            normalized_value=float(range_max) if range_max is not None else None,  # range_max holds std here
            range_min=None,
            range_max=None,
            data_points=None,
            **shared,
        ))

    elif value_type == "series" and data_points:
        # Emit one claim per data point
        for dp_i, dp_val in enumerate(data_points):
            claims.append(CanonicalClaim(
                claim_id=f"{base_id}::dp{dp_i}",
                value_type="scalar",
                normalized_value=float(dp_val),
                range_min=None,
                range_max=None,
                data_points=None,
                **shared,
            ))

    else:
        # Fallback: emit single claim with whatever we have
        claims.append(CanonicalClaim(
            claim_id=base_id,
            value_type=value_type,
            normalized_value=normalized_value,
            range_min=range_min,
            range_max=range_max,
            data_points=data_points,
            **shared,
        ))

    return claims


def gt_to_claims(gt: GTFile) -> List[CanonicalClaim]:
    claim_support = _support_snippets_by_entry(gt)
    claims: List[CanonicalClaim] = []
    for idx, entry in enumerate(gt.value_condition_mapping):
        claims.extend(_entry_to_claims(gt, idx, entry, claim_support.get(idx, [])))
    return claims


def prediction_to_claims(pred_obj: dict) -> List[CanonicalClaim]:
    """Parse predictions into canonical claims.

    Supported shapes:
    1. GT-like schema with value_condition_mapping
    2. A direct canonical-like schema with top-level `claims`
    """
    if isinstance(pred_obj, dict) and "claims" in pred_obj:
        claims: List[CanonicalClaim] = []
        for idx, c in enumerate(pred_obj["claims"]):
            measurement_conditions = tuple(sorted((normalize_text(k), normalize_text(v)) for k, v in c.get("measurement_conditions", [])))
            material_specifications = tuple(sorted((normalize_text(k), normalize_text(v)) for k, v in c.get("material_specifications", [])))
            object_class = normalize_string_list(c.get("object_class", []))
            prov = []
            for e in c.get("provenance", []):
                prov.append(
                    CanonicalFieldEvidence(
                        source_url=e.get("source_url"),
                        source_uid=e.get("source_uid"),
                        citation_snippet=e.get("citation_snippet"),
                        matched_snippet=e.get("matched_snippet"),
                        matched_snippet_pieces=tuple(e.get("matched_snippet_pieces", [])),
                        citation_index_range=tuple(e["citation_index_range"]) if e.get("citation_index_range") else None,
                        value_index_range=tuple(e["value_index_range"]) if e.get("value_index_range") else None,
                        confidence=e.get("confidence"),
                    )
                )
            payload = ValuePayload.model_validate(c["value"])
            vtype, nval, rmin, rmax, dps = _payload_to_fields(payload)
            base_id = c.get("claim_id", f"pred::{idx}")
            shared = dict(
                entry_index=idx,
                material_family=normalize_text(c.get("material_family", "")),
                material_class=normalize_text(c.get("material_class", "")),
                material_subclass=normalize_text(c.get("material_subclass", "unspecified")),
                haptic_property=normalize_text(c.get("haptic_property", "")),
                measurement_conditions=measurement_conditions,
                material_specifications=material_specifications,
                object_class=object_class,
                units=normalize_units(c.get("units") or c.get("normalized_units")),
                grounded=c.get("is_grounded"),
                provenance=tuple(prov),
                support_snippets=tuple(c.get("support_snippets", [])),
            )

            if vtype == "range":
                claims.append(CanonicalClaim(claim_id=f"{base_id}::min", value_type="scalar",
                    normalized_value=float(rmin) if rmin is not None else None,
                    range_min=None, range_max=None, data_points=None, **shared))
                claims.append(CanonicalClaim(claim_id=f"{base_id}::max", value_type="scalar",
                    normalized_value=float(rmax) if rmax is not None else None,
                    range_min=None, range_max=None, data_points=None, **shared))
            elif vtype == "series" and dps:
                for dp_i, dp_val in enumerate(dps):
                    claims.append(CanonicalClaim(claim_id=f"{base_id}::dp{dp_i}", value_type="scalar",
                        normalized_value=float(dp_val),
                        range_min=None, range_max=None, data_points=None, **shared))
            else:
                claims.append(CanonicalClaim(claim_id=base_id, value_type=vtype,
                    normalized_value=nval, range_min=rmin, range_max=rmax,
                    data_points=dps, **shared))
        return claims

    gt = GTFile.model_validate(pred_obj)
    return gt_to_claims(gt)
