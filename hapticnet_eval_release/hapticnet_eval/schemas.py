from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BaseIgnoreExtraModel(BaseModel):
    model_config = ConfigDict(extra="ignore")


class LabelValuePair(BaseIgnoreExtraModel):
    label: str
    val: str


class GroundingRecord(BaseIgnoreExtraModel):
    triplet_index: Optional[int] = None
    triplet_uid: Optional[str] = None
    source_url: Optional[str] = None
    source_uid: Optional[str] = None
    citation_snippet: Optional[str] = None
    matched_snippet: Optional[str] = None
    matched_snippet_pieces: Optional[List[str]] = None
    confidence: Optional[float] = None
    matched_snippet_score: Optional[float] = None
    matched_snippet_pieces_scores: Optional[List[float]] = None
    matched_snippet_pieces_indices_ranges: Optional[List[Union[str, List[int]]]] = None
    citation_index_range: Optional[List[int]] = None
    value_index_range: Optional[List[int]] = None
    indirect_parameter_contributions: Optional[Dict[str, str]] = None
    indirect_parameter_groundings: Optional[Dict[str, Any]] = None
    per_parameter_grounding: Optional[Dict[str, Any]] = None


class ValuePayload(BaseIgnoreExtraModel):
    value: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    data_points: Optional[List[float]] = None
    mean: Optional[float] = None
    std: Optional[float] = None

    @model_validator(mode="after")
    def validate_one_shape(self) -> "ValuePayload":
        present = [
            self.value is not None,
            self.min is not None or self.max is not None,
            self.data_points is not None,
            self.mean is not None,
        ]
        if sum(bool(x) for x in present) != 1:
            raise ValueError("Exactly one of {value}, {min/max}, {data_points}, or {mean/std} must be provided")
        if (self.min is None) ^ (self.max is None):
            raise ValueError("Range values require both min and max")
        if self.mean is not None and self.std is None:
            self.std = 0.0  # default std to 0 if only mean is given
        return self

    def shape(self) -> Literal["scalar", "range", "series", "mean_std"]:
        if self.value is not None:
            return "scalar"
        if self.data_points is not None:
            return "series"
        if self.mean is not None:
            return "mean_std"
        return "range"

    def normalized_scalar(self) -> Optional[float]:
        if self.value is not None:
            return float(self.value)
        if self.mean is not None:
            return float(self.mean)
        if self.min is not None and self.max is not None:
            return (float(self.min) + float(self.max)) / 2.0
        if self.data_points:
            return sum(self.data_points) / len(self.data_points)
        return None


class ValueConditionMappingEntry(BaseIgnoreExtraModel):
    material_subclass: Optional[str] = None
    measurement_conditions: List[LabelValuePair] = Field(default_factory=list)
    material_specifications: List[LabelValuePair] = Field(default_factory=list)
    object_class: List[str] = Field(default_factory=list)
    units: Optional[str] = None
    value: ValuePayload

    is_grounded: Optional[bool] = None
    successful_groundings: List[GroundingRecord] = Field(default_factory=list)
    normalized_value: Optional[Union[ValuePayload, float]] = None
    normalized_units: Optional[str] = None


class CitationRecord(BaseIgnoreExtraModel):
    verbatim_snippet: str
    supports_fields: List[str]


class SourceRecord(BaseIgnoreExtraModel):
    url: str
    uid: Optional[str] = None
    title: Optional[str] = None
    note: Optional[str] = None
    answer_exists: Optional[bool] = None
    answer_hierarchy_level: Optional[str] = None
    query_hierarchy_level: Optional[str] = None


class LabelerNote(BaseIgnoreExtraModel):
    is_valid: Optional[str] = None
    is_valid_explanation: Optional[str] = None
    labeler_id: Optional[str] = None


class PropertyStats(BaseIgnoreExtraModel):
    mean: Optional[float] = None
    median: Optional[float] = None
    standard_deviation: Optional[float] = None


class GTFile(BaseIgnoreExtraModel):
    material_family: str
    material_class: str
    haptic_property: str
    value_condition_mapping: List[ValueConditionMappingEntry]
    citations: List[CitationRecord] = Field(default_factory=list)
    sources: List[SourceRecord] = Field(default_factory=list)
    labeler_note: Optional[LabelerNote] = None
    property_stats: Optional[PropertyStats] = None


@dataclass(frozen=True)
class ParameterEvidence:
    """Structured evidence for a single formula parameter (e.g., thermal conductivity)."""
    parameter_name: str
    target_value: str
    grounding_type: str  # "direct", "logical_jump", "source_note_derived"
    root_value_spans: Tuple[Tuple[int, int], ...]  # character spans in source text
    source_uid: str


@dataclass(frozen=True)
class CanonicalFieldEvidence:
    source_url: Optional[str]
    source_uid: Optional[str]
    citation_snippet: Optional[str]
    matched_snippet: Optional[str]
    matched_snippet_pieces: Tuple[str, ...]
    citation_index_range: Optional[Tuple[int, int]]
    value_index_range: Optional[Tuple[int, int]]
    confidence: Optional[float]
    per_parameter_evidence: Optional[Tuple['ParameterEvidence', ...]] = None


@dataclass(frozen=True)
class CanonicalClaim:
    claim_id: str
    entry_index: int
    material_family: str
    material_class: str
    material_subclass: str
    haptic_property: str
    measurement_conditions: Tuple[Tuple[str, str], ...]
    material_specifications: Tuple[Tuple[str, str], ...]
    object_class: Tuple[str, ...]
    units: str
    value_type: Literal["scalar", "range", "series"]
    normalized_value: Optional[float]
    range_min: Optional[float]
    range_max: Optional[float]
    data_points: Optional[Tuple[float, ...]]
    grounded: Optional[bool]
    provenance: Tuple[CanonicalFieldEvidence, ...]
    support_snippets: Tuple[str, ...]


class MatchResult(BaseIgnoreExtraModel):
    gt_claim_id: Optional[str] = None
    pred_claim_id: Optional[str] = None
    similarity: float
    gt_only: bool = False
    pred_only: bool = False
    debug: Dict[str, Any] = Field(default_factory=dict)


class EvaluatorScore(BaseIgnoreExtraModel):
    name: str
    score: float
    max_score: float = 1.0
    details: Dict[str, Any] = Field(default_factory=dict)


class EvaluationReport(BaseIgnoreExtraModel):
    task_id: str
    regime: str
    aggregate_score: float
    scores: List[EvaluatorScore]
    matches: List[MatchResult]
    metadata: Dict[str, Any] = Field(default_factory=dict)
