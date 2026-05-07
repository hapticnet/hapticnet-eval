# Evaluator Observability Guide

Every HapticNetEval benchmark run now produces **evaluator observability** — per-match debugging data showing *what was compared* and *how each score was calculated*. This guide explains the observability mechanism for all 25 evaluators (9 core + 8 advanced + 5 LLM-judge + 3 coverage/diagnostic), with real examples and source code references.

---

## How Observability Works

After the `BenchmarkRunner` produces `EvaluatorScore` objects, the `_build_observability()` post-processor in `run_benchmark.py` enriches each score entry. For every matched GT↔Pred claim pair, it:

1. **Resolves claim IDs** to full claim content via `_claim_summary()` — value, units, conditions, material hierarchy, sources
2. **Builds calculation traces** using evaluator-specific trace builders that decompose the score into its component parts

The result is an `evaluator_observability` list added to each score in both the JSON and Markdown reports.

### JSON Structure

```json
{
  "name": "factual_value",
  "score": 0.2188,
  "evaluator_observability": [
    {
      "match_index": 0,
      "gt_claim": { "claim_id": "...", "value_display": "0.38 dimensionless", ... },
      "pred_claim": { "claim_id": "...", "value_display": "0.38 dimensionless", ... },
      "match_similarity": 0.5539,
      "calculation_trace": { "strict": {...}, "tolerant": {...}, "final": {...} }
    }
  ]
}
```

### Markdown Rendering

Each evaluator gets a section with side-by-side GT↔Pred tables and calculation trace tables:

| | GT Claim | Pred Claim |
|---|---|---|
| **Value** | `0.38 dimensionless` | `0.38 dimensionless` |
| **Conditions** | counterface=steel, method=flat-on-flat | counterface=steel |

| Component | Result | Detail |
|---|---:|---|
| strict | 1.000 | 0.38 == 0.38 |
| tolerant | 1.000 | relative_closeness(0.38, 0.38): err=0.0 |
| **final** | **1.000** | 0.25 × 1.0 + 0.75 × 1.0 |

---

## Real Example: 6061 Aluminum / Kinetic Friction

All examples below come from a real evaluation of the DB Builder pipeline's output against the ground-truth file `6061_aluminum_kinetic_friction.json`.

**Configuration**: GT has **7 claims** (7 distinct kinetic friction values under different conditions). The prediction produced **4 claims**. Hungarian matching found **3 matched pairs** + 4 GT-only + 1 pred-only.

---

## 1. Core Evaluators (`evaluators/core.py`)

### 1.1 FactualValueEvaluator — `factual_value`

**Weight**: 0.25 | **Operation**: Claim-level

Measures numeric correctness with a blend of strict and tolerant comparison.

#### Source Code (scoring logic)

```python
# core.py lines 50-66
if g.value_type == "scalar" and p.value_type == "scalar":
    strict = float(g.normalized_value == p.normalized_value)
    tolerant = relative_closeness(g.normalized_value, p.normalized_value)
    s = 0.25 * strict + 0.75 * tolerant
elif g.value_type == "range" and p.value_type == "range":
    tolerant = interval_iou(g.range_min, g.range_max, p.range_min, p.range_max)
    strict = float((g.range_min, g.range_max) == (p.range_min, p.range_max))
    s = 0.25 * strict + 0.75 * tolerant
else:
    s = 0.0  # type mismatch
```

Where `relative_closeness` is:

```python
# normalization.py lines 89-96
def relative_closeness(a, b, rel_tol=0.05, abs_tol=1e-6):
    if math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return 1.0
    denom = max(abs(a), abs(b), abs_tol)
    err = abs(a - b) / denom
    return max(0.0, 1.0 - err / max(rel_tol, 1e-9))
```

#### Observability Trace Fields

| Field | Type | Description |
|---|---|---|
| `strict.result` | float | 1.0 if values are exactly equal, 0.0 otherwise |
| `strict.reason` | string | e.g. `"0.38 == 0.38"` or `"0.42 ≠ 0.40"` |
| `tolerant.result` | float | Output of `relative_closeness()` |
| `tolerant.formula` | string | Shows `err` and `tol` values |
| `final.result` | float | Blended score |
| `final.formula` | string | `"0.25 × strict + 0.75 × tolerant = score"` |

#### Real Example

```
GT claim:   metals::aluminum::kinetic friction coefficient::6061-t6::1
            value = 0.38 dimensionless
            conditions: material b=steel (flat-on-flat geometry), test method=flat-on-flat

Pred claim: metals::6061 aluminum::kinetic friction (uk)::al 6061-t6::0
            value = 0.38 dimensionless
            conditions: sliding on=steel

Trace:
  strict:   1.0    (0.38 == 0.38)
  tolerant: 1.0    (relative_closeness(0.38, 0.38): err=0.000000, tol=0.05)
  final:    1.0    (0.25 × 1.0 + 0.75 × 1.0 = 1.0)
```

**Interpretation**: The method found the correct value from the correct source. The score is perfect because the numeric values match exactly. However, the evaluator score across all 7 GT claims is only **0.2188** because 4 GT claims were unmatched (score = 0.0 each), diluting the average.

---

### 1.2 ConditionsEvaluator — `conditions_and_specs`

**Weight**: 0.0 (disabled) | **Operation**: Claim-level

Measures recovery of measurement conditions, material specs, and object class using set-F1.

#### Source Code

```python
# core.py lines 88-116
cond = _set_f1(g.measurement_conditions, p.measurement_conditions)
specs = _set_f1(g.material_specifications, p.material_specifications)
obj = _set_f1(g.object_class, p.object_class)
score = (cond * 0.6) + (specs * 0.25) + (obj * 0.15)
```

#### Trace Fields

| Field | Description |
|---|---|
| `conditions_F1.result` | Set-F1 over `(label, value)` condition pairs |
| `conditions_F1.gt` / `.pred` | Actual condition tuples being compared |
| `specs_F1.result` | Set-F1 over material specifications |
| `object_class_F1.result` | Set-F1 over object class labels |
| `final.formula` | `"0.60×cond + 0.25×specs + 0.15×obj"` |

#### Real Example

```
GT conditions:  [("material b", "steel (flat-on-flat geometry)"), 
                 ("test method", "flat-on-flat (literature values)")]
Pred conditions: [("sliding on", "steel")]

GT specs:   [("temper", "t6 (precipitation-hardened)")]
Pred specs: [("sliding on", "steel")]

conditions_F1: 0.0    (no overlap in normalized (label,value) pairs)
specs_F1:      0.0
object_F1:     0.0    (gt=["flat coupon"], pred=["unspecified"])
final:         0.0    (0.60×0.0 + 0.25×0.0 + 0.15×0.0)
```

**Interpretation**: Even though the value is correct, the structured conditions representation is completely different between GT labeler notation and method output. This reveals a systematic gap: the extraction method uses different labeling conventions than the GT.

---

### 1.3 CitationSourceRecallEvaluator — `citation_source_recall`

**Weight**: 0.12 | **Operation**: Claim-level | **Mode**: At-least-one

#### Source Code

```python
# core.py lines 170-178
overlap = gt_sources & pred_sources
# "At least one" mode: full credit if pred found value from any GT source
score = 1.0 if overlap else 0.0
strict_recall = len(overlap) / len(gt_sources)
```

#### Trace Fields & Real Example

```
gt_sources:    ["roymech.org/...", "sensorprod.com/..."]
pred_sources:  ["sensorprod.com/..."]
overlap:       ["sensorprod.com/..."]
score:         1.0    (at-least-one mode: any overlap → full credit)
strict_recall: 0.5    (1 of 2 GT sources recovered)
```

**Interpretation**: The method cited 1 of 2 GT sources — full credit in at-least-one mode but only 50% strict recall. The observability shows exactly which sources were found vs missed.

---

### 1.4 CitationSourcePrecisionEvaluator — `citation_source_precision`

**Weight**: 0.02 | **Operation**: Claim-level

```python
score = len(gt_sources & pred_sources) / len(pred_sources)
```

Penalizes irrelevant citations. In the example: pred cited 1 source, that source is in GT → precision = 1.0.

---

### 1.5 GroundingLocalizationEvaluator — `grounding_localization`

**Weight**: 0.20 | **Operation**: Claim-level

Measures how precisely the prediction localizes evidence within source documents using character-level offset IoU.

#### Source Code

```python
# core.py lines 245-248
v = overlap_ratio(ge.value_index_range, pe.value_index_range)
c = overlap_ratio(ge.citation_index_range, pe.citation_index_range)
same_source = float(bool(ge.source_url and pe.source_url and ge.source_url == pe.source_url))
score = 0.5 * same_source + 0.3 * v + 0.2 * c
```

Where `overlap_ratio` is interval IoU:

```python
# normalization.py lines 110-119
def overlap_ratio(span_a, span_b):
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 1.0
```

#### Real Example

```
gt_source:         sensorprod.com/wp-content/uploads/2023/08/wp_dfm-2000-10.pdf
pred_source:       sensorprod.com/wp-content/uploads/2023/08/wp_dfm-2000-10.pdf
same_source:       1.0
value_offset_IoU:  1.0    (exact match of value character range)
citation_offset_IoU: 0.965 (96.5% overlap of citation span)
final:             0.993  (0.50×1.0 + 0.30×1.0 + 0.20×0.965)
```

**Interpretation**: Near-perfect localization — the method found the correct value span and citation span in the correct document. The 0.965 citation IoU means the predicted citation range is very close to the GT range with only minor boundary differences.

#### Indirect Claims (per-parameter span overlap)

For indirect/formula-derived claims (e.g., thermal effusivity), the evaluator uses per-parameter span overlap instead of a single value span. The trace includes additional fields:

| Field | Description |
|---|---|
| `indirect_mode` | `true` when per_parameter_evidence is present |
| `per_param_detail` | List of per-parameter overlap results |
| `per_param_detail[].gt_param` | GT parameter name (e.g., `thermal conductivity (W/(m·K))`) |
| `per_param_detail[].pred_param` | Matched pred parameter name |
| `per_param_detail[].overlap` | Span IoU for this parameter |
| `per_param_detail[].gt_spans` | GT root_value_spans for this parameter |
| `per_param_detail[].pred_spans` | Pred root_value_spans for this parameter |

Unmatched GT parameters get `overlap: 0.0` and `note: "no matching pred param"`.

---

### 1.6 CitationSnippetSupportEvaluator — `citation_snippet_support`

**Weight**: 0.0 (superseded by `citation_support_judgement`) | **Operation**: Claim-level

Coarse lexical recovery using `exact_match_ratio` (Jaccard):

```python
# normalization.py
def exact_match_ratio(a, b):
    sa, sb = set(a or []), set(b or [])
    return len(sa & sb) / len(sa | sb)
```

#### Real Example

```
gt_support_count:   1     (GT has 1 support snippet)
pred_support_count: 2     (pred has 2 evidence texts)
support_label:      "full" (both numeric and lexical thresholds met)
score:              0.0   (exact_match_ratio returns 0 — different text strings)
```

**Interpretation**: Even though the support label is "full" (meaning pred evidence numerically/lexically supports the claim), the Jaccard exact-match score is 0.0 because verbatim snippet text differs. This evaluator is backward-compatible but conservative.

---

### 1.7 ClaimCompletenessEvaluator — `claim_completeness`

**Weight**: 0.05 | **Operation**: Aggregate (no per-match trace)

```python
score = matched_gt_claims / total_gt_claims
```

#### Real Example

```json
{ "matched_gt_claims": 3, "total_gt_claims": 7 }
```

Score = 3/7 = **0.4286**. The method recovered 3 of 7 GT claims. The missing 4 represent values under different test conditions that the method failed to extract.

---

### 1.8 HallucinationEvaluator — `unsupported_claim_rate`

**Weight**: 0.03 | **Operation**: Aggregate

```python
unsupported_rate = pred_only_count / total_pred_count
score = 1.0 - unsupported_rate
```

#### Real Example

```json
{ "pred_only": 1, "total_pred": 4, "unsupported_rate": 0.25 }
```

Score = 1.0 − 0.25 = **0.75**. One of 4 predicted claims has no matching GT claim (potentially hallucinated or a valid value not in GT).

---

## 2. Advanced Evaluators (`evaluators/advanced.py`)

### 2.1 SourceEquivalenceEvaluator — `source_equivalence`

**Weight**: 0.0 (reported only)

Uses URL canonicalization, UID comparison, and title fuzzy matching to detect when pred cites an equivalent but different URL.

#### Trace: Shows pair-level equivalence assessments with match method.

---

### 2.2 CitationSupportJudgementEvaluator — `citation_support_judgement`

**Weight**: 0.0 (reported only)

Assigns **full / partial / none** support label based on lexical and numeric overlap:

```python
# evidence.py
# full:    lexical_overlap >= 0.4 AND numeric_overlap >= 0.5
# partial: either threshold met
# none:    neither met
```

#### Trace: Shows `label`, `lexical_overlap`, `numeric_overlap`, `best_gt_snippet`.

---

### 2.3 CitationFaithfulnessEvaluator — `citation_faithfulness`

**Weight**: 0.0 (reported only) | **Anti-gaming** evaluator

Detects post-rationalization: correct value but evidence doesn't actually support it.

```python
# Builds "anchors" from the claim:
text_anchors, numeric_anchors = build_claim_anchors(
    material_subclass, measurement_conditions, normalized_value, ...
)
# Checks anchor coverage in pred evidence
assessment = assess_citation_faithfulness(evidence_texts, text_anchors, numeric_anchors, ...)
```

#### Trace Fields

| Field | Description |
|---|---|
| `support_label` | full/partial/none |
| `numeric_anchor_coverage` | Fraction of numeric anchors found in evidence |
| `entity_anchor_coverage` | Fraction of entity/material anchors found |
| `likely_post_rationalized` | Boolean flag — true if value correct but anchors missing |
| `is_indirect` | Boolean — true if per_parameter_evidence was used for anchoring |
| `anchor_type` | `"per_parameter"` or `"final_value"` — shows which anchoring strategy was used |

---

### 2.4 NoAnswerEvaluator — `no_answer`

**Weight**: 0.0 (reported only) | **Operation**: File-level

```python
gt_no_answer = (len(gt_claims) == 0) or (all answer_exists flags are False)
pred_no_answer = pred has no_answer/abstain flag or empty claims
score = float(gt_no_answer == pred_no_answer)
```

#### Real Example

```json
{ "gt_no_answer": false, "pred_no_answer": false }
```

Score = 1.0. Both GT and pred found answers — correct behavior.

---

### 2.5 ConflictingEvidenceEvaluator — `conflicting_evidence`

**Weight**: 0.0 (reported only) | **Condition-dependent variation**

Groups GT claims by `(family, class, property, units)`, finds clusters with ≥2 distinct condition-value signatures, measures if pred preserved that multiplicity.

#### Source Code

```python
# advanced.py lines 231-237
cond_sigs = {condition_signature(c) for c in claims}
value_sigs = {value_signature(c) for c in claims}
if len(claims) >= 2 and len(cond_sigs) >= 2 and len(value_sigs) >= 2:
    matched_pred_ids = {match_map[c.claim_id] for c in claims if c.claim_id in match_map}
    score = len(matched_pred_ids) / len(claims)
```

#### Real Example

The 6061 Aluminum GT has 7 kinetic friction values under different conditions (different counterfaces, test methods). This is one cluster with 7 distinct condition-value signatures. The prediction matched 3 → score = 3/7 = **0.4286**.

---

### 2.6 CostEfficiencyEvaluator — `cost_efficiency`

**Weight**: 0.0 | **Operation**: File-level

```python
score = max(0, 1 - value/budget)  # budget default: $0.10
```

Score = 0.0 in the example (no cost metadata in prediction file).

---

### 2.7 LatencyEfficiencyEvaluator — `latency_efficiency`

**Weight**: 0.0 | Budget default: 30s. Same formula as cost efficiency.

---

### 2.8 ToolUsageEfficiencyEvaluator — `tool_usage_efficiency`

**Weight**: 0.0 | Composite across tool_calls (budget 10), model_calls (budget 8), pages_fetched (budget 20).

---

## 3. Evidence-Gated Evaluators

### 3.1 StrictGroundednessEvaluator — `strict_groundedness`

**Weight**: 0.10 | **Operation**: Claim-level | **FEVER-style all-or-nothing**

A claim passes **only if ALL** dimensions simultaneously pass:

```python
# strict_groundedness.py lines 28-51
value_score = float(relative_closeness(g.normalized_value, p.normalized_value) >= 0.999)
units_ok = normalize_units(g.units) == normalize_units(p.units)
citation_ok = float(bool(gt_urls & pred_urls)) if gt_urls else 1.0
passed = value_score > 0 and units_ok and citation_ok > 0
```

#### Trace Fields

| Field | Description |
|---|---|
| `passed` | Boolean — all gates satisfied? |
| `value_score` | 1.0 if near-exact (within 0.1%), 0.0 otherwise |
| `units_ok` | Boolean — normalized units match? |
| `citation_ok` | 1.0 if at least one shared citation URL, 0.0 otherwise |

#### Real Example

For the `0.38 dimensionless` claim: `value_score=1.0`, `units_ok=true`, `citation_ok=1.0` → **passed=true**. But the overall evaluator score is **0.25** because only 1 of 4 matched pairs passed all gates.

---

### 3.2 OpenWebApproxEvidenceEvaluator — `open_web_approx_evidence`

**Weight**: 0.0 | **Regime**: OPEN_WEB only

```python
# openweb_evidence.py lines 37-49
domain_match = float(pred_domain in gt_domains)
token_ratio = condition_token_overlap(gt_conditions, pred_evidence_text)
value_match = float(str(round(gt_value, 2)) in evidence_text)
score = max(domain_match, 0.55 * value_match + 0.45 * token_ratio)
```

Useful when pred cites completely different URLs than GT (open-web regime).

---

## 4. Coverage & Diagnostic Evaluators

### 4.1 ValueListCoverageEvaluator — `value_list_coverage`

**Weight**: 0.15 | **Operation**: File-level

Flattens all scalar values from both GT and pred, performs greedy numeric matching with 0.1% tolerance:

```python
# completeness.py lines 30-46
def greedy_numeric_match(pred, gt, rel_tol=1e-3):
    tp = 0
    for p in pred:
        for gi, g in enumerate(gt):
            if gi not in used_gt and abs(g - p) / max(abs(g), 1e-12) <= rel_tol:
                tp += 1; used_gt.add(gi); break
    return tp, len(pred) - tp, len(gt) - tp
```

Reports precision, recall, and F1 over the value inventory.

#### Real Example

```json
{
  "gt_scalar_value_count": 7,
  "pred_scalar_value_count": 4,
  "true_positives": 1,
  "false_positives": 3,
  "false_negatives": 6,
  "precision": 0.25,
  "recall": 0.143
}
```

F1 = **0.1818**. Only 1 of 4 predicted values matched any GT value. The method is extracting values, but mostly different ones than the GT.

---

### 4.2 StatementSupportEvaluator — `statement_support`

**Weight**: 0.0 | **Operation**: Global F1

ALCE/AIS-style: identifies which pred claims are supported by their own citations, then computes F1 against GT claims:

```python
# statement_support.py lines 30-53
# A pred claim is "supported" if:
# 1. Its own evidence text contains its numeric value (value_supported)
# 2. >= 50% of its condition tokens appear in the evidence (token_ratio >= 0.5)
if val_supported and token_ratio >= 0.5:
    supported_pred_ids.add(p.claim_id)

# Then: F1 = 2×P×R / (P+R) where P and R are over (matched_GT ∩ supported_pred)
```

#### Real Example

```json
{
  "precision": 1.0,
  "recall": 0.286,
  "f1": 0.4444,
  "supported_pred_ids": ["metals::6061 aluminum::kinetic friction (uk)::..."],
  "gt_claim_count": 7
}
```

Only 2 of 7 GT claims were matched by supported predictions → recall=0.286. But all supported predictions that matched GT were correct → precision=1.0.

---

## 5. LLM Evaluator Observability

LLM-based evaluators produce richer observability traces than deterministic evaluators, including per-call resource tracking and LLM judge reasoning.

### 5.1 SemanticConditionsEvaluator — `semantic_conditions`

**Weight**: 0.12 | **Operation**: Claim-level | **LLM-judge**

Uses an LLM-as-a-judge to assess semantic equivalence of measurement conditions.

#### Trace Fields

| Field | Description |
|---|---|
| `gt_conditions` | GT measurement conditions string |
| `gt_specs` | GT material specifications string |
| `gt_object_class` | GT object class string |
| `pred_conditions` | Pred measurement conditions string |
| `pred_specs` | Pred material specifications string |
| `pred_object_class` | Pred object class string |
| `score` | 0.0–1.0 semantic similarity score |
| `reasoning` | LLM-generated step-by-step justification |
| `call_resource` | Per-call LLM resource record (see §6) |

**Short-circuit behavior**: When both sides are empty or exact string match, score = 1.0 with no LLM call. The trace shows `reasoning = "Both GT and prediction have completely empty conditions..."` or `"Exact match — ..."` in these cases.

---

### 5.2 CalibratedClaimJudgeEvaluator — `verified_novel_claim_rate`

**Weight**: 0.0 (closed-docs), 0.40 (guided open-web), 0.90 (unguided open-web) | **LLM-judge**

The calibrated judge produces the most detailed traces of any evaluator. Each `pred_only` claim generates a trace row with:

#### Trace Fields

| Field | Description |
|---|---|
| `pred` | Pred claim ID |
| `value` | Extracted numeric value |
| `units` | Value units |
| `material` | Material class (subclass) |
| `conditions` | Measurement conditions string |
| `specs` | Material specifications string |
| `has_grounded_provenance` | Boolean — does the claim have any source_uid? |
| `evidence_available` | Boolean — was evidence text extractable? |
| `evidence_sources` | List of source UIDs used |
| `evidence_text` | First 2000 chars of evidence bundle text |
| `confidence` | 0.0–1.0 judge confidence |
| `is_verified` | Boolean — confidence ≥ threshold (0.95)? |
| `verdict` | `"supported"` / `"unsupported"` / `"insufficient_evidence"` |
| `failure_tags` | List of failure tags (e.g., `["value_mismatch", "hierarchy_specificity_error"]`) |
| `reasoning` | Full step-by-step LLM reasoning |
| `call_resource` | Per-call LLM resource record (see §6) |

#### Special Trace Patterns

| Pattern | `reasoning` prefix | Meaning |
|---|---|---|
| Param-level pre-check | `"PARAM-LEVEL PRE-CHECK: ..."` | All component params match a GT claim — matching failure, not novelty. Score = 0.75, no LLM call. |
| Ungrounded skip | `"SKIPPED — No grounded provenance. ..."` | No source_uid in provenance. Score = 0.0, no LLM call. |
| No evidence text | `"Claim has grounded provenance but..."` | Source document not found or empty. Score = 0.0, no LLM call. |
| LLM failure | `"LLM Judge failed to produce..."` | API error after retries. Score = 0.0. |

#### Aggregate Details

```json
{
  "pred_only_count": 5,
  "valid_novel_count": 3,
  "llm_resource_usage": {
    "calls": 4,
    "input_tokens": 45000,
    "output_tokens": 2400,
    "thinking_tokens": 4096,
    "total_tokens": 51496,
    "cost_usd": 0.006180,
    "latency_ms": 12340.5
  },
  "prompt_variant": "E5_E1_E3_calibrated",
  "judge_model": "gemini-3.1-flash-lite",
  "thinking_budget": 1024,
  "rows": [...]
}
```

---

### 5.3 VerifiedClaimsF1Evaluator — `verified_claims_f1`

**Weight**: 0.15 | **Operation**: Cross-evaluator

This evaluator runs **after** `factual_value` and `verified_novel_claim_rate` and reads their results from the context.

#### Trace Fields

| Field | Description |
|---|---|
| `precision` | Verified pred claims / total pred claims |
| `recall` | Matched GT with good value / total GT claims |
| `f1` | Harmonic mean of precision and recall |
| `verified_pred_count` | Count of verified prediction claims |
| `falsified_pred_count` | Count of falsified prediction claims |
| `matched_gt_with_value` | GT claims matched with factual_value ≥ threshold |
| `value_threshold` | Minimum factual_value score for "verified" (default: 0.3) |
| `novel_threshold` | Minimum novel claim confidence for "verified" (default: 0.75) |
| `rows[].type` | `"gt_only"` / `"pred_only"` / `"matched"` |
| `rows[].status` | `"missed"` / `"verified_novel"` / `"falsified"` / `"ungrounded"` / `"verified_match"` / `"falsified_match"` / `"weak_match"` |

---

### 5.4 UnverifiedClaimRateEvaluator — `unverified_claim_rate`

**Weight**: 0.0 (reported only) | **Operation**: Cross-evaluator

#### Trace Fields

| Field | Description |
|---|---|
| `unverified_pred_only` | Count of pred_only claims NOT verified by judge |
| `verified_pred_only` | Count of pred_only claims verified by judge |
| `total_pred` | Total prediction claims |
| `unverified_rate` | unverified_pred_only / total_pred |

---

## 6. LLM Resource Tracking System

All LLM-based evaluators share a global `LLMResourceTracker` (singleton per benchmark run) that records per-call token usage, latency, and cost.

### Architecture

```
LLMJudge.generate_structured()
    ↓
LLMResourceTracker.record(input_tokens, output_tokens, thinking_tokens, latency_ms, evaluator, model)
    ↓
per_call_records: [{evaluator, model, input_tokens, output_tokens, thinking_tokens, latency_ms, cost_usd}, ...]
```

### Per-Call Resource Record

Each LLM call produces a `call_resource` dict attached to the evaluator's detail row:

```json
{
  "evaluator": "verified_novel_claim_rate",
  "model": "gemini-3.1-flash-lite",
  "input_tokens": 11250,
  "output_tokens": 600,
  "thinking_tokens": 1024,
  "latency_ms": 3085.2,
  "cost_usd": 0.001545
}
```

### Per-Evaluator Summary

Each LLM evaluator includes an `llm_resource_usage` summary in its details:

```json
{
  "calls": 4,
  "input_tokens": 45000,
  "output_tokens": 2400,
  "thinking_tokens": 4096,
  "total_tokens": 51496,
  "cost_usd": 0.006180,
  "latency_ms": 12340.5
}
```

### Global Summary

The tracker's `summary()` method returns aggregate statistics across **all** LLM evaluators in a benchmark run:

```json
{
  "total_calls": 12,
  "total_input_tokens": 135000,
  "total_output_tokens": 7200,
  "total_thinking_tokens": 12288,
  "total_tokens": 154488,
  "total_cost_usd": 0.018540,
  "total_latency_ms": 37021.5,
  "avg_latency_ms": 3085.1
}
```

### Model Pricing

Pricing per 1M tokens is configured in `LLMResourceTracker.MODEL_PRICING`:

| Model | Input | Output | Thinking |
|---|---|---|---|
| `gemini-2.5-flash-lite` | $0.10 | $0.40 | $0.40 |
| `gemini-2.5-flash` | $0.15 | $0.60 | $0.60 |
| `gemini-3.1-flash-lite-preview` | $0.10 | $0.40 | $0.40 |
| `gemini-3.1-pro-preview` | $2.00 | $12.00 | $12.00 |

---

## 7. Evidence Fetcher Observability

The `EvidenceFetcher` produces per-URL `FetchRecord` entries that are included in the evaluation report's metadata under `source_fetch_stats`.

### FetchRecord Fields

| Field | Type | Description |
|---|---|---|
| `url` | string | Source URL |
| `uid` | string | MD5 hash UID of the URL |
| `status` | string | `"fetched"`, `"cached"`, `"pre_existing"`, `"failed"`, `"skipped"` |
| `source_dir` | string | `"source_index"` or `"cache"` — where content was found |
| `content_size` | int | Size of content file in bytes |
| `elapsed_s` | float | Time taken to fetch (fetched only) |
| `error` | string | Error message (failed/skipped only) |
| `tavily_extract_depth` | string | `"basic"` or `"advanced"` (fetched only) |
| `content_file` | string | Filename of content file |

### Aggregate Summary

```json
{
  "total_urls": 8,
  "by_status": {
    "pre_existing": 3,
    "cached": 2,
    "fetched": 2,
    "failed": 1
  },
  "total_content_bytes": 245000,
  "total_fetch_time_s": 4.52,
  "cache_dir": "~/.hapticnet_source_cache",
  "records": [...]
}
```

### Content Resolution Priority

The fetcher searches for content files in this order:
1. **Production SourceIndex** (`/mnt/cgm-atlas/ofri/HapticNet/SourceIndex/{uid}/content/`)
2. **Local cache** (`~/.hapticnet_source_cache/{uid}/content/`)

Within each directory, it tries these filenames in order:
1. `content.md`
2. `content_tavily.md`
3. `content_jina.md`

The first file found with >100 bytes is used.

---

## Practical Guide: Using Observability to Improve Methods

### Diagnosing Low Scores

| Symptom | Evaluator to Check | What to Look For |
|---|---|---|
| Low overall score | `claim_completeness` | How many GT claims were unmatched? |
| Value seems right but score is 0 | `factual_value` trace | Check `strict` vs `tolerant` — maybe a rounding difference |
| Good values but low aggregate | `grounding_localization` | Are offset ranges being captured? |
| High hallucination rate | `unsupported_claim_rate` | How many pred claims have no GT match? |
| Zero on conditions | `semantic_conditions` trace | Read the LLM reasoning — is it labeling convention or actual mismatch? |
| TE always post-rationalized | `citation_faithfulness` trace | Check `anchor_type` — if `final_value`, the computed TE never appears in cites; switch to `per_parameter` mode |
| Indirect matching fails | `matching` Hungarian output | Check if `_is_indirect_claim` detected it; review `indirect_param_similarity` trace |
| Novel claim is actually GT | `verified_novel_claim_rate` | Check if param-level pre-check triggered (`PARAM-LEVEL PRE-CHECK` in reasoning) |
| Judge rejecting valid claims | `verified_novel_claim_rate` rows | Check `failure_tags` and `evidence_text` — is the evidence window too narrow or the source doc missing? |
| High LLM judge cost | `llm_resource_usage` in evaluator details | Check `total_tokens` and `calls` — are too many claims going to the LLM? |
| Evidence not found | `source_fetch_stats.records` | Check `status` = `"failed"` entries — are URLs returning 403/404? |

### Evaluating the Evaluation

The observability data also reveals when evaluators may be **unfair or need reweighting**:

- **hierarchy** scoring 0 because GT uses "aluminum" but pred uses "6061 aluminum" → suggests string normalization should be looser
- **conditions_and_specs** scoring 0 despite semantically similar conditions → this is why `semantic_conditions` (LLM judge) replaced it at weight 0.12
- **citation_snippet_support** scoring 0 despite `support_label=full` → the Jaccard metric is too conservative; `citation_support_judgement` is more appropriate
- **strict_groundedness** scoring 0 for derived properties → this is expected for formula-computed values; use `INDIRECT_WEIGHTS` which zeroes it out

### Co-evolving Evaluation + Methods

1. Run evaluation with observability
2. Filter to evaluators with weight > 0 (these affect the aggregate score)
3. For each low-scoring evaluator, inspect the traces to determine: *is the method wrong, or is the evaluator too strict?*
4. Check `llm_resource_usage` — are LLM judge costs acceptable?
5. Check `source_fetch_stats` — are evidence documents available?
6. Adjust method (extraction pipeline) or evaluator (weights/thresholds) accordingly
7. Re-run and compare
