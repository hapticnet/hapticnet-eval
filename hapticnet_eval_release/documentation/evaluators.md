# Evaluator Reference

This document provides an in-depth explanation of every evaluator in the HapticNetEval Evaluator Framework: what it measures, how it scores, and which academic work inspired it.

---

## Evaluation Architecture

Every evaluator inherits from `BaseEvaluator` and receives:

| Argument | Type | Description |
|---|---|---|
| `gt_index` | `ClaimIndex` | Indexed ground-truth canonical claims |
| `pred_index` | `ClaimIndex` | Indexed prediction canonical claims |
| `matches` | `List[MatchResult]` | GT↔Pred claim alignment from Hungarian matching |
| `context` | `Dict` | Raw file-level objects (`gt_obj`, `pred_obj`, `task`, etc.) |

Each evaluator returns an `EvaluatorScore(name, score, max_score=1.0, details)`.

Evaluators fall into two operation modes:
- **Claim-level** — iterate over `matches`, compare paired GT/Pred `CanonicalClaim` objects.
- **File-level** — use `context["gt_obj"]` / `context["pred_obj"]` to access the full raw JSON schema.

---

## 1. Core Evaluators

Located in `evaluators/core.py`.

---

### 1.1 FactualValueEvaluator

| | |
|---|---|
| **Name** | `factual_value` |
| **Default weight** | 0.25 |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level |

**What it measures**: The numeric correctness of the predicted value compared to the ground-truth value.

**Scoring formula**: For each matched pair, the score is a blend of strict and tolerant comparison:

```
score = 0.25 × strict + 0.75 × tolerant
```

- **Scalar values** (including **mean±std**, which is decomposed into separate scalar claims for the mean and standard deviation): `strict` = exact equality; `tolerant` = `relative_closeness(gt, pred)` (1.0 when within 5% relative tolerance, decaying linearly beyond).
- **Range values**: `strict` = exact min/max match; `tolerant` = interval IoU (intersection-over-union of the two ranges).
- **Series values**: `strict` = exact point-by-point match; `tolerant` = `sequence_similarity` (mean pairwise relative closeness, truncated to shorter length).
- **Mismatched types**: score = 0.0.

The final evaluator score is the mean across all matches.

**Academic attribution**:
- **FActScore** (Min et al., 2023) — atomic fact verification at the claim level.
- **RAGChecker** — component-level scoring of retrieved structured fields.
- **ChemX** (Dunn et al.) — chemical property extraction accuracy with numeric tolerance.
- **MeasEval** (Harper et al., SemEval-2021) — measurement extraction with tolerance-based scoring.

---

### 1.2 ConditionsEvaluator

| | |
|---|---|
| **Name** | `conditions_and_specs` |
| **Default weight** | 0.0 (disabled) |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level |

**What it measures**: How well the prediction recovers the structured context surrounding a value — the measurement conditions, material specifications, and object class.

**Scoring formula**: For each matched pair, three set-F1 scores are computed over normalized `(label, value)` pairs:

```
score = 0.60 × conditions_F1 + 0.25 × specs_F1 + 0.15 × object_class_F1
```

**Academic attribution**:
- **RAGChecker** — fine-grained component scoring.
- **ChemX** — structured-field column-level evaluation for chemical property extraction.

---

### 1.3 HierarchyEvaluator

| | |
|---|---|
| **Name** | `hierarchy` |
| **Default weight** | 0.0 (disabled) |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level |

**What it measures**: Whether the prediction places the value at the correct level in the material taxonomy (family → class → subclass).

**Scoring formula**:

```
score = 0.20 × family_match + 0.30 × class_match + 0.50 × subclass_match
```

Each component is binary (1.0 if the normalized strings are identical, 0.0 otherwise). The weighting emphasizes subclass precision, because family/class are usually easy to get right.

**Academic attribution**:
- **ChemX** — hierarchy-sensitive column-level reporting in property extraction.
- **KILT** (Petroni et al., 2021) — entity-level knowledge retrieval requiring entity hierarchy awareness.

---

### 1.4 CitationSourceRecallEvaluator

| | |
|---|---|
| **Name** | `citation_source_recall` |
| **Default weight** | 0.12 |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level |

**What it measures**: The fraction of ground-truth source URLs that appear in the prediction's provenance for each matched claim.

**Scoring formula**:

```
recall = |GT_sources ∩ Pred_sources| / |GT_sources|
```

If both sets are empty, score = 1.0 (no sources to recover). If only GT has no sources, score = 1.0.

**Academic attribution**:
- **ALCE** (Gao et al., 2023) — citation recall measuring how much relevant evidence the system recovers.
- **KILT** — source recall in knowledge-intensive tasks.

---

### 1.5 CitationSourcePrecisionEvaluator

| | |
|---|---|
| **Name** | `citation_source_precision` |
| **Default weight** | 0.02 |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level |

**What it measures**: The fraction of the prediction's cited sources that are actually valid GT sources for the claim.

**Scoring formula**:

```
precision = |GT_sources ∩ Pred_sources| / |Pred_sources|
```

Penalizes irrelevant or surplus cited sources.

**Academic attribution**:
- **ALCE** — citation precision penalizing irrelevant citations.
- **GopherCite** (Menick et al., 2022) — supporting evidence precision in LLM responses.

---

### 1.6 GroundingLocalizationEvaluator

| | |
|---|---|
| **Name** | `grounding_localization` |
| **Default weight** | 0.20 |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level |

**What it measures**: How precisely the prediction localizes the evidence within source documents — using character-level offset spans for both the value and the citation.

**Scoring formula**: For each GT provenance record, the best-matching prediction provenance on the **same source URL** is found. The score combines:

```
score = 0.50 × same_source + 0.30 × value_offset_IoU + 0.20 × citation_offset_IoU
```

The offset IoU is the intersection-over-union of `[start, end]` character ranges.

**Indirect-aware behavior**: For indirect/formula-derived values (e.g., thermal effusivity), the evaluator uses **per-parameter span overlap** instead of a single value span. When either GT or pred provenance has `per_parameter_evidence`, the `_per_param_overlap()` method:
1. Matches GT parameters to pred parameters by name (with substring matching for synonyms like `thermal conductivity` ↔ `k`)
2. Compares `root_value_spans` using `overlap_ratio()` for each matched param pair
3. Averages across all GT params (unmatched params score 0)
4. Includes per-parameter observability detail in the trace output

This replaces the meaningless single-span comparison that would occur on a computed value.

**Academic attribution**:
- **LegalBench-RAG** — precise span retrieval evaluation in legal documents.
- **ALiiCE** (Guerreiro et al.) — fine-grained citation localization scoring.

---

### 1.7 CitationSnippetSupportEvaluator

| | |
|---|---|
| **Name** | `citation_snippet_support` |
| **Default weight** | 0.0 (disabled; superseded by `citation_support_judgement`) |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level |

**What it measures**: Coarse lexical recovery of the evidence snippets. Compares the set of GT support snippets against the set of predicted citation/matched snippets using exact match ratio (Jaccard-like).

This is a backward-compatible evaluator. The newer `CitationSupportJudgementEvaluator` provides richer full/partial/none labels and has replaced it in the default weight configuration.

**Academic attribution**:
- **ALCE / AIS** — attribution identification and snippet support checking.

---

### 1.8 ClaimCompletenessEvaluator

| | |
|---|---|
| **Name** | `claim_completeness` |
| **Default weight** | 0.05 |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level (aggregate) |

**What it measures**: Recall-style coverage — what fraction of GT claims were successfully matched to a prediction claim.

```
score = matched_gt_claims / total_gt_claims
```

**Academic attribution**:
- **FActScore** — recall of atomic facts.
- **RAGChecker** — claim coverage metrics.

---

### 1.9 HallucinationEvaluator

| | |
|---|---|
| **Name** | `unsupported_claim_rate` |
| **Default weight** | 0.03 |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level (aggregate) |

**What it measures**: The fraction of **prediction claims** that have no matching GT claim — i.e., unsupported or hallucinated claims. Score = 1.0 − unsupported rate.

```
unsupported_rate = pred_only_count / total_pred_count
score = 1.0 - unsupported_rate
```

**Academic attribution**:
- **FActScore** — hallucination rate as 1 − precision of supported facts.
- **RAGChecker** — false claim detection.
- **Ev2R** (Shuster et al.) — hallucination identification in grounded generation.

---

## 2. Advanced / LLM Evaluators

Located in `evaluators/advanced.py` and `evaluators/llm_evaluators.py`.

---

### 2.0 SemanticConditionsEvaluator

| | |
|---|---|
| **Name** | `semantic_conditions` |
| **Default weight** | 0.12 |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level |
| **Location** | `evaluators/llm_evaluators.py` |

**What it measures**: Semantic equivalence of measurement conditions using an LLM-as-a-judge. Addresses the extreme strictness of Set-F1 (`conditions_and_specs`) when comparing disparate labeling conventions (e.g. `material b=steel` vs `sliding on=steel`).

**Scoring formula**: Generates a 0.0 - 1.0 confidence score representing semantic equivalence via `gemini-2.5-flash`. Short-circuits to 1.0 when both sides are empty or when exact string match is detected (no LLM call needed).

**Indirect-aware behavior**: For indirect/formula-derived claims, the prompt includes Rule 7 sub-rules:
- **7a**: Formula notation equivalence (e.g., `b=(λ·ρ·c)^0.5` ↔ `e=√(k·ρ·c_p)`)
- **7b**: Parameter value matching with tolerance (same params + similar values → 0.70–1.0)
- **7c**: Mixed conditions (formula+params vs measurement conditions) → 0.40–0.60
- **7d**: Direct vs indirect claim comparison
- **7e**: GT unspecified + Pred has formula+params → 0.50–0.70 (rewards useful derivation info)

**Resource tracking**: Each LLM call records `input_tokens`, `output_tokens`, `thinking_tokens`, `latency_ms`, and `cost_usd` via the global `LLMResourceTracker`. The per-call record is attached to the evaluator detail row as `call_resource`.

**Academic attribution**:
- **LLM-as-a-Judge** (Zheng et al., 2023) — using powerful foundation models to assess semantic equivalence of textual data beyond lexical overlap.

---

### 2.0a CalibratedClaimJudgeEvaluator (Production)

| | |
|---|---|
| **Name** | `verified_novel_claim_rate` |
| **Default weight** | 0.0 (closed-docs), **0.40** (guided open-web), **0.90** (unguided open-web) |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level (aggregate) |
| **Location** | `evaluators/llm_evaluators.py` |
| **Judge model** | `gemini-3.1-flash-lite` (thinking_level=low, 1024 tokens) |

**What it measures**: Validates predictions that have no matching Ground Truth claim (i.e. novel or unmatched extractions). Uses a **calibrated** LLM-as-a-judge prompt (variant E5_E1_E2_E3_E4) to verify whether each `pred_only` claim is genuinely supported by its own cited source evidence. This is the **primary quality signal for open-web tracks**.

**Calibration**: The prompt was developed through systematic ablation over 5 feature variants (E1–E5) on a 176-item calibration set. The E5_E1_E2_E3_E4 variant achieved **0.735 AUROC** with `gemini-3.1-flash-lite` (thinking=low), compared to 0.648 for the uncalibrated baseline.

**Prompt features** (each verified independently during calibration):
- **E1**: Calibrated confidence scoring with explicit 0.0–1.0 range guidance
- **E2**: Permissive condition step (condition differences alone don't cause rejection)
- **E3**: Few-shot worked examples (4 exemplars: 2 supported, 2 unsupported)
- **E4**: Property-specific verification notes (Young's modulus, thermal effusivity)
- **E5**: Holistic rubric with 7-step evaluation procedure

**Evidence bundle format**: Structured `[EVIDENCE N]` blocks containing:
- Source metadata (UID, URL, title, grounding confidence)
- Citation/matched snippets from grounding
- Evidence window text (3000 chars around value span in source document)
- Indirect parameter contributions and per-parameter grounding details

**Output schema** (`CalibratedJudgeDecision`):
```python
class CalibratedJudgeDecision(BaseModel):
    is_valid: bool       # True only when verdict is "supported"
    confidence: float    # 0.0–1.0
    verdict: str         # "supported" | "unsupported" | "insufficient_evidence"
    support_mode: str    # "direct" | "derived" | "mixed" | "not_applicable"
    failure_tags: list    # e.g. ["hierarchy_specificity_error", "value_mismatch"]
    evidence_quality: str # "strong" | "medium" | "weak"
    reasoning: str       # Step-by-step explanation
```

**Indirect-aware behavior**: For indirect claims with `per_parameter_evidence`:
1. **Param-level pre-check**: Before calling the LLM, extracts component parameters (TC, density, Cp) from the pred_only claim and checks if ALL match some GT claim (using `relative_closeness ≥ 0.85`). If so, the claim is scored 0.75 without an LLM call — it's a matching failure, not genuine novelty.
2. **Evidence windowing**: Uses per-parameter span locations to extract labeled windows around each parameter's grounding in the source document. Overlapping windows are merged.
3. **Indirect section in prompt**: Instructs the judge to verify component parameter values (not the computed final value) and check material hierarchy consistency per parameter.

**Evidence gating**:
- Claims with **no grounded provenance** (no `source_uid`) are scored 0.0 immediately — no LLM call is made.
- Claims with grounded provenance but **no extractable evidence text** are also scored 0.0.

**Scoring formula**:
```
score = verified_novel_claims / total_pred_only_claims
verified = claims with confidence ≥ threshold (default: 0.95)
```

**Resource tracking**: Each LLM call records `input_tokens`, `output_tokens`, `thinking_tokens`, `latency_ms`, and `cost_usd` via the global `LLMResourceTracker`. Aggregate resource usage is included in the evaluator details as `llm_resource_usage`.

**Academic attribution**:
- **Attributed QA** / **RAGChecker** — automated fact-checking and entailment for hallucination detection.
- **LLM-as-a-Judge** (Zheng et al., 2023) — calibrated judge prompt methodology.

---

### 2.1 SourceEquivalenceEvaluator

| | |
|---|---|
| **Name** | `source_equivalence` |
| **Default weight** | 0.0 (reported only) |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level + file-level |

**What it measures**: Whether predicted sources are equivalent to GT sources, even if they appear under different URLs (mirrors, URL variants, same document via different hosts).

Uses URL canonicalization (stripping protocol, www prefix, trailing slashes), UID comparison, and title fuzzy matching to detect equivalence.

**Academic attribution**:
- **ALCE / ALiiCE** — citation-aware evaluation recognizing equivalent alternative sources.

---

### 2.2 CitationSupportJudgementEvaluator

| | |
|---|---|
| **Name** | `citation_support_judgement` |
| **Default weight** | 0.08 |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level |

**What it measures**: Assigns an explicit **full / partial / none** support label to each matched claim based on lexical and numeric overlap between GT support snippets and prediction evidence texts.

- **Full support**: both lexical overlap ≥ 0.4 and numeric overlap ≥ 0.5.
- **Partial support**: either overlap passes its threshold.
- **None**: neither passes.

Scores map to: full → 1.0, partial → 0.5, none → 0.0.

**Academic attribution**:
- **ALiiCE** — fine-grained citation evaluation with degree-of-support labels.
- **ALCE** — AIS (Attributed Information Support) classification.

---

### 2.3 CitationFaithfulnessEvaluator

| | |
|---|---|
| **Name** | `citation_faithfulness` |
| **Default weight** | 0.0 (reported only) |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level |

**What it measures**: Detects **post-rationalization** — cases where a system produces a correct value but cites evidence that doesn't actually support it. This is an anti-gaming metric.

The evaluator builds "anchors" from the claim (material name tokens, numeric values) and checks whether the prediction's evidence texts actually contain those anchors. A claim is flagged as "likely post-rationalized" if the factual value is correct but evidence anchor coverage is low.

**Indirect-aware behavior**: For claims with `per_parameter_evidence`, anchors are built from **component parameter values** (e.g., `0.0904` for thermal conductivity, `818` for density) rather than the final computed value (e.g., `305.24` for thermal effusivity). This is critical because the computed value never appears verbatim in any citation — only the source parameters do. Uses `number_variants_from_string()` to generate all numeric representations of each parameter value. Observability traces include `is_indirect` and `anchor_type` fields.

**Academic attribution**:
- Original research on citation faithfulness vs. citation correctness in LLM outputs.
- **GopherCite** — distinguishing genuine citation grounding from decorative citations.

---

### 2.4 NoAnswerEvaluator

| | |
|---|---|
| **Name** | `no_answer` |
| **Default weight** | 0.0 (reported only) |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | File-level |

**What it measures**: Correct abstention behavior. When human labelers could not find an answer (empty GT file, or all sources flagged `answer_exists: false`), a good system should explicitly abstain rather than hallucinate.

Score = 1.0 if `gt_no_answer == pred_no_answer`, else 0.0.

The evaluator checks for explicit abstention signals in predictions: `no_answer`, `abstain`, or `insufficient_evidence` flags, or an empty claims list.

**Academic attribution**:
- **GroUSE** — abstention and calibration evaluation for QA systems.
- **GopherCite** — "I don't know" evaluation for supported question answering.

---

### 2.5 ConflictingEvidenceEvaluator

| | |
|---|---|
| **Name** | `conflicting_evidence` |
| **Default weight** | 0.0 (reported only) |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level (cluster) |

**What it measures**: Whether systems correctly preserve **condition-dependent variation** — multiple distinct values for the same material+property under different measurement conditions. A typical example: the kinetic friction of 6061 Aluminum varies depending on the counterface material and test method.

The evaluator groups GT claims by `(material_family, material_class, haptic_property, units)`, identifies clusters with ≥ 2 distinct condition-value signatures, and measures the fraction of those GT claims that were independently matched.

**Academic attribution**:
- **Ev2R** — evidence handling in claims with conflicting sources.
- General structured extraction literature on condition-dependent multiplicity.

---

### 2.6 CostEfficiencyEvaluator

| | |
|---|---|
| **Name** | `cost_efficiency` |
| **Default weight** | 0.0 (reported only) |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | File-level (resource) |

**What it measures**: Whether the system stayed within a dollar cost budget. Looks for fields like `total_cost_usd`, `cost_usd`, `cost` in the prediction's system metrics.

Score = `max(0, 1 − value/budget)` with default budget of $0.10.

**Academic attribution**:
- Agentic evaluation literature on resource-aware benchmarking.
- Efficiency-aware evaluation norms in ML systems benchmarks.

---

### 2.7 LatencyEfficiencyEvaluator

| | |
|---|---|
| **Name** | `latency_efficiency` |
| **Default weight** | 0.0 (reported only) |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | File-level (resource) |

**What it measures**: Whether the system completed within a latency budget. Looks for `latency_seconds`, `wall_time_seconds`, `runtime_seconds` in system metrics.

Default budget: 30 seconds.

**Academic attribution**:
- Same as CostEfficiencyEvaluator — resource-aware agentic benchmarking.

---

### 2.8 ToolUsageEfficiencyEvaluator

| | |
|---|---|
| **Name** | `tool_usage_efficiency` |
| **Default weight** | 0.0 (reported only) |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | File-level (resource) |

**What it measures**: Composite tool efficiency score across three dimensions:
- **Tool calls**: budget = 10
- **Model calls**: budget = 8
- **Pages fetched**: budget = 20

Each dimension is scored independently. The final score is the mean of all available dimensions.

**Academic attribution**:
- Agentic evaluation frameworks measuring operational efficiency.
- Deep research agent benchmarking (tool-call counting, context window management).

---

## 3. Evidence-Gated Evaluators

---

### 3.1 StrictGroundednessEvaluator

| | |
|---|---|
| **Name** | `strict_groundedness` |
| **Default weight** | 0.10 |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level |
| **Location** | `evaluators/strict_groundedness.py` |

**What it measures**: FEVER-style all-or-nothing evidence-gated correctness. A claim passes only if **all** dimensions are simultaneously correct:
- Value score ≥ 0.999 (near-exact match)
- Units match
- At least one shared citation URL

Conditions are **not** gated — they are human-labeler outputs of structured extraction and are not treated as fact-checked GT.

The final score is the pass rate: `passed / total`.

**Academic attribution**:
- **FEVER** (Thorne et al., 2018) — Fact Extraction and VERification dataset using strict evidence-gated verdicts.
- **KILT** — evidence-gated knowledge retrieval evaluation.

---

### 3.2 OpenWebApproxEvidenceEvaluator

| | |
|---|---|
| **Name** | `open_web_approx_evidence` |
| **Default weight** | 0.0 (reported only) |
| **Regimes** | OPEN_WEB only |
| **Operation** | Claim-level |
| **Location** | `evaluators/openweb_evidence.py` |

**What it measures**: Approximate evidence quality in the OPEN_WEB scenario, where the system may cite completely different sources than the GT. Instead of requiring exact URL matches, it checks:
- **Domain match**: Is the prediction's source from the same domain as any GT source?
- **Value support**: Does the snippet text contain the GT numeric value?
- **Condition token ratio**: What fraction of the GT condition tokens appear in the snippet?

```
score = max(domain_match, 0.55 × value_match + 0.45 × token_ratio)
```

**Academic attribution**:
- **GopherCite** (Menick et al., 2022) — evaluating evidence quality when exact source matching is impossible.
- General open-web retrieval evaluation with approximate evidence scoring.

---

## 4. Coverage & Diagnostic Evaluators

---

### 4.1 ValueListCoverageEvaluator

| | |
|---|---|
| **Name** | `value_list_coverage` |
| **Default weight** | 0.15 |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | File-level |
| **Location** | `evaluators/completeness.py` |

**What it measures**: Overall numeric value inventory coverage at the file level, independent of record linkage. Flattens all scalar values (and min/max endpoints of ranges) from both GT and prediction, then performs greedy numeric matching.

**Tolerance modes**:
- **Direct extraction**: `rel_tol = 0.1%` (default) — for values extracted verbatim from sources
- **Indirect extraction**: `indirect_rel_tol = 5%` — for formula-computed values (e.g., thermal effusivity = √(k·ρ·cp)) where rounding of component parameters produces inherent variation

The evaluator auto-detects indirect mode by checking the prediction file for formula indicators in measurement conditions or `indirect_parameter_contributions` in grounding records.

Reports precision, recall, and F1 over the value inventory.

**Academic attribution**:
- **MeasEval** (SemEval-2021) — measurement extraction evaluation with value-level precision/recall.
- Inspired by value-list summaries in the GT `property_stats` and `value_list` sections.

---

### 4.2 StatementSupportEvaluator

| | |
|---|---|
| **Name** | `statement_support` |
| **Default weight** | 0.0 (reported only) |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level (global) |
| **Location** | `evaluators/statement_support.py` |

**What it measures**: ALCE/AIS-style atomic statement support. For each prediction claim, checks whether its own citation evidence actually supports the claim content (value presence + condition token overlap). Then compares the set of "citation-supported prediction claims" against the GT claim set.

Returns F1 between supported predictions and GT claims:

```
supported_pred = {claims whose evidence contains their value + ≥50% condition tokens}
tp = |matched_gt ∩ supported_pred|
precision = tp / (tp + fp)
recall = tp / (tp + fn)
F1 = 2 × precision × recall / (precision + recall)
```

**Academic attribution**:
- **ALCE** (Gao et al., 2023) — Automatic LLM Citation Evaluation using AIS.
- **FActScore** — atomic fact decomposition with per-fact support checking.
- **AIS** (Rashkin et al., 2022) — Attributable to Identified Sources framework.

---

## 5. Verification Evaluators

Located in `evaluators/llm_evaluators.py`.

---

### 5.1 VerifiedClaimsF1Evaluator

| | |
|---|---|
| **Name** | `verified_claims_f1` |
| **Default weight** | 0.15 |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level (cross-evaluator) |
| **Location** | `evaluators/llm_evaluators.py` |

**What it measures**: F1 score combining factual correctness and novel claim verification. A prediction claim is considered "verified" if either:
1. It matched a GT claim AND its `factual_value` score > threshold (human-verified via GT), OR
2. It is `pred_only` AND the `verified_novel_claim_rate` judge accepted it (novel but evidence-supported)

Claims that matched GT but had `factual_value = 0.0` are treated as false positives.

**Academic attribution**:
- Combines FActScore precision with novel claim verification for a unified F1.
- **RAGChecker** — factual precision/recall over verified claims.

---

### 5.2 UnverifiedClaimRateEvaluator

| | |
|---|---|
| **Name** | `unverified_claim_rate` |
| **Default weight** | 0.0 (reported only) |
| **Regimes** | FIXED_DOCS · URL_ONLY · OPEN_WEB |
| **Operation** | Claim-level (cross-evaluator) |
| **Location** | `evaluators/llm_evaluators.py` |

**What it measures**: The fraction of prediction claims that could NOT be verified — i.e., they neither matched a GT claim with correct value, nor passed the novel claim judge. Reports `score = 1.0 - unverified_rate`.

**Academic attribution**:
- Complement of `verified_claims_f1`. Provides a single hallucination-aware calibration metric.

---

## Weight Summary

Weights are defined in `benchmark.py`. The canonical source of truth is always the code. The framework provides **four weight profiles** for different evaluation scenarios:

### DEFAULT_WEIGHTS (Closed-Docs, Direct Properties)

| Evaluator | Weight | Category |
|---|---|---|
| `factual_value` | 0.25 | Core |
| `grounding_localization` | 0.15 | Core |
| `value_list_coverage` | 0.15 | Coverage |
| `verified_claims_f1` | 0.15 | Verification |
| `semantic_conditions` | 0.12 | LLM-Judge |
| `citation_source_recall` | 0.10 | Core |
| `citation_support_judgement` | 0.08 | Advanced |
| `strict_groundedness` | 0.05 | Evidence-Gated |
| `no_answer` | 0.04 | Advanced |
| `claim_completeness` | 0.03 | Core |
| `citation_source_precision` | 0.02 | Core |
| `unsupported_claim_rate` | 0.01 | Core |
| `conditions_and_specs` | 0.0 (superseded by `semantic_conditions`) | Core |
| `hierarchy` | 0.0 (disabled) | Core |
| `citation_snippet_support` | 0.0 (superseded by `citation_support_judgement`) | Core |
| `source_equivalence` | 0.0 | Advanced |
| `citation_faithfulness` | 0.0 | Advanced |
| `conflicting_evidence` | 0.0 | Advanced |
| `cost_efficiency` | 0.0 | Efficiency |
| `latency_efficiency` | 0.0 | Efficiency |
| `tool_usage_efficiency` | 0.0 | Efficiency |
| `statement_support` | 0.0 | Evidence-Gated |
| `open_web_approx_evidence` | 0.0 | Open-Web |
| `verified_novel_claim_rate` | 0.0 | Verification |
| `unverified_claim_rate` | 0.0 | Verification |

### INDIRECT_WEIGHTS (Closed-Docs, Derived/Formula Properties)

Same as DEFAULT_WEIGHTS except:
- `strict_groundedness`: **0.0** (too strict for formula-computed values)
- `citation_support_judgement`: **0.13** (+0.05 from strict_groundedness)

### OPEN_WEB_GUIDED_WEIGHTS (Track B — Guided DR Agents)

| Evaluator | Weight | Rationale |
|---|---|---|
| `verified_novel_claim_rate` | **0.40** | Primary quality signal — judge verifies each claim |
| `factual_value` | 0.25 | Fair: guided URLs produce similar values |
| `value_list_coverage` | 0.15 | Fair: guided URLs → expect same values |
| `claim_completeness` | 0.05 | Did the agent find enough claims? |
| `no_answer` | 0.05 | Correct abstention behavior |
| `cost_efficiency` | 0.05 | Track DR API costs |
| `latency_efficiency` | 0.05 | Track DR API latency |
| All others | 0.0 | Unfair — different document universe |

### OPEN_WEB_UNGUIDED_WEIGHTS (Track B — Autonomous DR Agents)

| Evaluator | Weight | Rationale |
|---|---|---|
| `verified_novel_claim_rate` | **0.90** | 100% of quality signal from judge |
| `cost_efficiency` | 0.05 | Track DR API costs |
| `latency_efficiency` | 0.05 | Track DR API latency |
| All others | 0.0 | Unfair — completely different source universe |

> [!IMPORTANT]
> `OPEN_WEB_WEIGHTS` is a backward-compatible alias that defaults to `OPEN_WEB_GUIDED_WEIGHTS`.

See [regimes.md](regimes.md) for detailed guidance on when to use each weight profile.

### Aggregate Scoring

The final aggregate is a **weighted geometric mean** of evaluator scores with weight > 0:

```
aggregate = exp(Σ(w_i × log(max(s_i, ε_floor))) / Σw_i)
```

The geometric mean penalizes poor performance on any single dimension multiplicatively, rewarding balanced performance. This is standard practice in multi-dimensional evaluation harnesses (e.g., SuperGLUE, BIG-bench).

A **score floor** (`eps_floor = 0.01`) prevents any single zero-score evaluator from collapsing the entire aggregate to near-zero, preserving discriminability between methods while still penalizing zero-scoring dimensions.

---

## Adding a Custom Evaluator

```python
from hapticnet_eval.evaluators.core import BaseEvaluator, ClaimIndex
from hapticnet_eval.schemas import EvaluatorScore, MatchResult

class MyCustomEvaluator(BaseEvaluator):
    name = "my_custom"

    def evaluate(self, gt_index, pred_index, matches, context=None):
        # your logic here
        return EvaluatorScore(name=self.name, score=0.95, details={})

# Register it
from hapticnet_eval.benchmark import BenchmarkRunner
runner = BenchmarkRunner(
    evaluators=[..., MyCustomEvaluator()],
    weights={..., "my_custom": 0.10},
)
```
