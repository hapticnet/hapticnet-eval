# Benchmark Regimes (Tracks)

The HapticNetEval Evaluator Framework supports three distinct **benchmark tracks**, each modeling a different level of system autonomy and source access. This document explains the design rationale, what each track measures, how it affects scoring, and how to configure weight profiles.

---

## Overview

| Track | Code name | System access | Key question |
|---|---|---|---|
| **Closed Documents** | `closed_docs` | Pre-supplied source texts | *Given the exact pages, can the system extract the right facts?* |
| **Guided Open-Web** | `open_web` (guided) | Source URL hints + unrestricted search | *Given URL pointers, can the agent find, extract, and cite facts?* |
| **Unguided Open-Web** | `open_web` (unguided) | Unrestricted search — no hints | *Can the agent independently find, extract, and verify facts?* |

> [!NOTE]
> The underlying `Regime` enum still defines `FIXED_DOCS`, `URL_ONLY`, and `OPEN_WEB` for backward compatibility. The **production pipeline** (Track B via `run_track_b_benchmark.py` and `DRRunner`) uses the **guided/unguided split** — a parameter on top of the `open_web` regime adapter — rather than `url_only` as a separate regime.

Each regime is implemented as a `RegimeAdapter` subclass with a `preprocess(gt_obj, pred_obj) → (gt_obj, pred_obj, metadata)` method that can transform inputs before evaluation.

---

## 1. CLOSED_DOCS (`closed_docs`)

### Description

In the **Closed Documents** track, the system under test receives exact source documents (full page text, or pre-scraped content) alongside the query. The challenge is purely about **extraction quality**: can the system identify the correct values, conditions, and citations from known, pre-loaded content?

### What it models

This track isolates the **information extraction** component of a deep-research pipeline, removing retrieval entirely. It is the most controlled setting and produces the most interpretable scores.

### Practical scenarios

- Testing extraction-only models (no retrieval component)
- Evaluating structured output extraction from known research papers
- A/B testing prompt engineering for value extraction
- Calibrating evaluator weights against a known-good source universe

### Evaluator behavior

All evaluators are active. Source matching is **strict** — the prediction must cite the exact source URLs present in the GT since the documents are provided.

### Weight profile

Uses `DEFAULT_WEIGHTS` (or `INDIRECT_WEIGHTS` for formula-derived properties).

| Dimension | Relevance |
|---|---|
| Factual value accuracy | **Primary** — the system has the text, so values should be exact |
| Conditions (semantic) | **Primary** — full context is available |
| Citation source recall/precision | **High** — exact URLs are expected |
| Grounding localization | **High** — character offsets should align |
| Value list coverage | **High** — all values should be recoverable |
| Open-web evidence | Not applicable |
| Calibrated judge | Not applicable (covered by factual_value + strict_groundedness) |

### Academic grounding

- **KILT** (Petroni et al., 2021) — knowledge-intensive language task benchmark with pre-defined evidence pages.
- **SQuAD** (Rajpurkar et al., 2016) — reading comprehension over given passages.
- **ChemX** — chemical property extraction from pre-selected papers.

---

## 2. GUIDED OPEN-WEB (`open_web`, `guided=True`)

### Description

In the **Guided Open-Web** track, the system receives the query along with **GT source URLs as guidance hints**, but must independently search the web, fetch content, and extract structured facts. The source URLs are embedded in the query as suggested starting points, but the system is free to search beyond them.

This is the **primary Track B variant** in the HapticNetEval benchmark.

### What it models

This track evaluates the agent's ability to do **guided deep research**: given rough pointers to relevant sources, can it navigate those pages, find the right values, and optionally discover additional information? The guidance URLs anchor the evaluation, making `factual_value` and `value_list_coverage` fair comparisons.

### Practical scenarios

- Evaluating deep-research agents with URL guidance (e.g., Gemini DR, OpenAI o3-deep-research)
- Benchmarking document parsing + extraction when the system must fetch its own pages
- Testing whether guided retrieval produces comparable results to closed-doc extraction

### Query construction

The `DRRunner` builds guided queries using the `GUIDED_QUERY` template:

```python
GUIDED_QUERY = (
    "What is the {property} of {material}? "
    "Provide specific numeric values with units, measurement conditions, "
    "and cite sources with verbatim quotes.\n\n"
    "The following source URLs may contain relevant information:\n"
    "{urls}\n\n"
    "You may also search for additional sources."
)
```

### Evaluator behavior

Because the system is **guided toward GT sources** but uses its own retrieval:

- **`factual_value`** (0.25): **Active** — guided URLs produce similar values.
- **`value_list_coverage`** (0.15): **Active** — expect reasonable value recall.
- **`verified_novel_claim_rate`** (0.40): **Primary quality signal** — the calibrated LLM judge verifies each extracted claim against its own cited evidence.
- **`claim_completeness`** (0.05): Active.
- **`no_answer`** (0.05): Active.
- **`cost_efficiency`** (0.05) / **`latency_efficiency`** (0.05): Track DR API resource usage.
- Citation-level evaluators (`citation_source_recall`, `grounding_localization`, etc.): **Zeroed** — the system retrieves from different document parses.

### Weight profile: `OPEN_WEB_GUIDED_WEIGHTS`

```python
OPEN_WEB_GUIDED_WEIGHTS = {
    "factual_value": 0.25,
    "value_list_coverage": 0.15,
    "verified_novel_claim_rate": 0.40,   # primary quality signal
    "claim_completeness": 0.05,
    "no_answer": 0.05,
    "cost_efficiency": 0.05,
    "latency_efficiency": 0.05,
    # Everything else: 0.0
}
```

### Academic grounding

- **GopherCite** (Menick et al., 2022) — open-domain supported QA where systems must find and cite evidence from the web.
- **ALCE** (Gao et al., 2023) — citation evaluation where systems process cited documents.
- **GroUSE** — holistic evaluation of grounded language generation including source diversity.

---

## 3. UNGUIDED OPEN-WEB (`open_web`, `guided=False`)

### Description

In the **Unguided Open-Web** track, the system receives **only the query** (material + property) and must independently search the web, identify relevant sources, fetch content, extract values, and ground them with evidence. This is the most challenging and most realistic setting — and the least constrained comparison against GT.

### What it models

This track evaluates **full autonomous research agent capability**: information need formulation, search strategy, source selection, parsing, extraction, normalization, and citation — all without any human-provided URL hints.

### Practical scenarios

- Evaluating autonomous research agents in a zero-guidance setting
- Leaderboard competitions on factual grounded QA
- Testing whether an agent can replicate expert-level curation from scratch

### Query construction

```python
UNGUIDED_QUERY = (
    "What is the {property} of {material}? "
    "Provide specific numeric values with units, measurement conditions, "
    "and cite sources with verbatim quotes."
)
```

### Evaluator behavior

In the unguided setting, the system explores a **different source universe** than GT. Consequently:

- **`verified_novel_claim_rate`** (0.90): **Dominant quality signal** — since we can't fairly compare factual values or source URLs, the calibrated LLM judge individually verifies each extracted claim against its own cited evidence. This is the only way to assess correctness without assuming source overlap.
- **`cost_efficiency`** (0.05) / **`latency_efficiency`** (0.05): Track DR API resource usage.
- All GT-dependent evaluators (`factual_value`, `value_list_coverage`, `citation_source_recall`, `grounding_localization`, etc.): **Zeroed** — unfair when source universe is unknown.

### Weight profile: `OPEN_WEB_UNGUIDED_WEIGHTS`

```python
OPEN_WEB_UNGUIDED_WEIGHTS = {
    "verified_novel_claim_rate": 0.90,   # 100% of quality signal
    "cost_efficiency": 0.05,
    "latency_efficiency": 0.05,
    # Everything else: 0.0
}
```

### Academic grounding

- **GopherCite** (Menick et al., 2022) — evaluating evidence quality when exact source matching is impossible.
- **FEVER** (Thorne et al., 2018) — evidence retrieval and verification from open document collections.
- **Ev2R** — evidence handling in open-domain fact verification.

---

## The Calibrated Novel Claim Judge

The `CalibratedClaimJudgeEvaluator` (evaluator name: `verified_novel_claim_rate`) is the cornerstone of open-web scoring. It uses a carefully calibrated LLM-as-a-judge prompt (variant **E5_E1_E2_E3_E4**) to assess whether each prediction claim is genuinely supported by its own cited evidence.

Key features:

| Feature | Description |
|---|---|
| **Calibrated prompt** | Developed through ablation study over 176 calibration items, achieving 0.735 AUROC |
| **Evidence bundles** | Structured `[EVIDENCE N]` blocks with citation snippets + document windows |
| **Indirect claim support** | Per-parameter verification for derived properties (e.g., thermal effusivity) |
| **Param-level pre-check** | Skips LLM call when pred_only claim parameters match a GT claim (matching failure, not novelty) |
| **Evidence gating** | Ungrounded claims (no `source_uid`) are scored 0.0 without an LLM call |
| **Resource tracking** | Per-call token/cost/latency tracking via `LLMResourceTracker` |
| **Judge model** | `gemini-3.1-flash-lite` with `thinking_level=low` (1024 token budget) |

See [evaluators.md](evaluators.md) for full documentation of the calibrated judge.

---

## Track Selection Guide

```
What kind of system are you evaluating?
├── Extraction-only model (no retrieval)
│   └── CLOSED_DOCS
│       Weight profile: DEFAULT_WEIGHTS (or INDIRECT_WEIGHTS for derived props)
│       Best for: prompt engineering, extraction quality, component testing
│
└── Deep-research agent (has retrieval/search)
    ├── Are you providing source URL hints?
    │   ├── Yes → GUIDED OPEN-WEB
    │   │         Weight profile: OPEN_WEB_GUIDED_WEIGHTS
    │   │         Best for: guided agent evaluation, parsing + retrieval testing
    │   │
    │   └── No → UNGUIDED OPEN-WEB
    │             Weight profile: OPEN_WEB_UNGUIDED_WEIGHTS
    │             Best for: full autonomous agent leaderboard
```

---

## Configuring Tracks

### In code — Closed-Docs (Track A)

```python
from hapticnet_eval.regimes.base import EvaluationTask
from hapticnet_eval.benchmark import BenchmarkRunner, DEFAULT_WEIGHTS

task = EvaluationTask(
    task_id="my_task",
    regime="closed_docs",
    gt_path="path/to/gt.json",
    pred_path="path/to/pred.json",
)
runner = BenchmarkRunner(weights=DEFAULT_WEIGHTS)
report = runner.evaluate_task(task)
```

### In code — Open-Web via DRRunner (Track B)

```python
from hapticnet_eval.dr_runner import DRRunner
from hapticnet_eval.benchmark import BenchmarkRunner, OPEN_WEB_GUIDED_WEIGHTS

# Guided variant
dr = DRRunner(provider="gemini", guided=True, timeout=600)
result = dr.run(
    query_material="Memory Foam",
    query_property="thermal_conductivity",
    gt_source_urls=["https://..."],
    output_dir="./runs/memory_foam_tc/gemini_guided",
)

# Evaluate with guided weights
runner = BenchmarkRunner(weights=OPEN_WEB_GUIDED_WEIGHTS)
task = EvaluationTask(
    task_id="memory_foam_tc",
    regime="open_web",
    gt_path="path/to/gt.json",
    pred_path="./runs/.../db_entry.json",
)
report = runner.evaluate_task(task)
```

### Via CLI — Track B Benchmark

```bash
# Full benchmark (24 queries × 4 providers × 2 variants = 192 runs)
python run_track_b_benchmark.py --gt_dir /path/to/gt_files

# Guided only, specific providers
python run_track_b_benchmark.py --gt_dir /path/to/gt_files \
    --providers gemini tavily --variants guided

# Smoke test (1 query)
python run_track_b_benchmark.py --gt_dir /path/to/gt_files \
    --providers perplexity --limit 1
```

---

## Weight Profiles Summary

Four weight profiles are defined in `benchmark.py`:

| Profile | Track | Primary signal | Secondary signals |
|---|---|---|---|
| `DEFAULT_WEIGHTS` | Closed-Docs (direct) | `factual_value` (0.25) | `grounding_localization` (0.15), `value_list_coverage` (0.15), `verified_claims_f1` (0.15), `semantic_conditions` (0.12), `citation_source_recall` (0.10) |
| `INDIRECT_WEIGHTS` | Closed-Docs (indirect/derived) | `factual_value` (0.25) | Same as above, but `strict_groundedness` → 0.0 (too strict for derived values), `citation_support_judgement` → 0.13 |
| `OPEN_WEB_GUIDED_WEIGHTS` | Guided Open-Web | `verified_novel_claim_rate` (0.40) | `factual_value` (0.25), `value_list_coverage` (0.15), `cost/latency_efficiency` (0.05 each) |
| `OPEN_WEB_UNGUIDED_WEIGHTS` | Unguided Open-Web | `verified_novel_claim_rate` (0.90) | `cost/latency_efficiency` (0.05 each) |

> [!IMPORTANT]
> `OPEN_WEB_WEIGHTS` is a backward-compatible alias that defaults to `OPEN_WEB_GUIDED_WEIGHTS`.

---

## Deep Research Providers

The `DRRunner` supports four DR API providers, each tested in both guided and unguided modes:

| Provider | Model | API style | Typical cost/query | Timeout |
|---|---|---|---|---|
| **Gemini** | `deep-research-preview-04-2026` | Interactions API (async poll) | ~$0.15 | 300s |
| **OpenAI** | `o3-deep-research` | Responses API (async poll) | ~$3.50 | 300s |
| **Tavily** | `pro` | `/research` endpoint | ~$0.30 | 300s |
| **Perplexity** | `sonar-deep-research` | Chat completions | ~$0.10 | 300s |

All providers produce a prose research report which is then structured into a `db_entry.json` by the `ReportAdapter` using constrained LLM extraction.

---

## Evidence Fetcher Integration

For open-web tracks, the `EvidenceFetcher` ensures that source documents cited by the DR agent are available for the calibrated judge to verify claims against. The pipeline:

1. **Scans** prediction claims for source URLs → derives UIDs via MD5 hash
2. **Checks** production SourceIndex for pre-existing content
3. **Checks** local `~/.hapticnet_source_cache` for previously fetched content
4. **Fetches** missing sources via Tavily Extract API (basic → advanced fallback)
5. **Writes** fetched content to the local cache as `content.md`

The fetcher produces full observability via `FetchRecord` entries — see [evaluator_observability_guide.md](evaluator_observability_guide.md) for details.

---

## Legacy: URL_ONLY Regime

The `url_only` regime adapter and its `Regime.URL_ONLY` enum value are preserved for backward compatibility but are not actively used in the production Track B pipeline. In practice, the **Guided Open-Web** track subsumes the URL-only use case: the DR agent receives URL hints and must fetch and parse them, but is also free to search for additional sources.

If you specifically want the strict URL-only behavior (system must fetch given URLs but cannot search freely), you can still use:

```python
task = EvaluationTask(regime="url_only", ...)
```

But note that no separate weight profile is provided — use `DEFAULT_WEIGHTS` as a starting point and adjust citation recall/precision expectations based on your parsing pipeline.

---

## Extending Regimes

To add a new regime, subclass `RegimeAdapter`:

```python
from hapticnet_eval.regimes.base import RegimeAdapter

class MyCustomRegime(RegimeAdapter):
    name = "my_regime"

    def preprocess(self, gt_obj, pred_obj):
        # Custom preprocessing logic
        # e.g., strip certain fields, inject metadata, validate format
        metadata = {"custom_field": "value"}
        return gt_obj, pred_obj, metadata
```

Then register it in `benchmark.py`:

```python
_REGIMES["my_regime"] = MyCustomRegime()
```

---

## Track Comparison Table

| Aspect | CLOSED_DOCS | GUIDED OPEN-WEB | UNGUIDED OPEN-WEB |
|---|---|---|---|
| Input to system | Query + full texts | Query + URL hints | Query only |
| Retrieval tested | ✗ | ✓ (guided) | ✓ (autonomous) |
| Parsing tested | ✗ | ✓ | ✓ |
| Extraction tested | ✓ | ✓ | ✓ |
| Source URL matching | Strict | Not scored | Not scored |
| Primary quality signal | `factual_value` | `verified_novel_claim_rate` | `verified_novel_claim_rate` |
| Expected factual recall | Highest | High | Variable |
| Grounding offsets | Exact | Different documents | Different documents |
| Resource tracking | ✗ | ✓ (cost + latency) | ✓ (cost + latency) |
| Difficulty | ★★☆ | ★★★★ | ★★★★★ |
| Use case | Component testing | Guided agent evaluation | Agent leaderboard |
| Weight profile | `DEFAULT_WEIGHTS` | `OPEN_WEB_GUIDED_WEIGHTS` | `OPEN_WEB_UNGUIDED_WEIGHTS` |
