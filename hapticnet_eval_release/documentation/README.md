# HapticNetEval Evaluator Framework

A unified, literature-grounded evaluation harness for **grounded, normalized, structured knowledge extraction**. This framework benchmarks deep-research agents, LLM-based fact-extraction systems, and RAG pipelines against human-curated ground-truth files from the HapticNet dataset.

---

## Overview

The HapticNetEval Evaluator Framework consolidates six prior prototype implementations into a single, modular Python package. It is designed around two core ideas:

1. **Claims-first evaluation** — Ground-truth files and predictions are decomposed into atomic *canonical claims*, each carrying a numeric value, measurement conditions, material hierarchy, units, and evidence provenance. Claims are matched via optimal assignment before scoring.
2. **Multi-dimensional scoring** — Instead of a single metric, the framework runs 25 independent evaluators spanning factual accuracy, citation quality, grounding localization, coverage, hallucination detection, LLM-judge verification, efficiency, and abstention handling. A weighted geometric mean produces the aggregate score, with four weight profiles for different evaluation tracks (closed-docs, guided open-web, unguided open-web).

Every evaluator is traceable to one or more published academic works (FActScore, ALCE, FEVER, KILT, ChemX, MeasEval, Ev2R, ALiiCE, LegalBench-RAG, GopherCite, GroUSE, RAGChecker — see [evaluators.md](evaluators.md)).

---

## Installation

```bash
# From the package root
pip install -e .
```

### Dependencies

| Package | Min version | Purpose |
|---|---|---|
| `pydantic` | ≥ 2.7 | Schema validation & serialization |
| `rapidfuzz` | ≥ 3.9 | Fuzzy string matching for claim similarity |
| `numpy` | ≥ 1.26 | Cost matrix for Hungarian matching |
| `scipy` | ≥ 1.13 | `linear_sum_assignment` for optimal matching |
| `pandas` | ≥ 2.2 | Result aggregation (optional) |

---

## Quick Start

```python
from hapticnet_eval.benchmark import BenchmarkRunner
from hapticnet_eval.regimes.base import EvaluationTask

runner = BenchmarkRunner()

task = EvaluationTask(
    task_id="6061_al_kinetic_friction",
    regime="closed_docs",                    # or "url_only" / "open_web"
    gt_path="path/to/gt_file.json",
    pred_path="path/to/prediction_file.json",
)

report = runner.evaluate_task(task)
print(f"Aggregate: {report.aggregate_score:.4f}")
for s in report.scores:
    print(f"  {s.name}: {s.score:.4f}")
```

---

## Package Layout

```
hapticneteval_evaluator_framework/
├── pyproject.toml
├── documentation/
│   ├── README.md              ← You are here
│   ├── evaluators.md          ← In-depth evaluator reference
│   └── regimes.md             ← Regime / track configuration guide
├── hapticnet_eval/
│   ├── __init__.py
│   ├── schemas.py             ← Pydantic models (GTFile, CanonicalClaim, etc.)
│   ├── canonicalize.py        ← GT → canonical claims conversion pipeline
│   ├── benchmark.py           ← BenchmarkRunner, DEFAULT_EVALUATORS, DEFAULT_WEIGHTS
│   ├── evaluators/
│   │   ├── base.py            ← BaseEvaluator abstract class
│   │   ├── core.py            ← 8 core evaluators
│   │   ├── advanced.py        ← 8 advanced evaluators
│   │   ├── matching.py        ← Hungarian-assignment claim matcher
│   │   ├── composite.py       ← Weighted geometric mean aggregator
│   │   ├── strict_groundedness.py
│   │   ├── statement_support.py
│   │   ├── completeness.py    ← ValueListCoverageEvaluator
│   │   └── openweb_evidence.py
│   ├── regimes/
│   │   ├── base.py            ← Regime enum + RegimeAdapter base
│   │   ├── closed_docs.py
│   │   ├── url_only.py
│   │   └── open_web.py
│   ├── utils/
│   │   ├── normalization.py   ← Text, unit, numeric normalization
│   │   ├── evidence.py        ← Citation support & faithfulness utilities
│   │   ├── numeric.py         ← Legacy numeric utils (greedy matching, etc.)
│   │   └── io.py              ← JSON loader
│   └── cli/
│       └── main.py            ← CLI entry point
├── examples/
│   └── run_example.py
└── tests/
```

---

## Core Concepts

### Canonical Claims

Every value–condition–mapping entry in a GT or prediction file is converted into a `CanonicalClaim` dataclass. This frozen, hashable object contains:

- **Identity**: `claim_id`, `entry_index`
- **Material hierarchy**: `material_family`, `material_class`, `material_subclass`
- **Property**: `haptic_property`
- **Structured fields**: `measurement_conditions`, `material_specifications`, `object_class`
- **Numeric payload**: `value_type` (scalar / range / series), `normalized_value`, `range_min`, `range_max`, `data_points`
- **Provenance**: `grounded` flag, `provenance` (list of `CanonicalFieldEvidence` with source URLs, snippets, character-level index ranges, confidence), `support_snippets`

### Claim Matching

Before evaluators run, GT and prediction claims are aligned using **Hungarian assignment** (`scipy.optimize.linear_sum_assignment`) on a multi-dimensional similarity matrix. The similarity function is a weighted sum of:

| Component | Weight | Method |
|---|---|---|
| Material family | 0.05 | Fuzzy string ratio |
| Material class | 0.08 | Fuzzy string ratio |
| Material subclass | 0.12 | Fuzzy string ratio |
| Haptic property | 0.15 | Fuzzy string ratio |
| Conditions | 0.20 | Set-F1 |
| Specifications | 0.07 | Set-F1 |
| Object class | 0.05 | Set-F1 |
| Units | 0.05 | Exact match |
| Value type | 0.03 | Exact match |
| Value | 0.07 | Type-aware closeness |
| Origin match | 0.05 | Structural origin (scalar/min/max/mean) |
| Source overlap | 0.08 | Source URL Jaccard |

Pairs below `min_similarity=0.45` are rejected. Unmatched GT claims become `gt_only`; unmatched predictions become `pred_only`.

### Aggregate Scoring

The final aggregate is a **weighted geometric mean** of the evaluator scores whose weight > 0 in `DEFAULT_WEIGHTS`. This penalizes any single zero-score dimension multiplicatively, rewarding balanced performance.

---

## Configuration

### Evaluator Weights

Default weights are defined in `benchmark.py` → `DEFAULT_WEIGHTS`. The 9 primary evaluators share a total weight of 1.0. The 12 advanced/diagnostic evaluators default to weight 0.0 (reported but not aggregated). Override by passing a custom `weights` dict to `BenchmarkRunner`:

```python
runner = BenchmarkRunner(weights={"factual_value": 0.30, "conditions_and_specs": 0.20, ...})
```

### Custom Evaluator Sets

```python
from hapticnet_eval.evaluators.core import FactualValueEvaluator, ConditionsEvaluator
runner = BenchmarkRunner(evaluators=[FactualValueEvaluator(), ConditionsEvaluator()])
```

---

## Ground-Truth File Format

GT files follow the HapticNet schema:

```json
{
  "material_family": "Metals",
  "material_class": "Aluminum",
  "haptic_property": "Kinetic friction coefficient",
  "value_condition_mapping": [ ... ],
  "citations": [ ... ],
  "sources": [ ... ],
  "labeler_note": { ... },
  "property_stats": { ... },
  "value_list": { ... }
}
```

Each entry in `value_condition_mapping` contains measurement conditions, material specifications, a value payload (`scalar`, `range`, or `series`), grounding information (`is_grounded`, `successful_groundings`, `normalized_value`, `normalized_units`).

Prediction files may follow the same schema **or** a direct `{"claims": [...]}` canonical format.

---

## Further Reading

- **[evaluators.md](evaluators.md)** — Complete reference for all 25 evaluators with scoring formulas and academic attributions.
- **[regimes.md](regimes.md)** — Detailed guide to the three benchmark tracks (closed-docs, guided open-web, unguided open-web) and their weight profiles.
- **[evaluator_observability_guide.md](evaluator_observability_guide.md)** — Deep-dive into the trace structures produced by each evaluator, including LLM judge reasoning, resource tracking, and evidence fetcher observability.
