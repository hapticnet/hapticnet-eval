# HapticNet Evaluation Harness

Benchmark harness for evaluating structured scientific material-property extraction against curated ground-truth records. This is the evaluation toolkit accompanying the **HapticNet** dataset paper (NeurIPS 2026, Datasets and Benchmarks Track).

## Overview

The harness evaluates a predicted `db_entry` JSON file against a ground-truth (GT) JSON file across **8 deterministic evaluators** and **3 optional LLM-based evaluators**:

| Evaluator | Type | What it measures |
|-----------|------|------------------|
| Factual Value | Deterministic | Numerical accuracy of reported property values |
| Value List Coverage | Deterministic | Fraction of GT values recovered by the prediction |
| Claim Completeness | Deterministic | Coverage of canonical claims (material + property + value + conditions) |
| Citation Source Recall | Deterministic | Fraction of GT source URLs cited in the prediction |
| Grounding Localization | Deterministic | Quality of citation snippet ↔ source text alignment |
| Strict Groundedness | Deterministic | Whether predicted values are supported by cited evidence |
| No Answer | Deterministic | Appropriate abstention when no evidence exists |
| Cost Efficiency | Deterministic | Score relative to API cost (if tracked) |
| Semantic Conditions | LLM | Semantic match of measurement conditions (temperature, method, etc.) |
| Verified Claims F1 | LLM | F1 between predicted and GT claims, verified by LLM judge |
| Verified Novel Claim Rate | LLM | Fraction of novel predictions verifiable against source evidence |

The **aggregate score** is a weighted geometric mean over all evaluators, with regime-specific weight profiles (see `benchmark.py`).

## Quick Start

### 1. Install

```bash
# Clone or extract this directory
cd hapticnet_eval_release

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install core dependencies
pip install -r requirements.txt

# Or install as a package (editable mode)
pip install -e .
```

### 2. Run the Example

```bash
python examples/run_example.py
```

This evaluates a sample copper thermal conductivity prediction against the curated GT and prints per-evaluator scores. No API keys are needed for this — all **deterministic evaluators** run locally.

### 3. CLI Usage

After installation, you can use the CLI:

```bash
# Evaluate a single prediction
hapticnet-eval evaluate \
  --gt path/to/gt_file.json \
  --pred path/to/prediction.json \
  --regime closed_docs \
  --out report.json

# Evaluate a batch via manifest
hapticnet-eval evaluate-manifest \
  --manifest manifest.json \
  --out batch_report.json
```

**Manifest format** — a JSON array of task objects:

```json
[
  {
    "task_id": "copper_thermal_conductivity",
    "regime": "closed_docs",
    "gt_path": "path/to/gt.json",
    "pred_path": "path/to/pred.json"
  }
]
```

## Evaluation Regimes

The harness supports three retrieval regimes, each with a tailored weight profile:

| Regime | Description | Key weight |
|--------|-------------|------------|
| `closed_docs` | Agent used only pre-compiled evidence (no web access) | Factual accuracy (high) |
| `url_only` | Agent was given source URLs but no document content | Grounding quality (high) |
| `open_web` | Agent performed unrestricted web search | Novel claim verification (high) |

## Enabling LLM Evaluators (Optional)

Three evaluators require a Google Gemini API key to run. They are **skipped gracefully** when no key is configured.

```bash
# Install the LLM dependency
pip install google-genai>=1.0

# Set your API key
export GOOGLE_API_KEY="your-api-key-here"
```

When a key is present, the harness automatically activates:
- **SemanticConditionsEvaluator** — LLM-based condition matching
- **VerifiedClaimsF1Evaluator** — LLM judge for claim verification
- **CalibratedClaimJudgeEvaluator** — Calibrated novel claim judge (AUROC 0.735)

## File Format

### Ground Truth (`gt_file.json`)

```json
{
  "material_family": "Metals",
  "material_class": "Copper",
  "haptic_property": "Thermal conductivity (k)",
  "value_condition_mapping": [
    {
      "material_subclass": "unspecified",
      "measurement_conditions": [{"label": "temperature", "val": "20°C"}],
      "material_specifications": [{"label": "purity", "val": "99.9%"}],
      "units": "W/(m·K)",
      "value": {"value": 401},
      "is_grounded": true,
      "successful_groundings": [
        {
          "source_url": "https://en.wikipedia.org/wiki/Copper",
          "citation_snippet": "401 W/(m·K)",
          "confidence": 0.95
        }
      ]
    }
  ],
  "sources": [
    {"url": "https://en.wikipedia.org/wiki/Copper", "title": "Copper - Wikipedia"}
  ]
}
```

### Prediction (`pred_file.json`)

Same schema as the GT file. The harness canonicalizes both into comparable claim sets, then runs evaluators on the aligned pairs.

## Directory Structure

```
hapticnet_eval_release/
├── README.md                 # This file
├── requirements.txt          # Core Python dependencies
├── pyproject.toml            # Package metadata (pip install -e .)
├── hapticnet_eval/           # Main package
│   ├── benchmark.py          # BenchmarkRunner — main entry point
│   ├── canonicalize.py       # GT/prediction → canonical claims
│   ├── schemas.py            # Pydantic data models
│   ├── evidence_fetcher.py   # Source content resolver
│   ├── report_adapter.py     # Adapter for DR report → db_entry
│   ├── dr_runner.py          # Deep Research runner (for production use)
│   ├── cli/                  # CLI entry point
│   │   └── main.py
│   ├── evaluators/           # All evaluator implementations
│   │   ├── core.py           # Deterministic evaluators
│   │   ├── advanced.py       # Citation & efficiency evaluators
│   │   ├── completeness.py   # Value list coverage
│   │   ├── composite.py      # Weighted geometric mean
│   │   ├── matching.py       # Claim matching (GT ↔ pred)
│   │   ├── llm_evaluators.py # LLM-based evaluators (optional)
│   │   ├── llm_judge.py      # Gemini API wrapper for judge
│   │   └── ...
│   ├── regimes/              # Evaluation regime definitions
│   │   ├── base.py           # EvaluationTask dataclass
│   │   ├── closed_docs.py
│   │   ├── open_web.py
│   │   └── url_only.py
│   └── utils/                # Shared utilities
│       ├── io.py             # JSON I/O
│       ├── normalization.py  # Unit normalization
│       ├── numeric.py        # Numeric comparison
│       └── evidence.py       # Evidence utilities
└── examples/
    ├── gt_copper_thermal_conductivity.json   # Sample GT
    ├── pred_copper_thermal_conductivity.json # Sample prediction
    └── run_example.py                        # Quick-start script
```

## Requirements

- **Python** ≥ 3.10
- Core: `pydantic`, `rapidfuzz`, `numpy`, `scipy`, `pandas`
- Optional (LLM evaluators): `google-genai` + `GOOGLE_API_KEY` env var

## License

See the main HapticNet repository for license terms.
