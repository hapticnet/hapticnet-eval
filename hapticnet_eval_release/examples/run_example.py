#!/usr/bin/env python3
"""
Quick-start example: evaluate a single prediction against a ground-truth file.

Usage:
    python examples/run_example.py
"""
import json
import sys
from pathlib import Path

# Ensure the package is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hapticnet_eval.benchmark import BenchmarkRunner
from hapticnet_eval.regimes.base import EvaluationTask


def main():
    examples_dir = Path(__file__).resolve().parent

    gt_path = examples_dir / "gt_copper_thermal_conductivity.json"
    pred_path = examples_dir / "pred_copper_thermal_conductivity.json"

    if not gt_path.exists() or not pred_path.exists():
        print("ERROR: example GT or prediction file not found.")
        print(f"  Expected GT:   {gt_path}")
        print(f"  Expected Pred: {pred_path}")
        sys.exit(1)

    # --- Run evaluation ---
    runner = BenchmarkRunner()
    task = EvaluationTask(
        task_id="copper_thermal_conductivity",
        regime="closed_docs",
        gt_path=str(gt_path),
        pred_path=str(pred_path),
    )

    print("Running HapticNet evaluation...")
    print(f"  GT:   {gt_path.name}")
    print(f"  Pred: {pred_path.name}")
    print(f"  Regime: closed_docs")
    print()

    report = runner.evaluate_task(task)

    # --- Print results ---
    print(f"{'='*60}")
    print(f"  AGGREGATE SCORE: {report.aggregate_score:.4f}")
    print(f"{'='*60}")
    print()
    print(f"{'Evaluator':<35s}  {'Score':>8s}  {'Max':>5s}")
    print(f"{'-'*35}  {'-'*8}  {'-'*5}")
    for s in sorted(report.scores, key=lambda x: -x.score):
        print(f"  {s.name:<33s}  {s.score:>8.4f}  {s.max_score:>5.1f}")

    # --- Save report ---
    out_path = examples_dir / "example_report.json"
    with open(out_path, "w") as f:
        json.dump(report.model_dump(), f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to: {out_path}")


if __name__ == "__main__":
    main()
