from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
from typing import List

from ..benchmark import BenchmarkRunner
from ..regimes.base import EvaluationTask
from ..utils.io import dump_json, load_json


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run HapticNet structured extraction benchmark")
    sub = p.add_subparsers(dest="command", required=True)

    one = sub.add_parser("evaluate", help="Evaluate one prediction against one GT file")
    one.add_argument("--gt", required=True, help="Path to GT JSON")
    one.add_argument("--pred", required=True, help="Path to prediction JSON")
    one.add_argument("--regime", default="closed_docs", choices=["closed_docs", "url_only", "open_web"])
    one.add_argument("--task-id", default="single_task")
    one.add_argument("--out", required=True, help="Path to write report JSON")

    batch = sub.add_parser("evaluate-manifest", help="Evaluate a manifest of tasks")
    batch.add_argument("--manifest", required=True, help="JSON list of tasks")
    batch.add_argument("--out", required=True, help="Path to write batch report JSON")
    return p


def _evaluate_one(args: argparse.Namespace) -> None:
    runner = BenchmarkRunner()
    report = runner.evaluate_task(EvaluationTask(task_id=args.task_id, regime=args.regime, gt_path=args.gt, pred_path=args.pred))
    dump_json(report.model_dump(), args.out)
    print(f"Aggregate score: {report.aggregate_score:.4f}")
    for s in report.scores:
        print(f"  {s.name}: {s.score:.4f}")


def _evaluate_manifest(args: argparse.Namespace) -> None:
    runner = BenchmarkRunner()
    manifest_raw = load_json(args.manifest)
    tasks: List[EvaluationTask] = [EvaluationTask(**row) for row in manifest_raw]
    reports = runner.evaluate_manifest(tasks)
    aggregate = mean(r.aggregate_score for r in reports) if reports else 0.0
    payload = {
        "mean_aggregate_score": aggregate,
        "num_tasks": len(reports),
        "reports": [r.model_dump() for r in reports],
    }
    dump_json(payload, args.out)
    print(f"Mean aggregate score: {aggregate:.4f} across {len(reports)} tasks")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "evaluate":
        _evaluate_one(args)
    elif args.command == "evaluate-manifest":
        _evaluate_manifest(args)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
