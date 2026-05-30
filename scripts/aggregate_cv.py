#!/usr/bin/env python3
"""Aggregate cv5 fold metrics into a single mean ± std row.

Usage:
    python scripts/aggregate_cv.py --model spectranet_lite
    python scripts/aggregate_cv.py --model spectrum_cnn
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def collect(runs_dir: Path, model: str, phase: int) -> list[dict]:
    rows = []
    for run in sorted(runs_dir.glob("*")):
        if not run.is_dir():
            continue
        m = run / "metrics.json"
        if not m.is_file():
            continue
        d = json.loads(m.read_text())
        if d.get("model") != model:
            continue
        if int(d.get("phase", 1)) != phase:
            continue
        split = d.get("split", "")
        if "cv5_fold" not in split:
            continue
        rows.append(d)
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--phase", type=int, default=2, help="Filter on phase field (default 2)")
    args = p.parse_args()
    rows = collect(Path(args.runs_dir), args.model, args.phase)
    if len(rows) == 0:
        print(f"No cv5 rows found for model {args.model} phase {args.phase}")
        return 1
    print(f"Aggregating {len(rows)} folds for {args.model} (phase {args.phase}):")
    metrics = {
        "test macro-F1 (constrained)": [r["test"]["macro_f1"] for r in rows],
        "test macro-F1 (unconstrained)": [
            r.get("test_unconstrained", r["test"]).get("macro_f1", r["test"]["macro_f1"])
            for r in rows
        ],
        "test accuracy": [r["test"]["accuracy"] for r in rows],
        "test balanced_accuracy": [r["test"]["balanced_accuracy"] for r in rows],
        "test coarse_accuracy": [r["test"]["coarse_accuracy"] for r in rows],
        "test top2_accuracy": [r["test"].get("top2_accuracy") or 0.0 for r in rows],
    }
    for name, vals in metrics.items():
        mu = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        print(f"  {name:35s} {mu:.3f} ± {sd:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
