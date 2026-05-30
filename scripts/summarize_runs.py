#!/usr/bin/env python3
"""Print a one-row-per-run summary of metrics.json files in runs/."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def fmt_run(metrics_path: Path) -> str:
    d = json.loads(metrics_path.read_text())
    val_c = d["val"]["macro_f1"]
    val_u = d["val_unconstrained"]["macro_f1"]
    test = d.get("test") or {}
    test_unc = d.get("test_unconstrained") or {}
    test_f1_c = test.get("macro_f1", 0.0)
    test_f1_u = test_unc.get("macro_f1", 0.0)
    acc = test.get("accuracy", 0.0)
    coarse_acc = test.get("coarse_accuracy", 0.0)
    bal_acc = test.get("balanced_accuracy", 0.0)
    top2 = test.get("top2_accuracy") or 0.0
    n_test = d.get("n_test", 0)
    model = d.get("model", "?")
    split = d.get("split", "?")
    return (
        f"{model:18s} {split:14s} val_macroF1={val_c:.3f}/{val_u:.3f} "
        f"test_macroF1={test_f1_c:.3f}/{test_f1_u:.3f} "
        f"acc={acc:.3f} bal_acc={bal_acc:.3f} coarse_acc={coarse_acc:.3f} "
        f"top2={top2:.3f} n_test={n_test}"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs-dir", default="runs", help="Directory of run subdirs (default: runs)"
    )
    p.add_argument(
        "--prefix", default="2026", help="Only include run dirs starting with this prefix"
    )
    args = p.parse_args()
    runs = sorted(
        Path(args.runs_dir).glob(f"{args.prefix}*"), key=lambda p: p.name
    )
    rows = []
    for run in runs:
        m = run / "metrics.json"
        if m.is_file():
            try:
                rows.append((run.name, fmt_run(m)))
            except Exception as e:
                rows.append((run.name, f"ERROR: {e}"))
    if not rows:
        print(f"No runs with metrics.json under {args.runs_dir}/{args.prefix}*")
        return 1
    width = max(len(name) for name, _ in rows)
    for name, summary in rows:
        print(f"{name.ljust(width)}  {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
