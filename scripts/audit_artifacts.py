#!/usr/bin/env python3
"""Audit GP spectra for std-based artifact regions.

Prints the fraction of points kept under the mask rule for each spectrum and
the top-N flagged files. Optional ``--flag-threshold`` controls which spectra
are listed in detail.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asteroid_ml.config import load_config
from asteroid_ml.spectrum_io import PreprocessConfig, valid_fraction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flag-threshold", type=float, default=0.6,
                        help="Flag spectra whose valid fraction is below this (default 0.6)")
    parser.add_argument("--top", type=int, default=20, help="Show this many worst spectra")
    parser.add_argument("--out", type=str, default="data/artifact_audit.csv",
                        help="Output CSV path (relative to repo root)")
    parser.add_argument("--include-unlabeled", action="store_true",
                        help="Audit every GP file, not just those in the manifest")
    args = parser.parse_args()

    cfg = load_config()
    pre_cfg = PreprocessConfig.from_dict(cfg.get("preprocess", {}))
    data_root = ROOT / cfg["data_root"]

    paths: list[Path] = []
    if args.include_unlabeled:
        for sub in (cfg["demeo_gp_dir"], cfg["binzel_gp_dir"]):
            d = data_root / sub
            if d.is_dir():
                paths.extend(sorted(d.glob("a*.txt")))
    else:
        manifest = data_root / cfg["manifest_file"]
        if not manifest.is_file():
            print(f"Manifest not found: {manifest}; pass --include-unlabeled to audit raw files")
            return
        with manifest.open(newline="") as f:
            for row in csv.DictReader(f):
                paths.append(data_root / row["spectrum_path"])

    rows: list[tuple[str, float]] = []
    for p in paths:
        if not p.is_file():
            continue
        rows.append((str(p.relative_to(data_root)), valid_fraction(p, pre_cfg)))

    rows.sort(key=lambda r: r[1])
    flagged = [r for r in rows if r[1] < args.flag_threshold]
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["spectrum_path", "valid_fraction"])
        for path, vf in rows:
            w.writerow([path, f"{vf:.4f}"])

    summary = {
        "total": len(rows),
        "flagged_below": args.flag_threshold,
        "n_flagged": len(flagged),
        "mean_valid_fraction": (sum(vf for _, vf in rows) / max(1, len(rows))),
        "min_valid_fraction": rows[0][1] if rows else None,
        "max_valid_fraction": rows[-1][1] if rows else None,
    }
    print(json.dumps(summary, indent=2))
    print(f"\nWrote per-spectrum CSV to {out_path}")
    print(f"\nWorst {min(args.top, len(rows))} spectra:")
    for path, vf in rows[: args.top]:
        print(f"  {vf:.3f}  {path}")


if __name__ == "__main__":
    main()
