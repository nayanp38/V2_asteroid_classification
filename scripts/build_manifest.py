#!/usr/bin/env python3
"""Build data/labels_manifest.csv from demeotax.tab and Binzel_classes.txt."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asteroid_ml.config import load_config
from asteroid_ml.labels import build_manifest, write_manifest_csv
from asteroid_ml.spectrum_io import PreprocessConfig, valid_fraction


def main() -> None:
    cfg = load_config()
    data_root = ROOT / cfg["data_root"]
    mars_path = data_root / cfg.get("mars_crossers_file", "mars_crossers.txt")

    pre_cfg_dict = cfg.get("preprocess", {})
    pre_cfg = PreprocessConfig.from_dict(pre_cfg_dict)
    min_valid = float(pre_cfg_dict.get("min_valid_fraction", 0.0))
    quality_filter = None
    if min_valid > 0:
        def quality_filter(path: Path) -> bool:
            try:
                return valid_fraction(path, pre_cfg) >= min_valid
            except Exception:
                return False

    records, stats, alias_log = build_manifest(
        data_root=data_root,
        demeo_gp_dir=data_root / cfg["demeo_gp_dir"],
        binzel_gp_dir=data_root / cfg["binzel_gp_dir"],
        demeotax_path=data_root / cfg["demeotax_file"],
        binzel_classes_path=data_root / cfg["binzel_classes_file"],
        aliases=cfg["class_aliases"],
        bd_classes=cfg["bd_classes"],
        excluded_classes=cfg.get("excluded_classes", []),
        mars_crossers_path=mars_path if mars_path.is_file() else None,
        quality_filter=quality_filter,
    )

    out_path = data_root / cfg["manifest_file"]
    write_manifest_csv(out_path, records)

    report = {
        "manifest": str(out_path),
        "stats": stats,
        "preprocess": {
            "std_mask_enabled": pre_cfg.std_mask_enabled,
            "std_mask_floor": pre_cfg.std_mask_floor,
            "std_mask_k": pre_cfg.std_mask_k,
            "min_valid_fraction": min_valid,
        },
        "alias_log_sample": alias_log[:20],
    }
    report_path = data_root / "manifest_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    alias_path = data_root / "alias_log.txt"
    alias_path.write_text("\n".join(alias_log) + ("\n" if alias_log else ""))

    print(f"Wrote {len(records)} rows to {out_path}")
    print(f"  DeMeo: {stats['demeo_rows']}, Binzel: {stats['binzel_rows']}")
    print(f"  Unique asteroids: {stats['unique_asteroids']}")
    print(f"  Class aliases applied: {stats['alias_applications']}")
    print(f"  Skipped (no label): {stats.get('skipped_no_label', 0)}")
    print(f"  Dropped (low quality): {stats.get('dropped_low_quality', 0)}")
    print(f"  Binzel multi-class entries (raw file): {stats['multi_class_binzel']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
