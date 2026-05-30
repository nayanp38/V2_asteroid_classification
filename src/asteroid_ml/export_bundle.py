"""Export a trained run as a portable model bundle."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from asteroid_ml.config import ROOT


BUNDLE_README = """# Asteroid BD classifier bundle

## Contents
- `best.pt` — PyTorch weights + class_to_index
- `config.yaml` — preprocessing and class list
- `predict.py` — minimal inference example

## Usage
```bash
pip install torch numpy pyyaml
python predict.py --spectrum path/to/spectrum_gp.txt
```
"""


PREDICT_PY = '''#!/usr/bin/env python3
"""Minimal inference script for exported bundle."""
import argparse
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from asteroid_ml.spectrum_io import PreprocessConfig, preprocess_spectrum
from asteroid_ml.models import build_model
from asteroid_ml.infer import prediction_entropy
from asteroid_ml.metrics import constrained_fine_argmax


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--spectrum", required=True)
    args = p.parse_args()
    ckpt = torch.load(ROOT / "best.pt", map_location="cpu", weights_only=True)
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

    n_coarse = len(ckpt.get("coarse_to_index") or {}) if ckpt.get("hierarchical", False) else 0
    model = build_model(
        ckpt["model_name"],
        n_classes=len(ckpt["class_to_index"]),
        n_coarse=n_coarse,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    pre_cfg = PreprocessConfig.from_dict(ckpt.get("preprocess") or cfg.get("preprocess", {}))
    x = preprocess_spectrum(
        Path(args.spectrum), ckpt.get("normalize_wavelength", 0.55), cfg=pre_cfg
    )
    xt = torch.from_numpy(x).unsqueeze(0)

    with torch.no_grad():
        logits_fine, logits_coarse = model(xt)
        probs_fine = torch.softmax(logits_fine, dim=1).numpy()[0]
        probs_coarse = (
            torch.softmax(logits_coarse, dim=1).numpy() if logits_coarse is not None else None
        )

    if probs_coarse is not None and ckpt.get("coarse_to_index"):
        idx = int(constrained_fine_argmax(
            probs_fine[None, :],
            probs_coarse,
            ckpt.get("coarse_to_fine_indices") or {},
            ckpt.get("coarse_to_index") or {},
        )[0])
    else:
        idx = int(probs_fine.argmax())
    inv = {i: c for c, i in ckpt["class_to_index"].items()}
    print(
        "class:", inv[idx],
        "prob:", float(probs_fine[idx]),
        "entropy:", prediction_entropy(probs_fine),
    )


if __name__ == "__main__":
    main()
'''


def export_bundle(run_dir: Path, out_dir: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    if out_dir is None:
        out_dir = ROOT / "releases" / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ("best.pt", "config.yaml", "metrics.json", "manifest.json"):
        src = run_dir / name
        if src.is_file():
            shutil.copy(src, out_dir / name)

    # Copy source package for predict.py
    src_pkg = ROOT / "src" / "asteroid_ml"
    dst_pkg = out_dir / "src" / "asteroid_ml"
    if dst_pkg.exists():
        shutil.rmtree(dst_pkg)
    shutil.copytree(
        src_pkg,
        dst_pkg,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    (out_dir / "README.md").write_text(BUNDLE_README)
    (out_dir / "predict.py").write_text(PREDICT_PY)
    (out_dir / "requirements.txt").write_text("torch>=2.0\nnumpy>=1.24\nPyYAML>=6.0\n")

    meta = {
        "source_run": str(run_dir),
        "bundle_path": str(out_dir),
    }
    if (run_dir / "best_params.json").is_file():
        shutil.copy(run_dir / "best_params.json", out_dir / "best_params.json")
        meta["tuned"] = True
    (out_dir / "bundle_manifest.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = export_bundle(Path(args.run), Path(args.out) if args.out else None)
    print(f"Exported bundle to {out}")


if __name__ == "__main__":
    main()
