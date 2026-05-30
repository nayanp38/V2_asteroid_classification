"""Inference on unlabeled spectra with confidence and entropy."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch

from asteroid_ml.config import ROOT, load_config
from asteroid_ml.labels import asteroid_key_from_path, coarse_class_to_index, read_manifest_csv
from asteroid_ml.metrics import constrained_fine_argmax
from asteroid_ml.models import build_model
from asteroid_ml.spectrum_io import PreprocessConfig, preprocess_spectrum


def prediction_entropy(probs: np.ndarray) -> float:
    p = probs[probs > 1e-12]
    return float(-np.sum(p * np.log(p)))


def list_unlabeled_gp_files(data_root: Path, cfg: dict) -> List[Path]:
    """GP spectra files not present in labels_manifest."""
    manifest_path = data_root / cfg["manifest_file"]
    labeled_paths: Set[str] = set()
    if manifest_path.is_file():
        for rec in read_manifest_csv(manifest_path):
            labeled_paths.add(rec.spectrum_path)

    files: List[Path] = []
    gp_dirs = [cfg["demeo_gp_dir"], cfg["binzel_gp_dir"]]
    if cfg.get("marsset_gp_dir"):
        gp_dirs.append(cfg["marsset_gp_dir"])
    seen: set[str] = set()
    for sub in gp_dirs:
        gp_dir = data_root / sub
        if not gp_dir.is_dir():
            continue
        for pattern in ("a*.txt", "au*.txt"):
            for f in sorted(gp_dir.glob(pattern)):
                if f.name in seen:
                    continue
                seen.add(f.name)
                rel = str(f.relative_to(data_root))
                if rel not in labeled_paths:
                    files.append(f)
    return files


def predict_files(
    model: torch.nn.Module,
    files: List[Path],
    class_to_index: Dict[str, int],
    coarse_to_index: Dict[str, int],
    coarse_to_fine_indices: Dict[str, List[int]],
    data_root: Path,
    normalize_wavelength: float,
    device: torch.device,
    preprocess_cfg: PreprocessConfig,
    reject_prob: float = 0.35,
    reject_entropy: Optional[float] = None,
    top_k: int = 3,
) -> List[dict]:
    index_to_class = {i: c for c, i in class_to_index.items()}
    n_classes = len(class_to_index)
    if reject_entropy is None:
        reject_entropy = np.log(n_classes) * 0.85

    rows: List[dict] = []
    model.eval()
    with torch.no_grad():
        for path in files:
            rel = str(path.relative_to(data_root))
            ast_id = asteroid_key_from_path(path)
            x = preprocess_spectrum(path, normalize_wavelength, cfg=preprocess_cfg)
            xt = torch.from_numpy(x).unsqueeze(0).to(device)
            logits_fine, logits_coarse = model(xt)
            probs_fine = torch.softmax(logits_fine, dim=1).cpu().numpy()[0]
            probs_coarse = (
                torch.softmax(logits_coarse, dim=1).cpu().numpy()
                if logits_coarse is not None
                else None
            )
            if probs_coarse is not None and coarse_to_index:
                constrained = constrained_fine_argmax(
                    probs_fine[None, :],
                    probs_coarse,
                    coarse_to_fine_indices,
                    coarse_to_index,
                )
                pred_idx = int(constrained[0])
            else:
                pred_idx = int(probs_fine.argmax())
            max_prob = float(probs_fine[pred_idx])
            ent = prediction_entropy(probs_fine)
            rejected = max_prob < reject_prob or ent > reject_entropy
            top_idx = np.argsort(probs_fine)[-top_k:][::-1]
            top_labels = [index_to_class[int(i)] for i in top_idx]
            top_probs = [float(probs_fine[int(i)]) for i in top_idx]

            rows.append(
                {
                    "spectrum_path": rel,
                    "asteroid_id": ast_id,
                    "predicted_class": index_to_class[pred_idx] if not rejected else "UNCERTAIN",
                    "max_probability": max_prob,
                    "entropy": ent,
                    "rejected": rejected,
                    "top1": top_labels[0],
                    "top1_prob": top_probs[0],
                    "top2": top_labels[1] if len(top_labels) > 1 else "",
                    "top2_prob": top_probs[1] if len(top_probs) > 1 else "",
                    "top3": top_labels[2] if len(top_labels) > 2 else "",
                    "top3_prob": top_probs[2] if len(top_probs) > 2 else "",
                }
            )
    return rows


def write_predictions_csv(rows: List[dict], out_path: Path) -> None:
    if not rows:
        out_path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_inference(
    run_dir: Path,
    cfg: Optional[dict] = None,
    reject_prob: float = 0.35,
    limit: Optional[int] = None,
) -> Path:
    run_dir = run_dir.resolve()
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    if cfg is None:
        cfg_path = run_dir / "config.yaml"
        cfg = load_config(cfg_path if cfg_path.exists() else None)

    data_root = ROOT / cfg["data_root"]
    files = list_unlabeled_gp_files(data_root, cfg)
    if limit is not None:
        files = files[:limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    coarse_to_index = ckpt.get("coarse_to_index") or {}
    fine_to_coarse_index = {
        int(k): int(v) for k, v in (ckpt.get("fine_to_coarse_index") or {}).items()
    }
    coarse_to_fine_indices = ckpt.get("coarse_to_fine_indices") or {}
    if not coarse_to_index:
        coarse_to_index, fine_to_coarse_index, coarse_to_fine_indices = coarse_class_to_index(
            ckpt["class_to_index"], cfg.get("coarse_groups", {})
        )
    n_coarse = len(coarse_to_index) if ckpt.get("hierarchical", True) else 0
    model = build_model(
        ckpt["model_name"],
        n_classes=len(ckpt["class_to_index"]),
        n_coarse=n_coarse,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    preprocess_cfg = PreprocessConfig.from_dict(ckpt.get("preprocess") or cfg.get("preprocess", {}))

    p2 = cfg.get("phase2", {}).get("inference", {})
    reject_prob = float(p2.get("reject_probability", reject_prob))
    reject_ent = p2.get("reject_entropy", None)

    rows = predict_files(
        model,
        files,
        ckpt["class_to_index"],
        coarse_to_index,
        coarse_to_fine_indices,
        data_root,
        ckpt.get("normalize_wavelength", 0.55),
        device,
        preprocess_cfg,
        reject_prob=reject_prob,
        reject_entropy=float(reject_ent) if reject_ent is not None else None,
    )

    out_csv = run_dir / "predictions_unlabeled.csv"
    write_predictions_csv(rows, out_csv)

    summary = {
        "n_unlabeled_files": len(files),
        "n_predictions": len(rows),
        "n_rejected": sum(1 for r in rows if r["rejected"]),
        "reject_probability_threshold": reject_prob,
        "class_distribution": {},
    }
    for r in rows:
        c = r["predicted_class"]
        summary["class_distribution"][c] = summary["class_distribution"].get(c, 0) + 1
    (run_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2))
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict classes for unlabeled GP spectra")
    parser.add_argument("--run", required=True, help="Path to trained run directory")
    parser.add_argument("--reject-prob", type=float, default=0.35)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    out = run_inference(Path(args.run), reject_prob=args.reject_prob, limit=args.limit)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
