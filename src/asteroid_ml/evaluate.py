"""Evaluate a trained run: confusion matrix and metrics on test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from asteroid_ml.config import ROOT, load_config
from asteroid_ml.dataset import AsteroidSpectrumDataset
from asteroid_ml.labels import coarse_class_to_index, read_manifest_csv
from asteroid_ml.metrics import compute_metrics, constrained_fine_argmax
from asteroid_ml.gradcam import generate_gradcam_report
from asteroid_ml.models import build_model
from asteroid_ml.spectrum_io import PreprocessConfig
from asteroid_ml.splits import get_split


def _latest_run(runs_dir: Path) -> Path:
    subdirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
    if not subdirs:
        raise FileNotFoundError(f"No runs in {runs_dir}")
    return subdirs[0]


def save_confusion(
    cm: np.ndarray, class_names: List[str], out_dir: Path, name: str = "confusion_matrix"
) -> None:
    csv_path = out_dir / f"{name}.csv"
    with csv_path.open("w") as f:
        f.write("true_class," + ",".join(class_names) + "\n")
        for i, c in enumerate(class_names):
            f.write(c + "," + ",".join(str(cm[i, j]) for j in range(len(class_names))) + "\n")

    fig, ax = plt.subplots(
        figsize=(max(8, len(class_names) * 0.35), max(6, len(class_names) * 0.3))
    )
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_run(run_dir: Path, gradcam: bool = False, gradcam_samples: int = 12) -> None:
    run_dir = run_dir.resolve()
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    cfg_path = run_dir / "config.yaml"
    cfg = load_config(cfg_path if cfg_path.exists() else None)

    data_root = ROOT / cfg["data_root"]
    records = read_manifest_csv(data_root / cfg["manifest_file"])
    class_to_index = ckpt["class_to_index"]
    index_to_class = {i: c for c, i in class_to_index.items()}
    class_names = [index_to_class[i] for i in range(len(class_to_index))]

    coarse_to_index = ckpt.get("coarse_to_index") or {}
    fine_to_coarse_index = {
        int(k): int(v) for k, v in (ckpt.get("fine_to_coarse_index") or {}).items()
    }
    coarse_to_fine_indices = ckpt.get("coarse_to_fine_indices") or {}
    if not coarse_to_index:
        coarse_to_index, fine_to_coarse_index, coarse_to_fine_indices = coarse_class_to_index(
            class_to_index, cfg.get("coarse_groups", {})
        )
    coarse_class_names = sorted(coarse_to_index, key=coarse_to_index.get)

    preprocess_cfg = PreprocessConfig.from_dict(ckpt.get("preprocess") or cfg.get("preprocess", {}))
    ds = AsteroidSpectrumDataset(
        records,
        data_root,
        class_to_index,
        ckpt.get("normalize_wavelength", 0.55),
        augment=False,
        preprocess_config=preprocess_cfg,
        fine_to_coarse_index=fine_to_coarse_index,
    )

    split_name = ckpt.get("split_name", "run3")
    base_split = split_name.split("_fold")[0] if "_fold" in split_name else split_name
    split_result = get_split(base_split if base_split != "cv5" else "cv5", records, cfg)
    if base_split == "cv5" or "cv5_fold" in split_name:
        fold_idx = int(split_name.replace("cv5_fold", "")) if "fold" in split_name else 0
        split = split_result[fold_idx]
    else:
        split = split_result

    test_idx = split.test
    if not test_idx:
        print("No test indices for this split.")
        return

    loader = DataLoader(Subset(ds, test_idx), batch_size=32, shuffle=False)
    n_coarse = len(coarse_to_index) if ckpt.get("hierarchical", True) else 0
    model = build_model(
        ckpt["model_name"],
        n_classes=len(class_to_index),
        n_coarse=n_coarse,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    yt: List[int] = []
    yt_coarse: List[int] = []
    probs_fine_list: List[np.ndarray] = []
    probs_coarse_list: List[np.ndarray] = []
    with torch.no_grad():
        for x, y, y_coarse, _paths in loader:
            logits_fine, logits_coarse = model(x)
            probs_fine_list.append(torch.softmax(logits_fine, dim=1).numpy())
            if logits_coarse is not None:
                probs_coarse_list.append(torch.softmax(logits_coarse, dim=1).numpy())
            yt.extend(y.tolist())
            yt_coarse.extend(y_coarse.tolist())

    probs_fine = np.vstack(probs_fine_list) if probs_fine_list else np.zeros((0, len(class_names)))
    probs_coarse = np.vstack(probs_coarse_list) if probs_coarse_list else None

    constrained_pred = (
        constrained_fine_argmax(probs_fine, probs_coarse, coarse_to_fine_indices, coarse_to_index)
        if probs_coarse is not None
        else probs_fine.argmax(axis=1)
    )
    unconstrained_pred = probs_fine.argmax(axis=1)

    y_coarse_pred = probs_coarse.argmax(axis=1) if probs_coarse is not None else None
    metrics_constrained = compute_metrics(
        yt,
        constrained_pred,
        class_names,
        probs_fine,
        coarse_class_names=coarse_class_names if probs_coarse is not None else None,
        y_coarse_true=yt_coarse if probs_coarse is not None else None,
        y_coarse_pred=y_coarse_pred,
    )
    metrics_unconstrained = compute_metrics(yt, unconstrained_pred, class_names, probs_fine)

    cm_constrained = np.array(metrics_constrained.pop("confusion_matrix"))
    cm_unconstrained = np.array(metrics_unconstrained.pop("confusion_matrix"))
    save_confusion(cm_constrained, class_names, run_dir, name="confusion_matrix")
    save_confusion(
        cm_unconstrained, class_names, run_dir, name="confusion_matrix_unconstrained"
    )

    eval_out = {
        "constrained": metrics_constrained,
        "unconstrained": metrics_unconstrained,
        "n_test": len(yt),
    }
    (run_dir / "eval_test.json").write_text(json.dumps(eval_out, indent=2, default=str))
    print(f"Test accuracy (constrained): {metrics_constrained['accuracy']:.3f}")
    print(f"Test macro-F1 (constrained): {metrics_constrained['macro_f1']:.3f}")
    print(f"Coarse accuracy: {metrics_constrained['coarse_accuracy']:.3f}")
    if metrics_constrained.get("top2_accuracy") is not None:
        print(f"Top-2 accuracy: {metrics_constrained['top2_accuracy']:.3f}")
    if metrics_constrained["support_warnings"]:
        print("Warnings:", "; ".join(metrics_constrained["support_warnings"][:5]))

    if gradcam and loader:
        saved = generate_gradcam_report(
            model,
            ckpt["model_name"],
            loader,
            index_to_class,
            data_root,
            run_dir,
            max_samples=gradcam_samples,
            normalize_wavelength=ckpt.get("normalize_wavelength", 0.55),
        )
        diag_path = run_dir / "diagnostics.json"
        diag = {}
        if diag_path.is_file():
            diag = json.loads(diag_path.read_text())
        diag["gradcam_plots"] = saved
        diag_path.write_text(json.dumps(diag, indent=2))
        print(f"Grad-CAM: {len(saved)} plots in {run_dir / 'gradcam'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None, help="Path to run directory")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM plots")
    parser.add_argument("--gradcam-samples", type=int, default=12)
    args = parser.parse_args()
    run_dir = Path(args.run) if args.run else _latest_run(ROOT / "runs")
    evaluate_run(run_dir, gradcam=args.gradcam, gradcam_samples=args.gradcam_samples)


if __name__ == "__main__":
    main()
