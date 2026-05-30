"""Train 1D CNN models on asteroid spectra (hierarchical + Phase 2)."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from asteroid_ml.augmentation import AugmentConfig
from asteroid_ml.config import ROOT, load_config
from asteroid_ml.dataset import AsteroidSpectrumDataset, compute_class_weights
from asteroid_ml.gradcam import generate_gradcam_report
from asteroid_ml.infer import run_inference
from asteroid_ml.labels import (
    LabelRecord,
    class_to_index_from_manifest,
    coarse_class_to_index,
    read_manifest_csv,
)
from asteroid_ml.losses import FocalSmoothedCE
from asteroid_ml.metrics import compute_metrics, constrained_fine_argmax
from asteroid_ml.models import build_model
from asteroid_ml.spectrum_io import PreprocessConfig
from asteroid_ml.splits import SplitIndices, get_split


def _git_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return None


def _train_labels(dataset: AsteroidSpectrumDataset, indices: List[int]) -> List[int]:
    return [dataset.class_to_index[dataset.records[i].class_bd] for i in indices]


def _merge_tuned_params(cfg: dict, tuned_path: Path) -> dict:
    data = json.loads(tuned_path.read_text())
    cfg = deepcopy(cfg)
    params = data.get("best_params", data)
    tcfg = cfg.setdefault("training", {})
    for k in (
        "learning_rate",
        "weight_decay",
        "batch_size",
        "focal_gamma",
        "label_smoothing",
    ):
        if k in params:
            tcfg[k] = params[k]
    aug = cfg.setdefault("phase2", {}).setdefault("augmentation", {})
    aug["enabled"] = True
    for k in ("noise_std", "smooth_sigma"):
        if k in params:
            aug[k] = params[k]
    if "coarse_weight" in params:
        cfg.setdefault("hierarchical", {})["coarse_weight"] = params["coarse_weight"]
    return cfg


def _cosine_lr_lambda(epoch: int, warmup_epochs: int, max_epochs: int) -> float:
    if max_epochs <= 0:
        return 1.0
    if epoch < warmup_epochs:
        return (epoch + 1) / max(1, warmup_epochs)
    progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _eval_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: List[str],
    coarse_to_fine_indices: Dict[str, List[int]],
    coarse_to_index: Dict[str, int],
    constrained: bool,
) -> Tuple[Dict, Dict, np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Evaluate ``loader`` returning (constrained_metrics, unconstrained_metrics,
    probs_fine, probs_coarse_or_None, y_true)."""
    model.eval()
    y_true: List[int] = []
    y_true_coarse: List[int] = []
    probs_fine: List[np.ndarray] = []
    probs_coarse: List[np.ndarray] = []
    has_coarse = False
    with torch.no_grad():
        for x, y, y_coarse, _paths in loader:
            x = x.to(device)
            logits_fine, logits_coarse = model(x)
            probs_fine.append(torch.softmax(logits_fine, dim=1).cpu().numpy())
            if logits_coarse is not None:
                has_coarse = True
                probs_coarse.append(torch.softmax(logits_coarse, dim=1).cpu().numpy())
            y_true.extend(y.tolist())
            y_true_coarse.extend(y_coarse.tolist())

    probs_fine_arr = np.vstack(probs_fine) if probs_fine else np.zeros((0, len(class_names)))
    probs_coarse_arr = np.vstack(probs_coarse) if has_coarse else None
    y_arr = np.asarray(y_true)

    unconstrained_pred = (
        probs_fine_arr.argmax(axis=1) if probs_fine_arr.size else np.array([], dtype=np.int64)
    )
    if constrained and has_coarse:
        constrained_pred = constrained_fine_argmax(
            probs_fine_arr, probs_coarse_arr, coarse_to_fine_indices, coarse_to_index
        )
    else:
        constrained_pred = unconstrained_pred

    coarse_class_names = (
        sorted(coarse_to_index, key=coarse_to_index.get) if coarse_to_index else None
    )
    if has_coarse:
        y_coarse_pred = probs_coarse_arr.argmax(axis=1)
        y_coarse_true_arr = np.asarray(y_true_coarse)
    else:
        y_coarse_pred = None
        y_coarse_true_arr = None

    constrained_metrics = compute_metrics(
        y_arr,
        constrained_pred,
        class_names,
        probs_fine_arr if probs_fine_arr.size else None,
        coarse_class_names=coarse_class_names,
        y_coarse_true=y_coarse_true_arr,
        y_coarse_pred=y_coarse_pred,
    )
    unconstrained_metrics = compute_metrics(
        y_arr,
        unconstrained_pred,
        class_names,
        probs_fine_arr if probs_fine_arr.size else None,
    )
    return (
        constrained_metrics,
        unconstrained_metrics,
        probs_fine_arr,
        probs_coarse_arr,
        y_arr,
    )


def train_one_split(
    records: List[LabelRecord],
    split: SplitIndices,
    cfg: dict,
    model_name: str,
    run_dir: Path,
    device: torch.device,
    phase: int = 1,
    pretrained_path: Optional[Path] = None,
) -> Dict:
    data_root = ROOT / cfg["data_root"]
    class_to_index = class_to_index_from_manifest(records, cfg["bd_classes"])
    index_to_class = {i: c for c, i in class_to_index.items()}
    class_names = [index_to_class[i] for i in range(len(class_to_index))]

    coarse_groups = cfg.get("coarse_groups", {})
    coarse_to_index, fine_to_coarse_index, coarse_to_fine_indices = coarse_class_to_index(
        class_to_index, coarse_groups
    )
    coarse_class_names = sorted(coarse_to_index, key=coarse_to_index.get)
    hier_cfg = cfg.get("hierarchical", {}) or {}
    hier_enabled = bool(hier_cfg.get("enabled", True))
    coarse_weight = float(hier_cfg.get("coarse_weight", 0.5))
    constrained_inference = bool(hier_cfg.get("constrained_inference", True))
    n_coarse = len(coarse_to_index) if hier_enabled else 0

    preprocess_cfg = PreprocessConfig.from_dict(cfg.get("preprocess", {}))

    p2_aug = cfg.get("phase2", {}).get("augmentation", {})
    use_aug = bool(p2_aug.get("enabled", False)) and phase >= 2
    aug_cfg = AugmentConfig.from_dict(p2_aug) if use_aug else None

    tcfg = cfg.get("training", {})
    val_frac = tcfg.get("val_fraction", 0.25)
    seed = tcfg.get("split_seed", 42)
    rng = np.random.default_rng(seed)
    shuffled = list(split.train)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_frac))
    val_idx = shuffled[:n_val]
    train_idx_final = shuffled[n_val:]

    train_ds = AsteroidSpectrumDataset(
        records,
        data_root,
        class_to_index,
        cfg.get("normalize_wavelength", 0.55),
        augment=use_aug,
        augment_config=aug_cfg,
        seed=seed,
        preprocess_config=preprocess_cfg,
        fine_to_coarse_index=fine_to_coarse_index,
    )
    eval_ds = AsteroidSpectrumDataset(
        records,
        data_root,
        class_to_index,
        cfg.get("normalize_wavelength", 0.55),
        augment=False,
        augment_config=None,
        seed=seed,
        preprocess_config=preprocess_cfg,
        fine_to_coarse_index=fine_to_coarse_index,
    )

    train_subset = Subset(train_ds, train_idx_final)
    val_subset = Subset(eval_ds, val_idx)
    test_subset = Subset(eval_ds, split.test) if split.test else None

    batch_size = tcfg.get("batch_size", 16)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = (
        DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        if test_subset and len(split.test) > 0
        else None
    )

    n_fine = len(class_to_index)
    model = build_model(model_name, n_classes=n_fine, n_coarse=n_coarse).to(device)
    encoder_param_names: List[str] = []
    freeze_epochs = 0
    if pretrained_path is not None and pretrained_path.is_file():
        sd = torch.load(pretrained_path, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        encoder_param_names = list(sd.keys())
        freeze_epochs = int(cfg.get("pretrain", {}).get("freeze_encoder_epochs", 0))
        print(
            f"  loaded pretrained encoder from {pretrained_path}; "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"freeze_epochs={freeze_epochs}"
        )

    train_labels_fine = _train_labels(train_ds, train_idx_final)
    train_labels_coarse = [fine_to_coarse_index.get(l, 0) for l in train_labels_fine]
    cw_mode = tcfg.get("class_weight_mode", "linear")
    cw_beta = float(tcfg.get("class_weight_beta", 0.999))
    fine_weights = compute_class_weights(
        train_labels_fine, n_fine, mode=cw_mode, beta=cw_beta
    ).to(device)
    coarse_weights = (
        compute_class_weights(train_labels_coarse, n_coarse, mode=cw_mode, beta=cw_beta).to(device)
        if n_coarse > 0
        else None
    )
    gamma = float(tcfg.get("focal_gamma", 0.0))
    smoothing = float(tcfg.get("label_smoothing", 0.0))
    criterion_fine = FocalSmoothedCE(weight=fine_weights, gamma=gamma, label_smoothing=smoothing)
    criterion_coarse = (
        FocalSmoothedCE(weight=coarse_weights, gamma=gamma, label_smoothing=smoothing)
        if n_coarse > 0
        else None
    )

    lr = float(tcfg.get("learning_rate", 1e-3))
    wd = float(tcfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    max_epochs = int(tcfg.get("max_epochs", 50))
    warmup = int(tcfg.get("warmup_epochs", 0))
    use_cosine = bool(tcfg.get("cosine_lr", True))
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda e: _cosine_lr_lambda(e, warmup, max_epochs)
        )
    else:
        scheduler = None

    grad_clip = float(tcfg.get("grad_clip_norm", 0.0))
    patience = int(tcfg.get("early_stop_patience", 15))

    best_val = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    history: List[Dict] = []

    encoder_prefixes = {n.split(".")[0] for n in encoder_param_names}

    def _set_encoder_frozen(frozen: bool) -> None:
        if not encoder_prefixes:
            return
        for name, param in model.named_parameters():
            top = name.split(".")[0]
            if top in encoder_prefixes:
                param.requires_grad = not frozen

    if freeze_epochs > 0:
        _set_encoder_frozen(True)
        print(f"  encoder frozen for first {freeze_epochs} epochs")

    for epoch in range(1, max_epochs + 1):
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            _set_encoder_frozen(False)
            print(f"  encoder unfrozen at epoch {epoch}")
        model.train()
        train_loss = 0.0
        n_train = 0
        for x, y_fine, y_coarse, _paths in train_loader:
            x = x.to(device)
            y_fine = y_fine.to(device)
            y_coarse = y_coarse.to(device)

            optimizer.zero_grad()
            logits_fine, logits_coarse = model(x)
            loss = criterion_fine(logits_fine, y_fine)
            if criterion_coarse is not None and logits_coarse is not None and coarse_weight > 0:
                loss = loss + coarse_weight * criterion_coarse(logits_coarse, y_coarse)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += float(loss.item()) * x.size(0)
            n_train += x.size(0)
        if scheduler is not None:
            scheduler.step()

        val_metrics, val_unc, _, _, _ = _eval_model(
            model,
            val_loader,
            device,
            class_names,
            coarse_to_fine_indices,
            coarse_to_index,
            constrained=constrained_inference,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss / max(n_train, 1),
                "val_macro_f1": val_metrics["macro_f1"],
                "val_macro_f1_unconstrained": val_unc["macro_f1"],
                "val_coarse_acc": val_metrics["coarse_accuracy"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        print(
            f"  epoch {epoch:03d} loss={history[-1]['train_loss']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.3f} "
            f"(unc={val_unc['macro_f1']:.3f}) "
            f"coarse_acc={val_metrics['coarse_accuracy']:.3f} "
            f"lr={history[-1]['lr']:.4g}"
        )

        score = val_metrics["macro_f1"]
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_final, val_unc, _val_probs_fine, _val_probs_coarse, _val_y = _eval_model(
        model,
        val_loader,
        device,
        class_names,
        coarse_to_fine_indices,
        coarse_to_index,
        constrained=constrained_inference,
    )
    if test_loader is not None:
        test_final, test_unc, _, _, _ = _eval_model(
            model,
            test_loader,
            device,
            class_names,
            coarse_to_fine_indices,
            coarse_to_index,
            constrained=constrained_inference,
        )
    else:
        test_final, test_unc = {}, {}

    ckpt = {
        "model_state_dict": model.state_dict(),
        "class_to_index": class_to_index,
        "model_name": model_name,
        "split_name": split.name,
        "normalize_wavelength": cfg.get("normalize_wavelength", 0.55),
        "phase": phase,
        "augmentation_enabled": use_aug,
        "hierarchical": hier_enabled,
        "coarse_to_index": coarse_to_index,
        "fine_to_coarse_index": fine_to_coarse_index,
        "coarse_to_fine_indices": coarse_to_fine_indices,
        "preprocess": cfg.get("preprocess", {}),
    }
    torch.save(ckpt, run_dir / "best.pt")

    metrics_out = {
        "split": split.name,
        "split_description": split.description,
        "model": model_name,
        "phase": phase,
        "augmentation_enabled": use_aug,
        "hierarchical": hier_enabled,
        "coarse_weight": coarse_weight,
        "constrained_inference": constrained_inference,
        "val": val_final,
        "val_unconstrained": val_unc,
        "test": test_final,
        "test_unconstrained": test_unc,
        "best_val_macro_f1": best_val,
        "n_train": len(train_idx_final),
        "n_val": len(val_idx),
        "n_test": len(split.test),
        "n_classes_fine": n_fine,
        "n_classes_coarse": n_coarse,
        "coarse_class_names": coarse_class_names,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics_out, indent=2, default=str))

    plt.figure()
    plt.plot(
        [h["epoch"] for h in history], [h["train_loss"] for h in history], label="train_loss"
    )
    ax2 = plt.gca().twinx()
    ax2.plot(
        [h["epoch"] for h in history],
        [h["val_macro_f1"] for h in history],
        color="C1",
        label="val macro-F1",
    )
    plt.xlabel("Epoch")
    plt.gca().set_ylabel("Train loss")
    ax2.set_ylabel("Val macro-F1 (constrained)")
    plt.title(f"{model_name} / {split.name} (phase {phase})")
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    diagnostics: Dict = {"phase": phase}

    if phase >= 2 and test_loader is not None:
        p2 = cfg.get("phase2", {})
        if p2.get("gradcam", {}).get("enabled", True):
            max_s = int(p2.get("gradcam", {}).get("max_samples", 12))
            saved = generate_gradcam_report(
                model,
                model_name,
                test_loader,
                index_to_class,
                data_root,
                run_dir,
                max_samples=max_s,
                normalize_wavelength=cfg.get("normalize_wavelength", 0.55),
            )
            diagnostics["gradcam_plots"] = saved

        if p2.get("inference", {}).get("enabled", True):
            csv_path = run_inference(run_dir, cfg=cfg)
            diagnostics["predictions_unlabeled"] = str(csv_path.name)

    (run_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    return metrics_out


def _write_run_manifest(
    run_dir: Path,
    run_name: str,
    phase: int,
    model: str,
    split_name: str,
    records_len: int,
    extra: Optional[dict] = None,
) -> None:
    manifest = {
        "run_id": run_name,
        "phase": phase,
        "model": model,
        "split": split_name,
        "git_hash": _git_hash(),
        "n_manifest_rows": records_len,
    }
    if extra:
        manifest.update(extra)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train asteroid 1D CNN")
    parser.add_argument(
        "--model",
        default="spectranet_lite",
        choices=["spectrum_cnn", "spectranet_lite"],
    )
    parser.add_argument("--split", default="run3", help="run1, run2, run3, or cv5")
    parser.add_argument("--fold", type=int, default=None, help="Fold index for cv5")
    parser.add_argument(
        "--phase",
        type=int,
        default=1,
        choices=[1, 2],
        help="Phase 2 enables augmentation + post-train diagnostics",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Force training augmentation (implies phase 2 behavior for training)",
    )
    parser.add_argument(
        "--tuned-params",
        type=str,
        default=None,
        help="Path to best_params.json from Optuna study",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to a pretrained encoder state_dict (.pt) to warm-start from",
    )
    parser.add_argument(
        "--disable-hierarchical",
        action="store_true",
        help="Disable the coarse head and constrained inference (ablation)",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.tuned_params:
        cfg = _merge_tuned_params(cfg, Path(args.tuned_params))
    if args.disable_hierarchical:
        cfg.setdefault("hierarchical", {})["enabled"] = False
    phase = max(args.phase, 2 if args.augment else args.phase)
    if phase >= 2:
        cfg.setdefault("phase2", {}).setdefault("augmentation", {})["enabled"] = True

    data_root = ROOT / cfg["data_root"]
    records = read_manifest_csv(data_root / cfg["manifest_file"])
    split_result = get_split(args.split, records, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pretrained_path = Path(args.pretrained) if args.pretrained else None

    extra_manifest: Dict[str, str] = {}
    if args.tuned_params:
        extra_manifest["tuned_params"] = args.tuned_params
    if args.pretrained:
        extra_manifest["pretrained"] = args.pretrained

    if args.split.lower() == "cv5":
        folds = split_result
        if args.fold is not None:
            folds = [folds[args.fold]]
        for fold in folds:
            run_name = (
                datetime.now().strftime("%Y%m%d_%H%M%S")
                + f"_{args.model}_{fold.name}_p{phase}"
            )
            run_dir = ROOT / "runs" / run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(ROOT / "configs" / "default.yaml", run_dir / "config.yaml")
            if args.tuned_params:
                shutil.copy(args.tuned_params, run_dir / "best_params.json")
            _write_run_manifest(
                run_dir, run_name, phase, args.model, fold.name, len(records), extra_manifest
            )
            print(f"Training {args.model} on {fold.name} (phase {phase}) -> {run_dir}")
            train_one_split(
                records,
                fold,
                cfg,
                args.model,
                run_dir,
                device,
                phase=phase,
                pretrained_path=pretrained_path,
            )
        return

    split: SplitIndices = split_result
    run_name = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_{args.model}_{split.name}_p{phase}"
    )
    run_dir = ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "configs" / "default.yaml", run_dir / "config.yaml")
    if args.tuned_params:
        shutil.copy(args.tuned_params, run_dir / "best_params.json")
    _write_run_manifest(
        run_dir, run_name, phase, args.model, split.name, len(records), extra_manifest
    )
    print(f"Training {args.model} on {split.name} (phase {phase}) -> {run_dir}")
    train_one_split(
        records,
        split,
        cfg,
        args.model,
        run_dir,
        device,
        phase=phase,
        pretrained_path=pretrained_path,
    )


if __name__ == "__main__":
    main()
