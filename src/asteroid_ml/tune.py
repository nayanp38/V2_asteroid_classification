"""Optuna hyperparameter search for asteroid 1D CNNs (hierarchical aware)."""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import optuna
from optuna.trial import Trial

from asteroid_ml.config import ROOT, load_config, pretrain_enabled
from asteroid_ml.labels import read_manifest_csv
from asteroid_ml.splits import get_split
from asteroid_ml.train import _git_hash, train_one_split


def _apply_trial_params(cfg: dict, trial: Trial) -> dict:
    cfg = deepcopy(cfg)
    tcfg = cfg.setdefault("training", {})
    tcfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
    tcfg["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    tcfg["batch_size"] = trial.suggest_categorical("batch_size", [8, 16, 32])
    tcfg["focal_gamma"] = trial.suggest_float("focal_gamma", 0.0, 3.0)
    tcfg["label_smoothing"] = trial.suggest_float("label_smoothing", 0.0, 0.2)
    tcfg["max_epochs"] = cfg.get("phase2", {}).get("optuna", {}).get("max_epochs", 30)
    tcfg["early_stop_patience"] = cfg.get("phase2", {}).get("optuna", {}).get(
        "early_stop_patience", 10
    )

    aug = cfg.setdefault("phase2", {}).setdefault("augmentation", {})
    aug["enabled"] = True
    aug["noise_std"] = trial.suggest_float("noise_std", 0.005, 0.04)
    aug["smooth_sigma"] = trial.suggest_float("smooth_sigma", 0.0, 1.5)

    hier = cfg.setdefault("hierarchical", {})
    hier["coarse_weight"] = trial.suggest_float("coarse_weight", 0.0, 1.5)
    return cfg


def run_study(
    model_name: str = "spectranet_lite",
    split_name: str = "run3",
    n_trials: int = 20,
    study_name: str | None = None,
    pretrained_path: Path | None = None,
) -> Path:
    cfg = load_config()
    records = read_manifest_csv(ROOT / cfg["data_root"] / cfg["manifest_file"])
    split = get_split(split_name, records, cfg)
    if isinstance(split, list):
        split = split[0]

    study_dir = ROOT / "runs" / (
        study_name or f"optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_name}_{split_name}"
    )
    study_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "configs" / "default.yaml", study_dir / "config_base.yaml")

    def objective(trial: Trial) -> float:
        trial_cfg = _apply_trial_params(cfg, trial)
        trial_run = study_dir / f"trial_{trial.number:03d}"
        trial_run.mkdir(exist_ok=True)
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metrics = train_one_split(
            records,
            split,
            trial_cfg,
            model_name,
            trial_run,
            device,
            phase=2,
            pretrained_path=pretrained_path,
        )
        trial.set_user_attr("test_macro_f1", metrics.get("test", {}).get("macro_f1"))
        return float(metrics["best_val_macro_f1"])

    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "model": model_name,
        "split": split_name,
    }
    (study_dir / "best_params.json").write_text(json.dumps(best, indent=2))

    trials_log = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": str(t.state),
        }
        for t in study.trials
    ]
    (study_dir / "trials.json").write_text(json.dumps(trials_log, indent=2))

    manifest = {
        "study_dir": str(study_dir),
        "phase": 2,
        "git_hash": _git_hash(),
        **best,
    }
    (study_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Best val macro-F1: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Saved to {study_dir}")
    return study_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search")
    parser.add_argument("--model", default="spectranet_lite")
    parser.add_argument("--split", default="run3")
    parser.add_argument("--n-trials", type=int, default=None)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--pretrained", default=None,
                        help="Optional pretrained encoder used in each trial")
    args = parser.parse_args()
    cfg = load_config()
    n_trials = args.n_trials or cfg.get("phase2", {}).get("optuna", {}).get("n_trials", 20)
    pretrained = Path(args.pretrained) if args.pretrained else None
    if pretrained and not pretrain_enabled(cfg):
        print("pretrain.enabled is false; ignoring --pretrained for Optuna trials.")
        pretrained = None
    run_study(args.model, args.split, n_trials, args.study_name, pretrained_path=pretrained)


if __name__ == "__main__":
    main()
