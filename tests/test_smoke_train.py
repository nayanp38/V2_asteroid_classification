"""Short training smoke test (skipped if no torch/data)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.mark.slow
def test_smoke_train_run3():
    from asteroid_ml.config import load_config
    from asteroid_ml.labels import read_manifest_csv, class_to_index_from_manifest
    from asteroid_ml.train import train_one_split
    from asteroid_ml.splits import split_run3
    import torch

    cfg = load_config()
    data_root = ROOT / cfg["data_root"]
    records = read_manifest_csv(data_root / cfg["manifest_file"])
    if len(records) < 50:
        pytest.skip("manifest too small")
    split = split_run3(records)
    split.train = split.train[:64]
    split.test = split.test[:16]
    cfg = dict(cfg)
    cfg["training"] = {
        **cfg.get("training", {}),
        "max_epochs": 2,
        "early_stop_patience": 5,
        "batch_size": 8,
    }
    run_dir = ROOT / "runs" / "_smoke_test"
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    metrics = train_one_split(
        records, split, cfg, "spectrum_cnn", run_dir, device
    )
    assert "val" in metrics
