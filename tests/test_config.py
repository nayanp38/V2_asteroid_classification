import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asteroid_ml.config import load_config, pretrain_enabled


def test_pretrain_disabled_by_default():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    assert pretrain_enabled(cfg) is False


def test_pretrain_enabled_override():
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg.setdefault("pretrain", {})["enabled"] = True
    assert pretrain_enabled(cfg) is True
