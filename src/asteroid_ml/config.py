"""Load project configuration from configs/default.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"


def load_config(path: Path | None = None) -> Dict[str, Any]:
    path = path or DEFAULT_CONFIG_PATH
    text = path.read_text()
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required. Install with: pip install pyyaml"
        ) from exc
    return yaml.safe_load(text)


def pretrain_enabled(cfg: Dict[str, Any]) -> bool:
    """Whether SSL encoder pretraining and warm-start are allowed."""
    return bool(cfg.get("pretrain", {}).get("enabled", False))
