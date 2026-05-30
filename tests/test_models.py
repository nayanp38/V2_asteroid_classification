import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asteroid_ml.models import SpectrumCNN, SpectraNetLite, build_model  # noqa: F401


def test_forward_shapes_fine_only():
    x = torch.randn(4, 2, 401)
    for name in ("spectrum_cnn", "spectranet_lite"):
        m = build_model(name, n_classes=25)
        logits_fine, logits_coarse = m(x)
        assert logits_fine.shape == (4, 25)
        assert logits_coarse is None


def test_forward_shapes_with_coarse():
    x = torch.randn(4, 2, 401)
    for name in ("spectrum_cnn", "spectranet_lite"):
        m = build_model(name, n_classes=25, n_coarse=13)
        logits_fine, logits_coarse = m(x)
        assert logits_fine.shape == (4, 25)
        assert logits_coarse is not None
        assert logits_coarse.shape == (4, 13)
