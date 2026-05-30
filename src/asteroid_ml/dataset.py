"""PyTorch dataset over manifest rows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from asteroid_ml.augmentation import AugmentConfig, augment_spectrum
from asteroid_ml.labels import LabelRecord, read_manifest_csv
from asteroid_ml.spectrum_io import PreprocessConfig, preprocess_spectrum


class AsteroidSpectrumDataset(Dataset):
    """Loads preprocessed (2, L) spectra and emits ``(x, y_fine, y_coarse, path)``.

    ``fine_to_coarse_index`` enables the hierarchical head; pass an empty mapping
    to get only the fine label (``y_coarse`` will then be -1 and consumers must
    ignore it).
    """

    def __init__(
        self,
        records: Sequence[LabelRecord],
        data_root: Path,
        class_to_index: Dict[str, int],
        normalize_wavelength: float = 0.55,
        indices: Optional[Sequence[int]] = None,
        augment: bool = False,
        augment_config: Optional[AugmentConfig] = None,
        seed: int = 42,
        preprocess_config: Optional[PreprocessConfig] = None,
        fine_to_coarse_index: Optional[Dict[int, int]] = None,
    ) -> None:
        self.data_root = data_root
        self.class_to_index = class_to_index
        self.normalize_wavelength = normalize_wavelength
        self.augment = augment
        self.augment_config = augment_config or AugmentConfig()
        self.rng = np.random.default_rng(seed)
        if indices is None:
            self.records = list(records)
        else:
            self.records = [records[i] for i in indices]
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.fine_to_coarse_index = fine_to_coarse_index or {}

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, idx: int) -> np.ndarray:
        rec = self.records[idx]
        path = self.data_root / rec.spectrum_path
        return preprocess_spectrum(path, self.normalize_wavelength, cfg=self.preprocess_config)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, str]:
        rec = self.records[idx]
        x = self._load(idx)
        if self.augment:
            x = augment_spectrum(x, self.rng, self.augment_config)
        label = self.class_to_index[rec.class_bd]
        coarse_label = int(self.fine_to_coarse_index.get(label, -1))
        return (
            torch.from_numpy(x),
            label,
            coarse_label,
            rec.spectrum_path,
        )


def load_manifest_dataset(
    data_root: Path,
    manifest_path: Path,
    class_to_index: Dict[str, int],
    normalize_wavelength: float = 0.55,
    augment: bool = False,
    augment_config: Optional[AugmentConfig] = None,
    preprocess_config: Optional[PreprocessConfig] = None,
    fine_to_coarse_index: Optional[Dict[int, int]] = None,
) -> AsteroidSpectrumDataset:
    records = read_manifest_csv(manifest_path)
    return AsteroidSpectrumDataset(
        records,
        data_root,
        class_to_index,
        normalize_wavelength,
        augment=augment,
        augment_config=augment_config,
        preprocess_config=preprocess_config,
        fine_to_coarse_index=fine_to_coarse_index,
    )


def compute_class_weights(
    labels: Sequence[int],
    n_classes: int,
    mode: str = "linear",
    beta: float = 0.999,
) -> torch.Tensor:
    """Per-class loss weights.

    ``mode="linear"`` reproduces the original ``N / (K * n_c)`` weighting.
    ``mode="effective"`` is Cui et al. 2019 "effective number of samples":
    ``w_c = (1 - beta) / (1 - beta**n_c)`` then normalized so weights average 1.
    """
    counts = np.bincount(list(labels), minlength=n_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    if mode == "effective":
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    else:
        weights = counts.sum() / (n_classes * counts)
    weights = weights * (n_classes / weights.sum())
    return torch.tensor(weights, dtype=torch.float32)
