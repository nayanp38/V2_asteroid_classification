"""Self-supervised pretraining via masked-spectrum modeling.

Given the preprocessed ``(2, L)`` array (channel 0 = normalized reflectance,
channel 1 = validity mask), we randomly mask 1–3 wavelength windows totalling
~15 % of the valid points, replace them with the anchor value 1.0, and train
the encoder + a small upsampling decoder to reconstruct the original
reflectance values at the masked positions (MSE on masked positions only).

After training we save the **encoder-only** state dict at
``runs/pretrain_<ts>/encoder.pt`` so it can be loaded into a classifier model
via ``train.py --pretrained ...``. Layer names match the encoder modules in
:class:`SpectraNetLite` (``block1.*``, ``block2.*``, ``block3.*``) and in
:class:`SpectrumCNN` (``features.*``).
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from asteroid_ml.augmentation import AugmentConfig, augment_spectrum
from asteroid_ml.config import ROOT, load_config
from asteroid_ml.models import SpectraNetLite, SpectrumCNN
from asteroid_ml.spectrum_io import PreprocessConfig, preprocess_spectrum

ANCHOR_VALUE = 1.0


class _UnlabeledSpectraDataset(Dataset):
    """Yields preprocessed ``(2, L)`` arrays for every GP file we can find.

    Augmentation is optional and identical to the classifier-time recipe.
    """

    def __init__(
        self,
        paths: List[Path],
        normalize_wavelength: float,
        preprocess_cfg: PreprocessConfig,
        augment: bool,
        augment_cfg: Optional[AugmentConfig],
        seed: int,
    ) -> None:
        self.paths = paths
        self.norm = normalize_wavelength
        self.pre_cfg = preprocess_cfg
        self.augment = augment
        self.aug_cfg = augment_cfg or AugmentConfig()
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = preprocess_spectrum(self.paths[idx], self.norm, cfg=self.pre_cfg)
        if self.augment:
            x = augment_spectrum(x, self.rng, self.aug_cfg)
        return torch.from_numpy(x)


def _collect_gp_paths(cfg: dict) -> List[Path]:
    data_root = ROOT / cfg["data_root"]
    out: List[Path] = []
    for sub in (cfg["demeo_gp_dir"], cfg["binzel_gp_dir"]):
        d = data_root / sub
        if d.is_dir():
            out.extend(sorted(d.glob("a*.txt")))
    return out


def _apply_mask(
    refl: torch.Tensor,
    mask: torch.Tensor,
    rng: np.random.Generator,
    mask_fraction: float,
    min_bins: int,
    max_bins: int,
    n_windows_max: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask 1..n_windows_max windows totalling ~mask_fraction of valid points.

    Returns ``(refl_masked, mask_positions)`` where ``mask_positions`` is a
    boolean tensor (True = position was masked). Only points that were valid
    in the original mask are eligible for masking.
    """
    B, L = refl.shape
    refl_out = refl.clone()
    masked = torch.zeros_like(refl, dtype=torch.bool)
    for b in range(B):
        valid_count = int(mask[b].sum().item())
        if valid_count <= min_bins:
            continue
        target = max(min_bins, int(mask_fraction * valid_count))
        budget = target
        n_windows = int(rng.integers(1, n_windows_max + 1))
        for _ in range(n_windows):
            if budget <= 0:
                break
            hi = max(min_bins + 1, min(max_bins, max(budget, min_bins)) + 1)
            width = int(rng.integers(min_bins, hi))
            width = max(min_bins, min(width, L - 1))
            start = int(rng.integers(0, L - width + 1))
            window = slice(start, start + width)
            window_valid = mask[b, window] > 0.5
            if not window_valid.any():
                continue
            sel = torch.zeros(L, dtype=torch.bool)
            sel[window] = window_valid
            refl_out[b, sel] = ANCHOR_VALUE
            masked[b, sel] = True
            budget -= int(sel.sum().item())
    return refl_out, masked


class _Decoder(nn.Module):
    """Upsample per-position encoder features back to (B, 1, L) reflectance."""

    def __init__(self, channels: int, length: int) -> None:
        super().__init__()
        self.length = length
        self.proj = nn.Conv1d(channels, channels // 2, kernel_size=1)
        self.act = nn.GELU()
        self.out = nn.Conv1d(channels // 2, 1, kernel_size=1)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        h = F.interpolate(feats, size=self.length, mode="linear", align_corners=False)
        h = self.act(self.proj(h))
        return self.out(h).squeeze(1)


def _build_encoder(model_name: str) -> Tuple[nn.Module, nn.Module, int]:
    """Return (encoder_only_module, encoder_features_callable, out_channels)."""
    name = model_name.lower().replace("-", "_")
    if name in ("spectranet_lite", "spectranet"):
        full = SpectraNetLite(n_classes=1)
        encoder = nn.ModuleDict(
            {"block1": full.block1, "block2": full.block2, "block3": full.block3}
        )

        def encode(x: torch.Tensor) -> torch.Tensor:
            return full.encode(x)

        return encoder, encode, full.encoder_channels
    if name in ("spectrum_cnn", "spectrumcnn", "baseline"):
        full = SpectrumCNN(n_outputs=1)
        encoder = nn.ModuleDict({"features": full.features})

        def encode(x: torch.Tensor) -> torch.Tensor:
            return full.encode(x)

        return encoder, encode, full.encoder_channels
    raise ValueError(f"Unknown model: {model_name}")


def _cosine_lr(epoch: int, warmup: int, total: int) -> float:
    if epoch < warmup:
        return (epoch + 1) / max(1, warmup)
    progress = (epoch - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def pretrain(
    model_name: str,
    cfg: dict,
    out_dir: Path,
    device: torch.device,
    n_epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    augment: bool = True,
) -> Path:
    pcfg = cfg.get("pretrain", {})
    n_epochs = int(n_epochs or pcfg.get("max_epochs", 200))
    batch_size = int(batch_size or pcfg.get("batch_size", 32))
    lr = float(pcfg.get("learning_rate", 1e-3))
    wd = float(pcfg.get("weight_decay", 1e-4))
    warmup = int(pcfg.get("warmup_epochs", 5))
    mask_fraction = float(pcfg.get("mask_fraction", 0.15))
    min_bins = int(pcfg.get("mask_min_bins", 5))
    max_bins = int(pcfg.get("mask_max_bins", 30))
    n_windows_max = int(pcfg.get("n_windows_max", 3))

    paths = _collect_gp_paths(cfg)
    if not paths:
        raise RuntimeError("No GP files found for pretraining.")
    L = int((cfg["wavelength_max"] - cfg["wavelength_min"]) / cfg["wavelength_step"]) + 1

    pre_cfg = PreprocessConfig.from_dict(cfg.get("preprocess", {}))
    aug_cfg = AugmentConfig.from_dict(cfg.get("phase2", {}).get("augmentation", {})) if augment else None
    ds = _UnlabeledSpectraDataset(
        paths,
        cfg.get("normalize_wavelength", 0.55),
        pre_cfg,
        augment=augment,
        augment_cfg=aug_cfg,
        seed=cfg.get("training", {}).get("split_seed", 42),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    encoder, encode_fn, enc_channels = _build_encoder(model_name)
    decoder = _Decoder(enc_channels, length=L)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    optim = torch.optim.AdamW(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr, weight_decay=wd
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lr_lambda=lambda e: _cosine_lr(e, warmup, n_epochs)
    )

    rng = np.random.default_rng(cfg.get("training", {}).get("split_seed", 42) + 100)
    history: List[Dict] = []

    for epoch in range(1, n_epochs + 1):
        encoder.train()
        decoder.train()
        epoch_loss = 0.0
        n_seen = 0
        for x in loader:
            x = x.to(device)
            refl = x[:, 0]
            mask = x[:, 1]
            target = refl.clone()
            refl_masked, masked_pos = _apply_mask(
                refl, mask, rng, mask_fraction, min_bins, max_bins, n_windows_max
            )
            if not masked_pos.any():
                continue
            x_in = torch.stack([refl_masked, mask], dim=1)
            feats = encode_fn(x_in)
            pred = decoder(feats)
            loss = F.mse_loss(pred[masked_pos], target[masked_pos])

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 1.0
            )
            optim.step()
            epoch_loss += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
        sched.step()
        avg = epoch_loss / max(n_seen, 1)
        history.append({"epoch": epoch, "loss": avg, "lr": optim.param_groups[0]["lr"]})
        if epoch == 1 or epoch % 10 == 0 or epoch == n_epochs:
            print(f"  pretrain epoch {epoch:03d} loss={avg:.5f} lr={optim.param_groups[0]['lr']:.4g}")

    encoder_path = out_dir / "encoder.pt"
    flat = {}
    for name, mod in encoder.items():
        for k, v in mod.state_dict().items():
            flat[f"{name}.{k}"] = v.detach().cpu()
    torch.save(flat, encoder_path)

    plt.figure()
    plt.plot([h["epoch"] for h in history], [h["loss"] for h in history])
    plt.xlabel("Epoch")
    plt.ylabel("MSE on masked positions")
    plt.title(f"SSL pretraining ({model_name})")
    plt.savefig(out_dir / "pretrain_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    summary = {
        "model_name": model_name,
        "n_files": len(paths),
        "n_epochs": n_epochs,
        "batch_size": batch_size,
        "mask_fraction": mask_fraction,
        "final_loss": history[-1]["loss"] if history else None,
        "encoder_path": str(encoder_path.relative_to(ROOT)),
    }
    (out_dir / "pretrain_summary.json").write_text(json.dumps(summary, indent=2))
    return encoder_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-supervised masked-spectrum pretraining")
    parser.add_argument(
        "--model", default="spectranet_lite", choices=["spectrum_cnn", "spectranet_lite"]
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    out_dir = ROOT / "runs" / f"pretrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text((ROOT / "configs" / "default.yaml").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_path = pretrain(
        args.model,
        cfg,
        out_dir,
        device,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        augment=not args.no_augment,
    )
    print(f"Saved encoder weights to {encoder_path}")


if __name__ == "__main__":
    main()
