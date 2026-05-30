# Agent Handoff: Asteroid 1D CNN (v2)

**Purpose:** Onboard a new agent session to continue work on the Bus–DeMeo asteroid spectral classification project (journal revision).
**Workspace:** `/Users/nayanpatel/Asteroid Paper/v2`
**Package version:** `asteroid_ml` 0.3.0
**Last updated:** 2026-05-29

---

## 1. Project goal

Replace the authors' original **2D CNN on rasterized spectrum plots** (128×128 grayscale images) with a **1D CNN on raw reflectance vectors**, addressing reviewer feedback that:

- Convolutions should run along the **wavelength axis** on reflectance values, not on fake 2D images.
- ML practice should include tuning, augmentation, explainability, and clear data accounting.

**Original paper (for context only):**
`/Users/nayanpatel/Asteroid Paper/aastex701-1/Asteroid_Classification.tex`
- ~627 spectra (DeMeo 2009 + Binzel 2019), 25 BD classes (after artifact filtering, 24 classes remain in our manifest)
- Reported accuracies: Run 1 ~77 %, Run 2 ~79 %, Run 3 ~84 % (2D CNN; **not comparable** directly to v2 numbers)

**Reference architecture inspiration:** SpectraNet-1D from AppleCiDEr (Junell et al. 2025); implemented in scaled-down form as `SpectraNetLite`.

---

## 2. Repository layout

```
v2/
├── AGENT_HANDOFF.md          # this file
├── README.md                 # user-facing setup + methodology + commands
├── requirements.txt
├── pytest.ini
├── configs/default.yaml      # paths, classes, training, hierarchical, pretrain
├── data/
│   ├── demeotax.tab          # DeMeo BD labels — PRIMARY DeMeo labels
│   ├── Binzel_classes.txt    # Binzel labels
│   ├── labels_manifest.csv   # built; ~567 rows after artifact filter
│   ├── manifest_report.json  # build stats (incl. dropped_low_quality)
│   ├── alias_log.txt
│   ├── DeMeo2009data_gp/
│   └── Binzel2019data_gp/
├── src/asteroid_ml/          # main package (PYTHONPATH=src)
├── scripts/
│   ├── build_manifest.py
│   ├── interpolate_spectra_gp.py
│   ├── audit_artifacts.py
│   └── run_simplified.sh     # end-to-end orchestrator
├── tests/
├── runs/                     # gitignored training outputs
└── releases/                 # gitignored export bundles
```

---

## 3. Data pipeline

### 3.1 GP files

`wavelength reflectance gp_std`, 401 rows on `0.45–2.45 µm` (Δλ = 0.005). NaN reflectance outside coverage; `gp_std` is used for artifact masking.

### 3.2 Labels & manifest

- DeMeo: `demeotax.tab` (fixed-width + date regex parse).
- Binzel: `Binzel_classes.txt` (`asteroid_id class`).
- `classes.txt` is **not** training labels.
- Aliases (`configs/default.yaml.class_aliases`): `Sw→S, Sqw→Sq, Srw→Sr, Vw→V, Svw→Sv`.
- Multi-class Binzel rows: primary class only (no duplication).

Manifest is built by `scripts/build_manifest.py` and **drops spectra with valid fraction < `preprocess.min_valid_fraction`** (default 0.6) under the new artifact mask.

### 3.3 Preprocessing (`spectrum_io.py`)

Output `(2, 401)`:

- Channel 0: reflectance normalized to 1.0 at 0.55 µm.
- Channel 1: validity mask (1 = trustworthy point).

Validity combines three rules (configurable via `preprocess.*`):

1. `np.isfinite(reflectance)`.
2. `gp_std ≤ std_mask_k × median(gp_std)` (default `k = 6.0`, floor 0.02).
3. *Not* part of a constant-value "frozen run" of length ≥ `frozen_run_min` (default 8 bins, `frozen_eps = 1e-4`).

Masked points have reflectance set to `artifact_fill_value` (default 1.0 = anchor).

### 3.4 Coarse / fine mapping (`labels.coarse_class_to_index`)

`configs/default.yaml.coarse_groups` defines parent complexes (S, C, X, V, D, A, K, L, O, Q, R, T, B, Cgh). The mapping yields `coarse_to_index`, `fine_to_coarse_index`, and `coarse_to_fine_indices` and is persisted in every checkpoint.

---

## 4. Models (`src/asteroid_ml/models/`)

Both models share the same head structure:

```
encoder  ─►  GAP  ─►  head_shared  ─►  head_fine
                                   └►  head_coarse  (optional, if n_coarse > 0)
```

- **`SpectrumCNN`** — 3-layer baseline (`features`), conv kernels 5/5/31, GELU.
- **`SpectraNetLite`** — 3× `SpectraBlock` with kernels 5/15/31, dual max+avg pool, channel gate, residual skip; primary model.

`build_model(name, n_classes, n_coarse=0)` is the only constructor.

---

## 5. Splits (`splits.py`)

| Split | Train | Test |
|-------|-------|------|
| run1 | DeMeo + 4 Xn IDs from config | Binzel rows whose `asteroid_id` not in train |
| run2 | `dynamical_group != mars_crosser` | `dynamical_group == mars_crosser` |
| run3 | 80 % stratified (deduped per asteroid_id) | 20 % stratified |
| cv5 | 5-fold stratified (deduped) | Out-of-fold per fold |

---

## 6. Training (`train.py`)

```bash
export PYTHONPATH=src
# Phase 1 (no augmentation, no pretrain): for baselines/ablations
python -m asteroid_ml.train --model spectranet_lite --split run3
# Full simplified pipeline (recommended): hierarchical + augmentation (SSL pretrain disabled)
python -m asteroid_ml.train --model spectranet_lite --split run3 --phase 2
```

Loss: `FocalSmoothedCE(weight=effective_number)` on the fine head plus `coarse_weight × FocalSmoothedCE` on the coarse head (default `coarse_weight = 0.5`). Optimizer: AdamW + cosine LR with linear warmup + gradient clip 1.0.

Early stop on **val macro-F1 (constrained)**. Inner-CV split: `training.val_fraction` (default 0.25) of train indices held out as val.

**Hierarchical inference:** constrained-argmax (fine prediction restricted to children of predicted coarse class) when `hierarchical.constrained_inference = true`; both constrained and unconstrained metrics are reported.

**Checkpoint keys:** `model_state_dict`, `class_to_index`, `model_name`, `split_name`, `normalize_wavelength`, `phase`, `augmentation_enabled`, `hierarchical`, `coarse_to_index`, `fine_to_coarse_index`, `coarse_to_fine_indices`, `preprocess`.

---

## 7. Self-supervised pretraining (`pretrain.py`) — **disabled by default**

`pretrain.enabled: false` in `configs/default.yaml`. `pretrain.py` exits unless `--force`; `train.py --pretrained` is ignored while disabled.

When enabled, masked-spectrum modelling on **every** GP file (labeled + unlabeled):

1. Preprocess to `(2, L)` exactly as for training.
2. Randomly mask 1–3 wavelength windows totalling ~15 % of valid points; fill with the anchor value.
3. Pass through the encoder + a small upsampling decoder (`F.interpolate(linear) → 1×1 conv`).
4. Loss: `MSE(reconstruction, original_reflectance)` **on masked positions only**.
5. Save **encoder-only** state dict (layer names match `block*` / `features` modules in the classifier) at `runs/pretrain_<ts>_<model>/encoder.pt`.

In `train.py`, `--pretrained encoder.pt` warm-starts the encoder via `load_state_dict(..., strict=False)`. Set `pretrain.freeze_encoder_epochs` > 0 to freeze the encoder for the first N epochs of fine-tuning.

---

## 8. Evaluation (`evaluate.py`, `metrics.py`)

```bash
python -m asteroid_ml.evaluate --run runs/<run_id>
python -m asteroid_ml.evaluate --run runs/<run_id> --gradcam --gradcam-samples 12
```

`metrics.compute_metrics` produces: accuracy, balanced accuracy, macro/weighted F1, coarse-class accuracy (constrained mode), top-2 accuracy, per-class recall, confusion matrix.

`metrics.constrained_fine_argmax` enforces the coarse → fine restriction at inference.

---

## 9. Augmentation (`augmentation.py`, train-only, phase ≥ 2)

Only two transforms remain after the v0.3 simplification:

- Mask-aware **Gaussian noise** on reflectance (`p_noise`, `noise_std`).
- Mask-aware **Gaussian smoothing** (`p_smooth`, `smooth_sigma`).

Removed in v0.3: circular wavelength shift, slope jitter, band-depth scaling, SpecAugment band masking, mixup, physics-informed feature branch.

---

## 10. End-to-end run

```bash
bash scripts/run_simplified.sh            # no SSL pretrain (pretrain.enabled: false)
```

Runs in this order:

1. For each model, train `run1`, `run2`, `run3`, `cv5` with `--phase 2` (random encoder init).

Outputs go to `runs/`; full session log at `runs/_simplified_session.log`.

---

## 11. Tests

```bash
PYTHONPATH=src MPLBACKEND=Agg pytest tests/ -q
PYTHONPATH=src MPLBACKEND=Agg pytest tests/ -q -m slow   # adds smoke training
```

Files: `test_labels.py`, `test_spectrum_io.py`, `test_models.py`, `test_augmentation.py`, `test_smoke_train.py`.

`MPLBACKEND=Agg` is required on macOS to avoid the GUI-backend crash.

---

## 12. Results (v0.3 simplified pipeline)

All numbers from `metrics.json` after running `bash scripts/run_simplified.sh 100`. SSL pretrain: 100 epochs masked-spectrum reconstruction on every GP file (labeled + unlabeled). Classifier: AdamW + cosine LR, plain class-weighted CE (linear weights), hierarchical loss with `coarse_weight = 0.3`, constrained fine inference.

| Model           | Split      | Test macro-F1 (constr / unc) | Test acc | Test coarse acc | Top-2 acc |
|-----------------|-----------|------------------------------|----------|-----------------|-----------|
| spectranet_lite | run1      | 0.126 / 0.138                | 0.211    | 0.282           | 0.335     |
| spectranet_lite | run2      | 0.155 / 0.185                | 0.246    | 0.360           | 0.403     |
| spectranet_lite | run3      | 0.236 / 0.221                | 0.288    | 0.559           | 0.495     |
| spectranet_lite | cv5 mean  | 0.255 ± 0.048 / 0.276 ± 0.046 | 0.320 ± 0.027 | 0.465 ± 0.047 | 0.501 ± 0.032 |
| spectrum_cnn    | run1      | 0.142 / 0.157                | 0.206    | 0.263           | 0.354     |
| spectrum_cnn    | run2      | 0.179 / 0.166                | 0.171    | 0.209           | 0.284     |
| spectrum_cnn    | run3      | 0.163 / 0.144                | 0.153    | 0.162           | 0.189     |
| spectrum_cnn    | cv5 mean  | 0.164 ± 0.020 / 0.172 ± 0.036 | 0.146 ± 0.018 | 0.206 ± 0.043 | 0.226 ± 0.035 |

v0.2 baseline reference (spectranet_lite on run3 only, no hierarchical / SSL / artifact filter):
`val macro-F1 = 0.209, test macro-F1 = 0.228, test accuracy = 0.288, val coarse_accuracy (first-letter) = 0.594`.

Reusable artifacts:

- Pretrained encoder weights: `runs/pretrain_<ts>_spectranet_lite/encoder.pt` and `runs/pretrain_<ts>_spectrum_cnn/encoder.pt`.
- Confusion matrices: `runs/<run>/confusion_matrix.png` (constrained) and `runs/<run>/confusion_matrix_unconstrained.png`.
- Grad-CAM overlays: `runs/<run>/gradcam/*.png`.
- Unlabeled predictions: `runs/<run>/predictions_unlabeled.csv`.

## 13. Suggested next work

1. Add the methodology paragraph to `aastex701-1/Asteroid_Classification.tex` (hierarchical fine + coarse head, GP-std artifact masking, masked-spectrum pretraining, constrained-argmax inference).
2. Regenerate confusion-matrix figure for the paper from `runs/20260529_215148_spectranet_lite_run3_p2/confusion_matrix.png` (the strongest single-split run).
3. Optional Optuna study with the simplified search space (`tune.py`) on `--split cv5 --fold 0` to fine-tune `learning_rate`, `weight_decay`, `coarse_weight`, augmentation noise.

---

## 14. Key design decisions (do not undo without discussion)

- **No 2D rasterization** — core reviewer response.
- **`classes.txt` is not training labels** — use `labels_manifest.csv` built from demeotax + Binzel.
- **Primary class only** — avoids duplicate-spectrum leakage in run3/CV.
- **run3 / cv5 dedupe by asteroid_id** — one spectrum per asteroid (demeo preferred).
- **Artifact-aware masking is on by default** — `preprocess.std_mask_enabled` and `preprocess.frozen_run_enabled` both default to `true`. Manifest will drop low-valid-fraction spectra accordingly.
- **Hierarchical head is on by default** — disable with `--disable-hierarchical` only for ablations.
- **Augmentation is intentionally minimal** — only Gaussian noise + smoothing, for paper clarity.
- **No ensembling** — single-model results only (decided in v0.3).

---

## 15. Quick command reference

```bash
export PYTHONPATH=src
python scripts/build_manifest.py
python -m asteroid_ml.pretrain --model spectranet_lite --epochs 100
python -m asteroid_ml.train --phase 2 --model spectranet_lite --split run3 \
       --pretrained runs/pretrain_<ts>_spectranet_lite/encoder.pt
python -m asteroid_ml.evaluate --run runs/<run_id> --gradcam
python -m asteroid_ml.infer --run runs/<run_id>
python -m asteroid_ml.export_bundle --run runs/<run_id>
bash scripts/run_simplified.sh 100
```

---

## 16. File → responsibility map

| File | Responsibility |
|------|----------------|
| `labels.py` | Parse demeotax/Binzel, aliases, coarse mapping, build/read manifest |
| `spectrum_io.py` | Load GP file, GP-std + frozen-run mask, normalize @ 0.55, `(2, L)` tensor |
| `dataset.py` | PyTorch Dataset (returns `(x, y_fine, y_coarse, path)`), class weights |
| `augmentation.py` | Gaussian noise + smoothing only |
| `splits.py` | run1/2/3/cv5 index lists |
| `losses.py` | Focal loss + label smoothing CE |
| `train.py` | Training loop, dual-head loss, cosine LR, grad clip, pretrained warm-start |
| `pretrain.py` | Masked-spectrum SSL pretraining |
| `evaluate.py` | Test metrics + confusion matrices (constrained + unconstrained) + optional Grad-CAM |
| `metrics.py` | accuracy, macro-F1, coarse, top-2, constrained fine argmax, CM |
| `tune.py` | Optuna study (simplified search space) |
| `gradcam.py` | 1D Grad-CAM + plots |
| `infer.py` | Unlabeled GP files, taxonomic-constrained predictions + confidence CSV |
| `export_bundle.py` | Portable `releases/` package |
| `config.py` | Load `configs/default.yaml` |

---

*End of handoff document.*
