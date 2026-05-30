"""Train/test split definitions for Runs 1–3 and 5-fold CV."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from asteroid_ml.labels import LabelRecord, normalize_asteroid_id


@dataclass
class SplitIndices:
    train: List[int]
    test: List[int]
    name: str
    description: str = ""


def _indices_by_filter(
    records: Sequence[LabelRecord], predicate
) -> List[int]:
    return [i for i, r in enumerate(records) if predicate(r)]


def dedupe_by_asteroid(records: Sequence[LabelRecord]) -> Tuple[List[LabelRecord], List[int]]:
    """Keep first occurrence per asteroid_id (prefer demeo over binzel)."""
    seen: Set[str] = set()
    deduped: List[LabelRecord] = []
    index_map: List[int] = []
    source_rank = {"demeo": 0, "binzel": 1, "marsset": 2}
    order = sorted(
        range(len(records)),
        key=lambda i: (source_rank.get(records[i].source, 9), i),
    )
    for i in order:
        aid = records[i].asteroid_id
        if aid in seen:
            continue
        seen.add(aid)
        deduped.append(records[i])
        index_map.append(i)
    return deduped, index_map


def split_run1(
    records: Sequence[LabelRecord],
    xn_training_ids: Sequence[str],
) -> SplitIndices:
    """Train: DeMeo spectra + Binzel Xn; test: Binzel (non-overlap IDs)."""
    xn_set = {normalize_asteroid_id(x) for x in xn_training_ids}
    train_ids: Set[str] = set()
    for r in records:
        if r.source == "demeo":
            train_ids.add(r.asteroid_id)
    train_ids.update(xn_set)

    train = _indices_by_filter(
        records,
        lambda r: r.source == "demeo" or r.asteroid_id in xn_set,
    )
    test = _indices_by_filter(
        records,
        lambda r: r.source == "binzel" and r.asteroid_id not in train_ids,
    )
    return SplitIndices(
        train=train,
        test=test,
        name="run1",
        description="DeMeo train + Xn; Binzel NEA/Mars-crosser test",
    )


def split_run2(records: Sequence[LabelRecord]) -> SplitIndices:
    """Train: non–Mars-crosser; test: Mars-crosser (Binzel-only heuristic)."""
    train = _indices_by_filter(
        records, lambda r: r.dynamical_group != "mars_crosser"
    )
    test = _indices_by_filter(
        records, lambda r: r.dynamical_group == "mars_crosser"
    )
    return SplitIndices(
        train=train,
        test=test,
        name="run2",
        description="Train DeMeo+Binzel NEA; test Mars-crossers",
    )


def split_run3(
    records: Sequence[LabelRecord],
    train_fraction: float = 0.8,
    min_class_size: int = 5,
    seed: int = 42,
) -> SplitIndices:
    """Stratified 80/20 per class_bd on deduplicated asteroids."""
    deduped, orig_indices = dedupe_by_asteroid(records)
    labels = [r.class_bd for r in deduped]
    classes = sorted(set(labels))
    rng = np.random.default_rng(seed)

    train_local: List[int] = []
    test_local: List[int] = []

    by_class: Dict[str, List[int]] = {c: [] for c in classes}
    for i, lab in enumerate(labels):
        by_class[lab].append(i)

    for _cls, idxs in by_class.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        if n < min_class_size:
            n_test = 1 if n > 1 else 0
        else:
            n_test = max(1, int(round(n * (1 - train_fraction))))
        test_local.extend(idxs[:n_test])
        train_local.extend(idxs[n_test:])

    train = [orig_indices[i] for i in train_local]
    test = [orig_indices[i] for i in test_local]
    return SplitIndices(
        train=train,
        test=test,
        name="run3",
        description="80/20 stratified by class (deduped asteroids)",
    )


def split_cv5(
    records: Sequence[LabelRecord],
    n_splits: int = 5,
    seed: int = 42,
) -> List[SplitIndices]:
    """Stratified K-fold on deduplicated asteroids."""
    deduped, orig_indices = dedupe_by_asteroid(records)
    y = [r.class_bd for r in deduped]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: List[SplitIndices] = []
    for fold_i, (train_local, test_local) in enumerate(skf.split(np.zeros(len(y)), y)):
        train = [orig_indices[i] for i in train_local]
        test = [orig_indices[i] for i in test_local]
        folds.append(
            SplitIndices(
                train=train,
                test=test,
                name=f"cv5_fold{fold_i}",
                description=f"Stratified 5-fold CV fold {fold_i}",
            )
        )
    return folds


def get_split(
    name: str,
    records: Sequence[LabelRecord],
    cfg: dict,
) -> SplitIndices | List[SplitIndices]:
    name = name.lower()
    if name == "run1":
        return split_run1(records, cfg.get("run1_xn_training_ids", []))
    if name == "run2":
        return split_run2(records)
    if name == "run3":
        r3 = cfg.get("run3", {})
        return split_run3(
            records,
            train_fraction=r3.get("train_fraction", 0.8),
            min_class_size=r3.get("min_class_size_for_80_20", 5),
            seed=cfg.get("training", {}).get("split_seed", 42),
        )
    if name == "cv5":
        cv = cfg.get("cv5", {})
        return split_cv5(
            records,
            n_splits=cv.get("n_splits", 5),
            seed=cv.get("seed", 42),
        )
    raise ValueError(f"Unknown split: {name}")
