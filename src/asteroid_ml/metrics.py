"""Classification metrics for evaluation, including hierarchical helpers."""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

warnings.filterwarnings(
    "ignore",
    message="y_pred contains classes not in y_true",
    category=UserWarning,
    module="sklearn",
)


def coarse_class(name: str) -> str:
    return name[0].upper() if name else ""


def constrained_fine_argmax(
    probs_fine: np.ndarray,
    probs_coarse: Optional[np.ndarray],
    coarse_to_fine_indices: Dict[str, List[int]],
    coarse_to_index: Dict[str, int],
) -> np.ndarray:
    """Constrain the fine argmax to the children of the predicted coarse class.

    If ``probs_coarse`` is ``None`` returns the plain fine argmax.
    """
    if probs_coarse is None or not coarse_to_fine_indices:
        return np.asarray(probs_fine.argmax(axis=1))

    index_to_coarse = {i: c for c, i in coarse_to_index.items()}
    coarse_preds = probs_coarse.argmax(axis=1)
    out = np.empty(len(coarse_preds), dtype=np.int64)
    for i, c_idx in enumerate(coarse_preds):
        coarse = index_to_coarse.get(int(c_idx), None)
        children = coarse_to_fine_indices.get(coarse, []) if coarse is not None else []
        if not children:
            out[i] = int(probs_fine[i].argmax())
        else:
            children_arr = np.asarray(children, dtype=np.int64)
            sub = probs_fine[i, children_arr]
            out[i] = int(children_arr[int(sub.argmax())])
    return out


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: List[str],
    y_prob: np.ndarray | None = None,
    coarse_class_names: Optional[List[str]] = None,
    y_coarse_true: Optional[Sequence[int]] = None,
    y_coarse_pred: Optional[Sequence[int]] = None,
) -> Dict:
    y_true_a = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred)
    n_classes = len(class_names)

    cm = confusion_matrix(
        y_true_a, y_pred_a, labels=list(range(n_classes))
    )
    macro_f1 = float(
        f1_score(y_true_a, y_pred_a, average="macro", zero_division=0)
    )
    acc = float(accuracy_score(y_true_a, y_pred_a))
    bal_acc = float(balanced_accuracy_score(y_true_a, y_pred_a))

    if coarse_class_names is not None and y_coarse_true is not None and y_coarse_pred is not None:
        coarse_acc = float(accuracy_score(y_coarse_true, y_coarse_pred))
        coarse_f1 = float(
            f1_score(y_coarse_true, y_coarse_pred, average="macro", zero_division=0)
        )
    else:
        coarse_true = [coarse_class(class_names[i]) for i in y_true_a]
        coarse_pred = [coarse_class(class_names[i]) for i in y_pred_a]
        coarse_acc = float(accuracy_score(coarse_true, coarse_pred))
        coarse_f1 = float(
            f1_score(coarse_true, coarse_pred, average="macro", zero_division=0)
        )

    top2_acc = None
    if y_prob is not None and y_prob.shape[1] >= 2:
        top2 = np.argsort(y_prob, axis=1)[:, -2:]
        hits = sum(1 for i, t in enumerate(y_true_a) if t in top2[i])
        top2_acc = float(hits / len(y_true_a)) if len(y_true_a) else 0.0

    per_class_recall: Dict[str, float] = {}
    support_warnings: List[str] = []
    for i, name in enumerate(class_names):
        mask = y_true_a == i
        n = int(mask.sum())
        if n == 0:
            per_class_recall[name] = float("nan")
            continue
        if n < 3:
            support_warnings.append(f"{name}: test n={n}")
        per_class_recall[name] = float((y_pred_a[mask] == i).sum() / n)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "coarse_accuracy": coarse_acc,
        "coarse_macro_f1": coarse_f1,
        "top2_accuracy": top2_acc,
        "confusion_matrix": cm.tolist(),
        "per_class_recall": per_class_recall,
        "support_warnings": support_warnings,
        "n_test": int(len(y_true_a)),
    }
