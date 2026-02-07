from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, auc as sk_auc


@dataclass
class AUCResult:
    per_class: List[float]
    mean_auc: float
    valid_mask: List[bool]


def compute_auc_per_class(
    y_true: np.ndarray,  # (N,C)
    y_score: np.ndarray, # (N,C)
) -> AUCResult:
    C = y_true.shape[1]
    per = []
    valid = []
    for c in range(C):
        yt = y_true[:, c]
        ys = y_score[:, c]
        # Need both classes present
        if (yt.max() == yt.min()):
            per.append(float("nan"))
            valid.append(False)
            continue
        try:
            per.append(float(roc_auc_score(yt, ys)))
            valid.append(True)
        except Exception:
            per.append(float("nan"))
            valid.append(False)
    valid_vals = [v for v, ok in zip(per, valid) if ok and np.isfinite(v)]
    mean = float(np.mean(valid_vals)) if len(valid_vals) else float("nan")
    return AUCResult(per_class=per, mean_auc=mean, valid_mask=valid)


def compute_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
):
    """
    Returns list of (fpr, tpr, auc) per class. Invalid classes -> None.
    """
    C = y_true.shape[1]
    curves = []
    for c in range(C):
        yt = y_true[:, c]
        ys = y_score[:, c]
        if yt.max() == yt.min():
            curves.append(None)
            continue
        fpr, tpr, _ = roc_curve(yt, ys)
        curves.append((fpr, tpr, float(sk_auc(fpr, tpr))))
    return curves
