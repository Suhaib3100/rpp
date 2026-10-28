"""Classification metrics.

Macro-averaged metrics are the primary reporting figures. NEU-CLS is exactly
balanced, so macro and micro averages coincide closely, but macro is reported
because it is the metric that would survive a class-imbalanced deployment and
because the base paper reports accuracy only.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


@dataclass
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_f1: float
    per_class_precision: list[float] = field(default_factory=list)
    per_class_recall: list[float] = field(default_factory=list)
    per_class_f1: list[float] = field(default_factory=list)
    per_class_support: list[int] = field(default_factory=list)
    confusion: list[list[int]] = field(default_factory=list)
    n_samples: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (f"acc={self.accuracy:.4f} macroP={self.macro_precision:.4f} "
                f"macroR={self.macro_recall:.4f} macroF1={self.macro_f1:.4f}")


def compute_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    num_classes: int = 6,
) -> ClassificationMetrics:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(range(num_classes))

    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(precision_score(y_true, y_pred, labels=labels,
                                              average="macro", zero_division=0)),
        macro_recall=float(recall_score(y_true, y_pred, labels=labels,
                                        average="macro", zero_division=0)),
        macro_f1=float(f1_score(y_true, y_pred, labels=labels,
                                average="macro", zero_division=0)),
        weighted_f1=float(f1_score(y_true, y_pred, labels=labels,
                                   average="weighted", zero_division=0)),
        per_class_precision=[float(v) for v in p],
        per_class_recall=[float(v) for v in r],
        per_class_f1=[float(v) for v in f1],
        per_class_support=[int(v) for v in sup],
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        n_samples=int(len(y_true)),
    )


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used for accuracy confidence bounds. Preferred over the normal
    approximation because the test set is small (360) and accuracies sit close
    to 1.0, where the normal interval over-covers and can exceed 1.
    """
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))
