"""Mesures de classification : précision, rappel et F1 par classe, moyennes macro et micro, exactitude, matrice de confusion, exactitude top-k."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

Labels = Sequence[Any] | np.ndarray


def classification_metrics(
    y_true: Sequence[Any], y_pred: Sequence[Any], labels: Sequence[Any] | None = None
) -> dict[str, Any]:
    """Toutes les mesures de classification d'une exécution."""
    truth = [str(y) for y in y_true]
    guess = [str(y) for y in y_pred]
    names = [str(label) for label in labels] if labels is not None else sorted(set(truth) | set(guess))
    metrics: dict[str, Any] = {"accuracy": float(accuracy_score(truth, guess)) if truth else 0.0, "n": len(truth)}
    for average in ("macro", "micro"):
        p, r, f, _ = precision_recall_fscore_support(truth, guess, labels=names, average=average, zero_division=0)
        metrics[f"precision_{average}"] = float(p)
        metrics[f"recall_{average}"] = float(r)
        metrics[f"f1_{average}"] = float(f)
    p, r, f, s = precision_recall_fscore_support(truth, guess, labels=names, average=None, zero_division=0)
    metrics["per_class"] = {
        label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i, label in enumerate(names)
    }
    metrics["confusion"] = {"labels": names, "matrix": confusion_matrix(truth, guess, labels=names).tolist()}
    return metrics


def f1_macro(y_true: Labels, y_pred: Labels) -> float:
    """F1 macro (mesure principale de l'attribution)."""
    if len(y_true) == 0:
        return 0.0
    labels = sorted({str(y) for y in y_true} | {str(y) for y in y_pred})
    _, _, f, _ = precision_recall_fscore_support(
        [str(y) for y in y_true], [str(y) for y in y_pred], labels=labels, average="macro", zero_division=0
    )
    return float(f)


def accuracy(y_true: Labels, y_pred: Labels) -> float:
    """Exactitude."""
    if len(y_true) == 0:
        return 0.0
    return float(np.mean([str(a) == str(b) for a, b in zip(y_true, y_pred, strict=True)]))


def topk_accuracy(y_true: Labels, rankings: Sequence[Sequence[tuple[str, float]]], k: int) -> float:
    """Part des exemples dont l'auteur réel figure parmi les `k` premiers."""
    if not y_true:
        return 0.0
    hits = [str(t) in {label for label, _ in ranking[:k]} for t, ranking in zip(y_true, rankings, strict=True)]
    return float(np.mean(hits))
