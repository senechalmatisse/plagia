"""Mesures de détection binaire (paires de passages, vérification d'auteur) et de
recherche : précision, rappel, F1, précision moyenne, AUC, c@1, rappel et exactitude @k."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def binary_metrics(
    y_true: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray, threshold: float
) -> dict[str, float]:
    """Mesures au seuil donné, précision moyenne et AUC (si les deux classes sont présentes)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    predicted = (s >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, predicted, average="binary", zero_division=0)
    both = 0 < y.sum() < len(y)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "average_precision": float(average_precision_score(y, s)) if both else 0.0,
        "auc": float(roc_auc_score(y, s)) if both else 0.5,
        "threshold": float(threshold),
        "positives": int(y.sum()),
        "negatives": int(len(y) - y.sum()),
    }


def c_at_1(y_true: Sequence[int], decisions: Sequence[int | None]) -> float:
    """c@1 (Peñas et Rodrigo, 2011) : les abstentions (`None`) sont créditées au taux
    d'exactitude : (n_c + n_u · n_c / n) / n."""
    n = len(y_true)
    if n == 0:
        return 0.0
    correct = sum(1 for t, d in zip(y_true, decisions, strict=True) if d is not None and int(d) == int(t))
    unanswered = sum(1 for d in decisions if d is None)
    return (correct + unanswered * correct / n) / n


def verification_metrics(
    y_true: Sequence[int], probabilities: Sequence[float], decisions: Sequence[int | None]
) -> dict[str, float]:
    """AUC, F1 (abstentions comptées comme erreurs) et c@1 de la vérification d'auteur."""
    y = np.asarray(y_true, dtype=int)
    answered = np.array([d if d is not None else 1 - t for t, d in zip(y, decisions, strict=True)], dtype=int)
    _, _, f, _ = precision_recall_fscore_support(y, answered, average="binary", zero_division=0)
    both = 0 < y.sum() < len(y)
    return {
        "auc": float(roc_auc_score(y, probabilities)) if both else 0.5,
        "f1": float(f),
        "c_at_1": c_at_1(list(y), list(decisions)),
        "abstentions": int(sum(d is None for d in decisions)),
    }


def accuracy_at_k(truth: Sequence[str], rankings: Sequence[Sequence[str]], k: int) -> float:
    """Part des passages dont la vraie source figure parmi les `k` premières."""
    if not truth:
        return 0.0
    return float(np.mean([t in list(r)[:k] for t, r in zip(truth, rankings, strict=True)]))


def recall_at_k(relevant: Sequence[set[str]], retrieved: Sequence[Sequence[str]], k: int) -> float:
    """Rappel de la recherche des sources parmi les `k` meilleures candidates (micro)."""
    found = total = 0
    for wanted, got in zip(relevant, retrieved, strict=True):
        top = set(list(got)[:k])
        found += len(wanted & top)
        total += len(wanted)
    return found / total if total else 1.0


def mean_or_nan(values: Sequence[Any]) -> float:
    """Moyenne (NaN si vide)."""
    return float(np.mean(values)) if len(values) else float("nan")
