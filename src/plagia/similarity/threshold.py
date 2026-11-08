"""Choix d'un seuil de décision sur le jeu de validation : maximisation de F1 ou précision cible, avec la courbe précision-rappel correspondante."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve


@dataclass(frozen=True)
class ThresholdChoice:
    """Seuil retenu et performances associées sur le jeu de validation."""

    threshold: float
    precision: float
    recall: float
    f1: float
    criterion: str

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON."""
        return asdict(self)


def select_threshold(
    y_true: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    criterion: str = "f1",
    target_precision: float = 0.9,
) -> ThresholdChoice:
    """Seuil maximisant F1, ou plus petit seuil atteignant `target_precision` (rappel maximal)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        threshold = float(np.median(s)) if len(s) else 0.5
        return ThresholdChoice(threshold, 0.0, 0.0, 0.0, f"{criterion} (dégénéré)")
    precision, recall, thresholds = precision_recall_curve(y, s)
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros_like(precision), where=(precision + recall) > 0
    )
    if criterion == "precision":
        ok = np.where(precision >= target_precision)[0]
        index = int(ok[np.argmax(recall[ok])]) if len(ok) else int(np.argmax(precision))
    else:
        index = int(np.argmax(f1))
    # Seuil au milieu de l'intervalle vide sous le score retenu : mêmes décisions sur la
    # validation, marge maximale pour les exemples de test.
    chosen = float(thresholds[index])
    below = s[s < chosen]
    threshold = (chosen + float(below.max())) / 2 if below.size else chosen
    return ThresholdChoice(threshold, float(precision[index]), float(recall[index]), float(f1[index]), criterion)


def pr_curve(y_true: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> dict[str, list[float]]:
    """Points de la courbe précision-rappel (pour les rapports)."""
    precision, recall, thresholds = precision_recall_curve(
        np.asarray(y_true, dtype=int), np.asarray(scores, dtype=float)
    )
    return {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": thresholds.tolist()}
