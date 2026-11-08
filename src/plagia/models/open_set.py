"""Attribution en ensemble ouvert : « auteur inconnu » sous le seuil τ."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from plagia.models.base import Classifier, Explanation

UNKNOWN = "<unknown>"


class OpenSetClassifier(Classifier):
    """Rejette la prédiction (`<unknown>`) quand la confiance maximale est inférieure à τ."""

    def __init__(self, base: Classifier | None = None, threshold: float = 0.5, unknown_label: str = UNKNOWN) -> None:
        self.base = base
        self.threshold = threshold
        self.unknown_label = unknown_label

    def fit(self, X: Any, y: Sequence[Any]) -> OpenSetClassifier:
        if self.base is None:
            raise ValueError("Aucun classificateur de base.")
        self.base.fit(X, y)
        self.classes_ = self.base.classes_
        return self

    def predict_scores(self, X: Any) -> np.ndarray:
        assert self.base is not None
        return self.base.predict_scores(X)

    def predict(self, X: Any) -> np.ndarray:
        scores = self.predict_scores(X)
        best = np.argmax(scores, axis=1)
        labels = np.asarray(self.classes_, dtype=object)[best]
        labels[scores.max(axis=1) < self.threshold] = self.unknown_label
        return labels

    def calibrate(self, X: Any, y: Sequence[Any], candidates: Sequence[float] | None = None) -> float:
        """Choisit τ maximisant le F1 macro (auteurs connus et classe inconnue) sur `(X, y)`.

        Les étiquettes de `y` absentes de l'entraînement sont comptées comme inconnues.
        """
        from sklearn.metrics import f1_score

        known = set(map(str, self.classes_))
        truth = np.array([label if str(label) in known else self.unknown_label for label in y], dtype=object)
        scores = self.predict_scores(X)
        confidence = scores.max(axis=1)
        raw = np.asarray(self.classes_, dtype=object)[np.argmax(scores, axis=1)]
        grid = candidates if candidates is not None else np.unique(np.concatenate([[0.0], confidence]))
        best_tau, best_f1 = self.threshold, -1.0
        for tau in grid:
            predicted = raw.copy()
            predicted[confidence < tau] = self.unknown_label
            score = f1_score(truth.astype(str), predicted.astype(str), average="macro", zero_division=0)
            if score > best_f1 + 1e-12:
                best_tau, best_f1 = float(tau), float(score)
        self.threshold = best_tau
        return best_tau

    def explain(self, feature_names: Sequence[str], top_k: int = 20) -> Explanation:
        assert self.base is not None
        return self.base.explain(feature_names, top_k)
