"""Interface commune `Classifier` et utilitaires de classement.

Le naïve Bayes écrit à la main et les modèles scikit-learn partagent la même interface :
`fit`, `predict`, `predict_scores` (probabilités, ou scores de décision ramenés dans [0, 1] par softmax),
`explain` (attributs les plus discriminants) et `rank` (k meilleurs auteurs avec un score de confiance).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin

from plagia.core.registry import Registry

CLASSIFIERS: Registry[Any] = Registry("classificateur")

Explanation = dict[str, list[tuple[str, float]]]


class Classifier(ClassifierMixin, BaseEstimator, ABC):
    """Classificateur compatible scikit-learn doté d'une explication de ses décisions."""

    classes_: np.ndarray

    @abstractmethod
    def fit(self, X: Any, y: Sequence[Any]) -> Classifier:
        """Apprend le modèle."""

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Classes prédites."""

    @abstractmethod
    def predict_scores(self, X: Any) -> np.ndarray:
        """Scores de confiance par classe (lignes de somme 1)."""

    @abstractmethod
    def explain(self, feature_names: Sequence[str], top_k: int = 20) -> Explanation:
        """Attributs les plus discriminants de chaque classe (nom, poids)."""

    def rank(self, X: Any, k: int = 3) -> list[list[tuple[str, float]]]:
        """Les `k` classes les plus probables de chaque exemple avec leur score."""
        return rank_from_scores(self.predict_scores(X), self.classes_, k)


def rank_from_scores(scores: np.ndarray, classes: Sequence[Any] | np.ndarray, k: int) -> list[list[tuple[str, float]]]:
    """Classement décroissant des classes à partir d'une matrice de scores."""
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return [[(str(classes[j]), float(scores[i, j])) for j in row] for i, row in enumerate(order)]


def decision_to_scores(decision: np.ndarray, n_classes: int) -> np.ndarray:
    """Scores de décision → pseudo-probabilités (softmax ; cas binaire géré)."""
    decision = np.asarray(decision, dtype=float)
    if decision.ndim == 1 or n_classes == 2:
        flat = decision.reshape(-1)
        decision = np.column_stack([-flat, flat])
    scores: np.ndarray = softmax(decision, axis=1)
    return scores


def top_features(weights: np.ndarray, feature_names: Sequence[str], top_k: int) -> list[tuple[str, float]]:
    """Attributs de plus fort poids positif."""
    if weights.size == 0:
        return []
    k = min(top_k, weights.size)
    index = np.argpartition(-weights, kth=k - 1)[:k]
    index = index[np.argsort(-weights[index], kind="stable")]
    return [(str(feature_names[i]), float(weights[i])) for i in index]
