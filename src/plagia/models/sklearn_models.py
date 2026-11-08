"""Modèles scikit-learn derrière l'interface `Classifier`.

SVM linéaire et régression logistique ; k plus proches voisins (cosinus),
Complement Naive Bayes, forêt aléatoire, gradient boosting et perceptron multicouche ; classe majoritaire.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

from plagia.models.base import CLASSIFIERS, Classifier, Explanation, decision_to_scores, top_features
from plagia.models.naive_bayes import NaiveBayes


class SklearnClassifier(Classifier):
    """Enveloppe d'un estimateur scikit-learn (hyperparamètres : `estimator__<nom>`)."""

    def __init__(self, estimator: BaseEstimator | None = None) -> None:
        self.estimator = estimator

    def fit(self, X: Any, y: Sequence[Any]) -> SklearnClassifier:
        if self.estimator is None:
            raise ValueError("Aucun estimateur fourni.")
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        prediction: np.ndarray = self.estimator_.predict(X)
        return prediction

    def predict_scores(self, X: Any) -> np.ndarray:
        if hasattr(self.estimator_, "predict_proba"):
            proba: np.ndarray = self.estimator_.predict_proba(X)
            return proba
        return decision_to_scores(self.estimator_.decision_function(X), len(self.classes_))

    def decision_function(self, X: Any) -> np.ndarray:
        """Scores de décision bruts (ou log-probabilités à défaut)."""
        if hasattr(self.estimator_, "decision_function"):
            decision: np.ndarray = self.estimator_.decision_function(X)
            return decision
        return np.log(np.clip(self.predict_scores(X), 1e-300, None))

    def coefficients(self) -> np.ndarray | None:
        """Matrice des coefficients (une ligne par classe) pour les modèles linéaires."""
        coef = getattr(self.estimator_, "coef_", None)
        if coef is None:
            return None
        coef = np.asarray(coef.toarray() if hasattr(coef, "toarray") else coef, dtype=float)
        if coef.shape[0] == 1 and len(self.classes_) == 2:
            coef = np.vstack([-coef[0], coef[0]])
        return coef

    def explain(self, feature_names: Sequence[str], top_k: int = 20) -> Explanation:
        coef = self.coefficients()
        if coef is not None:
            return {str(c): top_features(coef[i], feature_names, top_k) for i, c in enumerate(self.classes_)}
        if hasattr(self.estimator_, "feature_log_prob_"):
            nb = NaiveBayes()
            nb.feature_log_prob_ = np.asarray(self.estimator_.feature_log_prob_)
            nb.classes_ = self.classes_
            return nb.explain(feature_names, top_k)
        importances = getattr(self.estimator_, "feature_importances_", None)
        if importances is not None:
            return {"*": top_features(np.asarray(importances, dtype=float), feature_names, top_k)}
        return {}


def _wrap(factory: Callable[..., BaseEstimator], **defaults: Any) -> Callable[..., SklearnClassifier]:
    def build(**params: Any) -> SklearnClassifier:
        return SklearnClassifier(factory(**{**defaults, **params}))

    return build


CLASSIFIERS.add("linear_svm", _wrap(LinearSVC, C=1.0))
CLASSIFIERS.add("logreg", _wrap(LogisticRegression, C=10.0, max_iter=3000))
CLASSIFIERS.add("knn", _wrap(KNeighborsClassifier, n_neighbors=5, metric="cosine", weights="distance"))
CLASSIFIERS.add("complement_nb", _wrap(ComplementNB, alpha=1.0))
CLASSIFIERS.add("random_forest", _wrap(RandomForestClassifier, n_estimators=300, random_state=0, n_jobs=1))
CLASSIFIERS.add("gradient_boosting", _wrap(GradientBoostingClassifier, random_state=0))
CLASSIFIERS.add(
    "mlp", _wrap(MLPClassifier, hidden_layer_sizes=(256,), early_stopping=True, max_iter=300, random_state=0)
)
CLASSIFIERS.add("majority", _wrap(DummyClassifier, strategy="most_frequent"))


def build_classifier(name: str, **params: Any) -> Classifier:
    """Classificateur enregistré sous `name`."""
    classifier: Classifier = CLASSIFIERS.create(name, **params)
    return classifier


def param_prefix(classifier: Classifier) -> str:
    """Préfixe des hyperparamètres dans une grille (`estimator__` pour les enveloppes)."""
    return "estimator__" if isinstance(classifier, SklearnClassifier) else ""


def set_random_state(classifier: Classifier, seed: int) -> Classifier:
    """Fixe `random_state` si l'estimateur en possède un (reproductibilité)."""
    params = classifier.get_params(deep=True)
    key = f"{param_prefix(classifier)}random_state"
    if key in params:
        classifier.set_params(**{key: seed})
    return classifier
