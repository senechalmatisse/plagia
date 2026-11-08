"""Classification bayésienne naïve écrite à la main avec NumPy.

Variante multinomiale (et, en option, de Bernoulli) avec lissage de Laplace (paramètre α) et calcul en log-probabilités, selon Manning, Raghavan et Schütze, *Introduction to
Information Retrieval*, chap. 13.
Les prédictions et les log-probabilités coïncident avec `MultinomialNB` / `BernoulliNB` de scikit-learn (écart inférieur à 1e-6, vérifié par les tests).

Multinomial : log P(c | d) ∝ log P(c) + Σⱼ xⱼ log P(tⱼ | c), avec
P(tⱼ | c) = (Nⱼc + α) / (Σₖ Nₖc + α |V|).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse
from scipy.special import logsumexp

from plagia.models.base import CLASSIFIERS, Classifier, Explanation, top_features


class NaiveBayes(Classifier):
    """Naïve Bayes multinomial ou de Bernoulli."""

    def __init__(
        self, alpha: float = 1.0, fit_prior: bool = True, variant: str = "multinomial", binarize: float = 0.0
    ) -> None:
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.variant = variant
        self.binarize = binarize

    def _prepare(self, X: Any) -> Any:
        X = sparse.csr_matrix(X, dtype=np.float64) if sparse.issparse(X) else np.asarray(X, dtype=np.float64)
        if self.variant == "bernoulli":
            X = (self.binarize < X).astype(np.float64)
            return sparse.csr_matrix(X) if sparse.issparse(X) else X
        minimum = X.min() if X.shape[0] and X.shape[1] else 0.0
        if minimum < 0:
            raise ValueError("Le naïve Bayes multinomial exige des attributs positifs ou nuls.")
        return X

    def fit(self, X: Any, y: Sequence[Any]) -> NaiveBayes:
        if self.variant not in {"multinomial", "bernoulli"}:
            raise ValueError(f"Variante inconnue : {self.variant}")
        if self.alpha < 0:
            raise ValueError("alpha doit être positif ou nul.")
        X = self._prepare(X)
        self.classes_, y_index = np.unique(np.asarray(y), return_inverse=True)
        n_classes = len(self.classes_)
        Y = np.zeros((X.shape[0], n_classes), dtype=np.float64)
        Y[np.arange(X.shape[0]), y_index] = 1.0
        self.class_count_ = Y.sum(axis=0)
        counts = Y.T @ X
        self.feature_count_ = np.asarray(counts.toarray() if sparse.issparse(counts) else counts, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        smoothed = self.feature_count_ + self.alpha
        with np.errstate(divide="ignore"):
            if self.variant == "multinomial":
                self.feature_log_prob_ = np.log(smoothed) - np.log(smoothed.sum(axis=1, keepdims=True))
            else:
                self.feature_log_prob_ = np.log(smoothed) - np.log(self.class_count_ + 2 * self.alpha)[:, None]
            if self.fit_prior:
                self.class_log_prior_ = np.log(self.class_count_) - np.log(self.class_count_.sum())
            else:
                self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))
        return self

    def joint_log_likelihood(self, X: Any) -> np.ndarray:
        """log P(c) + log P(d | c) pour chaque classe."""
        X = self._prepare(X)
        if self.variant == "multinomial":
            jll = X @ self.feature_log_prob_.T
        else:
            with np.errstate(divide="ignore"):
                negative = np.log1p(-np.exp(self.feature_log_prob_))
            jll = X @ (self.feature_log_prob_ - negative).T + negative.sum(axis=1)
        return np.asarray(jll) + self.class_log_prior_

    def predict_log_proba(self, X: Any) -> np.ndarray:
        """log P(c | d) normalisées."""
        jll = self.joint_log_likelihood(X)
        result: np.ndarray = jll - logsumexp(jll, axis=1, keepdims=True)
        return result

    def predict_proba(self, X: Any) -> np.ndarray:
        """P(c | d)."""
        return np.exp(self.predict_log_proba(X))

    def predict_scores(self, X: Any) -> np.ndarray:
        return self.predict_proba(X)

    def predict(self, X: Any) -> np.ndarray:
        result: np.ndarray = self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]
        return result

    def log_likelihood_ratios(self) -> np.ndarray:
        """log P(t | c) − log moyenne des P(t | c') des autres classes (attributs discriminants)."""
        flp = self.feature_log_prob_
        n_classes = flp.shape[0]
        if n_classes == 1:
            return flp.copy()
        total = logsumexp(flp, axis=0)
        ratios = np.empty_like(flp)
        for c in range(n_classes):
            others = np.log(np.maximum(np.exp(total) - np.exp(flp[c]), 1e-300)) - np.log(n_classes - 1)
            ratios[c] = flp[c] - others
        return ratios

    def explain(self, feature_names: Sequence[str], top_k: int = 20) -> Explanation:
        ratios = self.log_likelihood_ratios()
        return {str(c): top_features(ratios[i], feature_names, top_k) for i, c in enumerate(self.classes_)}


CLASSIFIERS.add("naive_bayes", NaiveBayes)
CLASSIFIERS.add("bernoulli_nb", lambda **params: NaiveBayes(variant="bernoulli", **params))
