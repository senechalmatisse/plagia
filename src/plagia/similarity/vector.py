"""Cosinus sur sacs de mots ou TF-IDF."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from plagia.similarity.base import SIMILARITIES, Similarity, text_of


def cosine_rows(A: Any, B: Any) -> np.ndarray:
    """Cosinus ligne à ligne entre deux matrices de même forme."""
    A = normalize(sparse.csr_matrix(A))
    B = normalize(sparse.csr_matrix(B))
    return np.asarray(A.multiply(B).sum(axis=1)).ravel()


@SIMILARITIES.register("cosine")
class Cosine(Similarity):
    """Cosinus entre vecteurs de mots (`tf`, `binary`) ou TF-IDF (`tfidf`, IDF ajusté)."""

    name = "cosine"

    def __init__(
        self,
        weighting: str = "tfidf",
        analyzer: str = "word",
        ngram_range: Sequence[int] = (1, 1),
        sublinear_tf: bool = True,
    ) -> None:
        self.weighting = weighting
        self.analyzer = analyzer
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        self.sublinear_tf = sublinear_tf
        self._vectorizer: Any = None

    def _make(self) -> Any:
        if self.weighting == "tfidf":
            return TfidfVectorizer(
                analyzer=self.analyzer, ngram_range=self.ngram_range, sublinear_tf=self.sublinear_tf, dtype=np.float64
            )
        return CountVectorizer(
            analyzer=self.analyzer, ngram_range=self.ngram_range, binary=self.weighting == "binary", dtype=np.float64
        )

    def fit(self, items: Sequence[Any]) -> Cosine:
        self._vectorizer = self._make().fit([text_of(x) for x in items])
        return self

    def vectorize(self, items: Sequence[Any]) -> sparse.csr_matrix:
        """Vecteurs (normalisés L2) des textes."""
        texts = [text_of(x) for x in items]
        if self._vectorizer is None:  # sans ajustement : vocabulaire local (pondération tf)
            vectorizer = CountVectorizer(analyzer=self.analyzer, ngram_range=self.ngram_range, dtype=np.float64)
            return normalize(vectorizer.fit_transform(texts))
        return normalize(self._vectorizer.transform(texts))

    def score(self, a: Any, b: Any) -> float:
        matrix = self.vectorize([a, b])
        return float(matrix[0].multiply(matrix[1]).sum())

    def score_many(self, pairs: Any) -> np.ndarray:
        pairs = list(pairs)
        if not pairs:
            return np.zeros(0)
        left = self.vectorize([a for a, _ in pairs])
        right = self.vectorize([b for _, b in pairs])
        if self._vectorizer is None:
            return np.array([self.score(a, b) for a, b in pairs])
        return np.asarray(left.multiply(right).sum(axis=1)).ravel()
