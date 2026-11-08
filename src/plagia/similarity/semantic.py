"""Similarité sémantique : cosinus de représentations LSA, Doc2Vec ou plongements de phrases ; distance de Word Mover."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from plagia.core.errors import MissingDependencyError
from plagia.core.text import words
from plagia.features.base import FEATURE_BLOCKS
from plagia.similarity.base import SIMILARITIES, Similarity, text_of


@SIMILARITIES.register("semantic")
class SemanticSimilarity(Similarity):
    """Cosinus entre plongements produits par un bloc sémantique ajusté (LSA par défaut)."""

    name = "semantic"

    def __init__(self, block: str = "lsa", **params: Any) -> None:
        self.block = block
        self.params = params
        self._model: Any = None

    def fit(self, items: Sequence[Any]) -> SemanticSimilarity:
        self._model = FEATURE_BLOCKS.create(self.block, **self.params).fit(list(items))
        return self

    def embed(self, items: Sequence[Any]) -> np.ndarray:
        """Vecteurs normalisés des textes."""
        if self._model is None:
            raise RuntimeError("SemanticSimilarity doit être ajustée (fit) avant usage.")
        matrix = self._model.transform(list(items))
        dense = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        return np.divide(dense, norms, out=np.zeros_like(dense), where=norms > 0)

    def score(self, a: Any, b: Any) -> float:
        vectors = self.embed([a, b])
        return float(np.clip(vectors[0] @ vectors[1], 0.0, 1.0))

    def score_many(self, pairs: Any) -> np.ndarray:
        pairs = list(pairs)
        if not pairs:
            return np.zeros(0)
        left = self.embed([a for a, _ in pairs])
        right = self.embed([b for _, b in pairs])
        return np.clip((left * right).sum(axis=1), 0.0, 1.0)


@SIMILARITIES.register("word_mover")
class WordMoverSimilarity(Similarity):
    """1 / (1 + distance de Word Mover) avec des vecteurs de mots Gensim (`KeyedVectors`)."""

    name = "word_mover"

    def __init__(self, vectors: Any = None) -> None:
        self.vectors = vectors

    def score(self, a: Any, b: Any) -> float:
        if self.vectors is None:
            raise MissingDependencyError("gensim", "distance de Word Mover", "Fournissez des KeyedVectors Gensim.")
        ta = [w.lower() for w in words(text_of(a))]
        tb = [w.lower() for w in words(text_of(b))]
        distance = float(self.vectors.wmdistance(ta, tb))
        return 0.0 if not np.isfinite(distance) else 1.0 / (1.0 + distance)
