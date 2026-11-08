"""Attributs stylométriques : mots-outils, ponctuation, longueurs, richesse lexicale.

Les n-grammes d'étiquettes grammaticales, également stylométriques, forment un bloc à part (:mod:`plagia.features.pos`) afin d'être isolés dans les ablations.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin

from plagia.annotation.document import AnnotatedDocument
from plagia.core.lexicon import PUNCTUATION_MARKS, function_words
from plagia.core.text import is_word
from plagia.features.base import as_csr, make_scaler, register_block, to_documents


def yule_k(counts: Counter[str]) -> float:
    """Caractéristique K de Yule : 10⁴ (Σ i² Vᵢ − N) / N²."""
    n = sum(counts.values())
    if n == 0:
        return 0.0
    spectrum = Counter(counts.values())
    return 1e4 * (sum(i * i * v for i, v in spectrum.items()) - n) / (n * n)


def mattr(tokens: Sequence[str], window: int = 100) -> float:
    """Rapport types/occurrences moyen sur fenêtre glissante (peu sensible à la longueur)."""
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return len(set(tokens)) / n
    counts: Counter[str] = Counter(tokens[:window])
    total = len(counts)
    for i in range(window, n):
        out, inc = tokens[i - window], tokens[i]
        counts[out] -= 1
        if counts[out] == 0:
            del counts[out]
        counts[inc] += 1
        total += len(counts)
    return total / ((n - window + 1) * window)


@register_block("stylometry", family="stylometry")
class Stylometry(TransformerMixin, BaseEstimator):
    """Vecteur stylométrique dense, centré-réduit sur l'entraînement."""

    def __init__(
        self,
        function_words: bool = True,
        punctuation: bool = True,
        lengths: bool = True,
        richness: bool = True,
        max_word_length: int = 15,
        mattr_window: int = 100,
        language: str = "en",
        scale: str = "minmax",
    ) -> None:
        self.function_words = function_words
        self.punctuation = punctuation
        self.lengths = lengths
        self.richness = richness
        self.max_word_length = max_word_length
        self.mattr_window = mattr_window
        self.language = language
        self.scale = scale

    def _names(self) -> list[str]:
        names: list[str] = []
        if self.function_words:
            names += [f"fw:{w}" for w in sorted(function_words(self.language))]
        if self.punctuation:
            names += [f"punct:{p}" for p in PUNCTUATION_MARKS]
        if self.lengths:
            names += ["len:word_mean", "len:word_std", "len:sentence_mean", "len:sentence_std"]
            names += [f"len:word={k}" for k in range(1, self.max_word_length)] + [f"len:word>={self.max_word_length}"]
        if self.richness:
            names += ["rich:ttr", "rich:mattr", "rich:hapax", "rich:dis", "rich:yule_k", "char:upper", "char:digit"]
        return names

    def _vector(self, doc: AnnotatedDocument) -> list[float]:
        tokens = list(doc.tokens)
        words = [t for t in tokens if is_word(t)]
        lower = [w.lower() for w in words]
        n_words = max(len(words), 1)
        values: list[float] = []
        if self.function_words:
            counts = Counter(lower)
            values += [counts[w] / n_words for w in sorted(function_words(self.language))]
        if self.punctuation:
            punct = Counter(t for t in tokens if not is_word(t))
            values += [punct[p] / n_words for p in PUNCTUATION_MARKS]
        if self.lengths:
            lengths = np.array([len(w) for w in words] or [0], dtype=float)
            sentence_lengths = np.array(
                [sum(1 for t in s if is_word(t)) for s in doc.sentence_tokens()] or [0], dtype=float
            )
            values += [lengths.mean(), lengths.std(), sentence_lengths.mean(), sentence_lengths.std()]
            histogram = np.zeros(self.max_word_length)
            for length in lengths.astype(int):
                if length > 0:
                    histogram[min(length, self.max_word_length) - 1] += 1
            values += list(histogram / n_words)
        if self.richness:
            counts = Counter(lower)
            spectrum = Counter(counts.values())
            text = doc.text
            n_chars = max(len(text), 1)
            values += [
                len(counts) / n_words,
                mattr(lower, self.mattr_window),
                spectrum.get(1, 0) / n_words,
                spectrum.get(2, 0) / n_words,
                yule_k(counts),
                sum(c.isupper() for c in text) / n_chars,
                sum(c.isdigit() for c in text) / n_chars,
            ]
        return [0.0 if (math.isnan(v) or math.isinf(v)) else float(v) for v in values]

    def _matrix(self, X: Sequence[Any]) -> np.ndarray:
        return np.array([self._vector(d) for d in to_documents(X)], dtype=float).reshape(len(X), -1)

    def fit(self, X: Sequence[Any], y: Any = None) -> Stylometry:
        self.feature_names_ = self._names()
        scaler = make_scaler(self.scale)
        self.scaler_ = scaler.fit(self._matrix(X)) if scaler is not None else None
        return self

    def transform(self, X: Sequence[Any]) -> sparse.csr_matrix:
        values = self._matrix(X)
        if self.scaler_ is not None:
            values = self.scaler_.transform(values)
        return as_csr(values)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.feature_names_, dtype=object)
