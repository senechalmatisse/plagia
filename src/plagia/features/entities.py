"""Caractéristiques d'entités et de types issus des bases de connaissances.

* :class:`EntityFeatures` : comptes et proportions par type, ensemble des entités (formes de surface ou URI) avec leurs fréquences, n-grammes de types (`PER ORG LOC`) ;
* :class:`TypeFeatures` : sac des classes DBpedia ou YAGO des entités liées, à une profondeur réglable de la hiérarchie (repli sur le type de reconnaissance sinon).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from plagia.annotation.document import AnnotatedDocument
from plagia.annotation.schema import MINIMAL_LABELS
from plagia.features.base import as_csr, make_scaler, ngrams, register_block, to_documents


class EntityFormAnalyzer:
    """Formes des entités (surface en minuscules, ou URI si `level='uri'` et disponible)."""

    def __init__(self, level: str = "surface") -> None:
        self.level = level

    def __call__(self, doc: AnnotatedDocument) -> list[str]:
        forms = []
        for entity in doc.entities:
            if self.level == "uri" and entity.uri:
                forms.append(entity.uri)
            else:
                forms.append(" ".join(entity.text.lower().split()))
        return forms


class EntityTypeSequenceAnalyzer:
    """N-grammes de la suite des types d'entités dans l'ordre du texte."""

    def __init__(self, ngram_range: Sequence[int]) -> None:
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))

    def __call__(self, doc: AnnotatedDocument) -> list[str]:
        return ngrams([e.kind for e in doc.entities], self.ngram_range)


class EntityClassAnalyzer:
    """Classes de la base de connaissances à la profondeur `depth`."""

    def __init__(self, depth: int, fallback: str = "kind") -> None:
        self.depth = depth
        self.fallback = fallback

    def __call__(self, doc: AnnotatedDocument) -> list[str]:
        out: list[str] = []
        for entity in doc.entities:
            classes = entity.types_at_depth(self.depth) if entity.types else ()
            if classes:
                out.extend(classes)
            elif self.fallback == "kind":
                out.append(f"kind:{entity.kind}")
        return out


def _tfidf(analyzer: Any, min_df: int | float, sublinear_tf: bool) -> TfidfVectorizer:
    return TfidfVectorizer(analyzer=analyzer, min_df=min_df, sublinear_tf=sublinear_tf, dtype=np.float64)


def _fit_optional(vectorizer: TfidfVectorizer, docs: list[AnnotatedDocument]) -> TfidfVectorizer | None:
    try:
        return vectorizer.fit(docs)
    except ValueError:  # vocabulaire vide (aucune entité) : bloc ignoré
        return None


@register_block("entities", family="entities")
class EntityFeatures(TransformerMixin, BaseEstimator):
    """Comptes, proportions, formes et n-grammes de types d'entités."""

    def __init__(
        self,
        counts: bool = True,
        proportions: bool = True,
        forms: bool = True,
        type_ngrams: bool = True,
        ngram_range: Sequence[int] = (1, 3),
        level: str = "surface",
        min_df: int | float = 1,
        sublinear_tf: bool = True,
        labels: Sequence[str] = MINIMAL_LABELS,
        scale: str = "minmax",
    ) -> None:
        self.counts = counts
        self.proportions = proportions
        self.forms = forms
        self.type_ngrams = type_ngrams
        self.ngram_range = ngram_range
        self.level = level
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf
        self.labels = labels
        self.scale = scale

    def _dense(self, docs: list[AnnotatedDocument]) -> np.ndarray:
        rows = []
        for doc in docs:
            kinds = Counter(e.kind for e in doc.entities)
            n_words = max(len(doc.words), 1)
            total = max(sum(kinds.values()), 1)
            row: list[float] = []
            if self.counts:
                row += [100.0 * kinds[label] / n_words for label in self.labels]
                row.append(100.0 * sum(kinds.values()) / n_words)
            if self.proportions:
                row += [kinds[label] / total for label in self.labels]
            rows.append(row)
        return np.array(rows, dtype=float).reshape(len(docs), -1)

    def fit(self, X: Sequence[Any], y: Any = None) -> EntityFeatures:
        docs = to_documents(X)
        dense = self._dense(docs)
        scaler = make_scaler(self.scale)
        self.scaler_ = scaler.fit(dense) if (scaler is not None and dense.shape[1]) else None
        self.forms_ = (
            _fit_optional(_tfidf(EntityFormAnalyzer(self.level), self.min_df, self.sublinear_tf), docs)
            if self.forms
            else None
        )
        self.sequence_ = (
            _fit_optional(_tfidf(EntityTypeSequenceAnalyzer(self.ngram_range), self.min_df, self.sublinear_tf), docs)
            if self.type_ngrams
            else None
        )
        return self

    def transform(self, X: Sequence[Any]) -> sparse.csr_matrix:
        docs = to_documents(X)
        blocks = []
        dense = self._dense(docs)
        if dense.shape[1]:
            blocks.append(as_csr(self.scaler_.transform(dense) if self.scaler_ is not None else dense))
        for vectorizer in (self.forms_, self.sequence_):
            if vectorizer is not None:
                blocks.append(as_csr(vectorizer.transform(docs)))
        if not blocks:
            return sparse.csr_matrix((len(docs), 0))
        return sparse.hstack(blocks, format="csr")

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        names: list[str] = []
        if self.counts or self.proportions:
            if self.counts:
                names += [f"ent_density:{label}" for label in self.labels] + ["ent_density:all"]
            if self.proportions:
                names += [f"ent_share:{label}" for label in self.labels]
        if self.forms_ is not None:
            names += [f"ent[{f}]" for f in self.forms_.get_feature_names_out()]
        if self.sequence_ is not None:
            names += [f"ent_seq[{f}]" for f in self.sequence_.get_feature_names_out()]
        return np.asarray(names, dtype=object)


@register_block("types", family="types")
class TypeFeatures(TransformerMixin, BaseEstimator):
    """Sac des classes (DBpedia ou YAGO) des entités à la profondeur `depth`."""

    def __init__(
        self, depth: int = 2, fallback: str = "kind", min_df: int | float = 1, sublinear_tf: bool = True
    ) -> None:
        self.depth = depth
        self.fallback = fallback
        self.min_df = min_df
        self.sublinear_tf = sublinear_tf

    def fit(self, X: Sequence[Any], y: Any = None) -> TypeFeatures:
        docs = to_documents(X)
        self.vectorizer_ = _fit_optional(
            _tfidf(EntityClassAnalyzer(self.depth, self.fallback), self.min_df, self.sublinear_tf), docs
        )
        return self

    def transform(self, X: Sequence[Any]) -> sparse.csr_matrix:
        docs = to_documents(X)
        if self.vectorizer_ is None:
            return sparse.csr_matrix((len(docs), 0))
        return as_csr(self.vectorizer_.transform(docs))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if self.vectorizer_ is None:
            return np.asarray([], dtype=object)
        return np.asarray([f"type[{f}]" for f in self.vectorizer_.get_feature_names_out()], dtype=object)
