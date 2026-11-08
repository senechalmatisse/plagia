"""N-grammes d'étiquettes grammaticales."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from plagia.annotation.document import AnnotatedDocument
from plagia.core.errors import ConfigurationError
from plagia.features.base import as_csr, ngrams, register_block, to_documents


class PosAnalyzer:
    """Analyseur picklable : n-grammes d'étiquettes, sans franchir les frontières de phrase."""

    def __init__(self, tagset: str, ngram_range: Sequence[int]) -> None:
        self.tagset = tagset
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))

    def __call__(self, doc: AnnotatedDocument) -> list[str]:
        tags = doc.tags if self.tagset == "fine" else doc.pos
        if tags is None:
            raise ConfigurationError(
                "Les n-grammes d'étiquettes nécessitent un annotateur grammatical "
                "(annotation.engine: spacy, stanford avec POS, ou heuristic)."
            )
        bounds = doc.sentences or ((0, len(tags)),)
        out: list[str] = []
        for start, end in bounds:
            out.extend(ngrams(list(tags[start:end]), self.ngram_range))
        return out


@register_block("pos_ngrams", family="pos")
class PosNgrams(TransformerMixin, BaseEstimator):
    """TF-IDF de n-grammes d'étiquettes universelles (`upos`) ou fines (`fine`, PTB)."""

    def __init__(
        self,
        ngram_range: Sequence[int] = (1, 3),
        tagset: str = "upos",
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        sublinear_tf: bool = True,
        use_idf: bool = True,
        norm: str | None = "l2",
    ) -> None:
        self.ngram_range = ngram_range
        self.tagset = tagset
        self.min_df = min_df
        self.max_df = max_df
        self.sublinear_tf = sublinear_tf
        self.use_idf = use_idf
        self.norm = norm

    def _make(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer=PosAnalyzer(self.tagset, self.ngram_range),
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            use_idf=self.use_idf,
            norm=self.norm,
            dtype=np.float64,
        )

    def fit(self, X: Sequence[Any], y: Any = None) -> PosNgrams:
        self.vectorizer_ = self._make().fit(to_documents(X))
        return self

    def transform(self, X: Sequence[Any]) -> sparse.csr_matrix:
        return as_csr(self.vectorizer_.transform(to_documents(X)))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray([f"pos[{f}]" for f in self.vectorizer_.get_feature_names_out()], dtype=object)
