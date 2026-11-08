"""Sac de mots, TF-IDF et n-grammes de mots et de caractères.

Vocabulaire et IDF sont ajustés sur les données passées à `fit` uniquement (jeu d'entraînement ou pli d'entraînement de la validation croisée).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from plagia.annotation.document import AnnotatedDocument
from plagia.annotation.preprocess import Preprocessor
from plagia.features.base import as_csr, ngrams, register_block, to_documents


class TokenAnalyzer:
    """Analyseur scikit-learn (objet picklable) : prétraitement puis n-grammes de jetons."""

    def __init__(self, preprocessor: Preprocessor, ngram_range: Sequence[int]) -> None:
        self.preprocessor = preprocessor
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))

    def __call__(self, doc: AnnotatedDocument) -> list[str]:
        return ngrams(self.preprocessor.tokens(doc), self.ngram_range)


class _WordBlock(TransformerMixin, BaseEstimator):
    def _preprocessor(self) -> Preprocessor:
        return Preprocessor(
            lowercase=self.lowercase,
            punctuation=self.punctuation,
            stopwords=self.stopwords,
            normalization=self.normalization,
            numbers=self.numbers,
            language=self.language,
            stopwords_source=self.stopwords_source,
        )

    def _make_vectorizer(self) -> CountVectorizer:
        raise NotImplementedError

    def fit(self, X: Sequence[Any], y: Any = None) -> _WordBlock:
        self.vectorizer_ = self._make_vectorizer()
        self.vectorizer_.fit(to_documents(X))
        return self

    def fit_transform(self, X: Sequence[Any], y: Any = None, **fit_params: Any) -> sparse.csr_matrix:
        self.vectorizer_ = self._make_vectorizer()
        return as_csr(self.vectorizer_.fit_transform(to_documents(X)))

    def transform(self, X: Sequence[Any]) -> sparse.csr_matrix:
        return as_csr(self.vectorizer_.transform(to_documents(X)))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray(self.vectorizer_.get_feature_names_out(), dtype=object)


@register_block("bow", family="words")
class BagOfWords(_WordBlock):
    """Sac de mots (comptes ou binaire), n-grammes de mots, bornes de fréquence."""

    def __init__(
        self,
        binary: bool = False,
        ngram_range: Sequence[int] = (1, 1),
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        max_features: int | None = None,
        lowercase: bool = True,
        punctuation: bool = False,
        stopwords: bool = True,
        normalization: str = "none",
        numbers: str = "keep",
        language: str = "en",
        stopwords_source: str = "nltk",
    ) -> None:
        self.binary = binary
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.lowercase = lowercase
        self.punctuation = punctuation
        self.stopwords = stopwords
        self.normalization = normalization
        self.numbers = numbers
        self.language = language
        self.stopwords_source = stopwords_source

    def _make_vectorizer(self) -> CountVectorizer:
        return CountVectorizer(
            analyzer=TokenAnalyzer(self._preprocessor(), self.ngram_range),
            binary=self.binary,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            dtype=np.float64,
        )


@register_block("tfidf", family="words")
class Tfidf(_WordBlock):
    """TF-IDF de n-grammes de mots (pondération sous-linéaire, normalisation L2)."""

    def __init__(
        self,
        ngram_range: Sequence[int] = (1, 2),
        min_df: int | float = 1,
        max_df: int | float = 1.0,
        max_features: int | None = None,
        sublinear_tf: bool = True,
        norm: str | None = "l2",
        use_idf: bool = True,
        binary: bool = False,
        lowercase: bool = True,
        punctuation: bool = False,
        stopwords: bool = True,
        normalization: str = "none",
        numbers: str = "keep",
        language: str = "en",
        stopwords_source: str = "nltk",
    ) -> None:
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.use_idf = use_idf
        self.binary = binary
        self.lowercase = lowercase
        self.punctuation = punctuation
        self.stopwords = stopwords
        self.normalization = normalization
        self.numbers = numbers
        self.language = language
        self.stopwords_source = stopwords_source

    def _make_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer=TokenAnalyzer(self._preprocessor(), self.ngram_range),
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            norm=self.norm,
            use_idf=self.use_idf,
            binary=self.binary,
            dtype=np.float64,
        )


@register_block("char_ngrams", family="chars")
class CharNgrams(TransformerMixin, BaseEstimator):
    """TF-IDF de n-grammes de caractères (2 à 5) calculés sur le texte brut ou masqué."""

    def __init__(
        self,
        ngram_range: Sequence[int] = (2, 5),
        analyzer: str = "char",
        lowercase: bool = False,
        min_df: int | float = 2,
        max_df: int | float = 1.0,
        max_features: int | None = None,
        sublinear_tf: bool = True,
        use_idf: bool = True,
        norm: str | None = "l2",
    ) -> None:
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.lowercase = lowercase
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.use_idf = use_idf
        self.norm = norm

    def _make_vectorizer(self) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=(int(self.ngram_range[0]), int(self.ngram_range[1])),
            lowercase=self.lowercase,
            min_df=self.min_df,
            max_df=self.max_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            use_idf=self.use_idf,
            norm=self.norm,
            dtype=np.float64,
        )

    @staticmethod
    def _texts(X: Sequence[Any]) -> list[str]:
        return [d.text for d in to_documents(X)]

    def fit(self, X: Sequence[Any], y: Any = None) -> CharNgrams:
        self.vectorizer_ = self._make_vectorizer().fit(self._texts(X))
        return self

    def fit_transform(self, X: Sequence[Any], y: Any = None, **fit_params: Any) -> sparse.csr_matrix:
        self.vectorizer_ = self._make_vectorizer()
        return as_csr(self.vectorizer_.fit_transform(self._texts(X)))

    def transform(self, X: Sequence[Any]) -> sparse.csr_matrix:
        return as_csr(self.vectorizer_.transform(self._texts(X)))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray([f"c[{f}]" for f in self.vectorizer_.get_feature_names_out()], dtype=object)
