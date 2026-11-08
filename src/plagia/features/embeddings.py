"""Représentations sémantiques : LSA, Word2Vec/FastText, Doc2Vec, plongements de phrases.

LSA ne dépend que de scikit-learn (socle obligatoire). Les modèles Gensim et sentence-transformers sont des dépendances facultatives chargées à la demande ;
aucun modèle de langage n'est entraîné ni ajusté (les plongements de phrases sont pré-entraînés, leur version est enregistrée).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from plagia.annotation.preprocess import Preprocessor
from plagia.core.errors import MissingDependencyError
from plagia.features.base import as_csr, register_block, to_documents
from plagia.features.lexical import Tfidf


@register_block("lsa", family="semantic")
class Lsa(TransformerMixin, BaseEstimator):
    """Analyse sémantique latente : TF-IDF de mots puis SVD tronquée (vecteurs normalisés)."""

    def __init__(
        self,
        n_components: int = 100,
        ngram_range: Sequence[int] = (1, 1),
        min_df: int | float = 2,
        stopwords: bool = False,
        stopwords_source: str = "sklearn",
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.stopwords = stopwords
        self.stopwords_source = stopwords_source
        self.random_state = random_state

    def fit(self, X: Sequence[Any], y: Any = None) -> Lsa:
        self.tfidf_ = Tfidf(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            stopwords=self.stopwords,
            stopwords_source=self.stopwords_source,
        )
        matrix = self.tfidf_.fit_transform(X)
        k = max(1, min(self.n_components, matrix.shape[1] - 1, matrix.shape[0] - 1))
        self.svd_ = TruncatedSVD(n_components=k, random_state=self.random_state).fit(matrix)
        return self

    def transform(self, X: Sequence[Any]) -> sparse.csr_matrix:
        return as_csr(normalize(self.svd_.transform(self.tfidf_.transform(X))))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray([f"lsa{i}" for i in range(self.svd_.n_components)], dtype=object)


def _gensim() -> Any:
    try:
        import gensim
    except ImportError as exc:
        raise MissingDependencyError(
            "gensim", "plongements Word2Vec/FastText/Doc2Vec", "uv sync --extra semantic"
        ) from exc
    return gensim


def _tokens(X: Sequence[Any]) -> list[list[str]]:
    pre = Preprocessor(punctuation=False, stopwords=True)
    return [pre.tokens(d) for d in to_documents(X)]


@register_block("doc2vec", family="semantic")
class Doc2VecEmbedding(TransformerMixin, BaseEstimator):
    """Doc2Vec (Gensim) entraîné sur le corpus d'entraînement ; vecteurs inférés ensuite."""

    def __init__(
        self, vector_size: int = 100, window: int = 5, min_count: int = 2, epochs: int = 20, dm: int = 1, seed: int = 0
    ) -> None:
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.dm = dm
        self.seed = seed

    def fit(self, X: Sequence[Any], y: Any = None) -> Doc2VecEmbedding:
        gensim = _gensim()
        tagged = [gensim.models.doc2vec.TaggedDocument(t, [i]) for i, t in enumerate(_tokens(X))]
        self.model_ = gensim.models.Doc2Vec(
            tagged,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            dm=self.dm,
            seed=self.seed,
            workers=1,
        )
        return self

    def transform(self, X: Sequence[Any]) -> sparse.csr_matrix:
        vectors = [self.model_.infer_vector(t, epochs=self.epochs) for t in _tokens(X)]
        return as_csr(normalize(np.array(vectors, dtype=float).reshape(len(X), -1)))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray([f"d2v{i}" for i in range(self.vector_size)], dtype=object)


@register_block("word_vectors", family="semantic")
class WordVectors(TransformerMixin, BaseEstimator):
    """Word2Vec ou FastText (Gensim) ; vecteur de document = moyenne pondérée par TF-IDF."""

    def __init__(
        self,
        algorithm: str = "word2vec",
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        epochs: int = 10,
        seed: int = 0,
    ) -> None:
        self.algorithm = algorithm
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.seed = seed

    def fit(self, X: Sequence[Any], y: Any = None) -> WordVectors:
        gensim = _gensim()
        sentences = _tokens(X)
        cls = gensim.models.FastText if self.algorithm == "fasttext" else gensim.models.Word2Vec
        self.model_ = cls(
            sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            seed=self.seed,
            workers=1,
        )
        n_docs = len(sentences)
        df: dict[str, int] = {}
        for tokens in sentences:
            for token in set(tokens):
                df[token] = df.get(token, 0) + 1
        self.idf_ = {t: float(np.log((1 + n_docs) / (1 + c)) + 1) for t, c in df.items()}
        return self

    def transform(self, X: Sequence[Any]) -> sparse.csr_matrix:
        vectors = []
        for tokens in _tokens(X):
            weights, rows = [], []
            for token in set(tokens):
                if token in self.model_.wv:
                    rows.append(self.model_.wv[token])
                    weights.append(tokens.count(token) * self.idf_.get(token, 1.0))
            if rows:
                vectors.append(np.average(np.array(rows), axis=0, weights=weights))
            else:
                vectors.append(np.zeros(self.vector_size))
        return as_csr(normalize(np.array(vectors, dtype=float).reshape(len(X), -1)))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return np.asarray([f"wv{i}" for i in range(self.vector_size)], dtype=object)


@register_block("sentence_embeddings", family="semantic")
class SentenceEmbeddings(TransformerMixin, BaseEstimator):
    """Plongements de phrases pré-entraînés."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 32) -> None:
        self.model = model
        self.batch_size = batch_size

    def fit(self, X: Sequence[Any], y: Any = None) -> SentenceEmbeddings:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise MissingDependencyError(
                "sentence-transformers", "plongements de phrases", "uv sync --extra transformers"
            ) from exc
        self.encoder_ = SentenceTransformer(self.model)
        return self

    def transform(self, X: Sequence[Any]) -> sparse.csr_matrix:
        texts = [d.text for d in to_documents(X)]
        vectors = self.encoder_.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)
        return as_csr(np.asarray(vectors, dtype=float))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        dim = self.encoder_.get_sentence_embedding_dimension()
        return np.asarray([f"st{i}" for i in range(dim)], dtype=object)
