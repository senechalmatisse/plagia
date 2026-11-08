"""Recherche des sources candidates d'un document suspect.

Les documents sont découpés en segments de quelques phrases ;
un passage repris ne couvrant qu'une partie du document, le score d'une source est le maximum sur les paires de segments.
Trois index : TF-IDF (cosinus), index in ersé de shingles, MinHash + LSH.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from plagia.features.shingles import positioned_shingles
from plagia.similarity.minhash import MinHashIndex
from plagia.tasks.plagiarism.passages import sentence_char_spans, sliding_windows


def chunks(document: Any, size: int) -> list[str]:
    """Segments disjoints de `size` phrases."""
    spans = sentence_char_spans(document)
    return [document.text[a:b] for a, b in sliding_windows(spans, size, step=size)] or [document.text]


class CandidateRetriever:
    """Index des sources ; `retrieve` renvoie les `k` meilleures sources et leur score."""

    def __init__(
        self,
        method: str = "tfidf",
        k: int = 10,
        chunk_sentences: int = 4,
        shingle_size: int = 5,
        num_perm: int = 128,
        bands: int = 32,
        seed: int = 1,
    ) -> None:
        if method not in {"tfidf", "shingles", "minhash"}:
            raise ValueError(f"Méthode de recherche inconnue : {method}")
        self.method = method
        self.k = k
        self.chunk_sentences = chunk_sentences
        self.shingle_size = shingle_size
        self.num_perm = num_perm
        self.bands = bands
        self.seed = seed

    def fit(self, sources: Sequence[Any]) -> CandidateRetriever:
        """Indexe les documents sources."""
        self.source_ids_ = [str(s.id) for s in sources]
        if self.method == "tfidf":
            texts, owners = [], []
            for index, source in enumerate(sources):
                for chunk in chunks(source, self.chunk_sentences):
                    texts.append(chunk)
                    owners.append(index)
            self.vectorizer_ = TfidfVectorizer(sublinear_tf=True, min_df=1, dtype=np.float64)
            self.chunk_matrix_ = normalize(self.vectorizer_.fit_transform(texts)).T.tocsr()
            self.chunk_owner_ = np.asarray(owners)
        elif self.method == "shingles":
            self.index_: dict[int, set[int]] = defaultdict(set)
            for index, source in enumerate(sources):
                for key, _, _ in positioned_shingles(source.text, self.shingle_size):
                    self.index_[key].add(index)
        else:
            self.lsh_ = MinHashIndex(self.num_perm, self.bands, shingle_size=3, seed=self.seed)
            for index, source in enumerate(sources):
                for c, chunk in enumerate(chunks(source, self.chunk_sentences)):
                    self.lsh_.add((index, c), chunk)
        return self

    def scores(self, document: Any) -> np.ndarray:
        """Score de chaque source pour un document suspect."""
        n = len(self.source_ids_)
        result = np.zeros(n)
        if self.method == "tfidf":
            query = normalize(self.vectorizer_.transform(chunks(document, self.chunk_sentences)))
            similarity = (query @ self.chunk_matrix_).toarray().max(axis=0)
            np.maximum.at(result, self.chunk_owner_, similarity)
        elif self.method == "shingles":
            keys = {key for key, _, _ in positioned_shingles(document.text, self.shingle_size)}
            for key in keys:
                for index in self.index_.get(key, ()):
                    result[index] += 1
            if keys:
                result /= len(keys)
        else:
            for chunk in chunks(document, self.chunk_sentences):
                for chunk_key, jaccard in self.lsh_.query(chunk):
                    owner = int(chunk_key[0])  # type: ignore[index]
                    result[owner] = max(result[owner], jaccard)
        return result

    def retrieve(self, document: Any, k: int | None = None) -> list[tuple[str, float]]:
        """Les `k` sources de score le plus élevé (score strictement positif)."""
        k = k or self.k
        scores = self.scores(document)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.source_ids_[i], float(scores[i])) for i in order if scores[i] > 0]
