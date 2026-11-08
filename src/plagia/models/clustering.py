"""Regroupeurs derrière l'interface `Clusterer`.

k-means après réduction par SVD tronquée, classification hiérarchique agglomérative (liaison moyenne, distance cosinus) et, en option, DBSCAN ou HDBSCAN.
Les vecteurs sont normalisés (L2) après réduction : distance euclidienne et cosinus ordonnent alors les paires de la même façon.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from plagia.core.registry import Registry

CLUSTERERS: Registry[Any] = Registry("algorithme de regroupement")


def reduce_dimensions(X: Any, n_components: int | None, random_state: int = 0) -> np.ndarray:
    """SVD tronquée puis normalisation L2 (`n_components=None` : pas de réduction)."""
    if n_components is None:
        dense = X.toarray() if sparse.issparse(X) else np.asarray(X, dtype=float)
        return normalize(dense)
    k = max(1, min(n_components, X.shape[1] - 1, X.shape[0] - 1))
    reduced = TruncatedSVD(n_components=k, random_state=random_state).fit_transform(X)
    return normalize(reduced)


class Clusterer(ABC):
    """Interface commune des algorithmes de regroupement."""

    deterministic: bool = True

    def __init__(self, n_components: int | None = 100, random_state: int = 0) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.embedding_: np.ndarray | None = None

    def reduce(self, X: Any) -> np.ndarray:
        """Représentation réduite utilisée pour le regroupement (conservée pour la silhouette)."""
        self.embedding_ = reduce_dimensions(X, self.n_components, self.random_state)
        return self.embedding_

    @abstractmethod
    def fit_predict(self, X: Any) -> np.ndarray:
        """Étiquettes de groupe (`-1` pour le bruit de DBSCAN/HDBSCAN)."""


@CLUSTERERS.register("kmeans")
class KMeansClusterer(Clusterer):
    """k-means sur la représentation réduite."""

    deterministic = False

    def __init__(
        self, n_clusters: int = 8, n_components: int | None = 100, n_init: int = 10, random_state: int = 0
    ) -> None:
        super().__init__(n_components, random_state)
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.inertia_: float | None = None

    def fit_predict(self, X: Any) -> np.ndarray:
        model = KMeans(n_clusters=self.n_clusters, n_init=self.n_init, random_state=self.random_state)
        labels: np.ndarray = model.fit_predict(self.reduce(X))
        self.inertia_ = float(model.inertia_)
        return labels


@CLUSTERERS.register("agglomerative")
class AgglomerativeClusterer(Clusterer):
    """Classification hiérarchique agglomérative (liaison moyenne, distance cosinus)."""

    def __init__(
        self,
        n_clusters: int = 8,
        n_components: int | None = 100,
        linkage: str = "average",
        metric: str = "cosine",
        random_state: int = 0,
    ) -> None:
        super().__init__(n_components, random_state)
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric

    def fit_predict(self, X: Any) -> np.ndarray:
        model = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage, metric=self.metric)
        labels: np.ndarray = model.fit_predict(self.reduce(X))
        return labels


@CLUSTERERS.register("dbscan")
class DbscanClusterer(Clusterer):
    """DBSCAN (distance cosinus)."""

    def __init__(
        self,
        eps: float = 0.3,
        min_samples: int = 5,
        n_components: int | None = 100,
        metric: str = "cosine",
        random_state: int = 0,
        n_clusters: int | None = None,
    ) -> None:
        super().__init__(n_components, random_state)
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def fit_predict(self, X: Any) -> np.ndarray:
        labels: np.ndarray = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric).fit_predict(
            self.reduce(X)
        )
        return labels


@CLUSTERERS.register("hdbscan")
class HdbscanClusterer(Clusterer):
    """HDBSCAN (vecteurs normalisés : distance euclidienne équivalente au cosinus)."""

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        n_components: int | None = 100,
        random_state: int = 0,
        n_clusters: int | None = None,
    ) -> None:
        super().__init__(n_components, random_state)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples

    def fit_predict(self, X: Any) -> np.ndarray:
        model = HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, copy=True)
        labels: np.ndarray = model.fit_predict(self.reduce(X))
        return labels


def build_clusterer(name: str, **params: Any) -> Clusterer:
    """Regroupeur enregistré sous `name`."""
    clusterer: Clusterer = CLUSTERERS.create(name, **params)
    return clusterer
