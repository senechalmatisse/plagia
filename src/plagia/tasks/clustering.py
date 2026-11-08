"""Regroupement de documents (T3) : choix du nombre de groupes, familles de quasi-doublons, interprétation des groupes et projection en deux dimensions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from plagia.annotation.document import AnnotatedDocument
from plagia.core.errors import MissingDependencyError
from plagia.models.clustering import reduce_dimensions
from plagia.similarity.matrix import knn_cosine, symmetrize


def choose_k(X: Any, k_values: Sequence[int], n_components: int | None = 100, seed: int = 0) -> dict[str, Any]:
    """Inertie (méthode du coude) et silhouette pour chaque k."""
    embedding = reduce_dimensions(X, n_components, seed)
    ks, inertias, silhouettes = [], [], []
    for k in k_values:
        if k < 2 or k >= embedding.shape[0]:
            continue
        model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(embedding)
        ks.append(int(k))
        inertias.append(float(model.inertia_))
        silhouettes.append(float(silhouette_score(embedding, model.labels_)))
    best_silhouette = ks[int(np.argmax(silhouettes))] if ks else None
    return {
        "k": ks,
        "inertia": inertias,
        "silhouette": silhouettes,
        "best_silhouette": best_silhouette,
        "elbow": elbow_point(ks, inertias),
    }


def elbow_point(ks: Sequence[int], inertias: Sequence[float]) -> int | None:
    """Coude : point le plus éloigné de la droite joignant les extrémités de la courbe."""
    if len(ks) < 3:
        return ks[0] if ks else None
    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertias, dtype=float)
    x = (x - x.min()) / (np.ptp(x) or 1.0)
    y = (y - y.min()) / (np.ptp(y) or 1.0)
    start, end = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    direction = (end - start) / (np.linalg.norm(end - start) or 1.0)
    points = np.column_stack([x, y]) - start
    distances = np.abs(points[:, 0] * direction[1] - points[:, 1] * direction[0])
    return int(ks[int(np.argmax(distances))])


def near_duplicate_families(
    X: Any, threshold: float = 0.8, k: int = 10, method: str = "components", seed: int = 0
) -> np.ndarray:
    """Familles de quasi-doublons : graphe des similarités supérieures au seuil,
    composantes connexes ou communautés de Louvain (NetworkX)."""
    graph = symmetrize(knn_cosine(X, k=k, min_similarity=threshold))
    if method == "components":
        _, labels = connected_components(graph, directed=False)
        return np.asarray(labels)
    try:
        import networkx as nx
    except ImportError as exc:
        raise MissingDependencyError("networkx", "communautés de Louvain", "uv sync --extra graph") from exc
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.shape[0]))
    coo = sparse.triu(graph, k=1).tocoo()
    nx_graph.add_weighted_edges_from(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist(), strict=True))
    labels = np.empty(graph.shape[0], dtype=int)
    for index, community in enumerate(nx.community.louvain_communities(nx_graph, weight="weight", seed=seed)):
        for node in community:
            labels[node] = index
    return labels


def characterize_clusters(
    X: Any, labels: Sequence[int] | np.ndarray, feature_names: Sequence[str], top_k: int = 10
) -> dict[int, list[tuple[str, float]]]:
    """Termes les plus caractéristiques de chaque groupe (centroïde moins moyenne globale)."""
    X = sparse.csr_matrix(X)
    labels_array = np.asarray(labels)
    overall = np.asarray(X.mean(axis=0)).ravel()
    result: dict[int, list[tuple[str, float]]] = {}
    for label in sorted(set(labels_array.tolist())):
        members = np.where(labels_array == label)[0]
        centroid = np.asarray(X[members].mean(axis=0)).ravel() - overall
        order = np.argsort(-centroid)[:top_k]
        result[int(label)] = [(str(feature_names[i]), float(centroid[i])) for i in order if centroid[i] > 0]
    return result


def entity_profiles(
    documents: Sequence[AnnotatedDocument], labels: Sequence[int] | np.ndarray, top_k: int = 5
) -> dict[int, dict[str, Any]]:
    """Entités et types d'entités les plus fréquents de chaque groupe."""
    profiles: dict[int, dict[str, Any]] = {}
    for label in sorted({int(x) for x in labels}):
        forms: Counter[str] = Counter()
        kinds: Counter[str] = Counter()
        for doc, lab in zip(documents, labels, strict=True):
            if int(lab) == label:
                forms.update(e.text for e in doc.entities)
                kinds.update(e.kind for e in doc.entities)
        profiles[label] = {"entities": forms.most_common(top_k), "types": kinds.most_common(top_k)}
    return profiles


def project_2d(X: Any, method: str = "tsne", seed: int = 0, n_components_before: int = 50) -> np.ndarray:
    """Projection en deux dimensions (t-SNE, UMAP ou SVD) pour la visualisation."""
    if method == "svd":
        reduced: np.ndarray = TruncatedSVD(n_components=2, random_state=seed).fit_transform(X)
        return reduced
    base = reduce_dimensions(X, min(n_components_before, max(2, X.shape[1] - 1)), seed)
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise MissingDependencyError("umap-learn", "projection UMAP", "uv sync --extra viz") from exc
        projected: np.ndarray = umap.UMAP(n_components=2, random_state=seed, metric="cosine").fit_transform(base)
        return projected
    perplexity = float(min(30.0, max(2.0, (base.shape[0] - 1) / 3)))
    projected = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca").fit_transform(base)
    return projected
