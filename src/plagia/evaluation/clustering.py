"""Évaluation du regroupement : pureté, NMI, ARI, V-mesure (externes, invariantes par permutation des étiquettes) et silhouette (interne)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
    silhouette_score,
)


def purity(labels_true: Sequence[Any] | np.ndarray, labels_pred: Sequence[Any] | np.ndarray) -> float:
    """Part des documents appartenant à la classe majoritaire de leur groupe."""
    if len(labels_true) == 0:
        return 0.0
    clusters: dict[Any, Counter[Any]] = {}
    for t, p in zip(labels_true, labels_pred, strict=True):
        clusters.setdefault(p, Counter())[t] += 1
    return sum(c.most_common(1)[0][1] for c in clusters.values()) / len(labels_true)


def clustering_metrics(
    labels_true: Sequence[Any] | np.ndarray,
    labels_pred: Sequence[Any] | np.ndarray,
    embedding: np.ndarray | None = None,
) -> dict[str, float]:
    """Mesures externes et, si une représentation est fournie, silhouette."""
    truth = [str(x) for x in labels_true]
    predicted = [int(x) for x in labels_pred]
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(truth, predicted)
    result = {
        "purity": purity(truth, predicted),
        "nmi": float(normalized_mutual_info_score(truth, predicted)),
        "ari": float(adjusted_rand_score(truth, predicted)),
        "v_measure": float(v_measure),
        "homogeneity": float(homogeneity),
        "completeness": float(completeness),
        "n_clusters": len(set(predicted) - {-1}),
        "noise": int(sum(1 for x in predicted if x == -1)),
    }
    if embedding is not None and 1 < len(set(predicted)) < len(predicted):
        result["silhouette"] = float(silhouette_score(embedding, predicted))
    return result
