"""Exécution d'une condition de l'étude C (regroupement)."""

from __future__ import annotations

from typing import Any

import numpy as np

from plagia.evaluation.clustering import clustering_metrics
from plagia.experiments.context import ExperimentContext
from plagia.experiments.plan import Condition
from plagia.experiments.result import RunResult
from plagia.features.representation import build_representation
from plagia.models.clustering import build_clusterer
from plagia.tasks.clustering import (
    characterize_clusters,
    choose_k,
    entity_profiles,
    near_duplicate_families,
    project_2d,
)


def run_clustering(ctx: ExperimentContext, condition: Condition, seed: int) -> RunResult:
    """Représentation, réduction, regroupement et évaluation externe et interne."""
    section = ctx.config.clustering
    assert section is not None
    params = condition.params
    corpus_config = section.corpora[params["corpus"]]
    corpus = ctx.corpus(corpus_config.path)
    if corpus_config.split:
        corpus = corpus.split(corpus_config.split)
    annotations = ctx.annotations(corpus_config.path)
    documents = [annotations[d.id] for d in corpus]
    union = build_representation(section.representations[params["representation"]])
    X = union.fit_transform(documents)
    dimensions = params["dimensions"]

    metrics: dict[str, Any] = {"n_documents": len(documents)}
    artifacts: dict[str, Any] = {}
    if section.k == "true":
        k = len(set(corpus.labels))
    elif isinstance(section.k, int):
        k = section.k
    else:
        selection = choose_k(X, section.k_range, dimensions, seed)
        artifacts["k_selection"] = selection
        k = selection["best_silhouette"] if section.k == "silhouette" else selection["elbow"]
        k = k or len(set(corpus.labels))
    metrics["k"] = k
    algorithm = section.algorithms[params["algorithm"]]
    clusterer = build_clusterer(
        algorithm.type, n_clusters=k, n_components=dimensions, random_state=seed, **algorithm.params
    )
    labels = clusterer.fit_predict(X)
    for reference in corpus_config.labels:
        truth = corpus.labels if reference == "author" else corpus.groups
        values = clustering_metrics(truth, labels, clusterer.embedding_)
        for name, value in values.items():
            metrics[f"{reference}_{name}"] = value
    if section.families_threshold is not None:
        families = near_duplicate_families(X, section.families_threshold, seed=seed)
        sizes = np.bincount(families)
        metrics["families"] = int(np.sum(sizes > 1))
        metrics["family_documents"] = int(np.sum(sizes[sizes > 1]))
    names = [str(n) for n in union.get_feature_names_out()]
    artifacts["characteristic_features"] = {str(k): v for k, v in characterize_clusters(X, labels, names, 10).items()}
    artifacts["entity_profiles"] = {str(k): v for k, v in entity_profiles(documents, labels).items()}
    if seed == condition.seeds[0] and len(documents) >= 5:
        coords = project_2d(X, "tsne" if len(documents) <= 3000 else "svd", seed)
        artifacts["projection"] = {
            "x": np.round(coords[:, 0], 4).tolist(),
            "y": np.round(coords[:, 1], 4).tolist(),
            "cluster": [int(x) for x in labels],
            "author": corpus.labels,
            "work": corpus.groups,
        }
    predictions = [
        {"id": d.id, "cluster": int(c), "author": d.author, "work": d.group}
        for d, c in zip(corpus, labels, strict=True)
    ]
    return RunResult("", condition.id, seed, dict(params), metrics, predictions=predictions, artifacts=artifacts)
