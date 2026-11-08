"""Exécution d'une condition de vérification d'auteur."""

from __future__ import annotations

from typing import Any

import numpy as np

from plagia.evaluation.bootstrap import bootstrap_ci
from plagia.evaluation.detection import verification_metrics
from plagia.experiments.context import ExperimentContext
from plagia.experiments.plan import Condition
from plagia.experiments.result import RunResult, add_ci
from plagia.similarity.threshold import select_threshold
from plagia.tasks.verification import AuthorVerifier, ImpostersVerifier, decide, make_pairs


def run_verification(ctx: ExperimentContext, condition: Condition, seed: int) -> RunResult:
    """Paires d'entraînement (jeu train) et de test (jeu test), mesures AUC, F1 et c@1."""
    section = ctx.config.verification
    protocol = ctx.config.protocol
    assert section is not None
    params = condition.params
    corpus = ctx.corpus(section.corpus)
    annotations = ctx.annotations(section.corpus)
    train, test = corpus.split("train"), corpus.split("test")
    train_docs = [annotations[d.id] for d in train]
    test_docs = [annotations[d.id] for d in test]
    train_pairs = make_pairs(train.labels, section.pairs_per_split, seed)
    test_pairs = make_pairs(test.labels, section.pairs_per_split, seed + 1)
    representation = section.representations[params["representation"]]
    labels = np.array([y for _, _, y in test_pairs])
    if params["method"] == "imposters":
        verifier = ImpostersVerifier(representation, n_iterations=50, seed=seed).fit(train_docs)
        calibration_pairs = train_pairs[: min(100, len(train_pairs))]
        calibration_scores = verifier.score(train_docs, [(i, j) for i, j, _ in calibration_pairs])
        threshold = select_threshold([y for _, _, y in calibration_pairs], calibration_scores).threshold
        scores = verifier.score(test_docs, [(i, j) for i, j, _ in test_pairs])
        probabilities = np.clip(0.5 + (scores - threshold), 0, 1)
    else:
        model = AuthorVerifier(representation, section.classifiers[params["method"]], seed).fit(train_docs, train_pairs)
        probabilities = model.predict_proba(test_docs, [(i, j) for i, j, _ in test_pairs])
    decisions = decide(probabilities, section.abstain_margin)
    metrics: dict[str, Any] = dict(verification_metrics(labels.tolist(), probabilities.tolist(), decisions))
    from sklearn.metrics import roc_auc_score

    def _auc(index: np.ndarray) -> float:
        y = labels[index]
        return float(roc_auc_score(y, probabilities[index])) if 0 < y.sum() < len(y) else 0.5

    add_ci(metrics, "auc", bootstrap_ci(_auc, len(labels), protocol.bootstrap, protocol.confidence, seed))
    metrics.update({"n_train_pairs": len(train_pairs), "n_test_pairs": len(test_pairs)})
    predictions = [
        {"a": test[i].id, "b": test[j].id, "label": int(y), "score": round(float(p), 6), "decision": d}
        for (i, j, y), p, d in zip(test_pairs, probabilities, decisions, strict=True)
    ]
    return RunResult("", condition.id, seed, dict(params), metrics, predictions=predictions)
