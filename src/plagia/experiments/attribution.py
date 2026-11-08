"""Exécution d'une condition de l'étude A (attribution d'auteur)."""

from __future__ import annotations

from typing import Any

import numpy as np

from plagia.core.config import BlockConfig, ModelConfig, RepresentationConfig
from plagia.core.seeds import make_random
from plagia.corpus.splits import subsample_authors, subsample_per_author
from plagia.evaluation.bootstrap import bootstrap_ci
from plagia.evaluation.classification import accuracy, classification_metrics, f1_macro, topk_accuracy
from plagia.evaluation.explain import confusion_pairs, error_sample, family_shares
from plagia.experiments.context import ExperimentContext
from plagia.experiments.plan import Condition, Plan
from plagia.experiments.result import RunResult, add_ci
from plagia.models.open_set import OpenSetClassifier
from plagia.tasks.attribution import AuthorshipAttributor

MAJORITY_REPRESENTATION = RepresentationConfig(blocks=[BlockConfig(type="bow", params={"stopwords_source": "builtin"})])


def _representations(ctx: ExperimentContext) -> dict[str, RepresentationConfig]:
    if "representations" not in ctx.shared:
        ctx.shared["representations"] = Plan.from_config(ctx.config).representations
    reps: dict[str, RepresentationConfig] = ctx.shared["representations"]
    return reps


def run_attribution(ctx: ExperimentContext, condition: Condition, seed: int) -> RunResult:
    """Entraîne (avec réglage par validation croisée) et évalue une condition."""
    config = ctx.config
    section = config.attribution
    protocol = config.protocol
    assert section is not None
    params = condition.params
    corpus = ctx.corpus(section.corpus)
    annotations = ctx.annotations(section.corpus)

    chosen = subsample_authors(corpus, params["n_authors"], seed)
    unknown: set[str] = set()
    if (
        section.open_set is not None
        and not condition.baseline
        and len(chosen.authors) > section.open_set.unknown_authors + 1
    ):
        unknown = set(make_random(seed, "open-set").sample(chosen.authors, section.open_set.unknown_authors))
    known = subsample_per_author(chosen.filter(lambda d: d.author not in unknown), params["n_train"], seed, "train")
    train, test = known.split("train"), known.split("test")
    length = params["text_length"]
    X_train = [annotations[d.id] for d in train]
    y_train = train.labels
    X_test = [annotations[d.id].truncate_words(length) if length else annotations[d.id] for d in test]
    y_test = test.labels
    groups = [d.group for d in train] if protocol.group_by == "work" else None

    if condition.baseline:
        representation, classifier = MAJORITY_REPRESENTATION, ModelConfig(type="majority")
    else:
        representation = _representations(ctx)[params["representation"]]
        classifier = section.classifiers[params["classifier"]]
    model = AuthorshipAttributor(representation, classifier, seed=seed)
    if protocol.tune and classifier.grid:
        summary = model.tune(X_train, y_train, classifier.grid, groups, protocol.cv_folds, protocol.scoring)
    else:
        model.fit(X_train, y_train)
        summary = {}

    predicted = model.predict(X_test)
    ranking = model.rank(X_test, section.top_k)
    full = classification_metrics(y_test, predicted)
    metrics: dict[str, Any] = {k: v for k, v in full.items() if isinstance(v, (int, float))}
    truth, guess = np.asarray(y_test), np.asarray(predicted)
    n = len(truth)
    add_ci(
        metrics,
        "f1_macro",
        bootstrap_ci(lambda i: f1_macro(truth[i], guess[i]), n, protocol.bootstrap, protocol.confidence, seed),
    )
    add_ci(
        metrics,
        "accuracy",
        bootstrap_ci(lambda i: accuracy(truth[i], guess[i]), n, protocol.bootstrap, protocol.confidence, seed),
    )
    metrics[f"top{section.top_k}_accuracy"] = topk_accuracy(y_test, ranking, section.top_k)
    metrics.update(
        {
            "n_train": len(y_train),
            "n_test": n,
            "n_authors": len(set(y_train)),
            "cv_best_score": summary.get("best_score"),
        }
    )

    artifacts: dict[str, Any] = {
        "confusion": full["confusion"],
        "per_class": full["per_class"],
        "best_params": summary.get("best_params", {}),
        "confusions": confusion_pairs(y_test, predicted),
        "errors": error_sample(
            [d.id for d in test], y_test, predicted, [d.text for d in X_test], section.error_sample, seed
        ),
    }
    if not condition.baseline:
        explanation = model.explain(section.explain_top_k)
        artifacts["explanation"] = {k: [[f, w] for f, w in v] for k, v in explanation.items()}
        artifacts["family_shares"] = family_shares(explanation)
        contributions = model.block_contributions()
        artifacts["block_contributions"] = contributions
        for block, share in contributions.items():
            metrics[f"contribution_{block}"] = share
    if unknown:
        metrics.update(_open_set(model, chosen, annotations, unknown, test, seed, section.open_set.threshold))  # type: ignore[union-attr]

    predictions = [
        {"id": d.id, "true": t, "pred": p, "top": [[label, round(score, 6)] for label, score in r]}
        for d, t, p, r in zip(test, y_test, predicted, ranking, strict=True)
    ]
    return RunResult(
        "",
        condition.id,
        seed,
        dict(params),
        metrics,
        predictions=predictions,
        artifacts=artifacts,
        baseline=condition.baseline,
    )


def _open_set(
    model: AuthorshipAttributor,
    chosen: Any,
    annotations: Any,
    unknown: set[str],
    test: Any,
    seed: int,
    threshold: float | str,
) -> dict[str, Any]:
    """Évaluation en ensemble ouvert : moitié des textes pour calibrer τ, moitié pour l'évaluer."""
    rng = make_random(seed, "open-set-halves")
    unknown_docs = sorted((d for d in chosen if d.author in unknown and d.split == "test"), key=lambda d: d.id)
    known_docs = sorted(test, key=lambda d: d.id)
    rng.shuffle(unknown_docs)
    rng.shuffle(known_docs)
    half_u, half_k = len(unknown_docs) // 2, len(known_docs) // 2
    calibration = known_docs[:half_k] + unknown_docs[:half_u]
    evaluation = known_docs[half_k:] + unknown_docs[half_u:]
    features = model.pipeline.named_steps["features"]
    open_set = OpenSetClassifier(model.classifier, threshold=0.5 if threshold == "auto" else float(threshold))
    open_set.classes_ = model.classifier.classes_
    if threshold == "auto" and calibration:
        open_set.calibrate(
            features.transform([annotations[d.id] for d in calibration]), [d.author for d in calibration]
        )
    predicted = open_set.predict(features.transform([annotations[d.id] for d in evaluation]))
    truth = [d.author if d.author not in unknown else open_set.unknown_label for d in evaluation]
    unknown_mask = [t == open_set.unknown_label for t in truth]
    unknown_recall = (
        float(np.mean([p == open_set.unknown_label for p, u in zip(predicted, unknown_mask, strict=True) if u]))
        if any(unknown_mask)
        else float("nan")
    )
    return {
        "open_set_f1_macro": f1_macro(truth, [str(p) for p in predicted]),
        "open_set_unknown_recall": unknown_recall,
        "open_set_threshold": open_set.threshold,
    }
