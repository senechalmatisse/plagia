"""Exécution d'une condition de l'étude B (détection de plagiat).

Pour une cellule (niveau d'obfuscation, taille de passage), un jeu de caractéristiques de paires et un classificateur : paires équilibrées et à prévalence réaliste,
identification de l'origine, rappel de la recherche des sources et, au niveau des passages, mesures de PAN (plagdet).
La recherche des sources et l'alignement par graines, indépendants de la condition, sont calculés une seule fois par processus.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from plagia.core.config import ModelConfig
from plagia.core.types import PassagePair
from plagia.evaluation.bootstrap import bootstrap_ci
from plagia.evaluation.detection import accuracy_at_k, binary_metrics, recall_at_k
from plagia.evaluation.pan import case_detection_rates, detects, pan_measures
from plagia.experiments.context import ExperimentContext
from plagia.experiments.plan import Condition
from plagia.experiments.result import RunResult, add_ci
from plagia.similarity.threshold import pr_curve, select_threshold
from plagia.tasks.plagiarism.alignment import Extender, Filter, SeedMatcher, TextAligner
from plagia.tasks.plagiarism.dataset import PairExample, build_pair_examples, subsample_prevalence
from plagia.tasks.plagiarism.origin import OriginIdentifier
from plagia.tasks.plagiarism.pairs import PairClassifier, PairFeatureExtractor
from plagia.tasks.plagiarism.passages import PassageView, detection_views
from plagia.tasks.plagiarism.pipeline import ExtrinsicDetector
from plagia.tasks.plagiarism.retrieval import CandidateRetriever

BASELINE_FEATURES = {"baseline_cosine": "cosine_tfidf", "baseline_containment": "containment_shingles"}


def _base(ctx: ExperimentContext) -> dict[str, Any]:
    """Calculs indépendants de la condition : index, candidates, alignements bruts."""
    if "plagiarism" in ctx.shared:
        base: dict[str, Any] = ctx.shared["plagiarism"]
        return base
    section = ctx.config.plagiarism
    assert section is not None
    corpus = ctx.corpus(section.corpus)
    annotations = ctx.annotations(section.corpus)
    suspicious = sorted((d for d in corpus if d.meta.get("role") == "suspicious"), key=lambda d: d.id)
    if section.max_suspicious is not None:
        suspicious = suspicious[: section.max_suspicious]
    sources = [annotations[d.id] for d in corpus if d.meta.get("role") == "source"]
    r = section.retrieval
    if r.method == "pairs":  # alignement de textes PAN : paires (suspect, source) fournies
        retriever = None
        candidates: dict[str, list[str]] = {d.id: [] for d in suspicious}
        for susp_id, source_id in corpus.pairs:
            if susp_id in candidates and source_id in annotations:
                candidates[susp_id].append(source_id)
    else:
        retriever = CandidateRetriever(r.method, r.k, r.chunk_sentences, r.shingle_size, r.num_perm, r.bands).fit(
            sources
        )
        candidates = {d.id: [s for s, _ in retriever.retrieve(annotations[d.id])] for d in suspicious}
    base = {
        "corpus": corpus,
        "annotations": annotations,
        "suspicious": suspicious,
        "retriever": retriever,
        "candidates": candidates,
        "extractors": {},
        "examples": {},
    }
    ctx.shared["plagiarism"] = base
    return base


def _raw_alignments(base: dict[str, Any], ctx: ExperimentContext) -> dict[str, list[PassagePair]]:
    """Alignements par graines des documents de test (avant vérification), calculés une fois."""
    if "raw_alignments" not in base:
        section = ctx.config.plagiarism
        assert section is not None
        a = section.alignment
        matcher = SeedMatcher(a.seeds, a.ngram_size, a.sentence_threshold, a.dice_threshold)
        aligner = TextAligner(
            matcher,
            Extender(a.max_gap, a.min_seeds, a.min_density),
            Filter(0),
            min_similarity=a.min_similarity,
            confirmation=a.confirmation,
        )
        annotations, candidates = base["annotations"], base["candidates"]
        base["raw_alignments"] = {
            d.id: [p for s in candidates[d.id] for p in aligner.candidates(annotations[d.id], annotations[s])]
            for d in base["suspicious"]
            if d.split == "test"
        }
    raw: dict[str, list[PassagePair]] = base["raw_alignments"]
    return raw


def _examples(
    base: dict[str, Any], ctx: ExperimentContext, split: str, level: str, size: int, ratio: int, seed: int
) -> list[PairExample]:
    section = ctx.config.plagiarism
    assert section is not None
    key = (split, level, size, ratio, seed)
    if key not in base["examples"]:
        allowed = {d.id for d in base["suspicious"]}
        corpus = base["corpus"].filter(lambda d: d.meta.get("role") == "source" or d.id in allowed)
        base["examples"][key] = build_pair_examples(
            corpus,
            base["annotations"],
            split,
            section.window_sentences,
            ratio,
            seed=seed,
            levels=[level],
            sizes=[size],
            candidates=base["candidates"],
        )
    examples: list[PairExample] = base["examples"][key]
    return examples


def _extractor(base: dict[str, Any], features: Sequence[str]) -> PairFeatureExtractor:
    key = tuple(features)
    if key not in base["extractors"]:
        texts = [d.text for d in base["corpus"] if d.meta.get("role") == "source" and d.split == "train"]
        base["extractors"][key] = PairFeatureExtractor(features).fit(texts or [d.text for d in base["corpus"]])
    extractor: PairFeatureExtractor = base["extractors"][key]
    return extractor


class _Scorer:
    """Score de paire : classificateur appris ou mesure brute (lignes de base)."""

    def __init__(self, classifier: PairClassifier | None, extractor: PairFeatureExtractor) -> None:
        self.classifier = classifier
        self.extractor = extractor
        self.threshold = 0.5

    def __call__(self, pairs: Sequence[tuple[PassageView, PassageView]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0)
        if self.classifier is not None:
            return self.classifier.predict_proba(pairs)
        values: np.ndarray = self.extractor.transform(pairs)[:, 0]
        return values


def run_plagiarism(ctx: ExperimentContext, condition: Condition, seed: int) -> RunResult:
    """Évalue une cellule de l'étude B."""
    section = ctx.config.plagiarism
    protocol = ctx.config.protocol
    assert section is not None
    params = condition.params
    level, size = params["level"], int(params["size"])
    base = _base(ctx)
    max_ratio = max(section.test_prevalence)
    train = _examples(base, ctx, "train", level, size, section.negatives_per_positive, seed)
    validation = _examples(base, ctx, "validation", level, size, section.negatives_per_positive, seed)
    test_pool = _examples(base, ctx, "test", level, size, max_ratio, seed)

    if condition.baseline:
        extractor = _extractor(base, [BASELINE_FEATURES[params["classifier"]]])
        scorer = _Scorer(None, extractor)
    else:
        extractor = _extractor(base, section.feature_sets[params["feature_set"]])
        classifier = PairClassifier(
            extractor, section.classifiers[params["classifier"]] or ModelConfig(type="logreg"), seed
        )
        classifier.fit([e.pair for e in train], [e.label for e in train])
        scorer = _Scorer(classifier, extractor)
    choice = select_threshold(
        [e.label for e in validation],
        scorer([e.pair for e in validation]),
        section.threshold_criterion,
        section.target_precision,
    )
    scorer.threshold = choice.threshold
    if scorer.classifier is not None:
        scorer.classifier.threshold = choice.threshold

    metrics: dict[str, Any] = {
        "threshold": choice.threshold,
        "n_train_pairs": len(train),
        "n_validation_pairs": len(validation),
    }
    artifacts: dict[str, Any] = {"calibration": choice.to_dict()}
    predictions: list[dict[str, Any]] = []
    for ratio in section.test_prevalence:
        subset = subsample_prevalence(test_pool, ratio)
        labels = np.array([e.label for e in subset])
        scores = scorer([e.pair for e in subset])
        result = binary_metrics(labels, scores, choice.threshold)
        prefix = f"pairs_1to{ratio}"
        for name in ("precision", "recall", "f1", "average_precision", "auc"):
            metrics[f"{prefix}_{name}"] = result[name]
        metrics[f"{prefix}_n"] = len(subset)
        predicted = (scores >= choice.threshold).astype(int)

        def _pair_f1(index: np.ndarray, labels: np.ndarray = labels, predicted: np.ndarray = predicted) -> float:
            return _f1(labels[index], predicted[index])

        interval = bootstrap_ci(_pair_f1, len(labels), protocol.bootstrap, protocol.confidence, seed)
        add_ci(metrics, f"{prefix}_f1", interval)
        if ratio == max_ratio:
            artifacts["pr_curve"] = pr_curve(labels, scores) if 0 < labels.sum() < len(labels) else {}
            predictions = [
                {
                    "susp": e.suspicious.document_id,
                    "s": [e.suspicious.start, e.suspicious.end],
                    "src": e.source.document_id,
                    "t": [e.source.start, e.source.end],
                    "label": int(e.label),
                    "kind": e.kind,
                    "score": round(float(s), 6),
                }
                for e, s in zip(subset, scores, strict=True)
            ]
    if scorer.classifier is not None:
        coef = scorer.classifier.classifier.explain(extractor.features, top_k=len(extractor.features))
        artifacts["explanation"] = {k: [[f, w] for f, w in v] for k, v in coef.items()}

    metrics.update(_origin_and_retrieval(base, scorer, level, size, section.origin_k, section.window_sentences))
    if section.passage_level:
        cell_metrics, detections = _passages(base, ctx, scorer, level, size, seed, condition.baseline)
        metrics.update(cell_metrics)
        artifacts["detections"] = [d.to_dict() for d in detections]
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


def _f1(labels: np.ndarray, predicted: np.ndarray) -> float:
    tp = float(np.sum((labels == 1) & (predicted == 1)))
    fp = float(np.sum((labels == 0) & (predicted == 1)))
    fn = float(np.sum((labels == 1) & (predicted == 0)))
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def _cell_cases(base: dict[str, Any], level: str | None, size: int | None) -> list[PassagePair]:
    """Cas de test de la cellule (tous les cas de test si `level` et `size` valent None)."""
    test_ids = {d.id for d in base["suspicious"] if d.split == "test"}
    return [
        c
        for c in base["corpus"].cases
        if c.suspicious.document_id in test_ids
        and (level is None or c.obfuscation == level)
        and (size is None or int(c.meta.get("size", 0)) == size)
    ]


def _origin_and_retrieval(
    base: dict[str, Any], scorer: _Scorer, level: str, size: int, k: int, window: int
) -> dict[str, float]:
    cases = _cell_cases(base, level, size)
    annotations, candidates = base["annotations"], base["candidates"]
    retrieved = [candidates.get(c.suspicious.document_id, []) for c in cases]
    relevant = [{c.source.document_id} if c.source else set() for c in cases]
    identifier = OriginIdentifier(scorer, window)
    rankings = []
    for case in cases:
        view = PassageView.of(annotations[case.suspicious.document_id], case.suspicious.start, case.suspicious.end)
        ranked = identifier.rank(view, [annotations[s] for s in candidates.get(case.suspicious.document_id, [])])
        rankings.append([s for s, _ in ranked])
    truth = [c.source.document_id if c.source else "" for c in cases]
    return {
        "retrieval_recall_at_k": recall_at_k(
            relevant, retrieved, len(retrieved[0]) if retrieved and retrieved[0] else 1
        ),
        "origin_accuracy_at_1": accuracy_at_k(truth, rankings, 1),
        f"origin_accuracy_at_{k}": accuracy_at_k(truth, rankings, k),
        "n_test_cases": len(cases),
    }


def _passages(
    base: dict[str, Any], ctx: ExperimentContext, scorer: _Scorer, level: str, size: int, seed: int, baseline: bool
) -> tuple[dict[str, Any], list[PassagePair]]:
    """Détection au niveau des passages sur les documents de test de la cellule et les documents sans plagiat."""
    section = ctx.config.plagiarism
    assert section is not None
    a = section.alignment
    cases = _cell_cases(base, level, size)
    all_cases = _cell_cases(base, None, None)
    cell_docs = {c.suspicious.document_id for c in cases}
    case_docs = {c.suspicious.document_id for c in all_cases}
    # Documents de la cellule et documents sans plagiat ; les détections d'un cas d'une autre
    # cellule (même document, PAN) ne comptent ni comme vrais ni comme faux positifs.
    documents = [d for d in base["suspicious"] if d.split == "test" and (d.id in cell_docs or d.id not in case_docs)]
    others: dict[str, list[PassagePair]] = {}
    for case in all_cases:
        if case.suspicious.document_id in cell_docs and (
            case.obfuscation != level or int(case.meta.get("size", 0)) != size
        ):
            others.setdefault(case.suspicious.document_id, []).append(case)
    final = Filter(a.min_length)
    annotations = base["annotations"]
    detections: list[PassagePair] = []
    per_doc: dict[str, list[PassagePair]] = {}
    windows = ExtrinsicDetector(
        pair_classifier=scorer.classifier, mode="windows", window_sentences=section.window_sentences
    )
    for doc in documents:
        found: list[PassagePair] = []
        if a.mode in {"seeds", "both"}:
            found = list(_raw_alignments(base, ctx).get(doc.id, []))
            if a.verify and not baseline and found and scorer.classifier is not None:
                probabilities = scorer(detection_views(annotations[doc.id], annotations.__getitem__, found))
                found = [
                    PassagePair(p.suspicious, p.source, float(s), meta=p.meta)
                    for p, s in zip(found, probabilities, strict=True)
                    if s >= scorer.threshold
                ]
        if a.mode in {"windows", "both"} and scorer.classifier is not None:
            for source_id in base["candidates"].get(doc.id, []):
                found.extend(windows.window_detections(annotations[doc.id], annotations[source_id]))
        kept = final.apply(found)
        per_doc[doc.id] = [p for p in kept if not any(detects(c, p) for c in others.get(doc.id, []))]
        detections.extend(per_doc[doc.id])
    measures = pan_measures(cases, detections)
    metrics: dict[str, Any] = {
        "pan_precision": measures["precision"],
        "pan_recall": measures["recall"],
        "pan_granularity": measures["granularity"],
        "plagdet": measures["plagdet"],
    }
    metrics.update(case_detection_rates(cases, detections))
    doc_ids = [d.id for d in documents]
    case_by_doc: dict[str, list[PassagePair]] = {}
    for case in cases:
        case_by_doc.setdefault(case.suspicious.document_id, []).append(case)

    def _plagdet(index: np.ndarray) -> float:
        chosen = [doc_ids[i] for i in index]
        sample_cases = [c for d in chosen for c in case_by_doc.get(d, [])]
        sample_dets = [p for d in chosen for p in per_doc.get(d, [])]
        return pan_measures(sample_cases, sample_dets)["plagdet"]

    resamples = min(ctx.config.protocol.bootstrap, 200)
    interval = bootstrap_ci(_plagdet, len(doc_ids), resamples, ctx.config.protocol.confidence, seed)
    metrics["plagdet_low"], metrics["plagdet_high"] = interval["low"], interval["high"]
    metrics["n_passage_documents"] = len(documents)
    return metrics, detections
