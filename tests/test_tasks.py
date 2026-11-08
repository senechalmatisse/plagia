"""Tests des tâches : attribution (réglage, classement, explication), vérification,
plagiat (détection des copies exactes > 95 % — critère du jalon J5 —, paires, origine,
intrinsèque, rapport) et regroupement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plagia.annotation.document import AnnotatedDocument
from plagia.core.config import BlockConfig, ModelConfig, RepresentationConfig
from plagia.core.types import Corpus, Document, Passage, PassagePair
from plagia.evaluation.classification import f1_macro
from plagia.evaluation.pan import case_detection_rates, pan_measures
from plagia.tasks import AuthorshipAttributor, AuthorVerifier, ImpostersVerifier, make_pairs
from plagia.tasks.clustering import (
    characterize_clusters,
    choose_k,
    elbow_point,
    entity_profiles,
    near_duplicate_families,
    project_2d,
)
from plagia.tasks.plagiarism import (
    CandidateRetriever,
    Extender,
    ExtrinsicDetector,
    Filter,
    IntrinsicDetector,
    OriginIdentifier,
    PairClassifier,
    PairFeatureExtractor,
    PassageView,
    Seed,
    SeedMatcher,
    TextAligner,
    build_pair_examples,
    build_report,
    sliding_windows,
    subsample_prevalence,
    word_windows,
)
from plagia.tasks.plagiarism.passages import windows_inside, windows_outside
from plagia.tasks.plagiarism.report import covered_length
from plagia.tasks.verification import decide

STYLE = RepresentationConfig(
    blocks=[BlockConfig(type="char_ngrams", params={"ngram_range": [2, 4]}), BlockConfig(type="stylometry", weight=0.5)]
)


def _split(literary: Corpus, annotated: dict[str, AnnotatedDocument]) -> tuple[list, list, list, list]:  # type: ignore[type-arg]
    train, test = literary.split("train"), literary.split("test")
    return [annotated[d.id] for d in train], train.labels, [annotated[d.id] for d in test], test.labels


class TestAttribution:
    def test_fit_predict_rank_explain(
        self, literary: Corpus, annotated: dict[str, AnnotatedDocument], tmp_path: Path
    ) -> None:
        X_train, y_train, X_test, y_test = _split(literary, annotated)
        model = AuthorshipAttributor(STYLE, ModelConfig(type="linear_svm")).fit(X_train, y_train)
        predicted = model.predict(X_test)
        assert f1_macro(y_test, predicted) > 0.5  # 4 auteurs : hasard ≈ 0,25
        ranking = model.rank(X_test, 3)
        assert len(ranking[0]) == 3 and ranking[0][0][1] >= ranking[0][1][1]
        explanation = model.explain(5)
        assert set(explanation) == set(y_train) and all(len(v) == 5 for v in explanation.values())
        contributions = model.block_contributions()
        assert set(contributions) == {"char_ngrams", "stylometry"} and sum(contributions.values()) == pytest.approx(1.0)
        assert model.predict_scores(X_test).shape == (len(X_test), 4)
        path = model.save(tmp_path / "model.joblib")
        assert AuthorshipAttributor.load(path).predict(X_test) == predicted

    def test_tuning_grouped_by_work(self, literary: Corpus, annotated: dict[str, AnnotatedDocument]) -> None:
        X_train, y_train, _, _ = _split(literary, annotated)
        groups = [d.group for d in literary.split("train")]
        model = AuthorshipAttributor(
            RepresentationConfig(blocks=[BlockConfig(type="bow", params={"stopwords_source": "builtin"})]),
            ModelConfig(type="naive_bayes"),
        )
        summary = model.tune(X_train, y_train, {"alpha": [0.1, 1.0], "features__bow__min_df": [1]}, groups=groups, cv=3)
        assert summary["folds"] == 3 and summary["best_params"]["clf__alpha"] in {0.1, 1.0}
        assert "features__bow__min_df" in summary["best_params"]
        assert model.block_contributions() == pytest.approx({"bow": 1.0})
        untuned = AuthorshipAttributor(STYLE, ModelConfig(type="logreg")).tune(X_train, y_train, {}, cv=3)
        assert untuned["best_params"] == {}

    def test_majority_and_tree_contributions(self, literary: Corpus, annotated: dict[str, AnnotatedDocument]) -> None:
        X_train, y_train, _, _ = _split(literary, annotated)
        assert (
            AuthorshipAttributor(STYLE, ModelConfig(type="majority")).fit(X_train, y_train).block_contributions() == {}
        )
        forest = AuthorshipAttributor(STYLE, ModelConfig(type="random_forest", params={"n_estimators": 10})).fit(
            X_train, y_train
        )
        assert forest.block_contributions()


class TestVerification:
    def test_pairs_and_classifier(self, literary: Corpus, annotated: dict[str, AnnotatedDocument]) -> None:
        X_train, y_train, X_test, y_test = _split(literary, annotated)
        pairs = make_pairs(y_train, 60, seed=1)
        labels = [label for _, _, label in pairs]
        assert len(pairs) == 60 and 20 <= sum(labels) <= 40
        assert all((y_train[i] == y_train[j]) == bool(lab) for i, j, lab in pairs)
        verifier = AuthorVerifier(STYLE, seed=0).fit(X_train, pairs)
        test_pairs = make_pairs(y_test, 30, seed=2)
        probabilities = verifier.predict_proba(X_test, [(i, j) for i, j, _ in test_pairs])
        from sklearn.metrics import roc_auc_score

        assert roc_auc_score([lab for _, _, lab in test_pairs], probabilities) > 0.6
        assert decide(np.array([0.5, 0.9, 0.1]), margin=0.05) == [None, 1, 0]
        assert decide(np.array([0.5]), margin=0.0) == [0]

    def test_imposters(self, literary: Corpus, annotated: dict[str, AnnotatedDocument]) -> None:
        X_train, _, X_test, y_test = _split(literary, annotated)
        verifier = ImpostersVerifier(STYLE, n_iterations=10, n_imposters=5, seed=0).fit(X_train)
        pairs = make_pairs(y_test, 12, seed=3)
        scores = verifier.score(X_test, [(i, j) for i, j, _ in pairs])
        same = np.mean([s for s, (_, _, lab) in zip(scores, pairs, strict=True) if lab == 1])
        different = np.mean([s for s, (_, _, lab) in zip(scores, pairs, strict=True) if lab == 0])
        assert ((scores >= 0) & (scores <= 1)).all() and same >= different


def _sources_and_test(corpus: Corpus, annotated: dict[str, AnnotatedDocument]) -> tuple[list, list]:  # type: ignore[type-arg]
    sources = [annotated[d.id] for d in corpus if d.meta["role"] == "source"]
    test = [annotated[d.id] for d in corpus if d.meta["role"] == "suspicious" and d.split in {"test", "validation"}]
    return sources, test


class TestPlagiarism:
    def test_exact_copies_detected_above_95_percent(
        self, plagiarism_corpus: Corpus, plagiarism_annotated: dict[str, AnnotatedDocument]
    ) -> None:
        """Jalon J5 : copie exacte (O0) détectée avec précision et rappel > 95 % au niveau des passages."""
        sources, suspicious = _sources_and_test(plagiarism_corpus, plagiarism_annotated)
        detector = ExtrinsicDetector(
            CandidateRetriever("tfidf", k=5), TextAligner(extender=Extender(min_density=3.0))
        ).fit(sources)
        o0_docs = [d for d in suspicious if plagiarism_corpus.get(d.id).meta["level"] in {"O0", None}]
        detections = [p for d in o0_docs for p in detector.detect(d)]
        cases = [
            c
            for c in plagiarism_corpus.cases
            if c.obfuscation == "O0" and c.suspicious.document_id in {d.id for d in o0_docs}
        ]
        rates = case_detection_rates(cases, detections)
        assert len(cases) >= 4
        assert rates["passage_precision"] > 0.95 and rates["passage_recall"] > 0.95
        measures = pan_measures(cases, detections)
        assert measures["plagdet"] > 0.9 and measures["granularity"] == pytest.approx(1.0)

    def test_retrieval_methods(
        self, plagiarism_corpus: Corpus, plagiarism_annotated: dict[str, AnnotatedDocument]
    ) -> None:
        sources, suspicious = _sources_and_test(plagiarism_corpus, plagiarism_annotated)
        truth = {c.suspicious.document_id: c.source.document_id for c in plagiarism_corpus.cases if c.source}
        for method, minimum in (("tfidf", 0.9), ("shingles", 0.9), ("minhash", 0.3)):
            retriever = CandidateRetriever(method, k=5).fit(sources)
            hits = [truth[d.id] in {s for s, _ in retriever.retrieve(d)} for d in suspicious if d.id in truth]
            assert np.mean(hits) >= minimum, method
        with pytest.raises(ValueError):
            CandidateRetriever("bm25")

    def test_seed_matching_extension_and_filter(self) -> None:
        source = Document(
            "src", "Alpha beta gamma delta epsilon zeta eta theta iota kappa. Unrelated words follow here now."
        )
        suspicious = Document("susp", "Intro text here. Alpha beta gamma delta epsilon zeta eta theta iota kappa. End.")
        seeds = SeedMatcher(method="ngrams", ngram_size=3).seeds(suspicious, source)
        assert seeds and all(isinstance(s, Seed) and s.kind == "ngram" for s in seeds)
        sentence_seeds = SeedMatcher(method="sentences").seeds(suspicious, source)
        assert any(s.kind == "sentence" for s in sentence_seeds)
        clusters = Extender(max_gap=50, min_seeds=2, min_density=0).extend(seeds)
        assert len(clusters) == 1 and clusters[0].n_seeds == len(seeds)
        detections = TextAligner(SeedMatcher(ngram_size=3), Extender(min_density=0), Filter(min_length=20)).align(
            suspicious, source
        )
        assert len(detections) == 1 and suspicious.text[
            detections[0].suspicious.start : detections[0].suspicious.end
        ].startswith("Alpha")
        overlapping = [
            PassagePair(Passage("d", 0, 100), Passage("s", 0, 100), 0.9),
            PassagePair(Passage("d", 10, 100), Passage("t", 0, 100), 0.5),
            PassagePair(Passage("d", 300, 10), Passage("t", 0, 10), 0.9),
        ]
        kept = Filter(min_length=50).apply(overlapping)
        assert [k.source.document_id for k in kept] == ["s"]  # type: ignore[union-attr]
        for confirmation in ("containment", "edit"):
            aligned = TextAligner(
                SeedMatcher(ngram_size=3), Extender(min_density=0), Filter(20), confirmation=confirmation
            )
            assert aligned.align(suspicious, source) and aligned.align(suspicious, source)[0].score > 0.5
        with pytest.raises(ValueError, match="Confirmation"):
            TextAligner(confirmation="magic")
        verified = TextAligner(
            SeedMatcher(ngram_size=3), Extender(min_density=0), Filter(20), verifier=lambda pairs: np.zeros(len(pairs))
        ).align(suspicious, source)
        assert verified == []

    def test_pair_dataset_classifier_and_origin(
        self, plagiarism_corpus: Corpus, plagiarism_annotated: dict[str, AnnotatedDocument]
    ) -> None:
        train = build_pair_examples(plagiarism_corpus, plagiarism_annotated, "train", 3, 3, seed=1)
        validation = build_pair_examples(plagiarism_corpus, plagiarism_annotated, "validation", 3, 3, seed=2)
        test = build_pair_examples(
            plagiarism_corpus, plagiarism_annotated, "test", 3, 10, seed=3, levels=["O0", "O1"], sizes=[100, 300]
        )
        positives = [e for e in train if e.label == 1]
        assert positives and {e.kind for e in train} >= {"positive", "same_source", "other_source"}
        assert all(e.level in {"O0", "O1"} and e.size in {100, 300} for e in test)
        balanced = subsample_prevalence(test, 1)
        assert sum(e.label for e in balanced) * 2 >= len(balanced)
        extractor = PairFeatureExtractor(
            [
                "cosine_tfidf",
                "containment_shingles",
                "lcs",
                "jaccard_entities",
                "type_divergence",
                "semantic",
                "jaccard_shingles",
                "dice_words",
                "levenshtein",
                "length_ratio",
                "cosine_entities",
                "kb_type_divergence",
            ]
        )
        extractor.fit([d.text for d in plagiarism_corpus if d.meta["role"] == "source"])
        features = extractor.transform([e.pair for e in train[:10]])
        assert features.shape == (10, 12) and np.isfinite(features).all()
        assert extractor.transform([]).shape == (0, 12)
        classifier = PairClassifier(
            PairFeatureExtractor().fit([d.text for d in plagiarism_corpus]), ModelConfig(type="logreg")
        )
        classifier.fit([e.pair for e in train], [e.label for e in train])
        choice = classifier.calibrate([e.pair for e in validation], [e.label for e in validation])
        assert choice.f1 > 0.8 and classifier.threshold == choice.threshold
        predicted = classifier.predict([e.pair for e in test])
        assert np.mean(predicted == np.array([e.label for e in test])) > 0.85
        assert classifier.predict_proba([]).size == 0 and classifier.as_verifier() == classifier.predict_proba
        case = next(
            c for c in plagiarism_corpus.cases if c.suspicious.document_id in {e.suspicious.document_id for e in test}
        )
        view = PassageView.of(
            plagiarism_annotated[case.suspicious.document_id], case.suspicious.start, case.suspicious.end
        )
        assert case.source is not None
        candidates = [
            plagiarism_annotated[case.source.document_id],
            *[plagiarism_annotated[d.id] for d in plagiarism_corpus if d.meta["role"] == "source"][:3],
        ]
        ranking = OriginIdentifier(classifier.predict_proba).rank(view, candidates)
        assert ranking[0][0] == case.source.document_id
        assert OriginIdentifier(classifier.predict_proba).rank(view, []) == []
        with pytest.raises(Exception, match="inconnues"):
            PairFeatureExtractor(["nope"])

    def test_windows_mode(self, plagiarism_corpus: Corpus, plagiarism_annotated: dict[str, AnnotatedDocument]) -> None:
        train = build_pair_examples(plagiarism_corpus, plagiarism_annotated, "train", 3, 3, seed=1)
        classifier = PairClassifier(PairFeatureExtractor().fit(plagiarism_corpus.texts)).fit(
            [e.pair for e in train], [e.label for e in train]
        )
        sources, suspicious = _sources_and_test(plagiarism_corpus, plagiarism_annotated)
        detector = ExtrinsicDetector(CandidateRetriever(k=2), pair_classifier=classifier, mode="both").fit(sources)
        plagiarized = next(d for d in suspicious if plagiarism_corpus.get(d.id).meta["level"] == "O0")
        detections = detector.detect_many([plagiarized])[plagiarized.id]
        assert detections
        case = next(c for c in plagiarism_corpus.cases if c.suspicious.document_id == plagiarized.id)
        assert case.source is not None
        windows = detector.window_detections(plagiarized, plagiarism_annotated[case.source.document_id])
        assert windows and all(p.meta["mode"] == "windows" for p in windows)
        assert any(p.suspicious.overlaps(case.suspicious) for p in windows)
        assert ExtrinsicDetector(mode="windows").window_detections(plagiarized, plagiarized) == []
        with pytest.raises(ValueError):
            ExtrinsicDetector(mode="magic")

    def test_intrinsic_detection(self) -> None:
        plain = " ".join(["the cat sat on the mat and the dog lay by the door ."] * 40)
        odd = " ".join(["Notwithstanding; extraordinarily, magnanimous juxtapositions! Indubitably?"] * 25)
        text = f"{plain} {odd} {plain}"
        document = Document("d", text)
        detections = IntrinsicDetector(window_words=60, step_words=30, k=1.0).detect(document)
        start = text.index("Notwithstanding")
        assert detections and any(p.suspicious.start <= start + 200 and p.suspicious.end >= start for p in detections)
        assert IntrinsicDetector(window_words=500).detect(Document("short", "few words")) == []
        assert IntrinsicDetector(window_words=60, step_words=30, method="kmeans").detect(document) is not None
        assert word_windows("a b c d e", 2, 2)[-1] == (6, 9) and word_windows("", 2, 1) == []

    def test_passages_and_report(self) -> None:
        spans = [(0, 5), (6, 10), (11, 15), (16, 20)]
        assert sliding_windows(spans, 2) == [(0, 10), (6, 15), (11, 20)] and sliding_windows([], 2) == []
        assert sliding_windows(spans[:1], 3) == [(0, 5)]
        assert windows_inside(spans, 0, 20, 3) == [(0, 20)] and windows_outside([(0, 5), (6, 10)], [(4, 7)]) == []
        detections = [
            PassagePair(Passage("d", 0, 10), Passage("s", 0, 10), 0.8),
            PassagePair(Passage("d", 5, 10), Passage("t", 0, 10), 0.4),
        ]
        report = build_report("d", "x" * 30, detections)
        assert report.coverage == pytest.approx(0.5) and report.score == pytest.approx(0.6)
        assert set(report.sources) == {"s", "t"} and report.to_dict()["document"] == "d"
        assert covered_length([(0, 10), (5, 12), (20, 25)]) == 17
        assert build_report("e", "", []).coverage == 0.0


class TestClustering:
    def test_choose_k_and_elbow(self, news: Corpus) -> None:
        from plagia.features import build_representation

        union = build_representation([BlockConfig(type="tfidf", params={"stopwords_source": "builtin"})])
        X = union.fit_transform(news.texts)
        selection = choose_k(X, range(2, 7), n_components=10)
        assert selection["k"] == [2, 3, 4, 5, 6] and selection["best_silhouette"] in selection["k"]
        assert selection["elbow"] in selection["k"]
        assert elbow_point([2, 3], [5.0, 4.0]) == 2 and elbow_point([], []) is None
        labels = np.array([i % 3 for i in range(len(news))])
        terms = characterize_clusters(X, labels, list(union.get_feature_names_out()), 3)
        assert set(terms) == {0, 1, 2}
        assert project_2d(X, "svd").shape == (len(news), 2) and project_2d(X, "tsne").shape == (len(news), 2)

    def test_families_and_profiles(self, annotated: dict[str, AnnotatedDocument]) -> None:
        X = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99], [0.7, 0.7]])
        components = near_duplicate_families(X, threshold=0.95, k=2)
        assert components[0] == components[1] and components[2] == components[3] and components[0] != components[2]
        louvain = near_duplicate_families(X, threshold=0.95, k=2, method="louvain")
        assert louvain[0] == louvain[1]
        docs = list(annotated.values())[:6]
        profiles = entity_profiles(docs, [0, 0, 0, 1, 1, 1])
        assert set(profiles) == {0, 1} and "entities" in profiles[0]
