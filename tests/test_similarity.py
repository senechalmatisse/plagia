"""Tests des mesures de similarité, de MinHash/LSH, du fenêtrage, des matrices kNN et du choix de seuil."""

from __future__ import annotations

import random

import numpy as np
import pytest

from plagia.annotation.document import AnnotatedDocument, Entity
from plagia.core.errors import MissingDependencyError
from plagia.similarity import (
    SIMILARITIES,
    Containment,
    Cosine,
    Dice,
    EntitySimilarity,
    Jaccard,
    Lcs,
    Levenshtein,
    MinHasher,
    MinHashIndex,
    Overlap,
    SemanticSimilarity,
    TypeDivergence,
    WordMoverSimilarity,
    containment,
    cosine_rows,
    dice,
    jaccard,
    jensen_shannon,
    knn_cosine,
    lcs_length,
    levenshtein_distance,
    pr_curve,
    select_threshold,
    symmetrize,
    top_k_per_row,
    winnow,
)
from plagia.similarity.minhash import text_shingle_hashes
from plagia.similarity.sequence import lcs_length_dp, levenshtein_dp

A = "the quick brown fox jumps over the lazy dog near the river bank today"
B = "the quick brown fox jumps over the lazy cat near the river bank"


class TestSetMeasures:
    def test_known_values(self) -> None:
        x, y = {"a", "b", "c"}, {"b", "c", "d", "e"}
        assert jaccard(x, y) == pytest.approx(2 / 5) and dice(x, y) == pytest.approx(4 / 7)
        assert containment(x, y) == pytest.approx(2 / 3) and containment(set(), y) == 0.0
        assert jaccard(set(), set()) == 0.0 and dice(set(), set()) == 0.0

    def test_classes(self) -> None:
        assert Jaccard(unit="word").score(A, A) == 1.0
        assert 0 < Jaccard(n=3).score(A, B) < 1 and Dice(unit="char", n=4).score(A, B) > 0.5
        assert Containment(n=3).score(B, A) > Containment(n=3).score(A, B) and not Containment.symmetric
        assert Overlap(n=2).score(A, B) <= 1.0 and Overlap(n=2).score("", "") == 0.0
        assert Jaccard().score({"x"}, {"x"}) == 1.0
        with pytest.raises(ValueError):
            Jaccard(unit="sentence")
        assert set(SIMILARITIES.names()) >= {
            "jaccard",
            "dice",
            "containment",
            "cosine",
            "lcs",
            "levenshtein",
            "entity",
            "type_divergence",
            "semantic",
        }


class TestSequenceMeasures:
    def test_bit_parallel_matches_dynamic_programming(self) -> None:
        rng = random.Random(0)
        for _ in range(400):
            a = [rng.choice("abcde") for _ in range(rng.randint(0, 90))]
            b = [rng.choice("abcdef") for _ in range(rng.randint(0, 90))]
            assert lcs_length(a, b) == lcs_length_dp(a, b)
            assert levenshtein_distance(a, b) == levenshtein_dp(a, b)

    def test_normalized_scores(self) -> None:
        assert Levenshtein().score(A, A) == 1.0 and Levenshtein().score("", "") == 0.0
        assert Levenshtein().score("a b c", "a x c") == pytest.approx(2 / 3)
        assert Lcs().score("a b c d", "a c") == 0.5 and Lcs(normalize="min").score("a b c d", "a c") == 1.0
        assert Lcs(normalize="first").score(["a", "b"], ["b"]) == 0.5 and not Lcs(normalize="first").symmetric
        with pytest.raises(ValueError):
            Lcs(normalize="avg")


class TestVectorAndSemantic:
    def test_cosine(self) -> None:
        unfitted = Cosine()
        assert unfitted.score(A, A) == pytest.approx(1.0) and 0 < unfitted.score(A, B) < 1
        assert len(unfitted.score_many([(A, B), (A, A)])) == 2 and unfitted.score_many([]).size == 0
        fitted = Cosine().fit([A, B, "something else entirely"])
        assert fitted.score_many([(A, A)])[0] == pytest.approx(1.0)
        assert Cosine(weighting="binary").fit([A, B]).score(A, B) > 0
        assert np.allclose(cosine_rows(np.eye(2), np.eye(2)), [1, 1])
        assert "Cosine(" in repr(fitted)

    def test_semantic(self, literary) -> None:  # type: ignore[no-untyped-def]
        texts = literary.texts[:30]
        semantic = SemanticSimilarity(n_components=5).fit(texts)
        assert semantic.score(texts[0], texts[0]) == pytest.approx(1.0)
        assert semantic.score_many([(texts[0], texts[1])]).shape == (1,) and semantic.score_many([]).size == 0
        with pytest.raises(RuntimeError):
            SemanticSimilarity().embed(["x"])
        with pytest.raises(MissingDependencyError):
            WordMoverSimilarity().score("a", "b")

        class FakeVectors:
            def wmdistance(self, a: list[str], b: list[str]) -> float:
                return 1.0

        assert WordMoverSimilarity(FakeVectors()).score("a", "b") == 0.5


class TestEntityMeasures:
    def test_entity_and_type_divergence(self) -> None:
        def doc(*entities: tuple[str, str]) -> AnnotatedDocument:
            return AnnotatedDocument.from_text("d", "x").replace(
                entities=tuple(Entity(t, 0, 1, k, k) for t, k in entities)
            )

        a = doc(("Paris", "LOC"), ("IBM", "ORG"))
        b = doc(("paris", "LOC"), ("Bob", "PER"))
        assert EntitySimilarity().score(a, b) == pytest.approx(1 / 3)
        assert EntitySimilarity(method="cosine").score(a, b) == pytest.approx(0.5)
        assert TypeDivergence().score(a, a) == 1.0 and 0 < TypeDivergence().score(a, b) < 1
        assert jensen_shannon({"x": 1}, {"y": 1}) == pytest.approx(1.0) and jensen_shannon({}, {}) == 0.0
        assert jensen_shannon({}, {"y": 1}) == 1.0
        with pytest.raises(ValueError):
            EntitySimilarity(method="dice")
        typed = AnnotatedDocument.from_text("d", "x").replace(
            entities=(Entity("Z", 0, 1, "P", "PER", "u", ("0:owl:Thing", "1:dbo:Agent")),)
        )
        assert TypeDivergence(depth=1).distribution(typed) == {"dbo:Agent": 1}
        assert EntitySimilarity(level="uri").score(typed, typed) == 1.0


class TestMinHash:
    def test_estimate_close_to_jaccard(self) -> None:
        hasher = MinHasher(num_perm=256, seed=3)
        x, y = text_shingle_hashes(A, 2), text_shingle_hashes(B, 2)
        exact = len(set(x) & set(y)) / len(set(x) | set(y))
        assert abs(MinHasher.jaccard(hasher.signature(x), hasher.signature(y)) - exact) < 0.15
        assert (
            hasher.signature(np.array([], dtype=np.uint64)) == hasher.signature(np.array([], dtype=np.uint64))
        ).all()

    def test_lsh_index(self, literary) -> None:  # type: ignore[no-untyped-def]
        index = MinHashIndex(num_perm=64, bands=16, shingle_size=3)
        for doc in literary[:20]:
            index.add(doc.id, doc.text)
        index.add_many([("extra", text_shingle_hashes("completely different words here", 3))])
        assert len(index) == 21 and 0 < index.threshold < 1
        hits = index.query(literary[5].text)
        assert hits[0] == (literary[5].id, 1.0)
        assert index.query(literary[5].text, min_jaccard=1.1) == []
        with pytest.raises(ValueError):
            MinHashIndex(num_perm=10, bands=3)

    def test_winnowing(self) -> None:
        assert winnow([5, 3, 8, 3, 9, 1, 1, 7], 3) == [(3, 1), (3, 3), (1, 5), (1, 6)]
        assert winnow([4], 3) == [(4, 0)] and winnow([], 2) == []
        with pytest.raises(ValueError):
            winnow([1], 0)


class TestMatricesAndThresholds:
    def test_knn_matches_dense(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.random((25, 6))
        S = knn_cosine(X, k=3, block_size=7)
        normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
        dense = normalized @ normalized.T
        np.fill_diagonal(dense, -1)
        for i in range(25):
            assert set(S.getrow(i).indices) == set(np.argsort(-dense[i])[:3])
        assert symmetrize(S).nnz >= S.nnz and top_k_per_row(S, 2)[0][0][1] >= top_k_per_row(S, 2)[0][1][1]
        assert knn_cosine(X[:3], X[3:], k=2, exclude_self=False).shape == (3, 22)
        assert knn_cosine(X[:1], k=5).nnz == 0

    def test_threshold_selection(self) -> None:
        labels = [0, 0, 1, 1, 1, 0, 1]
        scores = [0.1, 0.4, 0.35, 0.8, 0.9, 0.2, 0.7]
        choice = select_threshold(labels, scores)
        assert choice.f1 == pytest.approx(8 / 9) and 0.2 < choice.threshold <= 0.35
        strict = select_threshold(labels, scores, criterion="precision", target_precision=1.0)
        assert strict.precision == 1.0 and strict.recall == 0.75 and 0.4 < strict.threshold <= 0.7
        assert select_threshold([1, 1], [0.2, 0.4]).criterion.endswith("(dégénéré)")
        separable = select_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        assert separable.threshold == pytest.approx(0.5)  # milieu de l'intervalle vide
        assert set(pr_curve(labels, scores)) == {"precision", "recall", "thresholds"}
        assert choice.to_dict()["criterion"] == "f1"
