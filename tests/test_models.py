"""Tests des modèles : naïve Bayes maison identique à scikit-learn, enveloppes scikit-learn, classement top-k, ensemble ouvert et regroupeurs."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy import sparse
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from plagia.models import (
    CLASSIFIERS,
    CLUSTERERS,
    NaiveBayes,
    OpenSetClassifier,
    SklearnClassifier,
    build_classifier,
    build_clusterer,
    param_prefix,
    rank_from_scores,
    reduce_dimensions,
    set_random_state,
)
from plagia.models.base import decision_to_scores, top_features


@pytest.fixture(scope="module")
def counts() -> tuple[sparse.csr_matrix, np.ndarray]:
    rng = np.random.default_rng(1)
    X = sparse.random(240, 400, density=0.05, random_state=2, format="csr")
    X.data = np.floor(X.data * 10) + 1
    y = rng.choice(["anne", "bruno", "chloé", "dora"], size=240)
    return X, y


class TestNaiveBayes:
    @pytest.mark.parametrize("alpha", [0.01, 0.5, 1.0])
    def test_identical_to_multinomial_nb(self, counts, alpha: float) -> None:  # type: ignore[no-untyped-def]
        X, y = counts
        ours, ref = NaiveBayes(alpha=alpha).fit(X, y), MultinomialNB(alpha=alpha).fit(X, y)
        assert (ours.predict(X) == ref.predict(X)).all()
        assert np.abs(ours.predict_log_proba(X) - ref.predict_log_proba(X)).max() < 1e-6
        assert np.allclose(ours.feature_log_prob_, ref.feature_log_prob_) and np.allclose(
            ours.class_log_prior_, ref.class_log_prior_
        )
        dense = NaiveBayes(alpha=alpha).fit(X.toarray(), y)
        assert np.abs(dense.predict_log_proba(X.toarray()) - ref.predict_log_proba(X)).max() < 1e-6

    def test_identical_to_bernoulli_nb(self, counts) -> None:  # type: ignore[no-untyped-def]
        X, y = counts
        ours, ref = NaiveBayes(alpha=0.5, variant="bernoulli").fit(X, y), BernoulliNB(alpha=0.5).fit(X, y)
        assert (ours.predict(X) == ref.predict(X)).all()
        assert np.abs(ours.predict_log_proba(X) - ref.predict_log_proba(X)).max() < 1e-6

    def test_uniform_prior_explain_and_errors(self, counts) -> None:  # type: ignore[no-untyped-def]
        X, y = counts
        uniform = NaiveBayes(fit_prior=False).fit(X, y)
        assert np.allclose(uniform.class_log_prior_, -np.log(4))
        explanation = uniform.explain([f"f{i}" for i in range(400)], top_k=5)
        assert set(explanation) == {"anne", "bruno", "chloé", "dora"} and len(explanation["anne"]) == 5
        assert np.allclose(uniform.predict_proba(X).sum(axis=1), 1.0)
        assert uniform.rank(X[:2], 2)[0][0][0] in explanation
        assert NaiveBayes().fit(X[:5], ["a"] * 5).log_likelihood_ratios().shape == (1, 400)
        with pytest.raises(ValueError, match="positifs"):
            NaiveBayes().fit(np.array([[-1.0, 2.0]]), ["a"])
        with pytest.raises(ValueError, match="Variante"):
            NaiveBayes(variant="gaussian").fit(X, y)
        with pytest.raises(ValueError, match="alpha"):
            NaiveBayes(alpha=-1).fit(X, y)


class TestSklearnWrappers:
    def test_all_registered_classifiers(self, counts) -> None:  # type: ignore[no-untyped-def]
        X, y = counts
        names = [f"f{i}" for i in range(400)]
        for name in CLASSIFIERS.names():
            classifier = set_random_state(build_classifier(name), 7)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                classifier.fit(X[:180], y[:180])
            scores = classifier.predict_scores(X[180:])
            assert scores.shape == (60, 4) and np.allclose(scores.sum(axis=1), 1.0), name
            assert len(classifier.predict(X[180:])) == 60
            assert isinstance(classifier.explain(names, 3), dict)
        svm = build_classifier("linear_svm").fit(X, y)
        assert param_prefix(svm) == "estimator__" and param_prefix(NaiveBayes()) == ""
        assert svm.decision_function(X[:2]).shape == (2, 4)
        forest = build_classifier("random_forest").fit(X, y)
        assert "*" in forest.explain(names, 2) and forest.decision_function(X[:1]).shape == (1, 4)
        with pytest.raises(ValueError):
            SklearnClassifier().fit(X, y)

    def test_binary_coefficients(self) -> None:
        X = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
        y = ["no", "no", "yes", "yes"]
        model = build_classifier("logreg").fit(X, y)
        explanation = model.explain(["a", "b"], 1)
        assert explanation["yes"][0][0] == "b" and explanation["no"][0][0] == "a"

    def test_helpers(self) -> None:
        assert decision_to_scores(np.array([0.0, 2.0]), 2).shape == (2, 2)
        assert rank_from_scores(np.array([[0.1, 0.9]]), ["a", "b"], 1) == [[("b", 0.9)]]
        assert top_features(np.array([]), [], 3) == []


def test_open_set(counts) -> None:  # type: ignore[no-untyped-def]
    X, y = counts
    known = y != "dora"
    model = OpenSetClassifier(NaiveBayes(), threshold=0.99).fit(X[known], y[known])
    predictions = model.predict(X[~known])
    assert "<unknown>" in set(predictions)
    tau = model.calibrate(X, y)
    assert 0.0 <= tau <= 1.0 and model.threshold == tau
    assert model.calibrate(X, y, candidates=[0.5]) == 0.5
    assert model.explain([f"f{i}" for i in range(400)], 1)
    with pytest.raises(ValueError):
        OpenSetClassifier().fit(X, y)


class TestClusterers:
    def test_algorithms(self) -> None:
        rng = np.random.default_rng(0)
        centers = np.eye(3) * 10
        X = np.vstack([c + rng.normal(scale=0.3, size=(20, 3)) for c in centers])
        X = np.hstack([X, rng.random((60, 5))])
        for name in ("kmeans", "agglomerative"):
            labels = build_clusterer(name, n_clusters=3, n_components=None).fit_predict(X)
            assert len(set(labels)) == 3
        kmeans = build_clusterer("kmeans", n_clusters=3, n_components=4)
        kmeans.fit_predict(X)
        assert kmeans.inertia_ is not None and kmeans.embedding_.shape == (60, 4)
        assert len(build_clusterer("dbscan", eps=0.05, min_samples=3, n_components=4).fit_predict(X)) == 60
        assert len(build_clusterer("hdbscan", min_cluster_size=5, n_components=4).fit_predict(X)) == 60
        assert set(CLUSTERERS.names()) == {"kmeans", "agglomerative", "dbscan", "hdbscan"}
        assert reduce_dimensions(sparse.csr_matrix(X), None).shape == (60, 8)
