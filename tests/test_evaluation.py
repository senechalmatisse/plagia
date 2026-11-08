"""Tests de la couche evaluation : mesures de PAN sur des cas construits, regroupement invariant par permutation, bootstrap, tests statistiques, comparaisons appariées et aides à l'explication."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plagia.core.types import Passage, PassagePair
from plagia.evaluation import (
    ERROR_CATEGORIES,
    accuracy,
    accuracy_at_k,
    binary_metrics,
    bootstrap_ci,
    c_at_1,
    case_detection_rates,
    classification_metrics,
    clustering_metrics,
    cohen_kappa,
    confusion_pairs,
    error_sample,
    f1_macro,
    family_shares,
    granularity,
    holm,
    mcnemar,
    paired_bootstrap,
    paired_comparisons,
    pan_measures,
    plagdet,
    purity,
    recall_at_k,
    summarize,
    topk_accuracy,
    verification_metrics,
    wilcoxon_seeds,
)
from plagia.evaluation.pan import detects


def case(start: int, length: int, src_start: int = 0, susp: str = "s", src: str | None = "t") -> PassagePair:
    return PassagePair(Passage(susp, start, length), Passage(src, src_start, length) if src else None)


class TestPanMeasures:
    def test_perfect_detection_scores_one(self) -> None:
        cases = [case(0, 100), case(500, 50, 300)]
        assert pan_measures(cases, cases)["plagdet"] == pytest.approx(1.0)
        assert pan_measures([], [])["plagdet"] == 1.0

    def test_fragmented_detection_has_granularity_above_one(self) -> None:
        cases = [case(0, 100)]
        fragments = [case(0, 50), case(50, 50, 50)]
        measures = pan_measures(cases, fragments)
        assert measures["granularity"] == 2.0 and measures["recall"] == pytest.approx(1.0)
        assert measures["plagdet"] == pytest.approx(1.0 / math.log2(3))

    def test_partial_and_false_detections(self) -> None:
        cases = [case(0, 100)]
        half = pan_measures(cases, [case(0, 50)])
        assert half["recall"] == pytest.approx(0.5) and half["precision"] == pytest.approx(1.0)
        noise = pan_measures(cases, [case(0, 100), case(1000, 100, 1000)])
        assert noise["precision"] == pytest.approx(0.5) and noise["recall"] == pytest.approx(1.0)
        wrong_source = pan_measures(cases, [PassagePair(Passage("s", 0, 100), Passage("other", 0, 100))])
        assert wrong_source["recall"] == 0.0 and wrong_source["plagdet"] == 0.0
        assert pan_measures(cases, [])["recall"] == 0.0 and plagdet(0.5, 0.5, 0.5) == 0.0
        assert granularity(cases, []) == 1.0

    def test_intrinsic_cases_without_source(self) -> None:
        cases = [case(0, 100, src=None)]
        detection = [PassagePair(Passage("s", 50, 100))]
        measures = pan_measures(cases, detection)
        assert measures["recall"] == pytest.approx(0.5) and measures["precision"] == pytest.approx(0.5)
        assert detects(cases[0], detection[0])

    def test_passage_level_rates(self) -> None:
        cases = [case(0, 100), case(200, 100, 200)]
        rates = case_detection_rates(cases, [case(0, 90), case(500, 50, 500)])
        assert rates["passage_recall"] == 0.5 and rates["passage_precision"] == 0.5
        assert case_detection_rates([], [])["passage_precision"] == 1.0


class TestClassification:
    def test_metrics(self) -> None:
        truth, predicted = ["a", "a", "b", "c"], ["a", "b", "b", "c"]
        metrics = classification_metrics(truth, predicted)
        assert metrics["accuracy"] == 0.75 and metrics["confusion"]["matrix"][0] == [1, 1, 0]
        assert metrics["per_class"]["b"]["precision"] == 0.5 and metrics["f1_micro"] == 0.75
        assert f1_macro(truth, predicted) == pytest.approx(metrics["f1_macro"]) and f1_macro([], []) == 0.0
        assert accuracy(truth, predicted) == 0.75 and accuracy([], []) == 0.0
        rankings = [[("b", 0.6), ("a", 0.4)]] * 4
        assert topk_accuracy(truth, rankings, 1) == 0.25 and topk_accuracy(truth, rankings, 2) == 0.75
        assert topk_accuracy([], [], 1) == 0.0

    def test_detection_and_retrieval(self) -> None:
        metrics = binary_metrics([0, 1, 1, 0], [0.1, 0.9, 0.6, 0.7], 0.5)
        assert metrics["recall"] == 1.0 and metrics["precision"] == pytest.approx(2 / 3) and metrics["auc"] == 0.75
        assert binary_metrics([1, 1], [0.2, 0.9], 0.5)["auc"] == 0.5
        assert c_at_1([1, 0, 1, 1], [1, 0, None, 0]) == pytest.approx((2 + 1 * 2 / 4) / 4) and c_at_1([], []) == 0.0
        verification = verification_metrics([1, 0, 1, 0], [0.9, 0.2, 0.5, 0.4], [1, 0, None, 0])
        assert verification["abstentions"] == 1 and verification["auc"] == 1.0
        assert accuracy_at_k(["a", "b"], [["a", "x"], ["x", "b"]], 1) == 0.5 and accuracy_at_k([], [], 1) == 0.0
        assert recall_at_k([{"a"}, {"b", "c"}], [["a"], ["c", "x"]], 1) == pytest.approx(2 / 3)
        assert recall_at_k([], [], 3) == 1.0


class TestClustering:
    def test_invariance_to_label_permutation(self) -> None:
        truth = ["a", "a", "b", "b", "c", "c"]
        clusters = [0, 0, 1, 1, 2, 2]
        permuted = [5, 5, 3, 3, 9, 9]
        first, second = clustering_metrics(truth, clusters), clustering_metrics(truth, permuted)
        for key in ("purity", "nmi", "ari", "v_measure"):
            assert first[key] == pytest.approx(second[key]) == pytest.approx(1.0)
        assert purity(truth, [0] * 6) == pytest.approx(1 / 3) and purity([], []) == 0.0
        embedding = np.array([[0, 0], [0, 0.1], [5, 5], [5, 5.1], [9, 0], [9, 0.1]])
        assert clustering_metrics(truth, clusters, embedding)["silhouette"] > 0.8
        noisy = clustering_metrics(truth, [0, 0, 1, 1, -1, -1])
        assert noisy["noise"] == 2 and noisy["n_clusters"] == 2


class TestStatistics:
    def test_bootstrap(self) -> None:
        values = np.array([0.0, 1.0] * 50)
        interval = bootstrap_ci(lambda i: float(values[i].mean()), len(values), 500, 0.95, seed=1)
        assert interval["low"] < 0.5 < interval["high"] and interval["value"] == 0.5
        assert bootstrap_ci(lambda i: 1.0, 0)["low"] == 1.0
        better = np.array([1.0] * 80 + [0.0] * 20)
        worse = np.array([1.0] * 50 + [0.0] * 50)
        diff = paired_bootstrap(lambda i: float(better[i].mean()), lambda i: float(worse[i].mean()), 100, 500, seed=2)
        assert diff["diff"] == pytest.approx(0.3) and diff["low"] > 0 and diff["p_value"] < 0.05
        assert paired_bootstrap(lambda i: 1.0, lambda i: 1.0, 0)["p_value"] == 1.0

    def test_mcnemar_and_holm(self) -> None:
        a = [True] * 30 + [False] * 5
        b = [False] * 20 + [True] * 10 + [True] * 5
        result = mcnemar(a, b)
        assert (result["b"], result["c"]) == (20, 5) and result["exact"] is False and result["p_value"] < 0.05
        exact = mcnemar([True, True, False], [False, True, True])
        assert exact["exact"] and exact["p_value"] == 1.0
        assert mcnemar([True], [True])["p_value"] == 1.0
        corrected = holm([0.01, 0.04, 0.03, 0.005])
        assert corrected["adjusted"] == pytest.approx([0.03, 0.06, 0.06, 0.02]) and corrected["reject"] == [
            True,
            False,
            False,
            True,
        ]
        assert holm([]) == {"adjusted": [], "reject": []}

    def test_against_statsmodels_when_available(self) -> None:
        pytest.importorskip("statsmodels")
        from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar  # pragma: no cover
        from statsmodels.stats.multitest import multipletests  # pragma: no cover

        a = [True] * 30 + [False] * 5  # pragma: no cover
        b = [False] * 20 + [True] * 15  # pragma: no cover
        ours = mcnemar(a, b)  # pragma: no cover
        theirs = sm_mcnemar([[10, 20], [5, 0]], exact=False, correction=True)  # pragma: no cover
        assert ours["p_value"] == pytest.approx(theirs.pvalue)  # pragma: no cover
        p_values = [0.01, 0.04, 0.03, 0.005]  # pragma: no cover
        assert holm(p_values)["adjusted"] == pytest.approx(
            list(multipletests(p_values, method="holm")[1])
        )  # pragma: no cover

    def test_wilcoxon_requires_ten_seeds(self) -> None:
        with pytest.raises(ValueError, match="10 graines"):
            wilcoxon_seeds([0.1] * 5, [0.2] * 5)
        with pytest.raises(ValueError, match="longueur"):
            wilcoxon_seeds([0.1], [0.1, 0.2])
        assert wilcoxon_seeds([0.5] * 10, [0.5] * 10)["p_value"] == 1.0
        result = wilcoxon_seeds([0.9 + i / 100 for i in range(10)], [0.5 + i / 200 for i in range(10)])
        assert result["p_value"] < 0.01

    def test_kappa_and_summary(self) -> None:
        assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
        assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)
        assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0
        with pytest.raises(ValueError):
            cohen_kappa([], [])
        stats = summarize([0.8, 0.9, 1.0])
        assert stats["mean"] == pytest.approx(0.9) and stats["ci_low"] < 0.9 < stats["ci_high"]
        assert summarize([0.5])["std"] == 0.0 and summarize([])["n"] == 0


def test_paired_comparisons_with_holm() -> None:
    truth = [f"a{i % 3}" for i in range(60)]
    reference = {f"d{i}": (t, t) for i, t in enumerate(truth)}
    worse = {f"d{i}": (t, t if i % 3 else "zz") for i, t in enumerate(truth)}
    same = dict(reference)
    rows = paired_comparisons("ref", {"ref": reference, "worse": worse, "same": same}, n_resamples=200)
    by_name = {r["condition"]: r for r in rows}
    assert by_name["worse"]["diff"] < 0 and by_name["worse"]["diff_high"] < 0
    assert by_name["worse"]["mcnemar_p_significant"] and not by_name["same"]["mcnemar_p_significant"]
    assert by_name["same"]["diff"] == 0.0 and (by_name["worse"]["mcnemar_b"], by_name["worse"]["mcnemar_c"]) == (0, 20)
    with pytest.raises(KeyError):
        paired_comparisons("missing", {"x": reference})


def test_explanation_helpers() -> None:
    assert confusion_pairs(["a", "a", "b"], ["b", "b", "b"]) == [("a", "b", 2)]
    shares = family_shares(
        {
            "x": [("char_ngrams__c[ab]", 1.0), ("stylometry__fw:the", 0.5)],
            "y": [("char_ngrams__c[cd]", 1.0), ("raw", 0.1)],
        }
    )
    assert shares["char_ngrams"] == 0.5 and shares["other"] == 0.25
    sample = error_sample(
        ["1", "2", "3", "4"], ["a", "a", "b", "b"], ["b", "a", "a", "a"], ["t1", "t2", "t3", "t4"], n=2
    )
    assert len(sample) == 2 and {s["true"] for s in sample} == {"a", "b"} and sample[0]["category"] == ""
    assert "citation" in ERROR_CATEGORIES
