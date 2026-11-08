"""Tests des expériences : plans des études A, B et C, exécution des quatre tâches, reprise, reproductibilité, résultats CSV, comparaisons appariées et rapports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from plagia.core.config import ExperimentConfig, load_config
from plagia.core.io import read_jsonl
from plagia.core.types import Corpus
from plagia.corpus.store import save_corpus
from plagia.experiments import Plan, ResultStore, Runner
from plagia.experiments.context import get_context
from plagia.experiments.runner import execute
from plagia.experiments.store import safe_name
from plagia.reporting import generate_report

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("name", "conditions", "baselines", "runs"),
    [
        ("study_a_c50.yaml", 288, 12, 1500),
        ("study_a_literary.yaml", 24, 1, 125),
        ("study_b_synthetic.yaml", 48, 24, 360),
        ("study_b_synthetic_extended.yaml", 108, 24, 660),
        ("study_c_clustering.yaml", 40, 0, 120),
    ],
)
def test_study_plans_match_specification(name: str, conditions: int, baselines: int, runs: int) -> None:
    plan = Plan.from_config(load_config(ROOT / "configs" / "studies" / name))
    summary = plan.summary()
    assert (summary["conditions"], summary["baseline_conditions"], summary["runs"]) == (conditions, baselines, runs)
    assert len({run.key for run in plan.runs}) == len(plan.runs)


def test_all_shipped_configs_are_valid() -> None:
    for path in sorted((ROOT / "configs").rglob("*.yaml")):
        if path.parent.name in {"studies", "demo"}:
            assert Plan.from_config(load_config(path)).conditions, path


def _attribution_config(corpus: Path, **overrides: Any) -> ExperimentConfig:
    data: dict[str, Any] = {
        "name": "t_attr",
        "task": "attribution",
        "seed": 3,
        "output_dir": "results",
        "annotation": {"engine": "heuristic", "cache_path": "cache/ann.sqlite"},
        "protocol": {"cv_folds": 2, "seeds": 2, "bootstrap": 50, "group_by": "work"},
        "attribution": {
            "corpus": str(corpus),
            "representations": {
                "R1": {"blocks": [{"type": "bow", "params": {"stopwords_source": "builtin"}}]},
                "Style": {
                    "blocks": [
                        {"type": "char_ngrams", "params": {"ngram_range": [2, 3]}},
                        {"type": "stylometry", "weight": 0.5},
                        {"type": "entities", "weight": 0.5},
                    ]
                },
            },
            "classifiers": {"nb": {"type": "naive_bayes", "grid": {"alpha": [0.1, 1.0]}}},
            "n_authors": [3],
            "text_length": [None, 40],
            "error_sample": 5,
            "ablation": {"base": "Style", "families": ["entities"]},
            "open_set": {"unknown_authors": 1},
        },
    }
    data.update(overrides)
    return ExperimentConfig.model_validate(data)


@pytest.fixture
def corpora(workdir: Path, literary: Corpus, plagiarism_corpus: Corpus) -> dict[str, Path]:
    paths = {"literary": workdir / "corpora" / "literary", "plagiarism": workdir / "corpora" / "plagiarism"}
    save_corpus(literary, paths["literary"])
    save_corpus(plagiarism_corpus, paths["plagiarism"])
    return paths


class TestAttributionRuns:
    def test_run_resume_and_outputs(self, corpora: dict[str, Path]) -> None:
        config = _attribution_config(corpora["literary"])
        plan = Plan.from_config(config)
        assert plan.summary()["conditions"] == 8 and plan.summary()["baseline_conditions"] == 2
        store = Runner(config).run()
        frame = pd.read_csv(store.directory / "results.csv")
        assert len(frame) == len(plan.runs) and set(frame["status"]) == {"ok"}
        assert {
            "condition",
            "seed",
            "f1_macro",
            "f1_macro_low",
            "duration_s",
            "code_version",
            "param_representation",
        } <= set(frame.columns)
        assert frame.loc[frame["param_representation"] != "-", "open_set_f1_macro"].notna().any()
        summary = pd.read_csv(store.directory / "summary.csv")
        assert "f1_macro_mean" in summary and len(summary) == len(plan.conditions)
        comparisons = pd.read_csv(store.directory / "comparisons.csv")
        assert {"Style", "R1"} <= set(comparisons["reference"]) and "mcnemar_p_holm" in comparisons
        for name in ("config.yaml", "config.json", "plan.json", "environment.json", "runs.jsonl"):
            assert (store.directory / name).exists()
        record = store.records()[0]
        assert store.predictions(record) and store.artifacts(record)
        # Reprise : rien n'est recalculé.
        again = Runner(config)
        assert again.pending() == []
        assert len(list(read_jsonl(store.runs_path))) == len(plan.runs)
        report = generate_report(store.directory)
        html = report.read_text(encoding="utf-8")
        assert "Résultats principaux" in html and "data:image/png;base64" in html and "Comparaisons appariées" in html
        markdown = generate_report(store.directory, "md")
        assert (
            markdown.read_text(encoding="utf-8").startswith("# Rapport") and (store.directory / "report_files").exists()
        )

    def test_runs_are_reproducible(self, corpora: dict[str, Path]) -> None:
        config = _attribution_config(corpora["literary"], output_dir="first")
        Runner(config, limit=2).run()
        second = config.with_overrides(output_dir="second")
        Runner(second, limit=2).run()
        first_frame = pd.read_csv(Path("first/t_attr/results.csv"))
        second_frame = pd.read_csv(Path("second/t_attr/results.csv"))
        columns = ["condition", "seed", "f1_macro", "accuracy", "f1_macro_low", "f1_macro_high"]
        pd.testing.assert_frame_equal(first_frame[columns], second_frame[columns])

    def test_errors_are_recorded_not_fatal(self, corpora: dict[str, Path]) -> None:
        config = _attribution_config(corpora["literary"])
        plan = Plan.from_config(config)
        broken = config.model_copy(
            update={"attribution": config.attribution.model_copy(update={"corpus": Path("missing")})}
        )  # type: ignore[union-attr]
        result = execute(broken.model_dump(), plan.runs[0])
        assert result.status == "error" and "missing" in (result.error or "")

    def test_parallel_execution(self, corpora: dict[str, Path]) -> None:
        config = _attribution_config(corpora["literary"], n_jobs=2, output_dir="parallel")
        store = Runner(config, limit=3).run()
        assert len(store.records()) == 3


def test_plan_filters_and_errors(corpora: dict[str, Path]) -> None:
    config = _attribution_config(corpora["literary"], include=[{"representation": "R1"}], exclude=[{"text_length": 40}])
    plan = Plan.from_config(config)
    assert {c.params["representation"] for c in plan.conditions if not c.baseline} == {"R1"}
    assert all(c.params["text_length"] is None for c in plan.conditions)
    bad = _attribution_config(corpora["literary"])
    data = bad.model_dump()
    data["attribution"]["ablation"] = {"base": "unknown"}
    with pytest.raises(Exception, match="inconnue"):
        Plan.from_config(ExperimentConfig.model_validate(data))
    assert safe_name("rep=R1|clf=nb#3") != safe_name("rep=R1_clf=nb#3")


def test_plagiarism_run(corpora: dict[str, Path]) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "t_plag",
            "task": "plagiarism",
            "annotation": {"engine": "heuristic", "cache_path": "cache/ann.sqlite"},
            "protocol": {"seeds": 1, "bootstrap": 30},
            "plagiarism": {
                "corpus": str(corpora["plagiarism"]),
                "levels": ["O0", "O3"],
                "sizes": [100],
                "feature_sets": {
                    "L": ["cosine_tfidf", "containment_shingles", "lcs"],
                    "L+E": ["cosine_tfidf", "containment_shingles", "lcs", "jaccard_entities", "type_divergence"],
                },
                "classifiers": {"logreg": {"type": "logreg"}},
                "retrieval": {"method": "tfidf", "k": 4},
                "alignment": {"min_density": 3.0, "mode": "both"},
            },
        }
    )
    store = Runner(config).run()
    frame = pd.read_csv(store.directory / "results.csv")
    assert set(frame["status"]) == {"ok"} and len(frame) == 2 * (2 + 2)
    for column in (
        "pairs_1to1_f1",
        "pairs_1to10_f1",
        "pairs_1to10_average_precision",
        "plagdet",
        "origin_accuracy_at_1",
        "retrieval_recall_at_k",
        "threshold",
    ):
        assert column in frame and frame[column].notna().all(), column
    exact = frame[(frame["param_level"] == "O0") & (frame["param_classifier"] == "logreg")]
    assert (exact["passage_recall"] > 0.9).all()
    assert (store.directory / "comparisons.csv").exists()
    html = generate_report(store.directory).read_text(encoding="utf-8")
    assert "Mesures de PAN" in html


def test_plagiarism_with_given_pairs(corpora: dict[str, Path]) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "t_pairs",
            "task": "plagiarism",
            "annotation": {"engine": "none"},
            "protocol": {"seeds": 1, "bootstrap": 10},
            "plagiarism": {
                "corpus": str(corpora["plagiarism"]),
                "levels": ["O1"],
                "sizes": [50],
                "feature_sets": {"L": ["cosine_tfidf", "lcs"]},
                "classifiers": {"svm": {"type": "linear_svm"}},
                "retrieval": {"method": "pairs"},
                "baselines": False,
            },
        }
    )
    store = Runner(config).run()
    assert store.records()[0]["metrics"]["retrieval_recall_at_k"] == 1.0


def test_clustering_and_verification_runs(corpora: dict[str, Path]) -> None:
    clustering = ExperimentConfig.model_validate(
        {
            "name": "t_clu",
            "task": "clustering",
            "annotation": {"engine": "heuristic", "cache_path": "cache/ann.sqlite"},
            "protocol": {"seeds": 2},
            "clustering": {
                "corpora": {"lit": {"path": str(corpora["literary"]), "labels": ["author", "work"]}},
                "representations": {
                    "words": {"blocks": [{"type": "tfidf", "params": {"stopwords_source": "builtin"}}]}
                },
                "dimensions": [10],
                "algorithms": {"km": {"type": "kmeans"}, "agg": {"type": "agglomerative"}},
                "families_threshold": 0.9,
            },
        }
    )
    store = Runner(clustering).run()
    frame = pd.read_csv(store.directory / "results.csv")
    assert len(frame) == 3 and {"author_ari", "work_nmi", "author_silhouette", "families"} <= set(frame.columns)
    assert "Projection" in generate_report(store.directory).read_text(encoding="utf-8")
    auto = clustering.with_overrides(
        name="t_clu_auto", clustering={**clustering.clustering.model_dump(), "k": "silhouette", "k_range": [2, 3, 4]}
    )  # type: ignore[union-attr]
    assert Runner(auto, limit=1).run().records()[0]["metrics"]["k"] in {2, 3, 4}

    verification = ExperimentConfig.model_validate(
        {
            "name": "t_ver",
            "task": "verification",
            "annotation": {"engine": "none"},
            "protocol": {"seeds": 1, "bootstrap": 20},
            "verification": {
                "corpus": str(corpora["literary"]),
                "representations": {"Style": {"blocks": [{"type": "char_ngrams", "params": {"ngram_range": [2, 3]}}]}},
                "classifiers": {"logreg": {"type": "logreg"}},
                "imposters": True,
                "pairs_per_split": 20,
                "abstain_margin": 0.05,
            },
        }
    )
    frame = pd.read_csv(Runner(verification).run().directory / "results.csv")
    assert len(frame) == 2 and {"auc", "c_at_1", "auc_low"} <= set(frame.columns)


def test_context_is_shared_and_offline(corpora: dict[str, Path]) -> None:
    config = _attribution_config(corpora["literary"])
    context = get_context(config)
    assert get_context(config) is context
    context.prepare()
    assert len(context.annotations(corpora["literary"])) == len(context.corpus(corpora["literary"]))


def test_store_empty(workdir: Path) -> None:
    store = ResultStore(workdir / "empty")
    assert store.records() == [] and store.dataframe().empty and store.completed() == set()
