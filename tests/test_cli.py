"""Tests de la ligne de commande `plagia` (sous-commandes principales, hors réseau)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plagia.cli.main import build_parser, main
from plagia.core.io import write_yaml
from plagia.corpus import load_corpus


def run(*args: str) -> int:
    return main(list(args))


def test_parser_lists_all_commands() -> None:
    commands = build_parser()._subparsers._group_actions[0].choices  # type: ignore[union-attr]
    expected = {"setup", "corpus", "annotate", "run", "attribute", "detect", "cluster", "report", "demo", "bench"}
    assert expected <= set(commands)


def test_corpus_commands(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        run(
            "corpus",
            "toy",
            "--kind",
            "literary",
            "--authors",
            "3",
            "--docs-per-work",
            "2",
            "--words",
            "80",
            "--out",
            "lit",
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["splits"] == {"test": 6, "train": 30}
    assert (
        run(
            "corpus",
            "toy",
            "--kind",
            "news",
            "--authors",
            "3",
            "--docs-per-work",
            "2",
            "--words",
            "80",
            "--out",
            "news",
        )
        == 0
    )
    assert (
        run(
            "corpus",
            "toy",
            "--kind",
            "plagiarism",
            "--authors",
            "4",
            "--sources",
            "30",
            "--suspicious",
            "36",
            "--per-cell",
            "2",
            "--out",
            "plag",
        )
        == 0
    )
    assert run("corpus", "info", "plag") == 0
    assert "aucune fuite" in capsys.readouterr().out
    assert run("corpus", "clean", "--source", "lit", "--out", "lit_clean") == 0
    assert len(load_corpus("lit_clean")) == 36
    assert (
        run(
            "corpus",
            "synthetic",
            "--base",
            "lit",
            "--synonyms",
            "toy",
            "--n-sources",
            "5",
            "--n-suspicious",
            "500",
            "--out",
            "fail",
        )
        == 2
    )


def test_c50_pan_and_aida_commands(workdir: Path) -> None:
    for name, text in {
        "C50train/AnnaB/1.txt": "Markets rose today.",
        "C50train/AnnaB/2.txt": "Bonds fell.",
        "C50test/AnnaB/3.txt": "Gold was flat.",
        "C50train/BobC/1.txt": "Rain in Paris.",
        "C50train/BobC/2.txt": "Snow in Oslo.",
        "C50test/BobC/3.txt": "Sun in Rome.",
    }.items():
        path = workdir / "raw" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert run("corpus", "c50", "--source", "raw", "--validation", "0.5", "--out", "c50") == 0
    assert set(load_corpus("c50").split_counts()) == {"train", "validation", "test"}
    pan = workdir / "pan"
    (pan / "susp").mkdir(parents=True)
    (pan / "src").mkdir()
    (pan / "susp" / "suspicious-document00001.txt").write_text("Hello copied text.", encoding="utf-8")
    (pan / "src" / "source-document00001.txt").write_text("copied text", encoding="utf-8")
    (pan / "pairs").write_text("suspicious-document00001.txt source-document00001.txt\n", encoding="utf-8")
    assert (
        run("corpus", "pan", "--source", "pan", "--split", "train", "--validation", "0.5", "--out", "pan_corpus") == 0
    )
    (workdir / "aida.tsv").write_text(
        "-DOCSTART- (1 X)\nParis\tB\tParis\tParis\thttp://en.wikipedia.org/wiki/Paris\t1\t/m/1\nwins\n",
        encoding="utf-8",
    )
    assert run("corpus", "aida", "--tsv", "aida.tsv", "--out", "aida") == 0
    assert load_corpus("aida")[0].meta["gold_entities"][0]["text"] == "Paris"
    assert run("annotate", "aida", "--engine", "heuristic", "--cache", "c.sqlite", "--evaluate") == 0
    assert run("annotate", "c50", "--engine", "none", "--cache", "c.sqlite", "--evaluate") == 2


def test_run_report_attribute_detect_cluster(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run(
        "corpus", "toy", "--kind", "literary", "--authors", "3", "--docs-per-work", "3", "--words", "80", "--out", "lit"
    )
    run(
        "corpus",
        "toy",
        "--kind",
        "plagiarism",
        "--authors",
        "4",
        "--sources",
        "30",
        "--suspicious",
        "36",
        "--per-cell",
        "2",
        "--out",
        "plag",
    )
    config = {
        "name": "cli_demo",
        "task": "attribution",
        "annotation": {"engine": "heuristic", "cache_path": "cache.sqlite"},
        "protocol": {"cv_folds": 2, "seeds": 1, "bootstrap": 20},
        "attribution": {
            "corpus": "lit",
            "representations": {"R3": {"blocks": [{"type": "char_ngrams", "params": {"ngram_range": [2, 3]}}]}},
            "classifiers": {"nb": {"type": "naive_bayes"}},
        },
    }
    write_yaml("demo.yaml", config)
    assert run("run", "demo.yaml", "--dry-run") == 0
    assert "rep=R3" in capsys.readouterr().out
    assert run("run", "demo.yaml", "--report", "--output-dir", "out") == 0
    assert Path("out/cli_demo/report.html").exists()
    assert run("report", "out/cli_demo", "--format", "md") == 0 and Path("out/cli_demo/report.md").exists()

    test_docs = list(load_corpus("lit").split("test"))[:2]
    for doc in test_docs:
        Path(f"{doc.id}.txt").write_text(doc.text, encoding="utf-8")
    files = [f"{d.id}.txt" for d in test_docs]
    assert run("attribute", "--train", "lit", "--save", "model.joblib", "--config", "demo.yaml", *files) == 0
    assert run("attribute", "--model", "model.joblib", "--top-k", "2", *files) == 0
    assert capsys.readouterr().out.count("(") >= 4
    assert run("attribute", *files) == 2

    plag = load_corpus("plag")
    suspicious = next(d for d in plag if d.meta.get("level") == "O0")
    Path("suspect.txt").write_text(suspicious.text, encoding="utf-8")
    assert (
        run(
            "detect",
            "suspect.txt",
            "--sources",
            "plag",
            "--out",
            "detect",
            "--min-density",
            "3",
            "--intrinsic",
            "--window",
            "60",
        )
        == 0
    )
    report = json.loads(Path("detect/suspect.json").read_text(encoding="utf-8"))
    assert report["detections"] and Path("detect/suspect.html").exists()
    sources_dir = Path("sources")
    sources_dir.mkdir()
    source_id = next(c.source.document_id for c in plag.cases if c.suspicious.document_id == suspicious.id and c.source)
    (sources_dir / f"{source_id}.txt").write_text(plag.get(source_id).text, encoding="utf-8")
    assert run("detect", "suspect.txt", "--sources", "sources", "--out", "detect2", "--min-density", "3") == 0

    assert run("demo", "--model", "model.joblib", "--text", files[0], "--out", "demo/a.html") == 0
    page = Path("demo/a.html").read_text(encoding="utf-8")
    assert "Auteurs les plus probables" in page and "Aucune collection de sources" in page
    assert run("demo", "--sources", "plag", "--text", "suspect.txt", "--out", "demo/b.html") == 0
    assert "<mark" in Path("demo/b.html").read_text(encoding="utf-8")
    assert run("demo", "--text", "suspect.txt") == 2
    assert run("demo", "--sources", "plag") == 2

    assert run("cluster", "lit", "--blocks", "tfidf", "--k", "true", "--dimensions", "10", "--out", "clusters") == 0
    assert json.loads(Path("clusters/clusters.json").read_text(encoding="utf-8"))["k"] == 3
    assert run("cluster", "lit", "--split", "test", "--blocks", "char_ngrams", "--k", "auto", "--dimensions", "5") == 0
    assert run("cluster", "lit", "--k", "2", "--algorithm", "agglomerative", "--dimensions", "5") == 0


def test_annotate_setup_and_bench(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run("corpus", "toy", "--kind", "news", "--authors", "2", "--docs-per-work", "2", "--words", "60", "--out", "news")
    assert run("annotate", "news", "--engine", "heuristic", "--cache", "c.sqlite") == 0
    assert run("annotate", "news", "--engine", "heuristic", "--cache", "c.sqlite", "--offline") == 0
    write_yaml(
        "ann.yaml",
        {
            "name": "a",
            "task": "clustering",
            "annotation": {"engine": "heuristic", "cache_path": "c.sqlite"},
            "clustering": {
                "corpora": {"n": {"path": "news"}},
                "representations": {"w": {"blocks": [{"type": "tfidf"}]}},
                "algorithms": {"k": {"type": "kmeans"}},
            },
        },
    )
    assert run("annotate", "news", "--config", "ann.yaml") == 0
    assert run("setup", "--check-only") == 0
    assert "module:networkx" in capsys.readouterr().out
    assert run("bench", "--documents", "30", "--words", "50", "--authors", "2") == 0
    assert json.loads(capsys.readouterr().out)["documents"] == 30


def test_demo_serve(workdir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util
    import subprocess

    calls: list[list[str]] = []
    assert run("demo", "--model", "m.joblib", "--serve") == 2  # modèle absent : refus avant lancement
    Path("m.joblib").write_bytes(b"")
    Path("src").mkdir()
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert run("demo", "--model", "m.joblib", "--serve") == 2
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(subprocess, "call", lambda command: calls.append(command) or 0)
    assert run("demo", "--model", "m.joblib", "--sources", "missing", "--serve") == 2
    assert run("demo", "--model", "m.joblib", "--sources", "src", "--serve", "--top-k", "3") == 0
    command = calls[0]
    assert command[1:4] == ["-m", "streamlit", "run"] and command[4].endswith("demo_app.py")
    assert command[5:] == ["--browser.gatherUsageStats=false", "--", "--top-k=3", "--model=m.joblib", "--sources=src"]


def test_clear_errors_without_traceback(workdir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    def fails(*args: str) -> str:
        assert run(*args) == 2
        return capsys.readouterr().err

    # corpus absent : message et commande de construction, aucun répertoire de résultats créé
    config = {
        "name": "missing",
        "task": "attribution",
        "attribution": {
            "corpus": "data/corpora/c50",
            "representations": {"R1": {"blocks": [{"type": "bow"}]}},
            "classifiers": {"nb": {"type": "naive_bayes"}},
        },
    }
    write_yaml("missing.yaml", config)
    assert "plagia corpus c50" in fails("run", "missing.yaml") and not Path("results/missing").exists()
    assert "plagia corpus toy --kind literary" in fails("attribute", "--train", "data/corpora/toy_literary")
    # modèle, textes et sources absents
    assert "Fichier(s) introuvable(s) : texte.txt" in fails("attribute", "--model", "absent.joblib", "texte.txt")
    Path("texte.txt").write_text("Un texte.", encoding="utf-8")
    assert "Modèle introuvable : absent.joblib" in fails("attribute", "--model", "absent.joblib", "texte.txt")
    assert "après C50" in fails("detect", "texte.txt", "--sources", "data/corpora/synthetic")
    Path("vide").mkdir()
    assert "Aucune source dans vide" in fails("detect", "texte.txt", "--sources", "vide")
    assert "absent.txt" in fails("demo", "--sources", "vide", "--text", "absent.txt")
    # rapport sans exécution
    assert "introuvable" in fails("report", "results/none")
    Path("results/empty").mkdir(parents=True)
    assert "Aucune exécution dans results/empty" in fails("report", "results/empty")


def test_corpus_export(workdir: Path) -> None:
    toy = ["--kind", "plagiarism", "--authors", "3", "--sources", "30", "--suspicious", "36", "--per-cell", "2"]
    assert run("corpus", "toy", *toy, "--out", "plag") == 0
    selection = ["--split", "test", "--role", "suspicious", "--with-plagiarism", "--limit", "2"]
    assert run("corpus", "export", "plag", *selection, "--out", "ex") == 0
    import csv

    rows = list(csv.DictReader(Path("ex/index.csv").read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 2 and all(int(r["passages_plagies"]) > 0 and r["role"] == "suspicious" for r in rows)
    corpus = load_corpus("plag")
    assert Path("ex", rows[0]["fichier"]).read_text(encoding="utf-8") == corpus.get(rows[0]["id"]).text
    assert run("corpus", "export", "plag", "--ids", rows[0]["id"], "--out", "one") == 0
    assert len(list(Path("one").glob("*.txt"))) == 1
    assert run("corpus", "export", "plag", "--ids", "nope", "--out", "x") == 2
    assert run("corpus", "export", "plag", "--split", "nowhere", "--out", "x") == 2
