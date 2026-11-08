"""Fixtures partagées : corpus jouets, annotations heuristiques, corpus de plagiat."""

from __future__ import annotations

from pathlib import Path

import pytest

from plagia.annotation.document import AnnotatedDocument
from plagia.annotation.heuristic import HeuristicAnnotator
from plagia.core.types import Corpus
from plagia.corpus.splits import assign_document_splits, assign_group_splits
from plagia.corpus.synthetic import SyntheticConfig, generate_plagiarism_corpus
from plagia.corpus.toy import generate_toy_corpus, toy_synonyms


@pytest.fixture(scope="session")
def literary() -> Corpus:
    """Corpus jouet littéraire : 4 auteurs × 6 œuvres × 4 textes, découpé par œuvre (5/1)."""
    corpus = generate_toy_corpus(n_authors=4, works_per_author=6, docs_per_work=4, words_per_doc=150, seed=7)
    return assign_group_splits(corpus, {"train": 5 / 6, "test": 1 / 6}, seed=7)


@pytest.fixture(scope="session")
def news() -> Corpus:
    """Corpus jouet « presse » : chaque document est son propre groupe, moitié test."""
    corpus = generate_toy_corpus(n_authors=4, works_per_author=6, docs_per_work=3, words_per_doc=120, seed=11)
    return assign_document_splits(corpus.map(lambda d: d.replace(work=None)), {"train": 0.5, "test": 0.5}, seed=11)


@pytest.fixture(scope="session")
def annotated(literary: Corpus) -> dict[str, AnnotatedDocument]:
    """Annotations heuristiques du corpus littéraire."""
    documents = HeuristicAnnotator().annotate_many([(d.id, d.text) for d in literary])
    return {d.id: d for d in documents}


@pytest.fixture(scope="session")
def plagiarism_corpus() -> Corpus:
    """Petit corpus de plagiat synthétique (4 niveaux × 3 tailles)."""
    base = generate_toy_corpus(n_authors=6, works_per_author=6, docs_per_work=6, words_per_doc=350, seed=3)
    config = SyntheticConfig(n_sources=90, n_suspicious=110, passages_per_cell=6, seed=5)
    return generate_plagiarism_corpus(base, config, synonyms=toy_synonyms())


@pytest.fixture(scope="session")
def plagiarism_annotated(plagiarism_corpus: Corpus) -> dict[str, AnnotatedDocument]:
    """Annotations heuristiques du corpus de plagiat."""
    documents = HeuristicAnnotator().annotate_many([(d.id, d.text) for d in plagiarism_corpus])
    return {d.id: d for d in documents}


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Répertoire de travail temporaire (chemins relatifs des configurations)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
