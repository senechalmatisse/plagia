"""Contexte d'exécution : corpus et annotations chargés une fois par processus.

Les annotations sont calculées avant la campagne (`prepare`, appels externes autorisés) puis relues dans le cache par chaque exécution en mode hors ligne :
toutes les conditions lisent les mêmes annotations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plagia.annotation.cache import AnnotationCache
from plagia.annotation.document import AnnotatedDocument
from plagia.annotation.pipeline import annotate_documents
from plagia.core.config import ExperimentConfig
from plagia.core.logging import get_logger
from plagia.core.types import Corpus
from plagia.corpus.store import load_corpus

log = get_logger(__name__)


def corpus_paths(config: ExperimentConfig) -> list[Path]:
    """Corpus utilisés par l'expérience."""
    section = config.section
    if config.task == "clustering":
        return [c.path for c in section.corpora.values()]
    return [section.corpus]


class ExperimentContext:
    """Cache en mémoire des corpus, des annotations et des calculs partagés."""

    def __init__(self, config: ExperimentConfig, offline: bool = True) -> None:
        self.config = config
        self.offline = offline
        self._corpora: dict[str, Corpus] = {}
        self._annotations: dict[str, dict[str, AnnotatedDocument]] = {}
        self.shared: dict[Any, Any] = {}

    def corpus(self, path: str | Path) -> Corpus:
        """Corpus enregistré (vérification du manifeste au premier chargement)."""
        key = str(path)
        if key not in self._corpora:
            self._corpora[key] = load_corpus(path)
        return self._corpora[key]

    def annotations(self, path: str | Path) -> dict[str, AnnotatedDocument]:
        """Annotations de tous les documents du corpus, par identifiant."""
        key = str(path)
        if key not in self._annotations:
            corpus = self.corpus(path)
            cache = (
                None if self.config.annotation.engine == "none" else AnnotationCache(self.config.annotation.cache_path)
            )
            try:
                documents = annotate_documents(list(corpus), self.config.annotation, cache=cache, offline=self.offline)
            finally:
                if cache is not None:
                    cache.close()
            self._annotations[key] = {d.id: d for d in documents}
        return self._annotations[key]

    def prepare(self) -> None:
        """Calcule (ou vérifie) les annotations de tous les corpus avant la campagne."""
        offline, self.offline = self.offline, False
        try:
            for path in corpus_paths(self.config):
                log.info("Préparation des annotations de %s", path)
                self.annotations(path)
        finally:
            self.offline = offline


_CONTEXTS: dict[str, ExperimentContext] = {}


def get_context(config: ExperimentConfig, offline: bool = True) -> ExperimentContext:
    """Contexte du processus courant pour cette configuration (réutilisé entre exécutions)."""
    key = f"{config.fingerprint()}|{offline}"
    if key not in _CONTEXTS:
        _CONTEXTS.clear()
        _CONTEXTS[key] = ExperimentContext(config, offline=offline)
    return _CONTEXTS[key]
