"""Chaîne d'annotation d'un corpus : annotateur (avec cache), liage, typage.

##Les annotations sont calculées une fois, hors de la boucle d'apprentissage, puis lues dans le cache par toutes les conditions d'expérience.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from plagia.annotation.base import ANNOTATORS, Annotator
from plagia.annotation.cache import AnnotationCache, CachedAnnotator
from plagia.annotation.document import AnnotatedDocument
from plagia.annotation.heuristic import HeuristicAnnotator
from plagia.annotation.knowledge_base import DbpediaKnowledgeBase, KnowledgeBase, TypeHierarchy, YagoKnowledgeBase
from plagia.annotation.spacy_engine import SpacyPipeline
from plagia.annotation.spotlight import SpotlightLinker
from plagia.annotation.stanford import StanfordNer, StanfordPipeline, StanfordPosTagger
from plagia.core.config import AnnotationConfig
from plagia.core.errors import ConfigurationError
from plagia.core.hashing import sha256_file
from plagia.core.logging import get_logger
from plagia.core.progress import progress
from plagia.core.types import Document

log = get_logger(__name__)


def build_annotator(config: AnnotationConfig) -> Annotator:
    """Annotateur décrit par la configuration."""
    if config.engine == "spacy":
        return SpacyPipeline(config.model, batch_size=config.batch_size, n_process=config.n_process)
    if config.engine == "stanford":
        if not (config.stanford_ner_jar and config.stanford_ner_model):
            raise ConfigurationError("annotation.stanford_ner_jar et stanford_ner_model sont requis.")
        ner = StanfordNer(config.stanford_ner_jar, config.stanford_ner_model, java=config.java)
        pos = None
        if config.stanford_pos_jar and config.stanford_pos_model:
            pos = StanfordPosTagger(config.stanford_pos_jar, config.stanford_pos_model, java=config.java)
        return StanfordPipeline(ner, pos)
    if config.engine == "heuristic":
        return HeuristicAnnotator()
    return ANNOTATORS.create("none")


def build_linker(
    config: AnnotationConfig, cache: AnnotationCache | None, offline: bool = False
) -> SpotlightLinker | None:
    """Client Spotlight si le liage est demandé."""
    if config.linker != "spotlight":
        return None
    return SpotlightLinker(
        config.spotlight_url, config.spotlight_confidence, config.spotlight_support, cache=cache, offline=offline
    )


def build_knowledge_base(
    config: AnnotationConfig, cache: AnnotationCache | None, offline: bool = False
) -> KnowledgeBase | None:
    """Base de connaissances pour le typage, si demandée."""
    if config.knowledge_base == "none":
        return None
    if config.taxonomy_path is None:
        raise ConfigurationError("annotation.taxonomy_path est requis pour le typage par base de connaissances.")
    hierarchy = TypeHierarchy.load(config.taxonomy_path)
    if config.knowledge_base == "dbpedia":
        return DbpediaKnowledgeBase(hierarchy)
    endpoint = config.sparql_endpoint or "https://yago-knowledge.org/sparql/query"
    return YagoKnowledgeBase(hierarchy, endpoint=endpoint, cache=cache, offline=offline)


def annotate_documents(
    documents: Sequence[Document],
    config: AnnotationConfig,
    cache: AnnotationCache | None = None,
    offline: bool = False,
    chunk_size: int = 512,
) -> list[AnnotatedDocument]:
    """Annote des documents (cache consulté d'abord), puis lie et type les entités."""
    annotator = build_annotator(config)
    items = [(d.id, d.text) for d in documents]
    if config.engine == "none":
        results = annotator.annotate_many(items)
    else:
        own_cache = cache is None
        cache = cache or AnnotationCache(config.cache_path)
        proxy = CachedAnnotator(annotator, cache, offline=offline)
        results = []
        chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]
        for chunk in progress(chunks, desc=f"annotation {annotator.name}"):
            results.extend(proxy.annotate_many(chunk))
        log.info("Annotation %s : %d en cache, %d calculée(s).", annotator.name, proxy.hits, proxy.misses)
        linker = build_linker(config, cache, offline)
        if linker is not None:
            results = [linker.link_document(d) for d in progress(results, desc="liage Spotlight")]
        kb = build_knowledge_base(config, cache, offline)
        if kb is not None:
            results = [kb.type_document(d) for d in results]
        if own_cache:
            cache.close()
    return results


def annotation_versions(config: AnnotationConfig) -> dict[str, Any]:
    """Versions des outils d'annotation et des bases de connaissances."""
    info: dict[str, Any] = {"engine": config.engine}
    try:
        annotator = build_annotator(config)
        info["annotator"] = annotator.name
        info["annotator_version"] = annotator.version
    except Exception as exc:  # outil absent : on trace l'erreur sans interrompre
        info["annotator_error"] = str(exc)
    if config.linker == "spotlight":
        info["spotlight"] = {"endpoint": config.spotlight_url, "confidence": config.spotlight_confidence}
    if config.knowledge_base != "none" and config.taxonomy_path is not None:
        info["knowledge_base"] = config.knowledge_base
        try:
            info["taxonomy_sha256"] = sha256_file(config.taxonomy_path)
        except OSError:
            info["taxonomy_sha256"] = None
    return info
