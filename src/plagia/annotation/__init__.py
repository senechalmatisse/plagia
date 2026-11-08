"""Couche `annotation` : prétraitement, étiquettes grammaticales, entités nommées, liage et typage par bases de connaissances, cache des annotations."""

from plagia.annotation.base import ANNOTATORS, Annotator, EntityRecognizer, RegexAnnotator
from plagia.annotation.cache import AnnotationCache, CachedAnnotator, CacheMissError
from plagia.annotation.document import AnnotatedDocument, Entity, as_annotated
from plagia.annotation.heuristic import HeuristicAnnotator
from plagia.annotation.knowledge_base import (
    DbpediaKnowledgeBase,
    KnowledgeBase,
    TypeHierarchy,
    YagoKnowledgeBase,
)
from plagia.annotation.pipeline import annotate_documents, annotation_versions, build_annotator
from plagia.annotation.preprocess import Preprocessor, stopword_list
from plagia.annotation.schema import MINIMAL_LABELS, normalize_label
from plagia.annotation.spacy_engine import SpacyPipeline
from plagia.annotation.spotlight import SpotlightLinker
from plagia.annotation.stanford import StanfordNer, StanfordPipeline, StanfordPosTagger

__all__ = [
    "ANNOTATORS",
    "MINIMAL_LABELS",
    "AnnotatedDocument",
    "AnnotationCache",
    "Annotator",
    "CacheMissError",
    "CachedAnnotator",
    "DbpediaKnowledgeBase",
    "Entity",
    "EntityRecognizer",
    "HeuristicAnnotator",
    "KnowledgeBase",
    "Preprocessor",
    "RegexAnnotator",
    "SpacyPipeline",
    "SpotlightLinker",
    "StanfordNer",
    "StanfordPipeline",
    "StanfordPosTagger",
    "TypeHierarchy",
    "YagoKnowledgeBase",
    "annotate_documents",
    "annotation_versions",
    "as_annotated",
    "build_annotator",
    "normalize_label",
    "stopword_list",
]
