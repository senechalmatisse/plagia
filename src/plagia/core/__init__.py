"""Couche `core` : types, configuration, graines, cache disque et registres."""

from plagia.core.errors import (
    ConfigurationError,
    ExternalServiceError,
    LeakageError,
    MissingDependencyError,
    PlagiaError,
)
from plagia.core.registry import Registry
from plagia.core.seeds import derive_seed, make_random, make_rng
from plagia.core.types import SPLITS, Corpus, Document, Passage, PassagePair

__all__ = [
    "SPLITS",
    "ConfigurationError",
    "Corpus",
    "Document",
    "ExternalServiceError",
    "LeakageError",
    "MissingDependencyError",
    "Passage",
    "PassagePair",
    "PlagiaError",
    "Registry",
    "derive_seed",
    "make_random",
    "make_rng",
]
