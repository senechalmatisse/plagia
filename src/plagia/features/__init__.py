"""Couche `features` : blocs de caractéristiques compatibles scikit-learn."""

from plagia.features.base import FAMILIES, FAMILY_NAMES, FEATURE_BLOCKS, block_family, to_documents
from plagia.features.embeddings import Doc2VecEmbedding, Lsa, SentenceEmbeddings, WordVectors
from plagia.features.entities import EntityFeatures, TypeFeatures
from plagia.features.lexical import BagOfWords, CharNgrams, Tfidf
from plagia.features.masking import Masker
from plagia.features.pos import PosNgrams
from plagia.features.representation import (
    ablate,
    ablation_representations,
    block_slices,
    build_block,
    build_representation,
    representation_families,
)
from plagia.features.stylometry import Stylometry

__all__ = [
    "FAMILIES",
    "FAMILY_NAMES",
    "FEATURE_BLOCKS",
    "BagOfWords",
    "CharNgrams",
    "Doc2VecEmbedding",
    "EntityFeatures",
    "Lsa",
    "Masker",
    "PosNgrams",
    "SentenceEmbeddings",
    "Stylometry",
    "Tfidf",
    "TypeFeatures",
    "WordVectors",
    "ablate",
    "ablation_representations",
    "block_family",
    "block_slices",
    "build_block",
    "build_representation",
    "representation_families",
    "to_documents",
]
