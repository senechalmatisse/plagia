"""Couche `similarity` : mesures de similarité, MinHash/LSH, matrices kNN, seuils."""

from plagia.similarity.base import SIMILARITIES, Similarity
from plagia.similarity.entity import EntitySimilarity, TypeDivergence, jensen_shannon
from plagia.similarity.matrix import knn_cosine, symmetrize, top_k_per_row
from plagia.similarity.minhash import MinHasher, MinHashIndex, winnow
from plagia.similarity.semantic import SemanticSimilarity, WordMoverSimilarity
from plagia.similarity.sequence import Lcs, Levenshtein, lcs_length, levenshtein_distance
from plagia.similarity.sets import Containment, Dice, Jaccard, Overlap, containment, dice, jaccard
from plagia.similarity.threshold import ThresholdChoice, pr_curve, select_threshold
from plagia.similarity.vector import Cosine, cosine_rows

__all__ = [
    "SIMILARITIES",
    "Containment",
    "Cosine",
    "Dice",
    "EntitySimilarity",
    "Jaccard",
    "Lcs",
    "Levenshtein",
    "MinHashIndex",
    "MinHasher",
    "Overlap",
    "SemanticSimilarity",
    "Similarity",
    "ThresholdChoice",
    "TypeDivergence",
    "WordMoverSimilarity",
    "containment",
    "cosine_rows",
    "dice",
    "jaccard",
    "jensen_shannon",
    "knn_cosine",
    "lcs_length",
    "levenshtein_distance",
    "pr_curve",
    "select_threshold",
    "symmetrize",
    "top_k_per_row",
    "winnow",
]
