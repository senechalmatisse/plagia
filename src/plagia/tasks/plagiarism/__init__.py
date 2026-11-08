"""Détection de plagiat (T2) : extrinsèque (recherche, alignement, filtrage, paires, origine) et intrinsèque, rapport de similarité."""

from plagia.tasks.plagiarism.alignment import Extender, Filter, Seed, SeedMatcher, TextAligner
from plagia.tasks.plagiarism.dataset import PairExample, build_pair_examples, subsample_prevalence
from plagia.tasks.plagiarism.intrinsic import IntrinsicDetector, word_windows
from plagia.tasks.plagiarism.origin import OriginIdentifier
from plagia.tasks.plagiarism.pairs import FEATURE_NAMES, PairClassifier, PairFeatureExtractor
from plagia.tasks.plagiarism.passages import PassageView, sentence_char_spans, sliding_windows
from plagia.tasks.plagiarism.pipeline import ExtrinsicDetector
from plagia.tasks.plagiarism.report import SimilarityReport, build_report
from plagia.tasks.plagiarism.retrieval import CandidateRetriever

__all__ = [
    "FEATURE_NAMES",
    "CandidateRetriever",
    "Extender",
    "ExtrinsicDetector",
    "Filter",
    "IntrinsicDetector",
    "OriginIdentifier",
    "PairClassifier",
    "PairExample",
    "PairFeatureExtractor",
    "PassageView",
    "Seed",
    "SeedMatcher",
    "SimilarityReport",
    "TextAligner",
    "build_pair_examples",
    "build_report",
    "sentence_char_spans",
    "sliding_windows",
    "subsample_prevalence",
    "word_windows",
]
