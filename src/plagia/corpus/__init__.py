"""Couche `corpus` : chargeurs, nettoyage, découpages sans fuite, manifestes, générateur de plagiat synthétique et corpus jouet."""

from plagia.corpus.cleaning import CleaningConfig, clean_corpus, deduplicate
from plagia.corpus.splits import (
    assign_document_splits,
    assign_group_splits,
    carve_validation,
    check_no_leakage,
    leakage_report,
    subsample_authors,
    subsample_per_author,
    truncate_split,
)
from plagia.corpus.store import load_corpus, save_corpus
from plagia.corpus.synthetic import SyntheticConfig, cell_counts, generate_plagiarism_corpus
from plagia.corpus.toy import generate_toy_corpus, toy_synonyms

__all__ = [
    "CleaningConfig",
    "SyntheticConfig",
    "assign_document_splits",
    "assign_group_splits",
    "carve_validation",
    "cell_counts",
    "check_no_leakage",
    "clean_corpus",
    "deduplicate",
    "generate_plagiarism_corpus",
    "generate_toy_corpus",
    "leakage_report",
    "load_corpus",
    "save_corpus",
    "subsample_authors",
    "subsample_per_author",
    "toy_synonyms",
    "truncate_split",
]
