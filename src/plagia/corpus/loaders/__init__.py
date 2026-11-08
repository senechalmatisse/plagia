"""Chargeurs de corpus : C50, Gutenberg, PAN, AIDA CoNLL-YAGO."""

from plagia.corpus.loaders.aida import load_aida, read_aida_tsv, read_conll2003
from plagia.corpus.loaders.c50 import download_c50, load_c50
from plagia.corpus.loaders.gutenberg import GutenbergClient, GutenbergConfig, build_gutenberg_corpus
from plagia.corpus.loaders.pan import load_pan, parse_pan_xml, write_pan_detections

__all__ = [
    "GutenbergClient",
    "GutenbergConfig",
    "build_gutenberg_corpus",
    "download_c50",
    "load_aida",
    "load_c50",
    "load_pan",
    "parse_pan_xml",
    "read_aida_tsv",
    "read_conll2003",
    "write_pan_detections",
]
