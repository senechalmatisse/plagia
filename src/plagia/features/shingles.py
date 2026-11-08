"""Shingles (n-grammes de mots) et n-grammes de caractères pour les mesures d'ensembles et la détection de plagiat (avec leurs positions dans le texte)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

import numpy as np

from plagia.core.hashing import hash64
from plagia.core.lexicon import function_words
from plagia.core.text import WORD_PATTERN

_WORD_RE = re.compile(WORD_PATTERN)


def word_tokens(
    text: str, lowercase: bool = True, drop_function_words: bool = False, language: str = "en"
) -> list[tuple[str, int, int]]:
    """Mots avec leurs décalages ; option de retrait des mots-outils (shingles de contenu)."""
    stop = function_words(language) if drop_function_words else frozenset()
    out = []
    for m in _WORD_RE.finditer(text):
        word = m.group().lower() if lowercase else m.group()
        if word.lower() in stop:
            continue
        out.append((word, m.start(), m.end()))
    return out


def shingles(tokens: Sequence[str], n: int) -> list[str]:
    """Shingles de `n` mots consécutifs (joints par une espace)."""
    if n <= 1:
        return list(tokens)
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def shingle_set(text: str, n: int = 3, lowercase: bool = True, drop_function_words: bool = False) -> set[str]:
    """Ensemble des shingles de mots d'un texte."""
    return set(shingles([w for w, _, _ in word_tokens(text, lowercase, drop_function_words)], n))


def char_ngram_set(text: str, n: int = 4, lowercase: bool = True) -> set[str]:
    """Ensemble des n-grammes de caractères (blancs réduits)."""
    text = " ".join(text.split())
    if lowercase:
        text = text.lower()
    return {text[i : i + n] for i in range(max(len(text) - n + 1, 0))}


def hashed_shingles(tokens: Sequence[str], n: int) -> np.ndarray:
    """Empreintes 64 bits stables des shingles (ordre conservé)."""
    return np.fromiter((hash64(s) for s in shingles(tokens, n)), dtype=np.uint64)


def positioned_shingles(
    text: str, n: int, lowercase: bool = True, drop_function_words: bool = False
) -> list[tuple[int, int, int]]:
    """Shingles hachés avec leur empan de caractères : `(empreinte, début, fin)`."""
    words = word_tokens(text, lowercase, drop_function_words)
    out = []
    for i in range(len(words) - n + 1):
        key = " ".join(w for w, _, _ in words[i : i + n])
        out.append((hash64(key), words[i][1], words[i + n - 1][2]))
    return out


def union_size(sets: Iterable[set[str]]) -> int:
    """Taille de l'union de plusieurs ensembles."""
    total: set[str] = set()
    for item in sets:
        total |= item
    return len(total)
