"""Mesures sur séquences de mots : distance d'édition et plus longue sous-séquence commune, calculées par parallélisme de bits (entiers Python de taille arbitraire).

* LCS : algorithme d'Allison-Dix / Hyyrö, O(⌈m/w⌉·n) ;
* Levenshtein : algorithme de Myers (1999) adapté par Hyyrö à la distance globale.

Les deux sont validés contre la programmation dynamique classique dans les tests.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

from plagia.core.text import words
from plagia.similarity.base import SIMILARITIES, Similarity, text_of


def _match_masks(sequence: Sequence[Hashable]) -> dict[Hashable, int]:
    masks: dict[Hashable, int] = {}
    for i, item in enumerate(sequence):
        masks[item] = masks.get(item, 0) | (1 << i)
    return masks


def lcs_length(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    """Longueur de la plus longue sous-séquence commune."""
    if len(a) < len(b):
        a, b = b, a
    m = len(a)
    if m == 0 or not b:
        return 0
    masks = _match_masks(a)
    full = (1 << m) - 1
    v = full
    for item in b:
        u = v & masks.get(item, 0)
        v = ((v + u) | (v - u)) & full
    return m - v.bit_count()


def levenshtein_distance(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    """Distance d'édition (insertions, suppressions, substitutions de coût 1)."""
    m = len(a)
    if m == 0:
        return len(b)
    if not b:
        return m
    peq = _match_masks(a)
    full = (1 << m) - 1
    high = 1 << (m - 1)
    pv, mv, score = full, 0, m
    for item in b:
        eq = peq.get(item, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = (mv | ~(xh | pv)) & full
        mh = pv & xh
        if ph & high:
            score += 1
        elif mh & high:
            score -= 1
        ph = ((ph << 1) | 1) & full
        mh = (mh << 1) & full
        pv = (mh | ~(xv | ph)) & full
        mv = ph & xv
    return score


def lcs_length_dp(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    """Référence par programmation dynamique (tests)."""
    previous = [0] * (len(b) + 1)
    for x in a:
        current = [0]
        for j, y in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if x == y else max(previous[j], current[j - 1]))
        previous = current
    return previous[-1]


def levenshtein_dp(a: Sequence[Hashable], b: Sequence[Hashable]) -> int:
    """Référence par programmation dynamique (tests)."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        current = [i]
        for j, y in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def _tokens(item: Any, lowercase: bool) -> list[str]:
    if isinstance(item, (list, tuple)):
        return [str(t).lower() if lowercase else str(t) for t in item]
    seq = words(text_of(item))
    return [w.lower() for w in seq] if lowercase else seq


@SIMILARITIES.register("levenshtein")
class Levenshtein(Similarity):
    """1 − distance d'édition sur les mots / longueur de la plus longue séquence."""

    name = "levenshtein"

    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def score(self, a: Any, b: Any) -> float:
        ta, tb = _tokens(a, self.lowercase), _tokens(b, self.lowercase)
        longest = max(len(ta), len(tb))
        return 1.0 - levenshtein_distance(ta, tb) / longest if longest else 0.0


@SIMILARITIES.register("lcs")
class Lcs(Similarity):
    """Longueur de la LCS des mots normalisée (`max`, `min` ou longueur de `a`)."""

    name = "lcs"

    def __init__(self, normalize: str = "max", lowercase: bool = True) -> None:
        if normalize not in {"max", "min", "first"}:
            raise ValueError("normalize doit valoir max, min ou first.")
        self.normalize = normalize
        self.lowercase = lowercase
        self.symmetric = normalize != "first"

    def score(self, a: Any, b: Any) -> float:
        ta, tb = _tokens(a, self.lowercase), _tokens(b, self.lowercase)
        if self.normalize == "max":
            denominator = max(len(ta), len(tb))
        elif self.normalize == "min":
            denominator = min(len(ta), len(tb))
        else:
            denominator = len(ta)
        return lcs_length(ta, tb) / denominator if denominator else 0.0
