"""Mesures sur ensembles : Jaccard, Dice, recouvrement (containment), chevauchement.

Les unités sont les mots, les *shingles* de mots (n de 3 à 8) ou les n-grammes de caractères.
Le coefficient de recouvrement |A ∩ B| / |A| est asymétrique : il convient à un passage court contenu dans une source longue.
"""

from __future__ import annotations

from collections.abc import Set
from typing import Any

from plagia.features.shingles import char_ngram_set, shingle_set
from plagia.similarity.base import SIMILARITIES, Similarity, text_of


def jaccard(a: Set[Any], b: Set[Any]) -> float:
    """|A ∩ B| / |A ∪ B| (0 si les deux ensembles sont vides)."""
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def dice(a: Set[Any], b: Set[Any]) -> float:
    """2 |A ∩ B| / (|A| + |B|)."""
    total = len(a) + len(b)
    return 2 * len(a & b) / total if total else 0.0


def containment(a: Set[Any], b: Set[Any]) -> float:
    """|A ∩ B| / |A| : part de `a` contenue dans `b` (asymétrique)."""
    return len(a & b) / len(a) if a else 0.0


def overlap_coefficient(a: Set[Any], b: Set[Any]) -> float:
    """|A ∩ B| / min(|A|, |B|)."""
    smallest = min(len(a), len(b))
    return len(a & b) / smallest if smallest else 0.0


class SetSimilarity(Similarity):
    """Base des mesures d'ensembles sur mots, shingles ou n-grammes de caractères."""

    def __init__(
        self, unit: str = "shingle", n: int = 3, lowercase: bool = True, drop_function_words: bool = False
    ) -> None:
        if unit not in {"word", "shingle", "char"}:
            raise ValueError(f"Unité inconnue : {unit}")
        self.unit = unit
        self.n = n if unit != "word" else 1
        self.lowercase = lowercase
        self.drop_function_words = drop_function_words

    def units(self, item: Any) -> set[str]:
        """Ensemble des unités d'un texte (ou ensemble fourni tel quel)."""
        if isinstance(item, (set, frozenset)):
            return set(item)
        text = text_of(item)
        if self.unit == "char":
            return char_ngram_set(text, self.n, self.lowercase)
        return shingle_set(text, self.n, self.lowercase, self.drop_function_words)

    def compare(self, a: set[str], b: set[str]) -> float:
        raise NotImplementedError

    def score(self, a: Any, b: Any) -> float:
        return self.compare(self.units(a), self.units(b))


@SIMILARITIES.register("jaccard")
class Jaccard(SetSimilarity):
    """Indice de Jaccard."""

    name = "jaccard"

    def compare(self, a: set[str], b: set[str]) -> float:
        return jaccard(a, b)


@SIMILARITIES.register("dice")
class Dice(SetSimilarity):
    """Coefficient de Dice."""

    name = "dice"

    def compare(self, a: set[str], b: set[str]) -> float:
        return dice(a, b)


@SIMILARITIES.register("containment")
class Containment(SetSimilarity):
    """Coefficient de recouvrement |A ∩ B| / |A| (asymétrique)."""

    name = "containment"
    symmetric = False

    def compare(self, a: set[str], b: set[str]) -> float:
        return containment(a, b)


@SIMILARITIES.register("overlap")
class Overlap(SetSimilarity):
    """Coefficient de chevauchement |A ∩ B| / min(|A|, |B|)."""

    name = "overlap"

    def compare(self, a: set[str], b: set[str]) -> float:
        return overlap_coefficient(a, b)
