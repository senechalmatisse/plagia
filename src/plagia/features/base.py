"""Socle des blocs de caractéristiques : registre, familles et utilitaires communs.

Chaque bloc suit l'interface `fit` / `transform` de scikit-learn et s'ajuste à l'intérieur des plis de validation croisée (aucune fuite).
Les entrées sont des :class:`~plagia.annotation.document.AnnotatedDocument` (ou des textes bruts).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np
from scipy import sparse

from plagia.annotation.document import AnnotatedDocument, as_annotated
from plagia.core.registry import Registry

FEATURE_BLOCKS: Registry[Any] = Registry("bloc de caractéristiques")
FAMILIES: dict[str, str] = {}

# Familles de caractéristiques utilisées pour les ablations (F8.4).
FAMILY_NAMES = ("words", "chars", "stylometry", "pos", "entities", "types", "semantic")


def register_block(name: str, family: str) -> Callable[[type], type]:
    """Enregistre un bloc et sa famille."""

    def decorator(cls: type) -> type:
        FEATURE_BLOCKS.add(name, cls)
        FAMILIES[name] = family
        return cls

    return decorator


def block_family(block_type: str) -> str:
    """Famille d'un type de bloc (`words`, `chars`, `stylometry`…)."""
    return FAMILIES.get(block_type, block_type)


def to_documents(X: Iterable[Any]) -> list[AnnotatedDocument]:
    """Convertit les entrées (textes ou documents annotés) en documents annotés."""
    return [as_annotated(x, str(i)) for i, x in enumerate(X)]


def as_csr(matrix: Any) -> sparse.csr_matrix:
    """Matrice creuse CSR en float64."""
    if sparse.issparse(matrix):
        return sparse.csr_matrix(matrix, dtype=np.float64)
    return sparse.csr_matrix(np.asarray(matrix, dtype=np.float64))


def ngrams(sequence: Sequence[str], ngram_range: Sequence[int]) -> list[str]:
    """N-grammes (joints par une espace) de longueur `ngram_range[0]` à `ngram_range[1]`."""
    low, high = int(ngram_range[0]), int(ngram_range[1])
    out: list[str] = []
    for n in range(low, high + 1):
        if n == 1:
            out.extend(sequence)
        else:
            out.extend(" ".join(sequence[i : i + n]) for i in range(len(sequence) - n + 1))
    return out


class MinMaxScalerSafe:
    """Mise à l'échelle dans [0, 1] ajustée sur l'entraînement, bornée à l'application.

    Les attributs restent positifs : ils sont compatibles avec le naïve Bayes multinomial,
    croisé avec toutes les représentations dans l'étude A.
    """

    def __init__(self) -> None:
        self.min_: np.ndarray | None = None
        self.range_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> MinMaxScalerSafe:
        self.min_ = values.min(axis=0) if len(values) else np.zeros(values.shape[1])
        spread = (values.max(axis=0) - self.min_) if len(values) else np.ones(values.shape[1])
        self.range_ = np.where(spread > 1e-12, spread, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        assert self.min_ is not None
        assert self.range_ is not None
        return np.clip((values - self.min_) / self.range_, 0.0, 1.0)


def make_scaler(kind: str) -> MinMaxScalerSafe | StandardScalerSafe | None:
    """Échelle des attributs denses : `minmax` (défaut), `standard` ou `none`."""
    if kind == "minmax":
        return MinMaxScalerSafe()
    if kind == "standard":
        return StandardScalerSafe()
    if kind in {"none", ""}:
        return None
    raise ValueError(f"Mise à l'échelle inconnue : {kind}")


class StandardScalerSafe:
    """Centrage-réduction tolérant aux variances nulles (attributs stylométriques denses)."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> StandardScalerSafe:
        self.mean_ = values.mean(axis=0)
        std = values.std(axis=0)
        self.scale_ = np.where(std > 1e-12, std, 1.0)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None
        assert self.scale_ is not None
        return (values - self.mean_) / self.scale_
