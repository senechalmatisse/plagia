"""Interface commune des mesures de similarité et registre nom → classe."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from plagia.annotation.document import AnnotatedDocument
from plagia.core.registry import Registry

SIMILARITIES: Registry[Any] = Registry("mesure de similarité")


def text_of(item: Any) -> str:
    """Texte d'une entrée (texte brut ou document annoté)."""
    if isinstance(item, AnnotatedDocument):
        return item.text
    return str(item)


class Similarity(ABC):
    """Mesure `score(a, b) → [0, 1]` ; symétrique ou non selon la mesure."""

    name: str = "similarity"
    symmetric: bool = True

    def fit(self, items: Sequence[Any]) -> Similarity:
        """Ajuste la mesure (IDF, modèle sémantique…) sur des données d'entraînement."""
        return self

    @abstractmethod
    def score(self, a: Any, b: Any) -> float:
        """Similarité de `a` à `b` dans [0, 1]."""

    def score_many(self, pairs: Iterable[tuple[Any, Any]]) -> np.ndarray:
        """Similarités d'une suite de paires."""
        return np.array([self.score(a, b) for a, b in pairs], dtype=float)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({params})"
