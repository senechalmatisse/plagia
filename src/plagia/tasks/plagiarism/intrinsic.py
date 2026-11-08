"""Plagiat intrinsèque : ruptures de style à l'intérieur d'un document.

Le document est découpé en fenêtres de mots (par exemple 500 mots, pas de 250) ; chaque fenêtre reçoit un profil stylométrique ;
les fenêtres atypiques sont celles dont l'écart au profil du document dépasse `k` écarts-types (`zscore`) ou qui forment le groupe minoritaire d'un regroupement en deux classes (`kmeans`).
Les fenêtres retenues contiguës sont fusionnées en passages (évaluation au niveau des caractères).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from plagia.annotation.document import AnnotatedDocument
from plagia.core.text import word_spans
from plagia.core.types import Passage, PassagePair
from plagia.features.stylometry import Stylometry


def word_windows(text: str, size: int, step: int) -> list[tuple[int, int]]:
    """Fenêtres de `size` mots avec un pas de `step` mots (décalages de caractères)."""
    spans = word_spans(text)
    if not spans:
        return []
    if len(spans) <= size:
        return [(spans[0][0], spans[-1][1])]
    starts = list(range(0, len(spans) - size + 1, step))
    if starts[-1] + size < len(spans):
        starts.append(len(spans) - size)
    return [(spans[i][0], spans[i + size - 1][1]) for i in starts]


class IntrinsicDetector:
    """Détection de fenêtres stylistiquement atypiques."""

    def __init__(
        self, window_words: int = 500, step_words: int = 250, k: float = 2.0, method: str = "zscore", seed: int = 0
    ) -> None:
        self.window_words = window_words
        self.step_words = step_words
        self.k = k
        self.method = method
        self.seed = seed

    def profiles(self, text: str, windows: list[tuple[int, int]]) -> np.ndarray:
        """Profils stylométriques (non centrés) des fenêtres."""
        extractor = Stylometry(scale="none", mattr_window=50)
        docs = [AnnotatedDocument.from_text(str(i), text[a:b]) for i, (a, b) in enumerate(windows)]
        matrix = extractor.fit(docs).transform(docs).toarray()
        std = matrix.std(axis=0)
        keep = std > 1e-12
        return (matrix[:, keep] - matrix[:, keep].mean(axis=0)) / std[keep]

    def outlier_scores(self, profiles: np.ndarray) -> np.ndarray:
        """Distance de chaque fenêtre au profil moyen du document."""
        if profiles.size == 0:
            return np.zeros(profiles.shape[0])
        return np.linalg.norm(profiles, axis=1) / np.sqrt(profiles.shape[1])

    def detect(self, document: Any) -> list[PassagePair]:
        """Passages suspects (sans source) d'un document."""
        text = document.text
        windows = word_windows(text, self.window_words, self.step_words)
        if len(windows) < 3:
            return []
        profiles = self.profiles(text, windows)
        scores = self.outlier_scores(profiles)
        if self.method == "kmeans" and profiles.shape[1] > 0:
            labels = KMeans(n_clusters=2, n_init=10, random_state=self.seed).fit_predict(profiles)
            minority = int(np.argmin(np.bincount(labels, minlength=2)))
            flagged = (
                labels == minority if np.sum(labels == minority) < len(labels) / 2 else np.zeros(len(labels), bool)
            )
        else:
            flagged = scores > scores.mean() + self.k * scores.std()
        spans: list[tuple[int, int, float]] = []
        for (a, b), flag, score in zip(windows, flagged, scores, strict=True):
            if not flag:
                continue
            if spans and a <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(spans[-1][1], b), max(spans[-1][2], float(score)))
            else:
                spans.append((a, b, float(score)))
        return [PassagePair(Passage(str(document.id), a, b - a), None, score=s) for a, b, s in spans]
