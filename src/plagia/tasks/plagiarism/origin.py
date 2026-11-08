"""Identification de l'origine d'un passage détecté : classement des sources candidates par le meilleur score de paire sur leurs fenêtres de phrases."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from plagia.tasks.plagiarism.passages import PassageView, sentence_char_spans, sliding_windows

PairScorer = Callable[[Sequence[tuple[PassageView, PassageView]]], np.ndarray]


class OriginIdentifier:
    """Classe les sources candidates d'un passage (exactitude@1 et @k en évaluation)."""

    def __init__(self, scorer: PairScorer, window_sentences: int = 3) -> None:
        self.scorer = scorer
        self.window_sentences = window_sentences

    def rank(self, passage: PassageView, candidates: Sequence[Any]) -> list[tuple[str, float]]:
        """Sources triées par score décroissant (meilleure fenêtre de chaque source)."""
        pairs: list[tuple[PassageView, PassageView]] = []
        owners: list[str] = []
        for source in candidates:
            for a, b in sliding_windows(sentence_char_spans(source), self.window_sentences, 1):
                pairs.append((passage, PassageView.of(source, a, b)))
                owners.append(str(source.id))
        if not pairs:
            return []
        scores = self.scorer(pairs)
        best: dict[str, float] = {}
        for owner, score in zip(owners, scores, strict=True):
            best[owner] = max(best.get(owner, -np.inf), float(score))
        return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
