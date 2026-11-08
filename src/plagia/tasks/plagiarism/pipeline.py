"""Détecteur de plagiat extrinsèque complet : recherche → alignement → filtrage.

Deux modes d'alignement, combinables : par graines (`seeds`) et par fenêtres glissantes de phrases notées par le classificateur de paires (`windows`).
La sortie est une liste de :class:`~plagia.core.types.PassagePair` (passage suspect, passage source, score) avec les décalages de caractères.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from plagia.core.types import Passage, PassagePair
from plagia.tasks.plagiarism.alignment import Filter, TextAligner
from plagia.tasks.plagiarism.pairs import PairClassifier
from plagia.tasks.plagiarism.passages import PassageView, sentence_char_spans, sliding_windows
from plagia.tasks.plagiarism.retrieval import CandidateRetriever


class ExtrinsicDetector:
    """Plagiat extrinsèque au niveau du passage."""

    def __init__(
        self,
        retriever: CandidateRetriever | None = None,
        aligner: TextAligner | None = None,
        pair_classifier: PairClassifier | None = None,
        mode: str = "seeds",
        window_sentences: int = 3,
        final_filter: Filter | None = None,
    ) -> None:
        if mode not in {"seeds", "windows", "both"}:
            raise ValueError(f"Mode d'alignement inconnu : {mode}")
        self.retriever = retriever or CandidateRetriever()
        self.aligner = aligner or TextAligner()
        self.pair_classifier = pair_classifier
        self.mode = mode
        self.window_sentences = window_sentences
        self.final_filter = final_filter or Filter(min_length=0)

    def fit(self, sources: Sequence[Any]) -> ExtrinsicDetector:
        """Indexe les sources (documents bruts ou annotés)."""
        self.sources_ = {str(s.id): s for s in sources}
        self.retriever.fit(list(sources))
        return self

    def window_detections(self, susp: Any, src: Any) -> list[PassagePair]:
        """Fenêtres suspectes × fenêtres sources notées par le classificateur, puis fusion."""
        if self.pair_classifier is None:
            return []
        s_windows = sliding_windows(sentence_char_spans(susp), self.window_sentences, 1)
        t_windows = sliding_windows(sentence_char_spans(src), self.window_sentences, 1)
        pairs = [(PassageView.of(susp, a, b), PassageView.of(src, c, d)) for a, b in s_windows for c, d in t_windows]
        if not pairs:
            return []
        scores = self.pair_classifier.predict_proba(pairs)
        hits = [(p, float(s)) for p, s in zip(pairs, scores, strict=True) if s >= self.pair_classifier.threshold]
        hits.sort(key=lambda h: h[0][0].start)
        merged: list[list[Any]] = []
        for (sv, tv), score in hits:
            if merged and sv.start <= merged[-1][1]:
                last = merged[-1]
                last[1] = max(last[1], sv.end)
                last[2] = min(last[2], tv.start)
                last[3] = max(last[3], tv.end)
                last[4] = max(last[4], score)
            else:
                merged.append([sv.start, sv.end, tv.start, tv.end, score])
        return [
            PassagePair(
                Passage(str(susp.id), a, b - a), Passage(str(src.id), c, d - c), score=s, meta={"mode": "windows"}
            )
            for a, b, c, d, s in merged
        ]

    def detect(self, suspicious: Any, candidates: Sequence[str] | None = None) -> list[PassagePair]:
        """Passages repris d'un document suspect, avec leur source."""
        source_ids = list(candidates) if candidates is not None else [s for s, _ in self.retriever.retrieve(suspicious)]
        detections: list[PassagePair] = []
        for source_id in source_ids:
            source = self.sources_.get(source_id)
            if source is None:
                continue
            if self.mode in {"seeds", "both"}:
                detections.extend(self.aligner.align(suspicious, source))
            if self.mode in {"windows", "both"}:
                detections.extend(self.window_detections(suspicious, source))
        return self.final_filter.apply(detections)

    def detect_many(self, documents: Sequence[Any]) -> dict[str, list[PassagePair]]:
        """Détections pour plusieurs documents suspects."""
        return {str(d.id): self.detect(d) for d in documents}
