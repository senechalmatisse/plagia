"""Vues de passages (texte et entités d'un empan) et fenêtres de phrases."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from plagia.annotation.document import AnnotatedDocument, Entity
from plagia.core.text import sentence_spans
from plagia.core.types import PassagePair


@dataclass(frozen=True, slots=True)
class PassageView:
    """Passage d'un document : décalages absolus, texte et entités qu'il contient."""

    document_id: str
    start: int
    end: int
    text: str
    entities: tuple[Entity, ...] = ()

    @classmethod
    def of(cls, document: Any, start: int, end: int) -> PassageView:
        """Passage `[start, end)` d'un document (annoté ou non)."""
        entities: tuple[Entity, ...] = ()
        if isinstance(document, AnnotatedDocument):
            entities = tuple(e for e in document.entities if e.start >= start and e.end <= end)
        return cls(str(document.id), start, end, document.text[start:end], entities)

    @property
    def length(self) -> int:
        """Longueur en caractères."""
        return self.end - self.start

    def annotated(self) -> AnnotatedDocument:
        """Document annoté minimal (texte et entités, décalages relatifs) pour les mesures."""
        shifted = tuple(
            Entity(e.text, e.start - self.start, e.end - self.start, e.label, e.kind, e.uri, e.types, e.score)
            for e in self.entities
        )
        return AnnotatedDocument(self.document_id, self.text, (), (), entities=shifted, engine="passage")


def detection_views(
    suspicious: Any, resolve: Callable[[str], Any], detections: Sequence[PassagePair]
) -> list[tuple[PassageView, PassageView]]:
    """Couples (passage suspect, passage source) des détections ; `resolve` renvoie la source."""
    views = []
    for detection in detections:
        source = detection.source
        if source is None:
            raise ValueError("Détection sans source : vue de paire impossible.")
        views.append(
            (
                PassageView.of(suspicious, detection.suspicious.start, detection.suspicious.end),
                PassageView.of(resolve(source.document_id), source.start, source.end),
            )
        )
    return views


def sentence_char_spans(document: Any) -> list[tuple[int, int]]:
    """Phrases en décalages de caractères (annotation si disponible, sinon segmenteur intégré)."""
    if isinstance(document, AnnotatedDocument) and document.sentences and document.offsets:
        return [(document.offsets[a][0], document.offsets[b - 1][1]) for a, b in document.sentences if b > a]
    return sentence_spans(document.text)


def sliding_windows(spans: Sequence[tuple[int, int]], size: int, step: int = 1) -> list[tuple[int, int]]:
    """Fenêtres glissantes de `size` phrases (une seule fenêtre si le texte est plus court)."""
    if not spans:
        return []
    if len(spans) <= size:
        return [(spans[0][0], spans[-1][1])]
    return [(spans[i][0], spans[i + size - 1][1]) for i in range(0, len(spans) - size + 1, step)]


def windows_inside(spans: Sequence[tuple[int, int]], low: int, high: int, size: int) -> list[tuple[int, int]]:
    """Fenêtres disjointes de `size` phrases entièrement contenues dans `[low, high)`."""
    inside = [s for s in spans if s[0] >= low and s[1] <= high]
    windows: list[tuple[int, int]] = []
    for i in range(0, len(inside), size):
        group = inside[i : i + size]
        if windows and len(group) < (size + 1) // 2:
            windows[-1] = (windows[-1][0], group[-1][1])
        else:
            windows.append((group[0][0], group[-1][1]))
    return windows


def windows_outside(windows: Sequence[tuple[int, int]], intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    """Fenêtres sans recouvrement avec les intervalles donnés."""
    return [w for w in windows if all(w[1] <= a or w[0] >= b for a, b in intervals)]
