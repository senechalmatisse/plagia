"""Rapport de similarité d'un document : score global, taux de couverture, passages alignés et sources. Le rendu HTML (surlignage) est assuré par `reporting`.

Un score de similarité n'est pas une preuve de plagiat : le rapport présente des passages candidats à examiner (citations, formules et textes standardisés produisent des correspondances légitimes).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from plagia.core.types import PassagePair


def covered_length(intervals: Sequence[tuple[int, int]]) -> int:
    """Nombre de caractères couverts par l'union d'intervalles."""
    total, end = 0, -1
    for a, b in sorted(intervals):
        if b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


@dataclass
class SimilarityReport:
    """Synthèse des détections d'un document suspect."""

    document_id: str
    length: int
    detections: list[PassagePair]
    coverage: float
    score: float
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON."""
        return {
            "document": self.document_id,
            "length": self.length,
            "coverage": self.coverage,
            "score": self.score,
            "sources": self.sources,
            "detections": [d.to_dict() for d in self.detections],
        }


def build_report(document_id: str, text: str, detections: Sequence[PassagePair]) -> SimilarityReport:
    """Score global (moyenne des scores pondérée par la longueur), couverture et sources."""
    intervals = [(d.suspicious.start, d.suspicious.end) for d in detections]
    covered = covered_length(intervals)
    coverage = covered / len(text) if text else 0.0
    weight = sum(d.suspicious.length for d in detections)
    score = sum(d.score * d.suspicious.length for d in detections) / weight if weight else 0.0
    sources: dict[str, dict[str, Any]] = {}
    for det in detections:
        if det.source is None:
            continue
        entry = sources.setdefault(det.source.document_id, {"passages": 0, "characters": 0, "max_score": 0.0})
        entry["passages"] += 1
        entry["characters"] += det.suspicious.length
        entry["max_score"] = max(entry["max_score"], det.score)
    return SimilarityReport(
        document_id, len(text), sorted(detections, key=lambda d: d.suspicious.start), coverage, score, sources
    )
