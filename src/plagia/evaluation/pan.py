"""Mesures de PAN pour la détection de passages (Potthast et al., COLING 2010).

* précision et rappel macro-moyennés au niveau des caractères, comptés dans le document suspect ET dans la source (comme le script d'évaluation officiel de PAN) ;
* granularité : nombre moyen de détections par cas détecté ;
* plagdet = F1 / log₂(1 + granularité).

Une détection « détecte » un cas si elle le recouvre dans le document suspect et, pour le plagiat extrinsèque, dans le même document source.
Le plagiat intrinsèque (cas sans source) est évalué sur le seul document suspect. Ce module ne dépend que de `core`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from plagia.core.types import Passage, PassagePair


def _overlaps(a: Passage, b: Passage) -> bool:
    return a.document_id == b.document_id and a.start < b.end and b.start < a.end


def detects(case: PassagePair, detection: PassagePair) -> bool:
    """Vrai si `detection` recouvre `case` (dans la source aussi s'il y en a une)."""
    if not _overlaps(case.suspicious, detection.suspicious):
        return False
    if case.source is None:
        return True
    return detection.source is not None and _overlaps(case.source, detection.source)


def _covered(reference: Passage, others: Sequence[Passage]) -> int:
    """Caractères de `reference` couverts par l'union des passages `others`."""
    intervals = sorted(
        (max(reference.start, o.start), min(reference.end, o.end))
        for o in others
        if o.document_id == reference.document_id and o.start < reference.end and reference.start < o.end
    )
    total, end = 0, reference.start
    for a, b in intervals:
        if b <= end:
            continue
        total += b - max(a, end)
        end = b
    return total


def _index(pairs: Sequence[PassagePair]) -> dict[str, list[int]]:
    index: dict[str, list[int]] = defaultdict(list)
    for i, pair in enumerate(pairs):
        index[pair.suspicious.document_id].append(i)
    return index


def _macro_recall(references: Sequence[PassagePair], others: Sequence[PassagePair]) -> float:
    if not references:
        return 0.0
    by_doc = _index(others)
    total = 0.0
    for ref in references:
        matched = [
            others[j]
            for j in by_doc.get(ref.suspicious.document_id, [])
            if detects(ref, others[j]) or detects(others[j], ref)
        ]
        covered = _covered(ref.suspicious, [m.suspicious for m in matched])
        size = ref.suspicious.length
        if ref.source is not None:
            covered += _covered(ref.source, [m.source for m in matched if m.source is not None])
            size += ref.source.length
        total += covered / size if size else 0.0
    return total / len(references)


def granularity(cases: Sequence[PassagePair], detections: Sequence[PassagePair]) -> float:
    """Nombre moyen de détections par cas détecté (1 si aucun cas n'est détecté)."""
    by_doc = _index(detections)
    counts = [
        sum(1 for j in by_doc.get(case.suspicious.document_id, []) if detects(case, detections[j])) for case in cases
    ]
    detected = [c for c in counts if c > 0]
    return sum(detected) / len(detected) if detected else 1.0


def plagdet(precision: float, recall: float, gran: float) -> float:
    """plagdet = F1 / log₂(1 + granularité)."""
    if precision <= 0 or recall <= 0 or gran < 1:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return f1 / math.log2(1 + gran)


def pan_measures(cases: Sequence[PassagePair], detections: Sequence[PassagePair]) -> dict[str, float]:
    """Précision, rappel, F1, granularité et plagdet (macro-moyennes, niveau caractère)."""
    if not cases and not detections:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "granularity": 1.0,
            "plagdet": 1.0,
            "cases": 0,
            "detections": 0,
        }
    recall = _macro_recall(cases, detections) if detections else 0.0
    precision = _macro_recall(detections, cases) if cases and detections else 0.0
    gran = granularity(cases, detections)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "granularity": gran,
        "plagdet": plagdet(precision, recall, gran),
        "cases": len(cases),
        "detections": len(detections),
    }


def case_detection_rates(
    cases: Sequence[PassagePair], detections: Sequence[PassagePair], min_overlap: float = 0.5
) -> dict[str, float]:
    """Précision et rappel au niveau des passages : un cas est trouvé si une détection en
    couvre au moins `min_overlap` (dans le document suspect) ; une détection est correcte
    si elle couvre au moins `min_overlap` d'elle-même par un cas."""
    by_det = _index(detections)
    by_case = _index(cases)
    found = 0
    for case in cases:
        matched = [
            detections[j].suspicious
            for j in by_det.get(case.suspicious.document_id, [])
            if detects(case, detections[j])
        ]
        if case.suspicious.length and _covered(case.suspicious, matched) / case.suspicious.length >= min_overlap:
            found += 1
    correct = 0
    for det in detections:
        matched = [cases[i].suspicious for i in by_case.get(det.suspicious.document_id, []) if detects(cases[i], det)]
        if det.suspicious.length and _covered(det.suspicious, matched) / det.suspicious.length >= min_overlap:
            correct += 1
    recall = found / len(cases) if cases else 1.0
    precision = correct / len(detections) if detections else (1.0 if not cases else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"passage_precision": precision, "passage_recall": recall, "passage_f1": f1}
