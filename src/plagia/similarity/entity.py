"""Similarité d'entités : Jaccard et cosinus des ensembles d'entités (formes de surface, puis URI après liage) ; divergence de Jensen-Shannon entre distributions de types."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any

from plagia.annotation.document import AnnotatedDocument, as_annotated
from plagia.similarity.base import SIMILARITIES, Similarity
from plagia.similarity.sets import jaccard


def entity_forms(doc: AnnotatedDocument, level: str = "surface") -> Counter[str]:
    """Multiensemble des entités (surface en minuscules, ou URI si disponible)."""
    forms: Counter[str] = Counter()
    for entity in doc.entities:
        if level == "uri" and entity.uri:
            forms[entity.uri] += 1
        else:
            forms[" ".join(entity.text.lower().split())] += 1
    return forms


def counter_cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """Cosinus entre deux vecteurs creux représentés par des dictionnaires."""
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def jensen_shannon(p: Mapping[str, float], q: Mapping[str, float]) -> float:
    """Divergence de Jensen-Shannon (base 2, dans [0, 1]) entre deux distributions."""
    total_p, total_q = sum(p.values()), sum(q.values())
    if not total_p or not total_q:
        return 1.0 if (total_p or total_q) else 0.0
    keys = set(p) | set(q)
    divergence = 0.0
    for key in keys:
        pk, qk = p.get(key, 0.0) / total_p, q.get(key, 0.0) / total_q
        mk = (pk + qk) / 2
        if pk > 0:
            divergence += 0.5 * pk * math.log2(pk / mk)
        if qk > 0:
            divergence += 0.5 * qk * math.log2(qk / mk)
    return min(max(divergence, 0.0), 1.0)


@SIMILARITIES.register("entity")
class EntitySimilarity(Similarity):
    """Jaccard ou cosinus des entités de deux documents."""

    name = "entity"

    def __init__(self, method: str = "jaccard", level: str = "surface") -> None:
        if method not in {"jaccard", "cosine"}:
            raise ValueError("method doit valoir jaccard ou cosine.")
        self.method = method
        self.level = level

    def score(self, a: Any, b: Any) -> float:
        fa = entity_forms(as_annotated(a), self.level)
        fb = entity_forms(as_annotated(b), self.level)
        if self.method == "jaccard":
            return jaccard(set(fa), set(fb))
        return counter_cosine(fa, fb)


@SIMILARITIES.register("type_divergence")
class TypeDivergence(Similarity):
    """1 − divergence de Jensen-Shannon entre distributions de types d'entités.

    `depth=None` compare les types de reconnaissance (PER, ORG…) ; sinon les classes de
    la base de connaissances à la profondeur `depth`.
    """

    name = "type_divergence"

    def __init__(self, depth: int | None = None) -> None:
        self.depth = depth

    def distribution(self, doc: AnnotatedDocument) -> Counter[str]:
        """Distribution des types du document."""
        counts: Counter[str] = Counter()
        for entity in doc.entities:
            if self.depth is not None and entity.types:
                counts.update(entity.types_at_depth(self.depth))
            else:
                counts[entity.kind] += 1
        return counts

    def score(self, a: Any, b: Any) -> float:
        return 1.0 - jensen_shannon(self.distribution(as_annotated(a)), self.distribution(as_annotated(b)))
