"""Qualité des annotateurs : reconnaissance (par type) et liage sur AIDA.

Les entités de référence sont lues dans `Document.meta["gold_entities"]` (chargeur AIDA).
La reconnaissance est évaluée par correspondance exacte de l'empan et du type (schéma minimal) ;
le liage par correspondance de l'empan et de la ressource (titre Wikipedia et URI DBpedia sont ramenés au même identifiant).
"""

from __future__ import annotations

import urllib.parse
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from plagia.annotation.document import AnnotatedDocument
from plagia.annotation.schema import normalize_label

GOLD_LABELS = ("PER", "ORG", "LOC", "MISC")


def prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Précision, rappel et F1 à partir des comptes."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def resource_id(uri: str | None) -> str | None:
    """`http://en.wikipedia.org/wiki/X` et `http://dbpedia.org/resource/X` → `X`."""
    if not uri:
        return None
    return urllib.parse.unquote(uri.rstrip("/").rsplit("/", 1)[-1]).replace(" ", "_")


def _score(gold: Iterable[tuple[Any, ...]], predicted: Iterable[tuple[Any, ...]]) -> tuple[int, int, int]:
    g, p = Counter(gold), Counter(predicted)
    tp = sum((g & p).values())
    return tp, sum(p.values()) - tp, sum(g.values()) - tp


def ner_quality(
    gold_documents: Sequence[Sequence[Mapping[str, Any]]],
    predicted: Sequence[AnnotatedDocument],
    labels: Sequence[str] = GOLD_LABELS,
    gold_scheme: str = "conll",
) -> dict[str, Any]:
    """Précision, rappel et F1 globaux, par type et pour la seule détection des mentions."""
    wanted = set(labels)
    gold_typed, pred_typed, gold_spans, pred_spans = [], [], [], []
    for d, (gold, doc) in enumerate(zip(gold_documents, predicted, strict=True)):
        for g in gold:
            if g.get("label"):
                kind = normalize_label(g["label"], gold_scheme)
                gold_typed.append((d, g["start"], g["end"], kind))
            gold_spans.append((d, g["start"], g["end"]))
        for e in doc.entities:
            if e.kind in wanted:
                pred_typed.append((d, e.start, e.end, e.kind))
                pred_spans.append((d, e.start, e.end))
    result: dict[str, Any] = {
        "overall": prf(*_score(gold_typed, pred_typed)),
        "mentions": prf(*_score(gold_spans, pred_spans)),
        "by_type": {},
    }
    for label in labels:
        result["by_type"][label] = prf(
            *_score([x for x in gold_typed if x[3] == label], [x for x in pred_typed if x[3] == label])
        )
    return result


def linking_quality(
    gold_documents: Sequence[Sequence[Mapping[str, Any]]],
    predicted: Sequence[AnnotatedDocument],
) -> dict[str, float]:
    """Liage : empan et ressource identiques (mentions de référence non liables exclues)."""
    gold: list[tuple[int, int, int, str | None]] = []
    pred: list[tuple[int, int, int, str | None]] = []
    for d, (gold_doc, doc) in enumerate(zip(gold_documents, predicted, strict=True)):
        gold.extend((d, g["start"], g["end"], resource_id(g.get("uri"))) for g in gold_doc if g.get("uri"))
        pred.extend((d, e.start, e.end, resource_id(e.uri)) for e in doc.entities if e.uri)
    return prf(*_score(gold, pred))
