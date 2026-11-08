"""Liage d'entités par DBpedia Spotlight.

Spotlight réalise le repérage des mentions, la génération des candidats, la désambiguïsation et le filtrage (seuils ``confidence`` et ``support``) ; chaque mention
est associée à une URI DBpedia, ce qui ramène les variantes d'une même entité (« U.S. », « United States ») à la même ressource. Le déploiement local par Docker est
recommandé ; les réponses sont mises en cache dans SQLite.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from plagia.annotation.cache import AnnotationCache, CacheMissError
from plagia.annotation.document import AnnotatedDocument, Entity
from plagia.annotation.schema import kind_from_types
from plagia.core.errors import ExternalServiceError

HttpPost = Callable[[str, bytes, Mapping[str, str]], bytes]

TYPE_PREFIXES = {
    "dbpedia": "dbo",
    "schema": "schema",
    "wikidata": "wikidata",
    "http://xmlns.com/foaf/0.1/": "foaf",
    "dul": "dul",
}


def normalize_type(raw: str) -> str:
    """``DBpedia:Person`` → ``dbo:Person`` ; ``Http://xmlns.com/foaf/0.1/Person`` → ``foaf:Person``."""
    raw = raw.strip()
    lower = raw.lower()
    for prefix, short in TYPE_PREFIXES.items():
        if prefix.startswith("http") and lower.startswith(prefix):
            return f"{short}:{raw[len(prefix) :]}"
        if lower.startswith(prefix + ":"):
            return f"{short}:{raw.split(':', 1)[1]}"
    return raw


@dataclass(frozen=True, slots=True)
class LinkedMention:
    """Mention liée à une ressource DBpedia."""

    surface: str
    start: int
    end: int
    uri: str
    types: tuple[str, ...]
    similarity: float
    support: int


def parse_spotlight(payload: Mapping[str, Any]) -> list[LinkedMention]:
    """Analyse la réponse JSON de ``/annotate``."""
    resources = payload.get("Resources") or []
    if isinstance(resources, Mapping):
        resources = [resources]
    mentions = []
    for res in resources:
        surface = str(res.get("@surfaceForm", ""))
        start = int(res.get("@offset", 0))
        types = tuple(normalize_type(t) for t in str(res.get("@types", "")).split(",") if t.strip())
        mentions.append(
            LinkedMention(
                surface=surface,
                start=start,
                end=start + len(surface),
                uri=str(res.get("@URI", "")),
                types=types,
                similarity=float(res.get("@similarityScore", 0.0) or 0.0),
                support=int(res.get("@support", 0) or 0),
            )
        )
    return mentions


def _default_post(timeout: int) -> HttpPost:
    def post(url: str, data: bytes, headers: Mapping[str, str]) -> bytes:
        request = urllib.request.Request(url, data=data, headers=dict(headers), method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body: bytes = response.read()
            return body

    return post


class SpotlightLinker:
    """Client DBpedia Spotlight avec cache des réponses."""

    name = "spotlight"

    def __init__(
        self,
        endpoint: str = "http://localhost:2222/rest",
        confidence: float = 0.5,
        support: int = 20,
        cache: AnnotationCache | None = None,
        offline: bool = False,
        timeout: int = 60,
        http_post: HttpPost | None = None,
        model_version: str = "unknown",
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.confidence = confidence
        self.support = support
        self.cache = cache
        self.offline = offline
        self.http_post = http_post or _default_post(timeout)
        self.model_version = model_version

    @property
    def version(self) -> str:
        """Identifiant du service (point d'accès et version du modèle déclarée)."""
        return f"{self.endpoint}#{self.model_version}"

    def params(self) -> dict[str, Any]:
        """Paramètres de filtrage."""
        return {"confidence": self.confidence, "support": self.support}

    def link(self, text: str) -> list[LinkedMention]:
        """Mentions liées du texte (cache consulté d'abord)."""
        key = AnnotationCache.key(text, self.name, self.version, self.params())
        if self.cache is not None:
            stored = self.cache.get_response(key)
            if stored is not None:
                return parse_spotlight(json.loads(stored))
        if self.offline:
            raise CacheMissError("Réponse Spotlight absente du cache (mode hors ligne).")
        data = urllib.parse.urlencode({"text": text, "confidence": self.confidence, "support": self.support}).encode(
            "utf-8"
        )
        try:
            body = self.http_post(f"{self.endpoint}/annotate", data, {"Accept": "application/json"})
        except OSError as exc:
            raise ExternalServiceError(f"Spotlight injoignable ({self.endpoint}) : {exc}") from exc
        decoded = body.decode("utf-8")
        if self.cache is not None:
            self.cache.put_response(key, self.name, decoded)
        return parse_spotlight(json.loads(decoded))

    def link_document(self, doc: AnnotatedDocument) -> AnnotatedDocument:
        """Attache URI et types aux entités ; ajoute les mentions liées non reconnues."""
        return merge_links(doc, self.link(doc.text))


def merge_links(doc: AnnotatedDocument, mentions: Sequence[LinkedMention]) -> AnnotatedDocument:
    """Fusionne les mentions liées avec les entités reconnues (recouvrement de caractères)."""
    entities = list(doc.entities)
    used: set[int] = set()
    for mention in mentions:
        best, best_overlap = None, 0
        for i, entity in enumerate(entities):
            overlap = min(entity.end, mention.end) - max(entity.start, mention.start)
            if overlap > best_overlap:
                best, best_overlap = i, overlap
        raw_types = tuple(f"raw:{t}" for t in mention.types)
        if best is not None and best not in used:
            used.add(best)
            entities[best] = Entity(
                entities[best].text,
                entities[best].start,
                entities[best].end,
                entities[best].label,
                entities[best].kind,
                uri=mention.uri,
                types=raw_types,
                score=mention.similarity,
            )
        elif best is None:
            entities.append(
                Entity(
                    mention.surface,
                    mention.start,
                    mention.end,
                    "SPOTLIGHT",
                    kind_from_types(mention.types),
                    uri=mention.uri,
                    types=raw_types,
                    score=mention.similarity,
                )
            )
    return doc.replace(entities=tuple(sorted(entities, key=lambda e: (e.start, e.end))))
