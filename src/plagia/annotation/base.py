"""Interfaces des annotateurs et registre des moteurs."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from plagia.annotation.document import AnnotatedDocument, Entity
from plagia.core.registry import Registry


class EntityRecognizer(ABC):
    """Reconnaissance d'entités nommées, indépendante du moteur."""

    @abstractmethod
    def recognize_many(self, texts: Sequence[str]) -> list[list[Entity]]:
        """Entités de chaque texte."""

    def recognize(self, text: str) -> list[Entity]:
        """Entités d'un texte."""
        return self.recognize_many([text])[0]


class Annotator(EntityRecognizer):
    """Moteur d'annotation : segmentation, étiquettes grammaticales, entités.

    Le traitement est découpé en quatre étapes pour permettre la mise en cache par le
    proxy :class:`~plagia.annotation.cache.CachedAnnotator` : `process` (sortie brute
    du moteur), ``serialize`` / ``deserialize`` (stockage) et `convert` (vers :`AnnotatedDocument`).
    """

    name: str = "annotator"
    scheme: str = "minimal"
    payload_format: str = "json"

    @property
    def version(self) -> str:
        """Version de l'outil et du modèle (clé du cache)."""
        return "0"

    def params(self) -> dict[str, Any]:
        """Paramètres qui influencent le résultat (clé du cache)."""
        return {}

    @abstractmethod
    def process(self, texts: Sequence[str]) -> list[Any]:
        """Sortie brute du moteur pour chaque texte."""

    def serialize(self, raw: Any) -> bytes:
        """Sortie brute → octets (défaut : JSON d'un ``AnnotatedDocument``)."""
        return json.dumps(raw.to_dict(), ensure_ascii=False).encode("utf-8")

    def deserialize(self, payload: bytes) -> Any:
        """Octets → sortie brute."""
        return AnnotatedDocument.from_dict(json.loads(payload.decode("utf-8")))

    def convert(self, doc_id: str, text: str, raw: Any) -> AnnotatedDocument:
        """Sortie brute → document annoté."""
        document: AnnotatedDocument = raw
        return document.replace(id=doc_id)

    def annotate_many(self, items: Sequence[tuple[str, str]]) -> list[AnnotatedDocument]:
        """Annote des couples `(identifiant, texte)`."""
        raws = self.process([text for _, text in items])
        return [self.convert(doc_id, text, raw) for (doc_id, text), raw in zip(items, raws, strict=True)]

    def annotate(self, doc_id: str, text: str) -> AnnotatedDocument:
        """Annote un texte."""
        return self.annotate_many([(doc_id, text)])[0]

    def recognize_many(self, texts: Sequence[str]) -> list[list[Entity]]:
        return [list(doc.entities) for doc in self.annotate_many([("", t) for t in texts])]


ANNOTATORS: Registry[Annotator] = Registry("annotateur")


class RegexAnnotator(Annotator):
    """Segmentation seule (mots, ponctuation, phrases), sans étiquettes ni entités."""

    name = "regex"

    @property
    def version(self) -> str:
        return "1"

    def process(self, texts: Sequence[str]) -> list[AnnotatedDocument]:
        return [AnnotatedDocument.from_text("", t, engine=self.name) for t in texts]


ANNOTATORS.add("none", RegexAnnotator)
ANNOTATORS.add("regex", RegexAnnotator)
