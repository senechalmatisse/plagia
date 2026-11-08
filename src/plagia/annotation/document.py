"""Représentation légère et sérialisable d'un document annoté.

Les caractéristiques n'utilisent jamais directement les objets spaCy ou Stanford : elles lisent un :class:`AnnotatedDocument` (jetons et décalages, étiquettes
grammaticales, lemmes, phrases, entités typées), calculé une fois hors de la boucle d'apprentissage puis relu dans le cache.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from plagia.core.text import is_word, sentence_spans, tokenize_with_offsets


@dataclass(frozen=True, slots=True)
class Entity:
    """Mention d'entité nommée.

    `label` est l'étiquette du moteur (OntoNotes, CoNLL…), `kind` celle du schéma
    minimal commun (PER, ORG, LOC, MISC, DATE, NUM). `types` contient les classes de la
    base de connaissances sous la forme `"profondeur:classe"` (`"2:dbo:Person"`).
    """

    text: str
    start: int
    end: int
    label: str
    kind: str
    uri: str | None = None
    types: tuple[str, ...] = ()
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON compacte."""
        data: dict[str, Any] = {"t": self.text, "s": self.start, "e": self.end, "l": self.label, "k": self.kind}
        if self.uri:
            data["u"] = self.uri
        if self.types:
            data["ty"] = list(self.types)
        if self.score is not None:
            data["sc"] = self.score
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Entity:
        """Inverse de :meth:`to_dict`."""
        return cls(
            text=data["t"],
            start=int(data["s"]),
            end=int(data["e"]),
            label=data["l"],
            kind=data["k"],
            uri=data.get("u"),
            types=tuple(data.get("ty", ())),
            score=data.get("sc"),
        )

    def types_at_depth(self, depth: int) -> tuple[str, ...]:
        """Classes de profondeur `depth` (ou les plus profondes si l'entité est moins typée)."""
        parsed = [(int(d), c) for d, _, c in (t.partition(":") for t in self.types) if d.isdigit()]
        if not parsed:
            return ()
        target = min(depth, max(d for d, _ in parsed))
        return tuple(sorted({c for d, c in parsed if d == target}))


@dataclass(frozen=True, slots=True)
class AnnotatedDocument:
    """Document segmenté et annoté (jetons, étiquettes, lemmes, phrases, entités)."""

    id: str
    text: str
    tokens: tuple[str, ...]
    offsets: tuple[tuple[int, int], ...]
    pos: tuple[str, ...] | None = None
    tags: tuple[str, ...] | None = None
    lemmas: tuple[str, ...] | None = None
    sentences: tuple[tuple[int, int], ...] = ()
    entities: tuple[Entity, ...] = ()
    engine: str = "regex"
    meta: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        n = len(self.tokens)
        if len(self.offsets) != n:
            raise ValueError("tokens et offsets doivent avoir la même longueur.")
        for name in ("pos", "tags", "lemmas"):
            values = getattr(self, name)
            if values is not None and len(values) != n:
                raise ValueError(f"{name} doit avoir la même longueur que tokens.")

    @classmethod
    def from_text(cls, doc_id: str, text: str, engine: str = "regex") -> AnnotatedDocument:
        """Segmentation déterministe sans annotation (mots, ponctuation, phrases)."""
        triples = tokenize_with_offsets(text)
        starts = [s for _, s, _ in triples]
        sentences = token_sentences(starts, sentence_spans(text))
        return cls(
            id=doc_id,
            text=text,
            tokens=tuple(t for t, _, _ in triples),
            offsets=tuple((s, e) for _, s, e in triples),
            sentences=sentences,
            engine=engine,
        )

    @property
    def words(self) -> list[str]:
        """Jetons qui sont des mots (ponctuation exclue)."""
        return [t for t in self.tokens if is_word(t)]

    @property
    def has_pos(self) -> bool:
        """Vrai si les étiquettes grammaticales sont disponibles."""
        return self.pos is not None

    def replace(self, **changes: Any) -> AnnotatedDocument:
        """Copie modifiée."""
        return replace(self, **changes)

    def truncate_words(self, n_words: int) -> AnnotatedDocument:
        """Préfixe du document limité à `n_words` mots (annotations tronquées en cohérence)."""
        word_indices = [i for i, t in enumerate(self.tokens) if is_word(t)]
        if len(word_indices) <= n_words:
            return self
        last = word_indices[n_words - 1]
        cut, keep = self.offsets[last][1], last + 1
        return replace(
            self,
            text=self.text[:cut],
            tokens=self.tokens[:keep],
            offsets=self.offsets[:keep],
            pos=self.pos[:keep] if self.pos is not None else None,
            tags=self.tags[:keep] if self.tags is not None else None,
            lemmas=self.lemmas[:keep] if self.lemmas is not None else None,
            sentences=tuple((a, min(b, keep)) for a, b in self.sentences if a < keep),
            entities=tuple(e for e in self.entities if e.end <= cut),
        )

    def sentence_tokens(self) -> list[tuple[str, ...]]:
        """Jetons de chaque phrase."""
        return [self.tokens[a:b] for a, b in self.sentences] or [self.tokens]

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON (cache des annotations hors spaCy)."""
        return {
            "id": self.id,
            "text": self.text,
            "tokens": list(self.tokens),
            "offsets": [list(o) for o in self.offsets],
            "pos": list(self.pos) if self.pos is not None else None,
            "tags": list(self.tags) if self.tags is not None else None,
            "lemmas": list(self.lemmas) if self.lemmas is not None else None,
            "sentences": [list(s) for s in self.sentences],
            "entities": [e.to_dict() for e in self.entities],
            "engine": self.engine,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AnnotatedDocument:
        """Inverse de :meth:`to_dict`."""

        def _tuple(key: str) -> tuple[str, ...] | None:
            value = data.get(key)
            return tuple(value) if value is not None else None

        return cls(
            id=data["id"],
            text=data["text"],
            tokens=tuple(data["tokens"]),
            offsets=tuple((int(a), int(b)) for a, b in data["offsets"]),
            pos=_tuple("pos"),
            tags=_tuple("tags"),
            lemmas=_tuple("lemmas"),
            sentences=tuple((int(a), int(b)) for a, b in data.get("sentences", [])),
            entities=tuple(Entity.from_dict(e) for e in data.get("entities", [])),
            engine=data.get("engine", "regex"),
        )


def token_sentences(token_starts: Sequence[int], spans: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Convertit des phrases en caractères en intervalles de jetons `(début, fin exclue)`."""
    result: list[tuple[int, int]] = []
    i = 0
    n = len(token_starts)
    for start, end in spans:
        while i < n and token_starts[i] < start:
            i += 1
        j = i
        while j < n and token_starts[j] < end:
            j += 1
        if j > i:
            result.append((i, j))
        i = j
    return tuple(result)


def as_annotated(item: AnnotatedDocument | str, doc_id: str = "") -> AnnotatedDocument:
    """Accepte un texte brut ou un document annoté."""
    if isinstance(item, AnnotatedDocument):
        return item
    return AnnotatedDocument.from_text(doc_id, str(item))
