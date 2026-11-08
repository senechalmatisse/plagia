"""Types fondamentaux : `Document`, `Passage`, `PassagePair` et `Corpus`."""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, overload

from plagia.core.hashing import sha256_text

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class Document:
    """Texte et métadonnées d'un document du corpus.

    `work` désigne l'œuvre (corpus littéraire) ou le document source (plagiat) : c'est
    la clé de regroupement utilisée pour des découpages sans fuite.
    """

    id: str
    text: str
    author: str | None = None
    work: str | None = None
    language: str = "en"
    licence: str | None = None
    split: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @property
    def group(self) -> str:
        """Clé de regroupement (œuvre ou, à défaut, identifiant du document)."""
        return self.work or self.id

    @property
    def sha256(self) -> str:
        """Empreinte du texte."""
        return sha256_text(self.text)

    def replace(self, **changes: Any) -> Document:
        """Copie modifiée (les documents sont immuables)."""
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON."""
        return {
            "id": self.id,
            "text": self.text,
            "author": self.author,
            "work": self.work,
            "language": self.language,
            "licence": self.licence,
            "split": self.split,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Document:
        """Inverse de :meth:`to_dict`."""
        return cls(
            id=str(data["id"]),
            text=str(data["text"]),
            author=data.get("author"),
            work=data.get("work"),
            language=data.get("language") or "en",
            licence=data.get("licence"),
            split=data.get("split"),
            meta=dict(data.get("meta") or {}),
        )


@dataclass(frozen=True, slots=True, order=True)
class Passage:
    """Segment contigu d'un document repéré par ses décalages de caractères."""

    document_id: str
    start: int
    length: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.length < 0:
            raise ValueError(f"Passage invalide : début={self.start}, longueur={self.length}")

    @property
    def end(self) -> int:
        """Décalage de fin (exclu)."""
        return self.start + self.length

    def overlap(self, other: Passage) -> int:
        """Nombre de caractères communs avec `other` (0 si autre document)."""
        if self.document_id != other.document_id:
            return 0
        return max(0, min(self.end, other.end) - max(self.start, other.start))

    def overlaps(self, other: Passage) -> bool:
        """Vrai si les deux passages partagent au moins un caractère."""
        return self.overlap(other) > 0

    def text(self, document_text: str) -> str:
        """Texte du passage dans le document fourni."""
        return document_text[self.start : self.end]

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON."""
        return {"document": self.document_id, "start": self.start, "length": self.length}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Passage:
        """Inverse de :meth:`to_dict`."""
        return cls(str(data["document"]), int(data["start"]), int(data["length"]))


@dataclass(frozen=True, slots=True)
class PassagePair:
    """Cas de plagiat (vérité terrain) ou détection : passage suspect et passage source.

    `source` vaut `None` en plagiat intrinsèque. `obfuscation` vaut `O0` à `O4`
    pour le corpus synthétique ou reprend l'étiquette PAN.
    """

    suspicious: Passage
    source: Passage | None = None
    score: float = 1.0
    obfuscation: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def to_dict(self) -> dict[str, Any]:
        """Représentation JSON."""
        return {
            "suspicious": self.suspicious.to_dict(),
            "source": self.source.to_dict() if self.source else None,
            "score": self.score,
            "obfuscation": self.obfuscation,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PassagePair:
        """Inverse de :meth:`to_dict`."""
        source = data.get("source")
        return cls(
            suspicious=Passage.from_dict(data["suspicious"]),
            source=Passage.from_dict(source) if source else None,
            score=float(data.get("score", 1.0)),
            obfuscation=data.get("obfuscation"),
            meta=dict(data.get("meta") or {}),
        )


class Corpus(Sequence[Document]):
    """Collection ordonnée de documents à identifiants uniques.

    Un corpus de plagiat porte en plus ses cas (`cases`) et, éventuellement, la liste
    des paires (suspect, source) à aligner (`pairs`, format PAN).
    """

    def __init__(
        self,
        documents: Iterable[Document],
        name: str = "corpus",
        cases: Iterable[PassagePair] = (),
        pairs: Iterable[tuple[str, str]] = (),
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        self._documents: list[Document] = list(documents)
        self._index: dict[str, int] = {}
        for i, doc in enumerate(self._documents):
            if doc.id in self._index:
                raise ValueError(f"Identifiant de document dupliqué : {doc.id}")
            self._index[doc.id] = i
        self.name = name
        self.cases: list[PassagePair] = list(cases)
        self.pairs: list[tuple[str, str]] = [(str(a), str(b)) for a, b in pairs]
        self.meta: dict[str, Any] = dict(meta or {})

    # protocole Sequence
    def __len__(self) -> int:
        return len(self._documents)

    @overload
    def __getitem__(self, index: int) -> Document: ...

    @overload
    def __getitem__(self, index: slice) -> Corpus: ...

    def __getitem__(self, index: int | slice) -> Document | Corpus:
        if isinstance(index, slice):
            return self._derive(self._documents[index])
        return self._documents[index]

    def __iter__(self) -> Iterator[Document]:
        return iter(self._documents)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return item in self._index
        return isinstance(item, Document) and item.id in self._index

    def __repr__(self) -> str:
        return f"Corpus(name={self.name!r}, documents={len(self)}, cases={len(self.cases)})"

    # accès
    def get(self, doc_id: str) -> Document:
        """Document d'identifiant `doc_id`."""
        try:
            return self._documents[self._index[doc_id]]
        except KeyError:
            raise KeyError(f"Document inconnu : {doc_id}") from None

    @property
    def ids(self) -> list[str]:
        """Identifiants dans l'ordre du corpus."""
        return [d.id for d in self._documents]

    @property
    def texts(self) -> list[str]:
        """Textes dans l'ordre du corpus."""
        return [d.text for d in self._documents]

    @property
    def labels(self) -> list[str]:
        """Auteurs (étiquettes de classe) dans l'ordre du corpus."""
        return [d.author or "" for d in self._documents]

    @property
    def groups(self) -> list[str]:
        """Clés de regroupement (œuvre ou source) dans l'ordre du corpus."""
        return [d.group for d in self._documents]

    @property
    def authors(self) -> list[str]:
        """Auteurs distincts, triés."""
        return sorted({d.author for d in self._documents if d.author})

    def split_counts(self) -> dict[str, int]:
        """Effectif de chaque jeu."""
        return dict(Counter(d.split or "none" for d in self._documents))

    # dérivations
    def _derive(self, documents: Iterable[Document], name: str | None = None) -> Corpus:
        docs = list(documents)
        kept = {d.id for d in docs}
        cases = [c for c in self.cases if c.suspicious.document_id in kept]
        pairs = [p for p in self.pairs if p[0] in kept]
        return Corpus(docs, name=name or self.name, cases=cases, pairs=pairs, meta=self.meta)

    def filter(self, predicate: Callable[[Document], bool], name: str | None = None) -> Corpus:
        """Sous-corpus des documents satisfaisant `predicate` (cas filtrés en conséquence)."""
        return self._derive((d for d in self._documents if predicate(d)), name)

    def split(self, *names: str) -> Corpus:
        """Sous-corpus des jeux `names` (par exemple `corpus.split("train")`)."""
        wanted = set(names)
        return self.filter(lambda d: d.split in wanted)

    def with_documents(self, documents: Iterable[Document]) -> Corpus:
        """Nouveau corpus avec les mêmes métadonnées et cas, documents remplacés."""
        return Corpus(documents, name=self.name, cases=self.cases, pairs=self.pairs, meta=self.meta)

    def map(self, function: Callable[[Document], Document]) -> Corpus:
        """Applique `function` à chaque document."""
        return self.with_documents(function(d) for d in self._documents)
