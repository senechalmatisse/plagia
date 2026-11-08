"""Masquage du texte, text distortion, pour séparer le style du sujet.

Modes :

* `entities` : chaque mention d'entité est remplacée par son type (`PER`, `LOC`…) ou par un joker ;
* `propn` : chaque nom propre (étiquette `PROPN`) est remplacé ;
* `content` : distorsion de Stamatatos (2017) — tous les mots absents de la liste des mots fréquents (les `keep_top_k` mots les plus fréquents de l'entraînement, ou les
  mots-outils) sont masqués, caractère par caractère (`multiple`, DV-MA) ou par un seul astérisque (`single`, DV-SA) ; les chiffres deviennent `#`.

Le masquage est un transformateur de texte appliqué avant la vectorisation : il s'ajuste (liste des mots fréquents) sur l'entraînement uniquement.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from sklearn.base import BaseEstimator, TransformerMixin

from plagia.annotation.document import AnnotatedDocument
from plagia.core.errors import ConfigurationError
from plagia.core.lexicon import function_words
from plagia.core.text import is_word
from plagia.features.base import to_documents


def replace_token_ranges(
    doc: AnnotatedDocument, replacements: Mapping[int, tuple[int, str, str | None]]
) -> AnnotatedDocument:
    """Remplace des plages de jetons `[début, fin)` par un jeton unique.

    `replacements[début] = (fin, nouveau_jeton, nouvelle_étiquette)`. Les blancs entre
    jetons sont conservés ; les étiquettes des jetons inchangés aussi ; les entités sont
    retirées (leurs mentions ont pu disparaître).
    """
    tokens: list[str] = []
    offsets: list[tuple[int, int]] = []
    pos: list[str] = []
    mapping: dict[int, int] = {}
    pieces: list[str] = []
    cursor, position, i = 0, 0, 0
    n = len(doc.tokens)
    while i < n:
        start, end = doc.offsets[i]
        gap = doc.text[cursor:start]
        pieces.append(gap)
        position += len(gap)
        if i in replacements:
            stop, new_token, new_tag = replacements[i]
            stop = max(stop, i + 1)
            cursor = doc.offsets[stop - 1][1]
            label = new_tag if new_tag is not None else (doc.pos[i] if doc.pos else None)
        else:
            stop, new_token = i + 1, doc.tokens[i]
            cursor = end
            label = doc.pos[i] if doc.pos else None
        for k in range(i, stop):
            mapping[k] = len(tokens)
        tokens.append(new_token)
        offsets.append((position, position + len(new_token)))
        pos.append(label or "X")
        pieces.append(new_token)
        position += len(new_token)
        i = stop
    pieces.append(doc.text[cursor:])
    sentences = []
    for a, b in doc.sentences:
        if a in mapping and (b - 1) in mapping:
            sentences.append((mapping[a], mapping[b - 1] + 1))
    return AnnotatedDocument(
        id=doc.id,
        text="".join(pieces),
        tokens=tuple(tokens),
        offsets=tuple(offsets),
        pos=tuple(pos) if doc.pos is not None else None,
        sentences=tuple(sentences),
        entities=(),
        engine=f"{doc.engine}+masked",
    )


class Masker(TransformerMixin, BaseEstimator):
    """Transformateur texte → texte masqué (documents annotés en entrée et en sortie)."""

    def __init__(
        self,
        mode: str = "entities",
        placeholder: str = "type",
        keep_top_k: int | None = None,
        style: str = "multiple",
        language: str = "en",
    ) -> None:
        self.mode = mode
        self.placeholder = placeholder
        self.keep_top_k = keep_top_k
        self.style = style
        self.language = language

    def fit(self, X: Sequence[Any], y: Any = None) -> Masker:
        if self.mode not in {"none", "entities", "propn", "content"}:
            raise ConfigurationError(f"Mode de masquage inconnu : {self.mode}")
        if self.mode == "content" and self.keep_top_k:
            counts: Counter[str] = Counter()
            for doc in to_documents(X):
                counts.update(t.lower() for t in doc.tokens if is_word(t))
            self.keep_ = frozenset(w for w, _ in counts.most_common(self.keep_top_k))
        else:
            self.keep_ = function_words(self.language)
        return self

    def _mask_word(self, word: str) -> str:
        if self.style == "single":
            return "#" if word.isdigit() else "*"
        return "".join("#" if c.isdigit() else "*" for c in word)

    def mask(self, doc: AnnotatedDocument) -> AnnotatedDocument:
        """Version masquée d'un document."""
        replacements: dict[int, tuple[int, str, str | None]] = {}
        if self.mode == "entities":
            starts = [s for s, _ in doc.offsets]
            for entity in doc.entities:
                first = next((k for k, s in enumerate(starts) if doc.offsets[k][1] > entity.start), None)
                if first is None or first in replacements:
                    continue
                last = first
                while last + 1 < len(starts) and starts[last + 1] < entity.end:
                    last += 1
                token = entity.kind if self.placeholder == "type" else "*"
                replacements[first] = (last + 1, token, "PROPN")
        elif self.mode == "propn":
            if doc.pos is None:
                raise ConfigurationError("Le masquage des noms propres nécessite les étiquettes grammaticales.")
            for k, tag in enumerate(doc.pos):
                if tag == "PROPN":
                    replacements[k] = (k + 1, "PROPN" if self.placeholder == "type" else "*", None)
        elif self.mode == "content":
            for k, token in enumerate(doc.tokens):
                if is_word(token) and token.lower() not in self.keep_:
                    replacements[k] = (k + 1, self._mask_word(token), None)
        else:
            return doc
        return replace_token_ranges(doc, replacements)

    def transform(self, X: Sequence[Any]) -> list[AnnotatedDocument]:
        return [self.mask(doc) for doc in to_documents(X)]

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        return input_features
