"""Jeu de référence AIDA CoNLL-YAGO et format CoNLL 2003.

AIDA ne fournit que les annotations : le texte CoNLL 2003 (Reuters) s'obtient séparément et se fusionne avec l'outil de l'archive AIDA, qui produit
`AIDA-YAGO2-dataset.tsv`. Ce module lit ce fichier fusionné (mentions liées à YAGO et à Wikipedia) et, si on le fournit, les fichiers CoNLL 2003 d'origine pour récupérer les
types d'entités (PER, ORG, LOC, MISC) ; il en dérive un corpus dont chaque document porte ses entités de référence dans `meta["gold_entities"]`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from plagia.core.logging import get_logger
from plagia.core.types import Corpus, Document

log = get_logger(__name__)

AIDA_LICENCE = "CC BY-SA 3.0 (annotations AIDA) ; texte CoNLL 2003 sous licence Reuters"
_DOCSTART_RE = re.compile(r"^-DOCSTART-\s*(?:\((.*)\))?")


@dataclass
class GoldMention:
    """Mention de référence (positions en jetons, fin exclue)."""

    start: int
    end: int
    text: str
    label: str | None = None
    entity: str | None = None
    url: str | None = None


@dataclass
class TokenDocument:
    """Document segmenté en jetons et phrases, avec ses mentions de référence."""

    id: str
    tokens: list[str] = field(default_factory=list)
    sentence_starts: list[int] = field(default_factory=list)
    mentions: list[GoldMention] = field(default_factory=list)

    @property
    def split(self) -> str:
        """Jeu AIDA : `train`, `testa` (validation) ou `testb` (test)."""
        if "testa" in self.id:
            return "validation"
        if "testb" in self.id:
            return "test"
        return "train"


def read_aida_tsv(path: str | Path) -> list[TokenDocument]:
    """Lit `AIDA-YAGO2-dataset.tsv` (jeton[, B/I, mention, entité YAGO, URL Wikipedia…])."""
    documents: list[TokenDocument] = []
    current: TokenDocument | None = None
    open_mention: GoldMention | None = None
    new_sentence = True
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            match = _DOCSTART_RE.match(line)
            if match:
                current = TokenDocument(id=(match.group(1) or f"doc{len(documents)}").strip())
                documents.append(current)
                open_mention, new_sentence = None, True
                continue
            if current is None:
                continue
            if not line.strip():
                open_mention, new_sentence = None, True
                continue
            fields = line.split("\t")
            index = len(current.tokens)
            if new_sentence:
                current.sentence_starts.append(index)
                new_sentence = False
            current.tokens.append(fields[0])
            if len(fields) >= 3 and fields[1] in {"B", "I"}:
                if fields[1] == "B" or open_mention is None:
                    entity = fields[3] if len(fields) > 3 else None
                    open_mention = GoldMention(
                        start=index,
                        end=index + 1,
                        text=fields[2],
                        entity=None if entity in {None, "--NME--"} else entity,
                        url=fields[4] if len(fields) > 4 else None,
                    )
                    current.mentions.append(open_mention)
                else:
                    open_mention.end = index + 1
            else:
                open_mention = None
    return documents


def read_conll2003(path: str | Path) -> list[TokenDocument]:
    """Lit un fichier CoNLL 2003 (`jeton POS chunk NER`), schémas IOB1 et IOB2."""
    documents: list[TokenDocument] = []
    current: TokenDocument | None = None
    previous = "O"
    new_sentence = True
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("-DOCSTART-"):
                current = TokenDocument(id=f"conll{len(documents)}")
                documents.append(current)
                previous, new_sentence = "O", True
                continue
            if current is None:
                current = TokenDocument(id="conll0")
                documents.append(current)
            if not line:
                previous, new_sentence = "O", True
                continue
            fields = line.split()
            index = len(current.tokens)
            if new_sentence:
                current.sentence_starts.append(index)
                new_sentence = False
            current.tokens.append(fields[0])
            tag = fields[-1]
            if tag == "O":
                previous = "O"
                continue
            prefix, _, label = tag.partition("-")
            starts = prefix == "B" or previous == "O" or previous.partition("-")[2] != label
            if starts or not current.mentions:
                current.mentions.append(GoldMention(start=index, end=index + 1, text=fields[0], label=label))
            else:
                mention = current.mentions[-1]
                mention.end = index + 1
                mention.text = " ".join(current.tokens[mention.start : mention.end])
            previous = tag
    return documents


def merge_conll_labels(aida: list[TokenDocument], conll: list[TokenDocument]) -> int:
    """Reporte les types CoNLL sur les mentions AIDA de mêmes positions ; renvoie le nombre fusionné."""
    if len(aida) != len(conll):
        log.warning("AIDA (%d documents) et CoNLL (%d) ne s'alignent pas : fusion ignorée.", len(aida), len(conll))
        return 0
    merged = 0
    for a_doc, c_doc in zip(aida, conll, strict=True):
        if a_doc.tokens != c_doc.tokens:
            continue
        labels = {(m.start, m.end): m.label for m in c_doc.mentions}
        for mention in a_doc.mentions:
            label = labels.get((mention.start, mention.end))
            if label:
                mention.label = label
                merged += 1
    return merged


def token_documents_to_corpus(documents: list[TokenDocument], name: str = "aida") -> Corpus:
    """Texte reconstitué (jetons séparés par des espaces, une phrase par ligne) et entités
    de référence en décalages de caractères dans `meta["gold_entities"]`."""
    result = []
    for doc in documents:
        offsets: list[tuple[int, int]] = []
        pieces: list[str] = []
        position = 0
        starts = set(doc.sentence_starts)
        for i, token in enumerate(doc.tokens):
            if i > 0:
                separator = "\n" if i in starts else " "
                pieces.append(separator)
                position += 1
            offsets.append((position, position + len(token)))
            pieces.append(token)
            position += len(token)
        text = "".join(pieces)
        gold: list[dict[str, Any]] = [
            {
                "start": offsets[m.start][0],
                "end": offsets[m.end - 1][1],
                "text": text[offsets[m.start][0] : offsets[m.end - 1][1]],
                "label": m.label,
                "uri": m.url,
                "entity": m.entity,
            }
            for m in doc.mentions
            if m.end <= len(offsets)
        ]
        result.append(
            Document(
                id=f"aida/{doc.id}",
                text=text,
                language="en",
                licence=AIDA_LICENCE,
                split=doc.split,
                meta={"gold_entities": gold},
            )
        )
    return Corpus(result, name=name, meta={"source": "AIDA CoNLL-YAGO", "licence": AIDA_LICENCE})


def load_aida(tsv_path: str | Path, conll_path: str | Path | None = None) -> Corpus:
    """Charge AIDA CoNLL-YAGO (et les types CoNLL si `conll_path` est fourni)."""
    aida = read_aida_tsv(tsv_path)
    if conll_path is not None:
        merged = merge_conll_labels(aida, read_conll2003(conll_path))
        log.info("Types CoNLL reportés sur %d mention(s).", merged)
    return token_documents_to_corpus(aida)
