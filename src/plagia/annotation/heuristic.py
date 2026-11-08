"""Annotateur heuristique sans dépendance (mode dégradé, démonstrations hors ligne).

Étiquettes grammaticales universelles par lexique de mots-outils et règles de suffixes ; entités par séquences de mots à majuscule, indices de titre (« Mr. »),
suffixes d'organisation (« Bank », « Company »), prépositions de lieu ; dates et nombres par expressions régulières.
Sa qualité est très inférieure à celle de spaCy : il sert à exécuter la chaîne complète sans modèle installé, jamais à produire des résultats.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from plagia.annotation.base import ANNOTATORS, Annotator
from plagia.annotation.document import AnnotatedDocument, Entity, token_sentences
from plagia.core.text import is_punctuation, sentence_spans, tokenize_with_offsets

_LEXICON: dict[str, str] = {}
_CLOSED_CLASSES = {
    "DET": "the a an this that these those every each some any no his her its their our my your all both either "
    "neither",
    "PRON": "i you he she it we they me him us them who whom which what whose myself himself herself itself ourselves "
    "themselves something nothing anything everything someone one",
    "ADP": "in on at of to for with by from as into onto upon within without about above below beyond across beside "
    "behind toward towards among amongst along near under over through during after before since until via against",
    "CCONJ": "and or but nor yet so",
    "SCONJ": "because although while when since whereas though unless if whether that as until whilst",
    "AUX": "is are was were be been being am has have had do does did will would shall should can could may might must",
    "ADV": "not very also then still thus however moreover indeed perhaps quite rather often never always almost soon "
    "here there now already even just only too again ever once yet",
    "PART": "'s n't",
    "INTJ": "oh ah alas yes",
}
for _tag, _words in _CLOSED_CLASSES.items():
    for _word in _words.split():
        _LEXICON.setdefault(_word, _tag)

_IRREGULAR_VERBS = frozenset(
    "met found left sent wrote bought built sold wept slept bore made said took gave came went saw "
    "knew thought told became began brought held kept felt stood understood".split()
)
TITLES = frozenset(
    "mr mr. mrs mrs. ms ms. miss dr dr. sir lady lord captain colonel general president king queen "
    "prince princess professor prof. senator minister judge father mother sister brother saint st.".split()
)
ORG_SUFFIXES = frozenset(
    "company bank society railway mills gazette institute club inc inc. corp corp. ltd ltd. co co. "
    "group university association party ministry council agency commission corporation times "
    "news airlines motors holdings plc ag sa".split()
)
LOC_PREPOSITIONS = frozenset("in at near from to toward towards beyond across within into through".split())
LOCATIONS = frozenset(
    "london paris berlin rome madrid washington moscow tokyo beijing china japan france germany "
    "italy spain russia britain england scotland ireland america canada mexico india brazil "
    "europe asia africa australia york boston chicago texas california hong kong".split()
)
_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_DATE_RE = re.compile(
    rf"\b(?:(?:\d{{1,2}} )?(?:{_MONTHS})(?: \d{{1,2}})?(?:,? \d{{4}})?|"
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)|(?:1[5-9]|20)\d{2})\b"
)
_NUM_RE = re.compile(
    r"(?:[$£€]\s?\d[\d,.]*(?:\s(?:million|billion|thousand))?"
    r"|\b\d[\d,.]*\s?(?:%|percent|per cent|million|billion)"
    r"|\b\d[\d,.]*\b)"
)


def heuristic_pos(tokens: Sequence[str], sentence_starts: set[int]) -> list[str]:
    """Étiquettes universelles approximatives."""
    tags: list[str] = []
    for i, token in enumerate(tokens):
        lower = token.lower()
        if is_punctuation(token):
            tags.append("PUNCT")
        elif token.replace(",", "").replace(".", "").isdigit():
            tags.append("NUM")
        elif lower in _LEXICON:
            tags.append(_LEXICON[lower])
        elif token[:1].isupper() and i not in sentence_starts:
            tags.append("PROPN")
        elif lower in _IRREGULAR_VERBS or lower.endswith(("ed", "ing")):
            tags.append("VERB")
        elif lower.endswith("ly"):
            tags.append("ADV")
        elif lower.endswith(("ous", "ful", "ive", "able", "ible", "al", "ic", "less", "ish", "est")):
            tags.append("ADJ")
        elif token[:1].isupper():
            tags.append(_LEXICON.get(lower, "PROPN"))
        else:
            tags.append("NOUN")
    return tags


def _capitalized(token: str) -> bool:
    return token[:1].isupper() and any(c.isalpha() for c in token)


def heuristic_entities(
    text: str, tokens: Sequence[str], offsets: Sequence[tuple[int, int]], sentence_starts: set[int]
) -> list[Entity]:
    """Entités approximatives (schéma minimal)."""
    entities: list[Entity] = []
    taken: list[tuple[int, int]] = []
    for pattern, kind in ((_DATE_RE, "DATE"), (_NUM_RE, "NUM")):
        for m in pattern.finditer(text):
            if any(m.start() < e and m.end() > s for s, e in taken):
                continue
            taken.append((m.start(), m.end()))
            entities.append(Entity(m.group(), m.start(), m.end(), kind, kind))
    i, n = 0, len(tokens)
    while i < n:
        token = tokens[i]
        lower = token.lower()
        starts_name = _capitalized(token) and (
            i not in sentence_starts
            or lower in TITLES
            or (i + 1 < n and _capitalized(tokens[i + 1]) and lower not in _LEXICON)
        )
        if not starts_name or lower in _LEXICON:
            i += 1
            continue
        j = i + 1
        while j < n and (_capitalized(tokens[j]) or (tokens[j] == "." and tokens[j - 1].lower() + "." in TITLES)):
            j += 1
        span_tokens = [t.lower() for t in tokens[i:j]]
        start, end = offsets[i][0], offsets[j - 1][1]
        if any(start < e and end > s for s, e in taken):
            i = j
            continue
        if span_tokens[0] in TITLES or span_tokens[0].rstrip(".") + "." in TITLES:
            kind = "PER"
        elif span_tokens[-1] in ORG_SUFFIXES:
            kind = "ORG"
        elif any(t in LOCATIONS for t in span_tokens) or (i > 0 and tokens[i - 1].lower() in LOC_PREPOSITIONS):
            kind = "LOC"
        elif j - i >= 2:
            kind = "PER"
        else:
            kind = "MISC"
        entities.append(Entity(text[start:end], start, end, kind, kind))
        taken.append((start, end))
        i = j
    return sorted(entities, key=lambda e: (e.start, e.end))


class HeuristicAnnotator(Annotator):
    """Annotateur de repli : segmentation, étiquettes universelles, entités approximatives."""

    name = "heuristic"
    scheme = "minimal"

    @property
    def version(self) -> str:
        return "1"

    def process(self, texts: Sequence[str]) -> list[AnnotatedDocument]:
        documents = []
        for text in texts:
            triples = tokenize_with_offsets(text)
            tokens = [t for t, _, _ in triples]
            offsets = [(s, e) for _, s, e in triples]
            sentences = token_sentences([s for s, _ in offsets], sentence_spans(text))
            starts = {a for a, _ in sentences}
            documents.append(
                AnnotatedDocument(
                    id="",
                    text=text,
                    tokens=tuple(tokens),
                    offsets=tuple(offsets),
                    pos=tuple(heuristic_pos(tokens, starts)),
                    lemmas=tuple(t.lower() for t in tokens),
                    sentences=sentences,
                    entities=tuple(heuristic_entities(text, tokens, offsets, starts)),
                    engine=self.name,
                )
            )
        return documents


ANNOTATORS.add("heuristic", HeuristicAnnotator)
