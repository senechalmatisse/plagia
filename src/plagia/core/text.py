"""Segmentation légère en mots et en phrases, partagée par toutes les couches.

Ces fonctions n'ont aucune dépendance externe : elles servent de socle déterministe (décalages de caractères exacts) au générateur de plagiat,
à l'alignement de passages et aux caractéristiques stylométriques. Les segmenteurs spaCy ou NLTK restent utilisables via `plagia.annotation`.
"""

from __future__ import annotations

import re
import unicodedata

WORD_PATTERN = r"\d+(?:[.,]\d+)+|\w+(?:[-'’]\w+)*"
_TOKEN_RE = re.compile(rf"{WORD_PATTERN}|[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(WORD_PATTERN, re.UNICODE)
_PARAGRAPH_BREAK_RE = re.compile(r"\n[ \t\r\f\v]*\n\s*")
_SENTENCE_END_RE = re.compile(r"[.!?…]+[\"'’”»)\]]*(?=\s|$)")

ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "st",
        "jr",
        "sr",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "inc",
        "ltd",
        "co",
        "corp",
        "no",
        "fig",
        "mt",
        "u.s",
        "u.k",
        "u.n",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "gen",
        "col",
        "capt",
        "lt",
        "sgt",
        "rev",
        "hon",
        "messrs",
        "mme",
        "mlle",
        "cf",
        "approx",
        "dept",
        "est",
        "govt",
        "p",
        "pp",
        "vol",
        "ch",
    }
)


def tokenize(text: str) -> list[str]:
    """Renvoie les mots et signes de ponctuation (un signe par jeton)."""
    return _TOKEN_RE.findall(text)


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """Renvoie `(jeton, début, fin)` pour chaque mot ou signe de ponctuation."""
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def words(text: str) -> list[str]:
    """Renvoie uniquement les mots (sans ponctuation)."""
    return _WORD_RE.findall(text)


def word_spans(text: str) -> list[tuple[int, int]]:
    """Renvoie les décalages `(début, fin)` de chaque mot."""
    return [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]


def count_words(text: str) -> int:
    """Nombre de mots d'un texte."""
    return sum(1 for _ in _WORD_RE.finditer(text))


def is_punctuation(token: str) -> bool:
    """Vrai si le jeton n'est composé que de ponctuation ou de symboles."""
    return bool(token) and all(unicodedata.category(c)[0] in {"P", "S"} for c in token)


def is_word(token: str) -> bool:
    """Vrai si le jeton est un mot (contient au moins un caractère alphanumérique)."""
    return any(c.isalnum() for c in token)


def _strip_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Découpe en paragraphes (séparés par une ligne vide)."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _PARAGRAPH_BREAK_RE.finditer(text):
        span = _strip_span(text, start, m.start())
        if span[1] > span[0]:
            spans.append(span)
        start = m.end()
    span = _strip_span(text, start, len(text))
    if span[1] > span[0]:
        spans.append(span)
    return spans


def _is_abbreviation(text: str, dot_index: int) -> bool:
    k = dot_index
    while k > 0 and not text[k - 1].isspace():
        k -= 1
    previous = text[k:dot_index].lower().lstrip("\"'“‘(«[")
    if not previous:
        return False
    if previous in ABBREVIATIONS:
        return True
    # Initiale isolée (« J. Smith ») ou sigle avec points (« U.S. »).
    return (len(previous) == 1 and previous.isalpha()) or bool(re.fullmatch(r"(?:[a-z]\.)+[a-z]", previous))


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Découpe en phrases et renvoie les décalages `(début, fin)`.

    Règles : fin de phrase sur `. ! ? …` suivis d'un blanc puis d'une majuscule,
    d'un chiffre ou d'un guillemet ; les abréviations courantes et les initiales ne
    coupent pas ; une ligne vide termine toujours la phrase.
    """
    spans: list[tuple[int, int]] = []
    for p_start, p_end in paragraph_spans(text):
        start = p_start
        for m in _SENTENCE_END_RE.finditer(text, p_start, p_end):
            end = m.end()
            if m.group() == "." and _is_abbreviation(text, m.start()):
                continue
            nxt = end
            while nxt < p_end and text[nxt].isspace():
                nxt += 1
            if nxt < p_end:
                c = text[nxt]
                if not (c.isupper() or c.isdigit() or c in "\"'“‘«([—-"):
                    continue
            span = _strip_span(text, start, end)
            if span[1] > span[0]:
                spans.append(span)
            start = end
        span = _strip_span(text, start, p_end)
        if span[1] > span[0]:
            spans.append(span)
    return spans


def sentences(text: str) -> list[str]:
    """Renvoie le texte de chaque phrase."""
    return [text[s:e] for s, e in sentence_spans(text)]


def normalize_whitespace(text: str) -> str:
    """Réduit les blancs : espaces multiples → un espace, 3 sauts de ligne ou plus → 2."""
    text = re.sub(r"[ \t\f\v]+", " ", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detokenize(tokens: list[str]) -> str:
    """Recolle des jetons en texte lisible (espaces autour de la ponctuation)."""
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,;:!?%)\]}»”’])", r"\1", text)
    text = re.sub(r"([(\[{«“])\s+", r"\1", text)
    text = re.sub(r"\s+(['’])\s*(s|t|re|ve|ll|d|m)\b", r"\1\2", text)
    return text
