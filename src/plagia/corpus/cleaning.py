"""Nettoyage des textes : Unicode NFC, balises, en-têtes, signatures, mentions d'auteur révélatrices de l'étiquette, déduplication exacte."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

from plagia.core.hashing import sha256_text
from plagia.core.logging import get_logger
from plagia.core.text import count_words, normalize_whitespace
from plagia.core.types import Corpus, Document

log = get_logger(__name__)

_TAG_RE = re.compile(r"<!--.*?-->|<!\[CDATA\[.*?\]\]>|</?[A-Za-z][A-Za-z0-9:_-]*(?:\s[^<>]*)?/?>", re.S)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f​﻿]")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
_SIGNATURE_LINE_RES = [
    re.compile(r"^\s*(?:\(c\)|©|copyright\b).*$", re.I | re.M),
    re.compile(
        r"^\s*(?:[Bb]y|[Pp]ar|[Ww]ritten by|[Rr]eporting by|[Rr]eported by|[Ee]diting by)"
        r"\s+[A-Z][\w.'-]*(?:\s+[A-Z][\w.'-]*){0,3}\s*$",
        re.M,
    ),
    re.compile(r"^\s*(?:tel|phone|fax|e-?mail)\s*[:.].*$", re.I | re.M),
    re.compile(r"^.*\b(?:\+?\d[\d ()-]{7,}\d)\b.*(?:tel|phone|fax|newsroom).*$", re.I | re.M),
]
_SIGNATURE_DELIMITER_RE = re.compile(r"^-- ?$", re.M)

_GUTENBERG_START_RE = re.compile(
    r"^\s*\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG E-?(?:BOOK|TEXT).*?\*{3}\s*$", re.I | re.M
)
_GUTENBERG_END_RE = re.compile(
    r"^\s*\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG E-?(?:BOOK|TEXT).*?\*{3}\s*$", re.I | re.M
)
_GUTENBERG_LEGACY_END_RE = re.compile(r"^\s*End of (?:the )?Project Gutenberg'?s? ", re.I | re.M)
_GUTENBERG_SMALL_PRINT_RE = re.compile(r"\*END\*THE SMALL PRINT!.*?\*END\*", re.S)


def decode_bytes(data: bytes) -> str:
    """Décode des octets en texte : UTF-8 (avec ou sans BOM), sinon CP1252, sinon Latin-1."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Normalisation Unicode et suppression des caractères de contrôle invisibles."""
    text = unicodedata.normalize(form, text)  # type: ignore[arg-type]
    return _CONTROL_RE.sub("", text.replace("\r\n", "\n").replace("\r", "\n"))


def strip_markup(text: str) -> str:
    """Supprime les balises HTML/XML et décode les entités (`&amp;` → `&`)."""
    return html.unescape(_TAG_RE.sub(" ", text))


def strip_gutenberg_boilerplate(text: str) -> str:
    """Retire l'en-tête et la mention de licence du Project Gutenberg."""
    start = _GUTENBERG_START_RE.search(text)
    if start:
        text = text[start.end() :]
    end = _GUTENBERG_END_RE.search(text) or _GUTENBERG_LEGACY_END_RE.search(text)
    if end:
        text = text[: end.start()]
    return _GUTENBERG_SMALL_PRINT_RE.sub("", text)


def strip_signatures(text: str) -> str:
    """Retire signatures, adresses électroniques, URL, mentions de copyright et bylines."""
    delimiter = _SIGNATURE_DELIMITER_RE.search(text)
    if delimiter:
        text = text[: delimiter.start()]
    for pattern in _SIGNATURE_LINE_RES:
        text = pattern.sub("", text)
    text = _EMAIL_RE.sub(" ", text)
    return _URL_RE.sub(" ", text)


def split_camel_case(name: str) -> str:
    """`AaronPressman` → `Aaron Pressman` (noms d'auteurs du corpus C50)."""
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).replace("_", " ").strip()


def author_name_variants(author: str) -> list[str]:
    """Variantes d'un nom d'auteur à masquer : nom complet et nom de famille."""
    full = split_camel_case(author)
    parts = [p for p in re.split(r"[\s.]+", full) if len(p) > 2]
    variants = {full}
    if parts:
        variants.add(parts[-1])
    if "".join(full.split()) != full:
        variants.add("".join(full.split()))
    return sorted(variants, key=len, reverse=True)


def remove_author_mentions(text: str, author: str, replacement: str = "") -> str:
    """Supprime les mentions du nom de l'auteur (fuite de l'étiquette)."""
    for variant in author_name_variants(author):
        text = re.sub(rf"\b{re.escape(variant)}\b", replacement, text, flags=re.I)
    return text


@dataclass(frozen=True)
class CleaningConfig:
    """Étapes de nettoyage activables une à une."""

    unicode_form: str = "NFC"
    markup: bool = True
    gutenberg: bool = False
    signatures: bool = True
    author_mentions: bool = True
    whitespace: bool = True
    min_words: int = 1


def clean_text(text: str, config: CleaningConfig = CleaningConfig(), author: str | None = None) -> str:
    """Applique les étapes de nettoyage configurées à un texte."""
    text = normalize_unicode(text, config.unicode_form)
    if config.gutenberg:
        text = strip_gutenberg_boilerplate(text)
    if config.markup:
        text = strip_markup(text)
    if config.signatures:
        text = strip_signatures(text)
    if config.author_mentions and author:
        text = remove_author_mentions(text, author)
    if config.whitespace:
        text = normalize_whitespace(text)
    return text


def clean_document(doc: Document, config: CleaningConfig = CleaningConfig()) -> Document:
    """Nettoie un document (nouvelle instance)."""
    return doc.replace(text=clean_text(doc.text, config, doc.author))


def clean_corpus(corpus: Corpus, config: CleaningConfig = CleaningConfig()) -> Corpus:
    """Nettoie tous les documents ; écarte ceux qui deviennent trop courts."""
    kept: list[Document] = []
    dropped = 0
    for doc in corpus:
        cleaned = clean_document(doc, config)
        if count_words(cleaned.text) >= config.min_words:
            kept.append(cleaned)
        else:
            dropped += 1
    if dropped:
        log.warning("%d document(s) écarté(s) après nettoyage (moins de %d mots).", dropped, config.min_words)
    return corpus.with_documents(kept)


def dedup_key(text: str) -> str:
    """Clé de déduplication exacte : texte en minuscules, blancs réduits."""
    return sha256_text(" ".join(text.lower().split()))


def deduplicate(corpus: Corpus) -> tuple[Corpus, list[str]]:
    """Supprime les doublons exacts (première occurrence conservée).

    Les documents d'entraînement étant listés avant ceux de test dans les chargeurs,
    un texte présent dans les deux jeux n'est conservé qu'en entraînement.
    """
    order = {"train": 0, "validation": 1, "test": 2}
    ranked = sorted(range(len(corpus)), key=lambda i: (order.get(corpus[i].split or "", 3), i))
    seen: set[str] = set()
    keep: set[int] = set()
    removed: list[str] = []
    for i in ranked:
        key = dedup_key(corpus[i].text)
        if key in seen:
            removed.append(corpus[i].id)
        else:
            seen.add(key)
            keep.add(i)
    if removed:
        log.info("%d doublon(s) exact(s) supprimé(s).", len(removed))
    return corpus.with_documents(d for i, d in enumerate(corpus) if i in keep), removed
