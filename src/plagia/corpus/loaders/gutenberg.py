"""Corpus littéraire construit à partir du Project Gutenberg.

Les livres sont téléchargés depuis un miroir (politique d'accès automatisé du site), avec un délai entre deux requêtes et un cache local des fichiers bruts.
L'en-tête et la licence Gutenberg sont supprimés, puis chaque œuvre est découpée en extraits de 500 à 1 000 mots coupés en fin de phrase.
Découpage par œuvre : 5 œuvres en entraînement et 1 en test par auteur (configurable).

Les identifiants de livres peuvent être omis : ils sont alors résolus par titre et auteur dans le catalogue `pg_catalog.csv`.
Un identifiant fourni est vérifié contre le titre de l'en-tête du livre téléchargé.
"""

from __future__ import annotations

import csv
import io
import re
import time
import unicodedata
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import Field

from plagia.core.config import FrozenModel
from plagia.core.errors import ExternalServiceError, PlagiaError
from plagia.core.io import ensure_dir, read_yaml
from plagia.core.logging import get_logger
from plagia.core.seeds import make_random
from plagia.core.text import count_words, paragraph_spans, sentence_spans, word_spans
from plagia.core.types import Corpus, Document
from plagia.corpus.cleaning import (
    decode_bytes,
    normalize_unicode,
    remove_author_mentions,
    strip_gutenberg_boilerplate,
    strip_markup,
)

log = get_logger(__name__)

GUTENBERG_LICENCE = "Public domain (Project Gutenberg)"
DEFAULT_MIRROR = "https://aleph.gutenberg.org"
DEFAULT_TEMPLATES = (
    "{mirror}/{path}/{id}/{id}-0.txt",
    "{mirror}/{path}/{id}/{id}.txt",
    "{mirror}/{path}/{id}/{id}-8.txt",
)
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
_HEADER_RE = re.compile(r"^(Title|Author|Language)\s*:\s*(.+)$", re.M)

Fetcher = Callable[[str], bytes]


class GutenbergWork(FrozenModel):
    """Œuvre : titre (obligatoire) et identifiant Gutenberg (facultatif)."""

    title: str
    id: int | None = None


class GutenbergAuthor(FrozenModel):
    """Auteur, ses œuvres et, éventuellement, l'œuvre réservée au test."""

    name: str
    works: list[GutenbergWork] = Field(min_length=2)
    test_work: str | None = None


class GutenbergConfig(FrozenModel):
    """Paramètres de construction du corpus littéraire."""

    authors: list[GutenbergAuthor] = Field(min_length=2)
    mirror: str = DEFAULT_MIRROR
    url_templates: list[str] = Field(default_factory=lambda: list(DEFAULT_TEMPLATES))
    catalog_url: str = CATALOG_URL
    delay_seconds: float = Field(2.0, ge=0)
    timeout: int = 60
    excerpts_per_work: int | None = 20
    min_words: int = Field(500, ge=50)
    max_words: int = Field(1000, ge=60)
    test_works_per_author: int = Field(1, ge=1)
    min_paragraph_words: int = 4
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> GutenbergConfig:
        """Charge la configuration depuis un fichier YAML."""
        return cls.model_validate(read_yaml(path))


def mirror_path(book_id: int) -> str:
    """Chemin d'un livre sur un miroir : chiffres de l'identifiant sauf le dernier."""
    digits = str(book_id)
    return "/".join(digits[:-1]) if len(digits) > 1 else "0"


def _default_fetcher(timeout: int) -> Fetcher:
    def fetch(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "plagia-corpus-builder/0.1"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data: bytes = response.read()
            return data

    return fetch


def _normalize_title(title: str) -> str:
    title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    title = re.sub(r"[^a-z0-9 ]", " ", title.lower())
    title = re.sub(r"^(the|a|an) ", "", " ".join(title.split()))
    return title


def titles_match(expected: str, found: str) -> bool:
    """Comparaison tolérante de titres (casse, accents, article initial, sous-titre)."""
    a, b = _normalize_title(expected), _normalize_title(found)
    return bool(a) and bool(b) and (a == b or b.startswith(a) or a.startswith(b))


def parse_header(text: str) -> dict[str, str]:
    """Champs `Title`, `Author` et `Language` de l'en-tête d'un livre Gutenberg."""
    head = text[:5000]
    return {m.group(1).lower(): m.group(2).strip() for m in _HEADER_RE.finditer(head)}


class GutenbergClient:
    """Téléchargement poli (miroir, délai, cache local) et résolution par catalogue."""

    def __init__(self, raw_dir: str | Path, config: GutenbergConfig, fetcher: Fetcher | None = None) -> None:
        self.raw_dir = ensure_dir(raw_dir)
        self.config = config
        self.fetcher = fetcher or _default_fetcher(config.timeout)
        self._last_request = 0.0
        self._catalog: list[dict[str, str]] | None = None

    def _get(self, url: str) -> bytes:
        wait = self.config.delay_seconds - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        try:
            return self.fetcher(url)
        finally:
            self._last_request = time.monotonic()

    def fetch(self, book_id: int) -> str:
        """Texte brut d'un livre (lu dans le cache local s'il existe)."""
        cached = self.raw_dir / f"{book_id}.txt"
        if cached.exists():
            return decode_bytes(cached.read_bytes())
        errors = []
        for template in self.config.url_templates:
            url = template.format(mirror=self.config.mirror.rstrip("/"), path=mirror_path(book_id), id=book_id)
            try:
                data = self._get(url)
            except OSError as exc:
                errors.append(f"{url} ({exc})")
                continue
            cached.write_bytes(data)
            return decode_bytes(data)
        raise ExternalServiceError(f"Livre {book_id} introuvable : " + "; ".join(errors))

    def catalog(self) -> list[dict[str, str]]:
        """Catalogue `pg_catalog.csv` (téléchargé une fois, puis lu dans le cache)."""
        if self._catalog is None:
            cached = self.raw_dir / "pg_catalog.csv"
            if not cached.exists():
                try:
                    cached.write_bytes(self._get(self.config.catalog_url))
                except OSError as exc:
                    raise ExternalServiceError(f"Catalogue Gutenberg indisponible : {exc}") from exc
            reader = csv.DictReader(io.StringIO(decode_bytes(cached.read_bytes())))
            self._catalog = list(reader)
        return self._catalog

    def resolve(self, title: str, author: str) -> int:
        """Identifiant du livre de titre `title` écrit par `author` (plus petit identifiant)."""
        last_name = _normalize_title(author.split()[-1])
        candidates = []
        for row in self.catalog():
            if row.get("Type", "Text") != "Text" or "en" not in (row.get("Language") or ""):
                continue
            if last_name not in _normalize_title(row.get("Authors", "")):
                continue
            if titles_match(title, row.get("Title", "")):
                candidates.append(int(row["Text#"]))
        if not candidates:
            raise PlagiaError(f"« {title} » de {author} absent du catalogue Gutenberg.")
        return min(candidates)


def body_paragraphs(text: str, min_words: int = 4) -> list[str]:
    """Paragraphes du corps du livre, lignes recollées ; titres et lignes courtes écartés."""
    text = re.sub(r"\[(?:Illustration|Footnote)[^\]]*\]", " ", text, flags=re.I)
    paragraphs = []
    for start, end in paragraph_spans(text):
        paragraph = " ".join(text[start:end].split())
        letters = [c for c in paragraph if c.isalpha()]
        if count_words(paragraph) < min_words or (letters and all(c.isupper() for c in letters)):
            continue
        paragraphs.append(paragraph)
    return paragraphs


def make_excerpts(
    paragraphs: Sequence[str],
    min_words: int = 500,
    max_words: int = 1000,
) -> list[str]:
    """Extraits contigus et disjoints de `min_words` à `max_words` mots, coupés en fin de phrase."""
    excerpts: list[str] = []
    current: list[str] = []
    total = 0
    for paragraph in paragraphs:
        current.append(paragraph)
        total += count_words(paragraph)
        if total < min_words:
            continue
        excerpt = "\n\n".join(current)
        if total > max_words:
            limit = word_spans(excerpt)[max_words - 1][1]
            cut = 0
            for _, end in sentence_spans(excerpt):
                if end <= limit:
                    cut = end
            excerpt = excerpt[:cut] if cut and count_words(excerpt[:cut]) >= min_words else excerpt[:limit]
        excerpts.append(excerpt)
        current, total = [], 0
    return excerpts


def _slug(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "_", ascii_name.lower()).strip("_")


def build_gutenberg_corpus(
    config: GutenbergConfig,
    raw_dir: str | Path,
    client: GutenbergClient | None = None,
) -> Corpus:
    """Construit le corpus littéraire (extraits, métadonnées d'auteur et d'œuvre, jeux)."""
    client = client or GutenbergClient(raw_dir, config)
    documents: list[Document] = []
    for author in config.authors:
        slug = _slug(author.name)
        rng = make_random(config.seed, "gutenberg", slug)
        titles = [w.title for w in author.works]
        if author.test_work:
            test_titles = {author.test_work}
        else:
            test_titles = set(rng.sample(titles, min(config.test_works_per_author, len(titles) - 1)))
        for work in author.works:
            book_id = work.id if work.id is not None else client.resolve(work.title, author.name)
            raw = client.fetch(book_id)
            header_title = parse_header(raw).get("title", "")
            if header_title and not titles_match(work.title, header_title):
                log.warning(
                    "Livre %d : titre « %s » ≠ « %s » ; résolution par catalogue.", book_id, header_title, work.title
                )
                book_id = client.resolve(work.title, author.name)
                raw = client.fetch(book_id)
            body = strip_markup(strip_gutenberg_boilerplate(normalize_unicode(raw)))
            body = remove_author_mentions(body, author.name)
            excerpts = make_excerpts(
                body_paragraphs(body, config.min_paragraph_words), config.min_words, config.max_words
            )
            n_wanted = config.excerpts_per_work
            if n_wanted is not None and len(excerpts) > n_wanted:
                keep = sorted(make_random(config.seed, "excerpts", book_id).sample(range(len(excerpts)), n_wanted))
                excerpts = [excerpts[i] for i in keep]
            elif n_wanted is not None and len(excerpts) < n_wanted:
                log.warning("« %s » : %d extrait(s) seulement (%d demandés).", work.title, len(excerpts), n_wanted)
            split = "test" if work.title in test_titles else "train"
            for i, excerpt in enumerate(excerpts):
                documents.append(
                    Document(
                        id=f"gut/{slug}/{book_id}/{i:03d}",
                        text=excerpt,
                        author=slug,
                        work=f"gut-{book_id}",
                        language="en",
                        licence=GUTENBERG_LICENCE,
                        split=split,
                        meta={"title": work.title, "book_id": book_id, "author_name": author.name},
                    )
                )
    corpus = Corpus(documents, name="literary", meta={"source": "Project Gutenberg", "seed": config.seed})
    _check_minimums(corpus)
    return corpus


def _check_minimums(corpus: Corpus) -> None:
    works_per_author: dict[str, set[str]] = {}
    for doc in corpus:
        works_per_author.setdefault(doc.author or "", set()).add(doc.group)
    if len(works_per_author) < 10:
        log.warning("Corpus littéraire : %d auteur(s) (10 attendus au minimum).", len(works_per_author))
    short = [a for a, w in works_per_author.items() if len(w) < 6]
    if short:
        log.warning("Moins de 6 œuvres pour : %s", ", ".join(sorted(short)))
