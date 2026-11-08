"""Cache des annotations.

Une annotation est calculée une fois par document et par version d'outil. Le cache SQLite contient deux tables :

* `annotations` : sorties des annotateurs (``DocBin`` pour spaCy, JSON sinon), indexées par empreinte du texte, outil, version et paramètres ;
* `responses` : réponses des services externes (Spotlight, SPARQL).

En mode `offline`, un document absent du cache provoque une erreur au lieu d'un appel externe : une relance s'appuie entièrement sur le cache.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from plagia.annotation.base import Annotator
from plagia.annotation.document import AnnotatedDocument
from plagia.core.errors import PlagiaError
from plagia.core.hashing import sha256_text, stable_hash
from plagia.core.io import ensure_dir

_SCHEMA = """
CREATE TABLE IF NOT EXISTS annotations (
    key TEXT PRIMARY KEY, tool TEXT NOT NULL, version TEXT NOT NULL, params TEXT NOT NULL,
    format TEXT NOT NULL, payload BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS responses (
    key TEXT PRIMARY KEY, service TEXT NOT NULL, payload TEXT NOT NULL
);
"""


class CacheMissError(PlagiaError):
    """Annotation absente du cache en mode hors ligne."""


class AnnotationCache:
    """Cache SQLite des annotations et des réponses de services externes."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            ensure_dir(Path(self.path).parent)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def __getstate__(self) -> dict[str, Any]:
        return {"path": self.path}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        self.__init__(state["path"])  # type: ignore[misc]

    @staticmethod
    def key(text: str, tool: str, version: str, params: Mapping[str, Any]) -> str:
        """Clé : empreinte du texte, outil, version et paramètres."""
        return sha256_text(f"{sha256_text(text)}|{tool}|{version}|{stable_hash(dict(params))}")

    def get_annotation(self, key: str) -> tuple[str, bytes] | None:
        """``(format, contenu)`` ou ``None``."""
        with self._lock:
            row = self._connection.execute("SELECT format, payload FROM annotations WHERE key = ?", (key,)).fetchone()
        return (row[0], bytes(row[1])) if row else None

    def put_annotation(
        self, key: str, tool: str, version: str, params: Mapping[str, Any], fmt: str, payload: bytes
    ) -> None:
        """Enregistre une annotation."""
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO annotations VALUES (?, ?, ?, ?, ?, ?)",
                (key, tool, version, stable_hash(dict(params)), fmt, sqlite3.Binary(payload)),
            )
            self._connection.commit()

    def get_response(self, key: str) -> str | None:
        """Réponse d'un service externe ou `None`."""
        with self._lock:
            row = self._connection.execute("SELECT payload FROM responses WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def put_response(self, key: str, service: str, payload: str) -> None:
        """Enregistre la réponse d'un service externe."""
        with self._lock:
            self._connection.execute("INSERT OR REPLACE INTO responses VALUES (?, ?, ?)", (key, service, payload))
            self._connection.commit()

    def stats(self) -> dict[str, Any]:
        """Effectifs par outil et par service."""
        with self._lock:
            tools = dict(self._connection.execute("SELECT tool, COUNT(*) FROM annotations GROUP BY tool").fetchall())
            services = dict(
                self._connection.execute("SELECT service, COUNT(*) FROM responses GROUP BY service").fetchall()
            )
        return {"annotations": tools, "responses": services}

    def close(self) -> None:
        """Ferme la connexion (sans effet si elle l'est déjà)."""
        with self._lock:
            self._connection.close()

    def __enter__(self) -> AnnotationCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class CachedAnnotator:
    """Proxy : n'appelle l'annotateur que pour les documents absents du cache."""

    def __init__(self, annotator: Annotator, cache: AnnotationCache, offline: bool = False) -> None:
        self.annotator = annotator
        self.cache = cache
        self.offline = offline
        self.hits = 0
        self.misses = 0

    def annotate_many(self, items: Sequence[tuple[str, str]]) -> list[AnnotatedDocument]:
        """Annote des couples `(identifiant, texte)` en s'appuyant sur le cache."""
        annotator = self.annotator
        params = annotator.params()
        results: list[AnnotatedDocument | None] = [None] * len(items)
        missing: list[int] = []
        keys = []
        version: str | None = None
        for i, (doc_id, text) in enumerate(items):
            if version is None:
                version = annotator.version
            key = self.cache.key(text, annotator.name, version, params)
            keys.append(key)
            stored = self.cache.get_annotation(key)
            if stored is None:
                missing.append(i)
                continue
            self.hits += 1
            results[i] = annotator.convert(doc_id, text, annotator.deserialize(stored[1]))
        if missing:
            if self.offline:
                raise CacheMissError(
                    f"{len(missing)} document(s) absent(s) du cache pour {annotator.name} (mode hors ligne)."
                )
            self.misses += len(missing)
            raws = annotator.process([items[i][1] for i in missing])
            for i, raw in zip(missing, raws, strict=True):
                self.cache.put_annotation(
                    keys[i],
                    annotator.name,
                    version or annotator.version,
                    params,
                    annotator.payload_format,
                    annotator.serialize(raw),
                )
                results[i] = annotator.convert(items[i][0], items[i][1], raw)
        return [r for r in results if r is not None]
