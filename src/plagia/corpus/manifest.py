"""Manifeste des corpus : une ligne JSON par document et empreinte SHA-256."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from plagia.core.hashing import sha256_file
from plagia.core.io import atomic_write_text, write_jsonl
from plagia.core.text import count_words
from plagia.core.types import Corpus

MANIFEST_NAME = "manifest.jsonl"
DIGEST_NAME = "manifest.sha256"


def build_manifest(corpus: Corpus) -> list[dict[str, Any]]:
    """Entrées du manifeste : identifiant stable, auteur, œuvre ou source, langue, longueur,
    licence, jeu d'appartenance et empreinte du texte."""
    entries = []
    for doc in corpus:
        entry: dict[str, Any] = {
            "id": doc.id,
            "author": doc.author,
            "work": doc.work,
            "language": doc.language,
            "length": len(doc.text),
            "words": count_words(doc.text),
            "licence": doc.licence,
            "split": doc.split,
            "sha256": doc.sha256,
        }
        role = doc.meta.get("role")
        if role:
            entry["role"] = role
        entries.append(entry)
    return entries


def write_manifest(corpus: Corpus, directory: str | Path) -> str:
    """Écrit `manifest.jsonl` et `manifest.sha256` ; renvoie l'empreinte."""
    directory = Path(directory)
    path = directory / MANIFEST_NAME
    write_jsonl(path, build_manifest(corpus))
    digest = sha256_file(path)
    atomic_write_text(directory / DIGEST_NAME, f"{digest}  {MANIFEST_NAME}\n")
    return digest


def read_manifest_digest(directory: str | Path) -> str:
    """Empreinte enregistrée du manifeste."""
    return (Path(directory) / DIGEST_NAME).read_text(encoding="utf-8").split()[0]


def verify_manifest(directory: str | Path) -> bool:
    """Vrai si le manifeste correspond à son empreinte enregistrée."""
    directory = Path(directory)
    return sha256_file(directory / MANIFEST_NAME) == read_manifest_digest(directory)
