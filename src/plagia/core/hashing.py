"""Empreintes stables (indépendantes du processus et de la plateforme)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path, PurePath
from typing import Any


def sha256_bytes(data: bytes) -> str:
    """Empreinte SHA-256 hexadécimale d'une suite d'octets."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Empreinte SHA-256 d'un texte encodé en UTF-8."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Empreinte SHA-256 d'un fichier, lu par blocs."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, PurePath):
        return obj.as_posix()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=repr)
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if hasattr(obj, "item"):  # scalaires NumPy
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Mapping):
        return dict(obj)
    return repr(obj)


def stable_json(obj: Any) -> str:
    """Sérialisation JSON canonique (clés triées, séparateurs compacts)."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=_json_default)


def stable_hash(obj: Any, length: int = 16) -> str:
    """Empreinte courte et stable d'un objet sérialisable en JSON."""
    return sha256_text(stable_json(obj))[:length]


def hash64(value: str) -> int:
    """Hachage 64 bits stable d'une chaîne (BLAKE2b)."""
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "little")
