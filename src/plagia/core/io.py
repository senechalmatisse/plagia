"""Entrées-sorties déterministes : JSON Lines, YAML, écritures atomiques.

Les fichiers sont toujours écrits en UTF-8 avec des fins de ligne `\\n` afin que
deux reconstructions identiques produisent des octets identiques sur tous les systèmes d'exploitation.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import yaml

from plagia.core.hashing import _json_default


def ensure_dir(path: str | Path) -> Path:
    """Crée le répertoire (et ses parents) s'il n'existe pas."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    """Écrit dans un fichier temporaire puis le renomme (pas de fichier tronqué)."""
    target = Path(path)
    ensure_dir(target.parent)
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, target)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return target


def atomic_write_text(path: str | Path, text: str) -> Path:
    """Version texte (UTF-8, fins de ligne `\\n`) de :func:`atomic_write_bytes`."""
    return atomic_write_bytes(path, text.encode("utf-8"))


def dumps_json_line(record: Mapping[str, Any]) -> str:
    """Sérialise un enregistrement sur une ligne JSON (ordre des clés conservé)."""
    return json.dumps(record, ensure_ascii=False, default=_json_default)


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Écrit un fichier JSON Lines de façon atomique ; renvoie le nombre de lignes."""
    lines = [dumps_json_line(r) for r in records]
    atomic_write_text(path, "".join(f"{line}\n" for line in lines))
    return len(lines)


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    """Ajoute une ligne à un fichier JSON Lines (écriture incrémentale)."""
    target = Path(path)
    ensure_dir(target.parent)
    with open(target, "a", encoding="utf-8", newline="\n") as handle:
        handle.write(dumps_json_line(record) + "\n")
        handle.flush()


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Lit un fichier JSON Lines ligne par ligne (lignes vides ignorées)."""
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} : ligne JSON invalide ({exc})") from exc


def read_yaml(path: str | Path) -> Any:
    """Charge un fichier YAML (chargeur sûr)."""
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: str | Path, data: Any) -> Path:
    """Écrit un fichier YAML lisible (ordre des clés conservé)."""
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    return atomic_write_text(path, text)


def write_json(path: str | Path, data: Any, indent: int = 2) -> Path:
    """Écrit un fichier JSON indenté (clés triées pour la stabilité)."""
    text = json.dumps(data, ensure_ascii=False, indent=indent, sort_keys=True, default=_json_default)
    return atomic_write_text(path, text + "\n")


def read_json(path: str | Path) -> Any:
    """Charge un fichier JSON."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
