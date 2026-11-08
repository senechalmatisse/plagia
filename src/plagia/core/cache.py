"""Cache disque générique (matrices de caractéristiques, modèles, calculs coûteux).

Chaque entrée est indexée par l'empreinte stable de ses paramètres : corpus, bloc, paramètres du bloc, version du code.
Les annotations disposent de leur propre cache SQLite (`plagia.annotation.cache`)
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import joblib

from plagia.core.hashing import stable_hash
from plagia.core.io import ensure_dir
from plagia.core.logging import get_logger

T = TypeVar("T")
log = get_logger(__name__)


class DiskCache:
    """Cache clé → objet sérialisé par joblib, rangé par espace de noms."""

    def __init__(self, root: str | Path, namespace: str = "default", enabled: bool = True) -> None:
        self.root = Path(root)
        self.namespace = namespace
        self.enabled = enabled

    @property
    def directory(self) -> Path:
        """Répertoire de l'espace de noms."""
        return self.root / self.namespace

    def key(self, parts: Any) -> str:
        """Clé stable dérivée des paramètres."""
        return stable_hash(parts, length=24)

    def path(self, key: str) -> Path:
        """Chemin du fichier associé à une clé."""
        return self.directory / key[:2] / f"{key}.joblib"

    def __contains__(self, parts: Any) -> bool:
        return self.enabled and self.path(self.key(parts)).exists()

    def get_or_compute(self, parts: Any, compute: Callable[[], T]) -> T:
        """Renvoie la valeur en cache ou la calcule puis l'enregistre."""
        if not self.enabled:
            return compute()
        target = self.path(self.key(parts))
        if target.exists():
            try:
                value: T = joblib.load(target)
                return value
            except Exception as exc:  # cache corrompu : on recalcule
                log.warning("Entrée de cache illisible (%s) : recalcul.", exc)
        value = compute()
        ensure_dir(target.parent)
        tmp = target.with_suffix(".tmp")
        joblib.dump(value, tmp, compress=3)
        tmp.replace(target)
        return value

    def clear(self) -> None:
        """Vide l'espace de noms."""
        shutil.rmtree(self.directory, ignore_errors=True)
