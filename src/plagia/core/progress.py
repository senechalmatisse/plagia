"""Barre de progression (tqdm si disponible, sinon itération simple)."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any


def progress[T](iterable: Iterable[T], desc: str = "", total: int | None = None, **kwargs: Any) -> Iterator[T]:
    """Enveloppe `iterable` dans une barre tqdm si la bibliothèque est installée."""
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - dépend de l'environnement
        yield from iterable
        return
    yield from tqdm(iterable, desc=desc, total=total, leave=False, **kwargs)
