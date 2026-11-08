"""Registres nom → fabrique, pour instancier les composants depuis la configuration.

Un nouveau moteur d'entités, une mesure de similarité, un bloc de caractéristiques
ou un classificateur s'ajoute en l'enregistrant, sans modifier le reste du code.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from plagia.core.errors import ConfigurationError


class Registry[T]:
    """Registre de fabriques identifiées par un nom unique."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._factories: dict[str, Callable[..., T]] = {}

    def add(self, name: str, factory: Callable[..., T]) -> None:
        """Enregistre `factory` sous `name` (erreur si le nom est déjà pris)."""
        if name in self._factories:
            raise ConfigurationError(f"{self.kind} « {name} » déjà enregistré.")
        self._factories[name] = factory

    def register(self, name: str | None = None) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Décorateur d'enregistrement (`@REGISTRY.register("nom")`)."""

        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            self.add(name or getattr(factory, "__name__", None) or repr(factory), factory)
            return factory

        return decorator

    def get(self, name: str) -> Callable[..., T]:
        """Renvoie la fabrique enregistrée sous `name`."""
        try:
            return self._factories[name]
        except KeyError:
            available = ", ".join(sorted(self._factories)) or "aucun"
            raise ConfigurationError(f"{self.kind} inconnu : « {name} ». Disponibles : {available}.") from None

    def create(self, name: str, **params: Any) -> T:
        """Instancie le composant `name` avec les paramètres donnés."""
        factory = self.get(name)
        try:
            return factory(**params)
        except TypeError as exc:
            raise ConfigurationError(f"Paramètres invalides pour {self.kind} « {name} » : {exc}") from exc

    def names(self) -> list[str]:
        """Noms enregistrés, triés."""
        return sorted(self._factories)

    def __contains__(self, name: object) -> bool:
        return name in self._factories

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._factories)
