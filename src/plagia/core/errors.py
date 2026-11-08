"""Exceptions communes du projet."""

from __future__ import annotations

from pathlib import Path


class PlagiaError(Exception):
    """Erreur de base du projet."""


class ConfigurationError(PlagiaError):
    """Configuration invalide ou incohérente."""


class CorpusNotFoundError(PlagiaError):
    """Corpus absent : le répertoire ne contient pas de `documents.jsonl`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        super().__init__(f"Aucun corpus dans {self.path} (fichier documents.jsonl absent).")


class LeakageError(PlagiaError):
    """Fuite détectée entre jeux d'entraînement, de validation et de test."""


class MissingDependencyError(PlagiaError, ImportError):
    """Dépendance facultative absente (Gensim, Stanford, Spotlight…)."""

    def __init__(self, package: str, feature: str, hint: str | None = None) -> None:
        message = f"La fonctionnalité « {feature} » nécessite « {package} »."
        if hint:
            message += f" {hint}"
        super().__init__(message)
        self.package = package
        self.feature = feature


class ExternalServiceError(PlagiaError):
    """Service externe indisponible (Spotlight, SPARQL, téléchargement)."""
