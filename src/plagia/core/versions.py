"""Traçabilité : versions des bibliothèques, des modèles et du code (hash Git)."""

from __future__ import annotations

import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

import plagia

TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "spacy",
    "nltk",
    "gensim",
    "pydantic",
    "statsmodels",
    "joblib",
    "matplotlib",
    "rdflib",
    "networkx",
    "sentence-transformers",
    "umap-learn",
    "pyyaml",
)
_SPACY_MODEL_PREFIXES = ("en-core-web", "en_core_web", "fr-core-news", "fr_core_news")


def package_versions(names: Iterable[str] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Version installée de chaque paquet (`None` s'il est absent)."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def spacy_model_versions() -> dict[str, str]:
    """Versions des modèles spaCy installés comme paquets Python."""
    found: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name", "") or ""
        if name.lower().startswith(_SPACY_MODEL_PREFIXES):
            found[name.replace("-", "_")] = dist.version
    return dict(sorted(found.items()))


def git_revision(cwd: str | Path | None = None) -> dict[str, Any]:
    """Hash du commit courant et état du répertoire de travail."""

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def environment_info(extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Description complète de l'environnement d'exécution, archivée avec les résultats."""
    info: dict[str, Any] = {
        "plagia": plagia.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "spacy_models": spacy_model_versions(),
        "git": git_revision(),
    }
    if extra:
        info.update(extra)
    return info


def code_version() -> str:
    """Version courte du code : `<version>+<hash Git>[.dirty]`."""
    rev = git_revision()
    if not rev["commit"]:
        return plagia.__version__
    suffix = ".dirty" if rev["dirty"] else ""
    return f"{plagia.__version__}+{rev['commit'][:10]}{suffix}"
