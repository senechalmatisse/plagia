"""plagia : analyse d'identité d'auteurs et détection de plagiat.

Le paquet est organisé en couches (voir `docs/architecture.md`) :
`cli` → `reporting`, `experiments` → `tasks` → `models`, `similarity` → `features` → `annotation` → `corpus` → `core` ; `evaluation` ne dépend que de `core`.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
