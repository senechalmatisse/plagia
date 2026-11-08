"""Couche `tasks` : attribution et vérification d'auteur, détection de plagiat, regroupement."""

from plagia.tasks.attribution import AuthorshipAttributor, make_classifier
from plagia.tasks.verification import AuthorVerifier, ImpostersVerifier, make_pairs

__all__ = ["AuthorVerifier", "AuthorshipAttributor", "ImpostersVerifier", "make_classifier", "make_pairs"]
