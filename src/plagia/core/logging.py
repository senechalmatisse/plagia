"""Journalisation homogène du projet."""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s : %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Renvoie le journal du module `name`."""
    return logging.getLogger(name)


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure la sortie console (appelé par la ligne de commande)."""
    logging.basicConfig(level=level, format=_FORMAT, datefmt="%H:%M:%S", force=True)
