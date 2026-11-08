"""Gestion des graines : toute génération aléatoire dérive d'une graine maître unique.

Une sous-graine est dérivée de la graine maître et d'une suite de clés (nom de l'étape, identifiant de condition…)
par hachage : l'ordre d'exécution, le parallélisme ou l'ajout d'une étape n'influencent pas les autres tirages.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np

from plagia.core.hashing import stable_json


def derive_seed(master: int, *keys: object) -> int:
    """Dérive une graine 32 bits de la graine maître et des clés fournies."""
    payload = stable_json([int(master), *[str(k) for k in keys]]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=4).digest(), "little")


def make_rng(master: int, *keys: object) -> np.random.Generator:
    """Générateur NumPy indépendant pour l'étape désignée par `keys`."""
    return np.random.default_rng(derive_seed(master, *keys))


def make_random(master: int, *keys: object) -> random.Random:
    """Générateur `random.Random` indépendant pour l'étape désignée par `keys`."""
    return random.Random(derive_seed(master, *keys))
