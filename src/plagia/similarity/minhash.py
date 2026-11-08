"""Recherche de quasi-doublons à grande échelle : MinHash + LSH et winnowing.

Implémentation native NumPy, déterministe (empreintes BLAKE2b stables et graine fixée) :
elle évite toute comparaison de toutes les paires. Le seuil approximatif de détection d'un index à `b` bandes de `r` lignes vaut (1/b)^(1/r).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence

import numpy as np

from plagia.features.shingles import hashed_shingles, word_tokens

MERSENNE_PRIME = np.uint64((1 << 61) - 1)
MAX_HASH = np.uint64((1 << 32) - 1)


class MinHasher:
    """Signatures MinHash à `num_perm` permutations simulées par hachage universel."""

    def __init__(self, num_perm: int = 128, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, int(MERSENNE_PRIME), size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(MERSENNE_PRIME), size=num_perm, dtype=np.uint64)

    def signature(self, hashes: np.ndarray) -> np.ndarray:
        """Signature d'un ensemble d'empreintes 64 bits."""
        if len(hashes) == 0:
            return np.full(self.num_perm, MAX_HASH, dtype=np.uint64)
        values = np.asarray(hashes, dtype=np.uint64)[:, None]
        with np.errstate(over="ignore"):
            permuted = ((values * self.a + self.b) % MERSENNE_PRIME) & MAX_HASH
        signature: np.ndarray = permuted.min(axis=0)
        return signature

    @staticmethod
    def jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimation de l'indice de Jaccard par la proportion de minima égaux."""
        return float(np.mean(sig_a == sig_b))


def text_shingle_hashes(text: str, n: int = 5, drop_function_words: bool = False) -> np.ndarray:
    """Empreintes des shingles de mots d'un texte (ensemble, sans doublon)."""
    tokens = [w for w, _, _ in word_tokens(text, True, drop_function_words)]
    return np.unique(hashed_shingles(tokens, n))


class MinHashIndex:
    """Index LSH par bandes : renvoie les clés candidates d'une requête et leur Jaccard estimé."""

    def __init__(self, num_perm: int = 128, bands: int = 32, shingle_size: int = 5, seed: int = 1) -> None:
        if num_perm % bands:
            raise ValueError("num_perm doit être un multiple de bands.")
        self.hasher = MinHasher(num_perm, seed)
        self.bands = bands
        self.rows = num_perm // bands
        self.shingle_size = shingle_size
        self._buckets: list[dict[bytes, list[Hashable]]] = [defaultdict(list) for _ in range(bands)]
        self._signatures: dict[Hashable, np.ndarray] = {}

    @property
    def threshold(self) -> float:
        """Seuil approximatif de Jaccard au-delà duquel une paire devient candidate."""
        return float((1 / self.bands) ** (1 / self.rows))

    def _signature(self, item: str | np.ndarray) -> np.ndarray:
        hashes = text_shingle_hashes(item, self.shingle_size) if isinstance(item, str) else item
        return self.hasher.signature(hashes)

    def add(self, key: Hashable, item: str | np.ndarray) -> None:
        """Indexe un texte (ou ses empreintes de shingles)."""
        signature = self._signature(item)
        self._signatures[key] = signature
        for band in range(self.bands):
            chunk = signature[band * self.rows : (band + 1) * self.rows].tobytes()
            self._buckets[band][chunk].append(key)

    def add_many(self, items: Iterable[tuple[Hashable, str | np.ndarray]]) -> None:
        """Indexe plusieurs éléments."""
        for key, item in items:
            self.add(key, item)

    def query(self, item: str | np.ndarray, min_jaccard: float = 0.0) -> list[tuple[Hashable, float]]:
        """Clés candidates triées par Jaccard estimé décroissant."""
        signature = self._signature(item)
        candidates: set[Hashable] = set()
        for band in range(self.bands):
            chunk = signature[band * self.rows : (band + 1) * self.rows].tobytes()
            candidates.update(self._buckets[band].get(chunk, ()))
        scored = [(key, MinHasher.jaccard(signature, self._signatures[key])) for key in candidates]
        return sorted((s for s in scored if s[1] >= min_jaccard), key=lambda s: (-s[1], str(s[0])))

    def __len__(self) -> int:
        return len(self._signatures)


def winnow(hashes: Sequence[int], window: int) -> list[tuple[int, int]]:
    """Empreintes par fenêtrage (Schleimer et al., 2003) : minimum de chaque fenêtre de
    `window` empreintes (le plus à droite en cas d'égalité), sans doublon consécutif."""
    if window < 1:
        raise ValueError("window doit être positif.")
    selected: list[tuple[int, int]] = []
    last = -1
    for start in range(max(len(hashes) - window + 1, 1 if hashes else 0)):
        block = hashes[start : start + window]
        best = min(range(len(block)), key=lambda i: (block[i], -i))
        position = start + best
        if position != last:
            selected.append((int(hashes[position]), position))
            last = position
    return selected
