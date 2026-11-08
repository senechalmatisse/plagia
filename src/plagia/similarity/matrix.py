"""Matrices de similarité creuses calculées par blocs : seuls les k plus proches voisins de chaque ligne sont conservés, sans matrice dense complète en mémoire."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize


def knn_cosine(
    X: Any,
    Y: Any = None,
    k: int = 10,
    block_size: int = 1024,
    exclude_self: bool = True,
    min_similarity: float = 0.0,
) -> sparse.csr_matrix:
    """Similarités cosinus des `k` plus proches voisins de chaque ligne de `X` dans `Y`."""
    X = normalize(sparse.csr_matrix(X, dtype=np.float64))
    same = Y is None
    Y = X if same else normalize(sparse.csr_matrix(Y, dtype=np.float64))
    n, m = X.shape[0], Y.shape[0]
    k = min(k, m - (1 if same and exclude_self else 0))
    rows, cols, values = [], [], []
    if k <= 0:
        return sparse.csr_matrix((n, m))
    YT = Y.T.tocsc()
    for start in range(0, n, block_size):
        block = (X[start : start + block_size] @ YT).toarray()
        if same and exclude_self:
            for i in range(block.shape[0]):
                block[i, start + i] = -np.inf
        top = np.argpartition(-block, kth=k - 1, axis=1)[:, :k]
        for i in range(block.shape[0]):
            for j in top[i]:
                value = block[i, j]
                if np.isfinite(value) and value > min_similarity:
                    rows.append(start + i)
                    cols.append(int(j))
                    values.append(float(value))
    return sparse.csr_matrix((values, (rows, cols)), shape=(n, m))


def symmetrize(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    """Graphe non orienté : maximum de `S` et de sa transposée."""
    return matrix.maximum(matrix.T).tocsr()


def top_k_per_row(matrix: sparse.csr_matrix, k: int) -> list[list[tuple[int, float]]]:
    """Voisins triés par similarité décroissante pour chaque ligne."""
    result = []
    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        pairs = sorted(zip(row.indices.tolist(), row.data.tolist(), strict=True), key=lambda p: (-p[1], p[0]))
        result.append(pairs[:k])
    return result
