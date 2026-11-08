"""Intervalles de confiance par bootstrap : au moins 1 000 rééchantillonnages des exemples de test ;
intervalle bootstrap de la différence entre deux conditions appariées."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

IndexMetric = Callable[[np.ndarray], float]


def bootstrap_ci(
    metric: IndexMetric,
    n_items: int,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Valeur ponctuelle et intervalle percentile de `metric` (fonction des indices tirés)."""
    point = metric(np.arange(n_items))
    if n_items == 0 or n_resamples == 0:
        return {"value": point, "low": point, "high": point, "std": 0.0}
    rng = np.random.default_rng(seed)
    values = np.array([metric(rng.integers(0, n_items, size=n_items)) for _ in range(n_resamples)])
    alpha = (1 - confidence) / 2
    return {
        "value": float(point),
        "low": float(np.quantile(values, alpha)),
        "high": float(np.quantile(values, 1 - alpha)),
        "std": float(values.std(ddof=1)),
    }


def paired_bootstrap(
    metric_a: IndexMetric,
    metric_b: IndexMetric,
    n_items: int,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Différence `a − b` sur les mêmes rééchantillonnages : intervalle et p-valeur bilatérale."""
    diff = metric_a(np.arange(n_items)) - metric_b(np.arange(n_items))
    if n_items == 0 or n_resamples == 0:
        return {"diff": float(diff), "low": float(diff), "high": float(diff), "p_value": 1.0}
    rng = np.random.default_rng(seed)
    samples = np.empty(n_resamples)
    for i in range(n_resamples):
        index = rng.integers(0, n_items, size=n_items)
        samples[i] = metric_a(index) - metric_b(index)
    alpha = (1 - confidence) / 2
    p_value = min(1.0, 2 * min(np.mean(samples <= 0), np.mean(samples >= 0)))
    return {
        "diff": float(diff),
        "low": float(np.quantile(samples, alpha)),
        "high": float(np.quantile(samples, 1 - alpha)),
        "p_value": float(p_value),
    }
