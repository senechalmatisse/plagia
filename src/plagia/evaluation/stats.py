"""Tests statistiques : McNemar, correction de Holm, Wilcoxon sur les graines (à partir de 10 graines seulement), κ de Cohen, résumés moyenne ± écart-type.

Avec 5 paires, la plus petite p-valeur bilatérale possible du test de Wilcoxon vaut 2/2⁵ = 0,0625 > 0,05 : le test n'est donc proposé qu'à partir de 10 graines.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool], exact: bool | None = None) -> dict[str, Any]:
    """Test de McNemar sur les prédictions appariées de deux systèmes.

    `b` : A juste et B faux ; `c` : A faux et B juste. Test binomial exact si
    `b + c < 25` (ou `exact=True`), sinon χ² avec correction de continuité.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_arr = np.asarray(correct_b, dtype=bool)
    b = int(np.sum(a & ~b_arr))
    c = int(np.sum(~a & b_arr))
    n = b + c
    use_exact = n < 25 if exact is None else exact
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "b": b, "c": c, "exact": use_exact}
    if use_exact:
        p_value = float(stats.binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)
        statistic = float(min(b, c))
    else:
        statistic = (abs(b - c) - 1) ** 2 / n
        p_value = float(stats.chi2.sf(statistic, df=1))
    return {"statistic": float(statistic), "p_value": min(1.0, p_value), "b": b, "c": c, "exact": use_exact}


def holm(p_values: Sequence[float], alpha: float = 0.05) -> dict[str, list[Any]]:
    """Correction de Holm (pas à pas descendante) : p-valeurs ajustées et rejets."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return {"adjusted": [], "reject": []}
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return {"adjusted": adjusted.tolist(), "reject": (adjusted <= alpha).tolist()}


def wilcoxon_seeds(scores_a: Sequence[float], scores_b: Sequence[float], min_seeds: int = 10) -> dict[str, float]:
    """Test de Wilcoxon apparié sur les scores par graine (au moins `min_seeds` graines)."""
    if len(scores_a) != len(scores_b):
        raise ValueError("Les deux séries doivent avoir la même longueur.")
    if len(scores_a) < min_seeds:
        raise ValueError(
            f"Wilcoxon requiert au moins {min_seeds} graines ({len(scores_a)} fournies) : "
            "avec 5 paires, la plus petite p-valeur possible dépasse 0,05."
        )
    differences = np.asarray(scores_a) - np.asarray(scores_b)
    if np.allclose(differences, 0):
        return {"statistic": 0.0, "p_value": 1.0}
    result = stats.wilcoxon(scores_a, scores_b)
    return {"statistic": float(result.statistic), "p_value": float(result.pvalue)}


def cohen_kappa(annotator_a: Sequence[Any], annotator_b: Sequence[Any]) -> float:
    """κ de Cohen entre deux annotateurs (accord corrigé du hasard)."""
    if len(annotator_a) != len(annotator_b) or not annotator_a:
        raise ValueError("Annotations de même longueur non vide requises.")
    labels = sorted({*map(str, annotator_a), *map(str, annotator_b)})
    index = {label: i for i, label in enumerate(labels)}
    table = np.zeros((len(labels), len(labels)))
    for x, y in zip(annotator_a, annotator_b, strict=True):
        table[index[str(x)], index[str(y)]] += 1
    n = table.sum()
    observed = np.trace(table) / n
    expected = float((table.sum(axis=0) * table.sum(axis=1)).sum() / (n * n))
    return 1.0 if expected == 1 else float((observed - expected) / (1 - expected))


def summarize(values: Sequence[float], confidence: float = 0.95) -> dict[str, float]:
    """Moyenne, écart-type et intervalle de Student sur les graines."""
    array = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    n = len(array)
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0, "ci_low": float("nan"), "ci_high": float("nan")}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if n > 1 else 0.0
    half = float(stats.t.ppf((1 + confidence) / 2, n - 1) * std / np.sqrt(n)) if n > 1 else 0.0
    return {"mean": mean, "std": std, "n": n, "ci_low": mean - half, "ci_high": mean + half}
