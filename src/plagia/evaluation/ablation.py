"""Comparaisons appariées entre conditions (ablations).

Les conditions partagent les mêmes découpages (mêmes graines) :
leurs prédictions sur les mêmes documents sont appariées. Pour chaque condition comparée à la référence :
test de McNemar sur les prédictions, intervalle bootstrap de la différence de F1, puis correction de Holm sur l'ensemble des comparaisons.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from plagia.evaluation.bootstrap import paired_bootstrap
from plagia.evaluation.classification import f1_macro
from plagia.evaluation.stats import holm, mcnemar

Predictions = Mapping[str, tuple[str, str]]  # identifiant → (vérité, prédiction)


def align_predictions(reference: Predictions, other: Predictions) -> tuple[list[str], list[str], list[str]]:
    """Vérités et prédictions des deux conditions sur les identifiants communs (triés)."""
    common = sorted(set(reference) & set(other))
    return [reference[i][0] for i in common], [reference[i][1] for i in common], [other[i][1] for i in common]


def _indexed(
    metric: Callable[[Sequence[str], Sequence[str]], float], truth: np.ndarray, predicted: np.ndarray
) -> Callable[[np.ndarray], float]:
    """Mesure restreinte aux indices tirés par le bootstrap."""

    def compute(index: np.ndarray) -> float:
        return metric(truth[index].tolist(), predicted[index].tolist())

    return compute


def paired_comparisons(
    reference: str,
    predictions: Mapping[str, Predictions],
    metric: Callable[[Sequence[str], Sequence[str]], float] = f1_macro,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    alpha: float = 0.05,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Compare chaque condition à `reference` ; p-valeurs corrigées par Holm."""
    if reference not in predictions:
        raise KeyError(f"Condition de référence absente : {reference}")
    rows: list[dict[str, Any]] = []
    for name in sorted(predictions):
        if name == reference:
            continue
        truth, ref_pred, other_pred = align_predictions(predictions[reference], predictions[name])
        t, r, o = np.asarray(truth), np.asarray(ref_pred), np.asarray(other_pred)
        test = mcnemar(o == t, r == t)
        boot = paired_bootstrap(_indexed(metric, t, o), _indexed(metric, t, r), len(t), n_resamples, confidence, seed)
        rows.append(
            {
                "condition": name,
                "reference": reference,
                "n": len(t),
                "metric_condition": metric(truth, other_pred) if truth else float("nan"),
                "metric_reference": metric(truth, ref_pred) if truth else float("nan"),
                "diff": boot["diff"],
                "diff_low": boot["low"],
                "diff_high": boot["high"],
                "bootstrap_p": boot["p_value"],
                "mcnemar_p": test["p_value"],
                "mcnemar_b": test["b"],
                "mcnemar_c": test["c"],
            }
        )
    for key in ("mcnemar_p", "bootstrap_p"):
        corrected = holm([row[key] for row in rows], alpha)
        for row, adjusted, reject in zip(rows, corrected["adjusted"], corrected["reject"], strict=True):
            row[f"{key}_holm"] = adjusted
            row[f"{key}_significant"] = reject
    return rows
