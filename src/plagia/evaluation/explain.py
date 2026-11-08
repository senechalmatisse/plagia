"""Aide à l'explication des résultats : confusions fréquentes, contribution des familles d'attributs, échantillon d'erreurs à classer à la main (au moins 30)."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

ERROR_CATEGORIES = (
    "confusion entre auteurs proches",
    "sujet partagé",
    "entités communes",
    "texte court",
    "citation",
    "texte standardisé (formules, dépêches types)",
    "erreur d'annotation",
    "autre",
)


def confusion_pairs(y_true: Sequence[str], y_pred: Sequence[str], top: int = 10) -> list[tuple[str, str, int]]:
    """Confusions (vrai → prédit) les plus fréquentes."""
    counter = Counter((str(t), str(p)) for t, p in zip(y_true, y_pred, strict=True) if str(t) != str(p))
    return [(t, p, n) for (t, p), n in counter.most_common(top)]


def feature_family(feature: str) -> str:
    """Famille (nom du bloc) d'un attribut nommé `bloc__attribut`."""
    return feature.split("__", 1)[0] if "__" in feature else "other"


def family_shares(explanation: Mapping[str, Sequence[tuple[str, float]]]) -> dict[str, float]:
    """Part de chaque bloc parmi les attributs les plus discriminants (tous auteurs)."""
    counts: Counter[str] = Counter()
    for features in explanation.values():
        counts.update(feature_family(name) for name, _ in features)
    total = sum(counts.values()) or 1
    return {family: n / total for family, n in counts.most_common()}


def error_sample(
    ids: Sequence[str],
    y_true: Sequence[str],
    y_pred: Sequence[str],
    texts: Sequence[str],
    n: int = 30,
    seed: int = 0,
    max_chars: int = 400,
) -> list[dict[str, Any]]:
    """Échantillon d'erreurs (stratifié par auteur réel) à catégoriser à la main.

    Les colonnes `category` et `comment` sont laissées vides ; les catégories
    suggérées sont listées dans :data:`ERROR_CATEGORIES`.
    """
    errors = [i for i in range(len(ids)) if str(y_true[i]) != str(y_pred[i])]
    by_author: dict[str, list[int]] = defaultdict(list)
    for i in errors:
        by_author[str(y_true[i])].append(i)
    rng = np.random.default_rng(seed)
    for author in by_author:
        rng.shuffle(by_author[author])
    chosen: list[int] = []
    while len(chosen) < min(n, len(errors)):
        for author in sorted(by_author):
            if by_author[author] and len(chosen) < n:
                chosen.append(by_author[author].pop())
    return [
        {
            "id": ids[i],
            "true": str(y_true[i]),
            "predicted": str(y_pred[i]),
            "excerpt": " ".join(texts[i].split())[:max_chars],
            "category": "",
            "comment": "",
        }
        for i in chosen
    ]
