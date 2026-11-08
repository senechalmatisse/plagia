"""Détection de la langue par profils de mots-outils, sans dépendance."""

from __future__ import annotations

from collections import Counter

from plagia.core.text import words

PROFILES: dict[str, frozenset[str]] = {
    "en": frozenset(
        "the of and to in is that it was for on are as with his they at be this from have or "
        "by one had not but what all were when we there can an which their said if do will".split()
    ),
    "fr": frozenset(
        "le la les de des du et en un une est que qui dans pour pas sur au aux il elle ne se "
        "ce plus par avec son sa ses mais ou nous vous ils sont été être cette comme".split()
    ),
    "de": frozenset(
        "der die das und in den von zu mit sich des auf für ist im dem nicht ein eine als auch "
        "es an werden aus er hat dass sie nach wird bei einer um am sind noch wie".split()
    ),
    "es": frozenset(
        "de la que el en y a los se del las un por con no una su para es al lo como más pero "
        "sus le ya o este sí porque esta entre cuando muy sin sobre también".split()
    ),
    "it": frozenset(
        "di e il la che in a per un è non una le si con del da i al sono della ha come lo ma "
        "più anche gli nel alla questo se o ci delle mi suo".split()
    ),
}


def language_scores(text: str, max_words: int = 2000) -> dict[str, float]:
    """Proportion de mots du texte appartenant au profil de chaque langue."""
    tokens = [w.lower() for w in words(text)[:max_words]]
    if not tokens:
        return dict.fromkeys(PROFILES, 0.0)
    counts = Counter(tokens)
    total = len(tokens)
    return {lang: sum(n for w, n in counts.items() if w in profile) / total for lang, profile in PROFILES.items()}


def detect_language(text: str, default: str = "und") -> tuple[str, float]:
    """Langue la plus probable et marge de confiance (écart avec la deuxième)."""
    scores = language_scores(text)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    if best_score < 0.05:
        return default, 0.0
    margin = best_score - (ranked[1][1] if len(ranked) > 1 else 0.0)
    return best, round(margin, 4)
