"""Alignement de textes : graines, extension, filtrage.

* :class:`SeedMatcher` : graines = shingles de mots communs (copies exactes ou
  modifiées) et paires de phrases similaires (cosinus TF-IDF et Dice), robustes aux remplacements de synonymes et aux permutations de phrases ;
* :class:`Extender` : regroupe les graines proches dans les deux documents (écart maximal en caractères), par fusions successives jusqu'à stabilité ;
* :class:`Filter` : seuils de longueur, résolution des chevauchements.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from plagia.core.lexicon import ENGLISH_FUNCTION_WORDS
from plagia.core.text import words
from plagia.core.types import Passage, PassagePair
from plagia.features.shingles import positioned_shingles, shingle_set
from plagia.similarity.sequence import levenshtein_distance
from plagia.similarity.sets import containment, dice
from plagia.tasks.plagiarism.passages import PassageView, detection_views, sentence_char_spans

Verifier = Callable[[Sequence[tuple[PassageView, PassageView]]], np.ndarray]
_FUNCTION_WORDS = frozenset(ENGLISH_FUNCTION_WORDS)


@dataclass(frozen=True, slots=True)
class Seed:
    """Correspondance élémentaire entre un empan suspect et un empan source."""

    susp_start: int
    susp_end: int
    src_start: int
    src_end: int
    kind: str
    score: float = 1.0


class SeedMatcher:
    """Détection des graines entre un document suspect et une source."""

    def __init__(
        self,
        method: str = "both",
        ngram_size: int = 4,
        sentence_threshold: float = 0.5,
        dice_threshold: float = 0.5,
        max_occurrences: int = 5,
    ) -> None:
        self.method = method
        self.ngram_size = ngram_size
        self.sentence_threshold = sentence_threshold
        self.dice_threshold = dice_threshold
        self.max_occurrences = max_occurrences

    def ngram_seeds(self, susp_text: str, src_text: str) -> list[Seed]:
        """Shingles de mots communs aux deux textes."""
        positions: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for key, start, end in positioned_shingles(src_text, self.ngram_size):
            positions[key].append((start, end))
        seeds: list[Seed] = []
        for key, start, end in positioned_shingles(susp_text, self.ngram_size):
            matches = positions.get(key)
            if matches and len(matches) <= self.max_occurrences:
                seeds.extend(Seed(start, end, a, b, "ngram") for a, b in matches)
        return seeds

    def sentence_seeds(self, susp: Any, src: Any) -> list[Seed]:
        """Paires de phrases de cosinus TF-IDF et de Dice suffisants."""
        s_spans, t_spans = sentence_char_spans(susp), sentence_char_spans(src)
        if not s_spans or not t_spans:
            return []
        s_texts = [susp.text[a:b] for a, b in s_spans]
        t_texts = [src.text[a:b] for a, b in t_spans]
        try:
            vectorizer = TfidfVectorizer(sublinear_tf=True, dtype=np.float64).fit(s_texts + t_texts)
        except ValueError:
            return []
        cosine = (vectorizer.transform(s_texts) @ vectorizer.transform(t_texts).T).toarray()
        s_words = [{w.lower() for w in words(t)} for t in s_texts]
        t_words = [{w.lower() for w in words(t)} for t in t_texts]
        seeds = []
        for i, j in zip(*np.nonzero(cosine >= self.sentence_threshold), strict=True):
            if dice(s_words[i], t_words[j]) >= self.dice_threshold:
                seeds.append(
                    Seed(s_spans[i][0], s_spans[i][1], t_spans[j][0], t_spans[j][1], "sentence", float(cosine[i, j]))
                )
        return seeds

    def seeds(self, susp: Any, src: Any) -> list[Seed]:
        """Toutes les graines selon la méthode configurée."""
        found: list[Seed] = []
        if self.method in {"ngrams", "both"}:
            found.extend(self.ngram_seeds(susp.text, src.text))
        if self.method in {"sentences", "both"}:
            found.extend(self.sentence_seeds(susp, src))
        return found


def _gap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, max(a_start, b_start) - min(a_end, b_end))


@dataclass
class Cluster:
    """Groupe de graines et empans couverts dans chaque document."""

    susp_start: int
    susp_end: int
    src_start: int
    src_end: int
    n_seeds: int
    score: float

    def merge(self, other: Cluster) -> None:
        self.susp_start = min(self.susp_start, other.susp_start)
        self.susp_end = max(self.susp_end, other.susp_end)
        self.src_start = min(self.src_start, other.src_start)
        self.src_end = max(self.src_end, other.src_end)
        self.score = max(self.score, other.score)
        self.n_seeds += other.n_seeds


class Extender:
    """Extension et fusion des graines en passages (fusion tant que les écarts sont faibles).

    Un groupe n'est conservé que s'il réunit `min_seeds` graines et une densité d'au
    moins `min_density` graines pour 100 caractères du passage suspect : les
    correspondances fortuites entre textes de même sujet sont éparses.
    """

    def __init__(self, max_gap: int = 250, min_seeds: int = 2, min_density: float = 1.0) -> None:
        self.max_gap = max_gap
        self.min_seeds = min_seeds
        self.min_density = min_density

    def extend(self, seeds: Sequence[Seed]) -> list[Cluster]:
        clusters = [Cluster(s.susp_start, s.susp_end, s.src_start, s.src_end, 1, s.score) for s in seeds]
        clusters.sort(key=lambda c: (c.susp_start, c.src_start))
        changed = True
        while changed and len(clusters) > 1:
            changed = False
            merged: list[Cluster] = []
            for cluster in clusters:
                target = next(
                    (
                        m
                        for m in reversed(merged)
                        if _gap(m.susp_start, m.susp_end, cluster.susp_start, cluster.susp_end) <= self.max_gap
                        and _gap(m.src_start, m.src_end, cluster.src_start, cluster.src_end) <= self.max_gap
                    ),
                    None,
                )
                if target is None:
                    merged.append(cluster)
                else:
                    target.merge(cluster)
                    changed = True
            clusters = sorted(merged, key=lambda c: (c.susp_start, c.src_start))
        return [
            c
            for c in clusters
            if c.n_seeds >= self.min_seeds and 100 * c.n_seeds / max(c.susp_end - c.susp_start, 1) >= self.min_density
        ]


class Filter:
    """Seuil de longueur et résolution des chevauchements (on garde la détection la mieux notée)."""

    def __init__(self, min_length: int = 150, overlap_ratio: float = 0.5) -> None:
        self.min_length = min_length
        self.overlap_ratio = overlap_ratio

    def apply(self, detections: Sequence[PassagePair]) -> list[PassagePair]:
        kept: list[PassagePair] = []
        ordered = sorted(
            (d for d in detections if d.suspicious.length >= self.min_length),
            key=lambda d: (-d.score, -d.suspicious.length, d.suspicious.start),
        )
        for det in ordered:
            clash = any(
                det.suspicious.overlap(k.suspicious)
                > self.overlap_ratio * min(det.suspicious.length, k.suspicious.length)
                for k in kept
            )
            if not clash:
                kept.append(det)
        return sorted(kept, key=lambda d: (d.suspicious.document_id, d.suspicious.start))


def content_dice(a: str, b: str) -> float:
    """Dice des ensembles de mots pleins (mots-outils exclus) de deux passages."""
    wa = {w.lower() for w in words(a)} - _FUNCTION_WORDS
    wb = {w.lower() for w in words(b)} - _FUNCTION_WORDS
    return dice(wa, wb)


def shingle_containment(a: str, b: str, n: int = 3) -> float:
    """Recouvrement des shingles du plus court passage dans le plus long (copies modifiées)."""
    sa, sb = shingle_set(a, n), shingle_set(b, n)
    return containment(sa, sb) if len(sa) <= len(sb) else containment(sb, sa)


def edit_similarity(a: str, b: str) -> float:
    """1 − distance d'édition sur les mots / longueur maximale (confirmation des copies)."""
    wa, wb = [w.lower() for w in words(a)], [w.lower() for w in words(b)]
    longest = max(len(wa), len(wb))
    return 1.0 - levenshtein_distance(wa, wb) / longest if longest else 0.0


CONFIRMATIONS = {"dice": content_dice, "containment": shingle_containment, "edit": edit_similarity}


class TextAligner:
    """Graines → extension → confirmation → vérification éventuelle par le classificateur de paires → filtrage.

    La confirmation (`confirmation`) mesure la similarité des deux passages alignés :
    Dice des mots pleins (`dice`, robuste aux synonymes et aux permutations), recouvrement
    de shingles (`containment`) ou distance d'édition sur les mots (`edit`, copies
    exactes et modifiées). Elle sert de score sans classificateur ; les passages sous `min_similarity` sont écartés.
    """

    def __init__(
        self,
        matcher: SeedMatcher | None = None,
        extender: Extender | None = None,
        filter: Filter | None = None,
        verifier: Verifier | None = None,
        threshold: float = 0.5,
        min_similarity: float = 0.5,
        confirmation: str = "dice",
    ) -> None:
        if confirmation not in CONFIRMATIONS:
            raise ValueError(f"Confirmation inconnue : {confirmation} ({', '.join(CONFIRMATIONS)}).")
        self.confirmation = confirmation
        self.matcher = matcher or SeedMatcher()
        self.extender = extender or Extender()
        self.filter = filter or Filter()
        self.verifier = verifier
        self.threshold = threshold
        self.min_similarity = min_similarity

    def candidates(self, susp: Any, src: Any) -> list[PassagePair]:
        """Passages alignés et validés, avant vérification et filtrage."""
        clusters = self.extender.extend(self.matcher.seeds(susp, src))
        detections = []
        for c in clusters:
            confirm = CONFIRMATIONS[self.confirmation]
            similarity = confirm(susp.text[c.susp_start : c.susp_end], src.text[c.src_start : c.src_end])
            if similarity < self.min_similarity:
                continue
            detections.append(
                PassagePair(
                    Passage(str(susp.id), c.susp_start, c.susp_end - c.susp_start),
                    Passage(str(src.id), c.src_start, c.src_end - c.src_start),
                    score=float(similarity),
                    meta={"seeds": c.n_seeds},
                )
            )
        return detections

    def align(self, susp: Any, src: Any) -> list[PassagePair]:
        """Passages repris de `src` dans `susp`."""
        detections = self.candidates(susp, src)
        if self.verifier is not None and detections:
            probabilities = self.verifier(detection_views(susp, lambda _: src, detections))
            detections = [
                PassagePair(d.suspicious, d.source, float(p), meta=d.meta)
                for d, p in zip(detections, probabilities, strict=True)
                if p >= self.threshold
            ]
        return self.filter.apply(detections)
