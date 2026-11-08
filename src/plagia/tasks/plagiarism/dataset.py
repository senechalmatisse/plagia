"""Construction des paires de passages étiquetées pour le classificateur de paires.

* Positifs : pour chaque cas, chaque fenêtre de phrases du passage suspect est associée
  à la fenêtre du passage source qui partage le plus de mots (alignement oracle tiré de la vérité terrain).
* Négatifs difficiles, dans l'ordre : même document source hors du passage repris (même
  sujet, mêmes entités, sans reprise) ; autre source candidate (même jeu) ; fenêtre suspecte sans plagiat face à une source.

Chaque négatif est rattaché au positif qui l'a engendré (`group`), ce qui permet de sous-échantillonner les négatifs à prévalence équilibrée (1:1) ou réaliste (1:10).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from plagia.core.seeds import make_random
from plagia.core.text import words
from plagia.core.types import Corpus
from plagia.similarity.sets import jaccard
from plagia.tasks.plagiarism.passages import (
    PassageView,
    sentence_char_spans,
    sliding_windows,
    windows_inside,
    windows_outside,
)


@dataclass(frozen=True, slots=True)
class PairExample:
    """Paire de passages étiquetée."""

    suspicious: PassageView
    source: PassageView
    label: int
    kind: str
    level: str | None
    size: int | None
    group: int

    @property
    def pair(self) -> tuple[PassageView, PassageView]:
        """Couple (suspect, source)."""
        return self.suspicious, self.source


def build_pair_examples(
    corpus: Corpus,
    documents: Mapping[str, Any],
    split: str,
    window_sentences: int = 3,
    negatives_per_positive: int = 3,
    seed: int = 0,
    levels: Sequence[str] | None = None,
    sizes: Sequence[int] | None = None,
    candidates: Mapping[str, Sequence[str]] | None = None,
) -> list[PairExample]:
    """Paires positives et négatives du jeu `split` (documents annotés ou bruts par identifiant)."""
    rng = make_random(seed, "pairs", split)
    spans_cache: dict[str, list[tuple[int, int]]] = {}

    def spans(doc_id: str) -> list[tuple[int, int]]:
        if doc_id not in spans_cache:
            spans_cache[doc_id] = sentence_char_spans(documents[doc_id])
        return spans_cache[doc_id]

    split_docs = [d for d in corpus if d.split == split]
    sources = sorted(d.id for d in split_docs if d.meta.get("role") == "source")
    suspicious = sorted(d.id for d in split_docs if d.meta.get("role") == "suspicious")
    case_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    source_spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for case in corpus.cases:
        case_spans[case.suspicious.document_id].append((case.suspicious.start, case.suspicious.end))
        if case.source is not None:
            source_spans[case.source.document_id].append((case.source.start, case.source.end))
    clean_windows = [
        (doc_id, w)
        for doc_id in suspicious
        for w in windows_outside(sliding_windows(spans(doc_id), window_sentences, window_sentences), case_spans[doc_id])
    ]

    examples: list[PairExample] = []
    group = 0
    split_ids = {d.id for d in split_docs}
    for case in corpus.cases:
        susp_id = case.suspicious.document_id
        if susp_id not in split_ids or case.source is None:
            continue
        level, size = case.obfuscation, int(case.meta.get("size", 0)) or None
        if levels is not None and level not in levels:
            continue
        if sizes is not None and size not in sizes:
            continue
        src_id = case.source.document_id
        susp_doc, src_doc = documents[susp_id], documents[src_id]
        susp_windows = windows_inside(spans(susp_id), case.suspicious.start, case.suspicious.end, window_sentences)
        src_windows = windows_inside(spans(src_id), case.source.start, case.source.end, window_sentences)
        if not susp_windows:
            susp_windows = [(case.suspicious.start, case.suspicious.end)]
        if not src_windows:
            src_windows = [(case.source.start, case.source.end)]
        src_word_sets = [{w.lower() for w in words(src_doc.text[a:b])} for a, b in src_windows]
        for s_start, s_end in susp_windows:
            susp_view = PassageView.of(susp_doc, s_start, s_end)
            s_words = {w.lower() for w in words(susp_view.text)}
            best = max(range(len(src_windows)), key=lambda k: jaccard(s_words, src_word_sets[k]))
            t_start, t_end = src_windows[best]
            examples.append(
                PairExample(susp_view, PassageView.of(src_doc, t_start, t_end), 1, "positive", level, size, group)
            )
            examples.extend(
                _negatives(
                    susp_view,
                    src_id,
                    documents,
                    spans,
                    source_spans,
                    sources,
                    candidates,
                    clean_windows,
                    window_sentences,
                    negatives_per_positive,
                    rng,
                    level,
                    size,
                    group,
                    true_sources={src_id},
                )
            )
            group += 1
    return examples


def _negatives(
    susp_view: PassageView,
    src_id: str,
    documents: Mapping[str, Any],
    spans: Any,
    source_spans: Mapping[str, list[tuple[int, int]]],
    sources: list[str],
    candidates: Mapping[str, Sequence[str]] | None,
    clean_windows: list[tuple[str, tuple[int, int]]],
    window_sentences: int,
    count: int,
    rng: Any,
    level: str | None,
    size: int | None,
    group: int,
    true_sources: set[str],
) -> list[PairExample]:
    out: list[PairExample] = []
    same = windows_outside(sliding_windows(spans(src_id), window_sentences, 1), source_spans.get(src_id, []))
    others = [s for s in (candidates or {}).get(susp_view.document_id, []) if s not in true_sources and s in documents]
    if not others:
        others = [s for s in sources if s not in true_sources]
    kinds = ["same_source", "other_source", "clean"]
    attempts = 0
    while len(out) < count and attempts < 10 * count:
        kind = kinds[(len(out) + attempts) % len(kinds)]
        attempts += 1
        if kind == "same_source" and same:
            a, b = rng.choice(same)
            out.append(PairExample(susp_view, PassageView.of(documents[src_id], a, b), 0, kind, level, size, group))
        elif kind == "other_source" and others:
            other = rng.choice(others)
            windows = windows_outside(sliding_windows(spans(other), window_sentences, 1), source_spans.get(other, []))
            if windows:
                a, b = rng.choice(windows)
                out.append(PairExample(susp_view, PassageView.of(documents[other], a, b), 0, kind, level, size, group))
        elif kind == "clean" and clean_windows and others:
            doc_id, (a, b) = rng.choice(clean_windows)
            other = rng.choice(others)
            windows = sliding_windows(spans(other), window_sentences, 1)
            if windows:
                c, d = rng.choice(windows)
                out.append(
                    PairExample(
                        PassageView.of(documents[doc_id], a, b),
                        PassageView.of(documents[other], c, d),
                        0,
                        kind,
                        level,
                        size,
                        group,
                    )
                )
    return out


def subsample_prevalence(examples: Sequence[PairExample], negatives_per_positive: int) -> list[PairExample]:
    """Garde tous les positifs et au plus `negatives_per_positive` négatifs par positif."""
    kept: list[PairExample] = []
    taken: dict[int, int] = defaultdict(int)
    for example in examples:
        if example.label == 1:
            kept.append(example)
        elif taken[example.group] < negatives_per_positive:
            taken[example.group] += 1
            kept.append(example)
    return kept
