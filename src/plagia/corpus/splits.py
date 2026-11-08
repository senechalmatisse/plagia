"""Découpages entraînement / validation / test sans fuite (F1.5).

Les découpages sont stratifiés par auteur et groupés par œuvre (corpus littéraire) ou
par document source (plagiat) : un document, une œuvre ou une source n'apparaît que
dans un seul jeu. :func:`check_no_leakage` vérifie que les intersections sont vides.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from plagia.core.errors import LeakageError
from plagia.core.seeds import make_random
from plagia.core.text import word_spans
from plagia.core.types import Corpus, Document
from plagia.corpus.cleaning import dedup_key


def allocate(n_items: int, fractions: Mapping[str, float]) -> dict[str, int]:
    """Répartit `n_items` entre les jeux selon `fractions` (plus forts restes).

    Chaque jeu de fraction non nulle reçoit au moins un élément lorsque c'est possible.
    """
    names = [name for name, f in fractions.items() if f > 0]
    total = sum(fractions[name] for name in names)
    if not names or total <= 0:
        raise ValueError("Les fractions doivent contenir au moins une valeur positive.")
    raw = {name: n_items * fractions[name] / total for name in names}
    counts = {name: math.floor(value) for name, value in raw.items()}
    remainder = n_items - sum(counts.values())
    for name in sorted(names, key=lambda k: raw[k] - counts[k], reverse=True)[:remainder]:
        counts[name] += 1
    if n_items >= len(names):
        for name in names:
            while counts[name] == 0:
                donor = max(counts, key=lambda k: counts[k])
                counts[donor] -= 1
                counts[name] += 1
    return counts


def assign_group_splits(
    corpus: Corpus,
    fractions: Mapping[str, float],
    seed: int,
    stratify: bool = True,
) -> Corpus:
    """Affecte chaque groupe (œuvre ou source) à un jeu, stratifié par auteur."""
    strata: dict[str, dict[str, None]] = defaultdict(dict)
    for doc in corpus:
        key = (doc.author or "") if stratify else ""
        strata[key][doc.group] = None
    assignment: dict[str, str] = {}
    for key in sorted(strata):
        groups = sorted(strata[key])
        make_random(seed, "splits", key).shuffle(groups)
        position = 0
        for name, count in allocate(len(groups), fractions).items():
            for group in groups[position : position + count]:
                assignment[group] = name
            position += count
    return corpus.map(lambda d: d.replace(split=assignment[d.group]))


def assign_document_splits(corpus: Corpus, fractions: Mapping[str, float], seed: int) -> Corpus:
    """Affecte chaque document à un jeu (chaque document forme son propre groupe)."""
    keyed = Corpus([d.replace(work=d.id) for d in corpus], name=corpus.name)
    assigned = assign_group_splits(keyed, fractions, seed)
    splits = {d.id: d.split for d in assigned}
    return corpus.map(lambda d: d.replace(split=splits[d.id]))


def assign_pair_component_splits(corpus: Corpus, fractions: Mapping[str, float], seed: int) -> Corpus:
    """Découpage d'un corpus de plagiat par composantes connexes du graphe (suspect, source).

    Un suspect et toutes ses sources (paires et cas) tombent dans le même jeu : aucune
    source ne peut être partagée entre deux jeux.
    """
    parent: dict[str, str] = {d.id: d.id for d in corpus}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    links = list(corpus.pairs) + [
        (c.suspicious.document_id, c.source.document_id) for c in corpus.cases if c.source is not None
    ]
    for a, b in links:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    regrouped = corpus.map(lambda d: d.replace(work=find(d.id)))
    assigned = assign_group_splits(regrouped, fractions, seed, stratify=False)
    splits = {d.id: d.split for d in assigned}
    return corpus.map(lambda d: d.replace(split=splits[d.id]))


def carve_validation(corpus: Corpus, validation_size: float, seed: int, grouped: bool = False) -> Corpus:
    """Extrait un jeu de validation du jeu d'entraînement existant (stratifié, groupé ou non)."""
    train = corpus.split("train")
    fractions = {"train": 1.0 - validation_size, "validation": validation_size}
    carved = assign_group_splits(train, fractions, seed) if grouped else assign_document_splits(train, fractions, seed)
    new_splits = {d.id: d.split for d in carved}
    return corpus.map(lambda d: d.replace(split=new_splits.get(d.id, d.split)))


def leakage_report(corpus: Corpus) -> dict[str, list[str]]:
    """Intersections entre jeux : identifiants, groupes, textes identiques, sources de cas."""
    by_split: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for doc in corpus:
        split = doc.split or "none"
        by_split["ids"][doc.id].add(split)
        by_split["groups"][doc.group].add(split)
        by_split["texts"][dedup_key(doc.text)].add(split)
    split_of = {d.id: d.split for d in corpus}
    for case in corpus.cases:
        if case.source is None:
            continue
        susp_split = split_of.get(case.suspicious.document_id)
        src_split = split_of.get(case.source.document_id)
        if susp_split is not None and src_split is not None:
            by_split["case_sources"][case.source.document_id].update({susp_split, src_split})
    return {
        kind: sorted(key for key, splits in mapping.items() if len(splits) > 1) for kind, mapping in by_split.items()
    }


def check_no_leakage(corpus: Corpus) -> None:
    """Lève `LeakageError` si un document, un groupe, un texte ou une source est partagé."""
    report = leakage_report(corpus)
    problems = {kind: keys for kind, keys in report.items() if keys}
    if problems:
        detail = "; ".join(f"{kind}: {len(keys)} (ex. {keys[:3]})" for kind, keys in problems.items())
        raise LeakageError(f"Fuite entre jeux dans « {corpus.name} » : {detail}")


def subsample_authors(corpus: Corpus, n_authors: int | None, seed: int) -> Corpus:
    """Tire `n_authors` auteurs (par graine) ; `None` conserve tous les auteurs."""
    authors = corpus.authors
    if n_authors is None or n_authors >= len(authors):
        return corpus
    chosen = set(make_random(seed, "authors", n_authors).sample(authors, n_authors))
    return corpus.filter(lambda d: d.author in chosen)


def subsample_per_author(corpus: Corpus, n_per_author: int | None, seed: int, split: str = "train") -> Corpus:
    """Garde `n_per_author` documents par auteur dans `split` (autres jeux inchangés)."""
    if n_per_author is None:
        return corpus
    by_author: dict[str, list[str]] = defaultdict(list)
    for doc in corpus:
        if doc.split == split:
            by_author[doc.author or ""].append(doc.id)
    kept: set[str] = set()
    for author in sorted(by_author):
        ids = sorted(by_author[author])
        rng = make_random(seed, "per_author", split, author)
        kept.update(ids if len(ids) <= n_per_author else rng.sample(ids, n_per_author))
    return corpus.filter(lambda d: d.split != split or d.id in kept)


def truncate_words(text: str, n_words: int) -> str:
    """Tronque un texte après `n_words` mots (décalages du début conservés)."""
    spans = word_spans(text)
    if len(spans) <= n_words:
        return text
    return text[: spans[n_words - 1][1]]


def truncate_split(corpus: Corpus, n_words: int | None, split: str = "test") -> Corpus:
    """Tronque les textes du jeu `split` (longueur du texte à classer)."""
    if n_words is None:
        return corpus

    def _cut(doc: Document) -> Document:
        return doc.replace(text=truncate_words(doc.text, n_words)) if doc.split == split else doc

    return corpus.map(_cut)


def cv_splitter(n_splits: int, grouped: bool, seed: int) -> StratifiedKFold | StratifiedGroupKFold:
    """Validation croisée stratifiée (et groupée par œuvre si `grouped`)."""
    if grouped:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def effective_folds(labels: Sequence[str], groups: Sequence[str] | None, requested: int) -> int:
    """Nombre de plis réalisable (au plus le plus petit effectif par classe ou par groupe)."""
    per_class: dict[str, set[Any]] = defaultdict(set)
    for i, label in enumerate(labels):
        per_class[label].add(groups[i] if groups is not None else i)
    smallest = min((len(v) for v in per_class.values()), default=0)
    return max(2, min(requested, smallest)) if smallest >= 2 else 0


def split_ids(corpus: Corpus) -> dict[str, set[str]]:
    """Identifiants de documents par jeu."""
    result: dict[str, set[str]] = defaultdict(set)
    for doc in corpus:
        result[doc.split or "none"].add(doc.id)
    return dict(result)


def intersections(sets: Mapping[str, Iterable[str]]) -> dict[tuple[str, str], set[str]]:
    """Intersections deux à deux (utilisé par les tests de non-fuite)."""
    names = sorted(sets)
    materialized = {name: set(sets[name]) for name in names}
    return {(a, b): materialized[a] & materialized[b] for i, a in enumerate(names) for b in names[i + 1 :]}
