"""Générateur du corpus de plagiat synthétique.

À partir d'un corpus de base (C50, extraits Gutenberg, corpus jouet…) et d'une graine, le générateur :

1. sépare les documents en sources et en hôtes suspects disjoints ;
2. affecte sources et hôtes aux jeux (entraînement, validation, test), un hôte ne recevant que des passages de sources du même jeu (aucune fuite) ;
3. laisse une partie des hôtes sans plagiat (la moitié par défaut) et attribue à chaque hôte plagié une cellule (niveau d'obfuscation, taille de passage) ;
4. insère dans chaque hôte plagié des passages de `taille` mots copiés depuis une source (début de phrase), obfusqués au niveau de la cellule, entre deux phrases ;
5. enregistre les décalages de caractères du passage plagié et de sa source.

Une cellule par hôte permet d'évaluer chaque niveau et chaque taille séparément.

Dimensionnement : une portion de source n'est reprise qu'une fois. Avec les valeurs minimales (1 000 sources, 1 000 suspects dont 500 plagiés),
chaque hôte plagié reçoit 6 ou 7 passages et les sources s'épuisent pour les passages de 300 mots ;
les valeurs par défaut (1 500 sources, 3 000 suspects) donnent 2 passages par hôte et 250 passages par cellule sur un corpus de la taille de C50.
Une cellule sous l'objectif est signalée (journal et `meta["shortfall"]`).
"""

from __future__ import annotations

import bisect
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import Field

from plagia.core.config import FrozenModel
from plagia.core.errors import PlagiaError
from plagia.core.logging import get_logger
from plagia.core.seeds import make_random
from plagia.core.text import count_words, sentence_spans, word_spans
from plagia.core.types import Corpus, Document, Passage, PassagePair
from plagia.corpus.obfuscation import Paraphraser, SynonymProvider, Translator, build_obfuscator
from plagia.corpus.splits import allocate

log = get_logger(__name__)


class SyntheticConfig(FrozenModel):
    """Paramètres du corpus synthétique (valeurs par défaut, qui fixe des minimums : au moins 1 000 sources et 1 000 suspects)."""

    n_sources: int = Field(1500, ge=1)
    n_suspicious: int = Field(3000, ge=2)
    no_plagiarism_fraction: float = Field(0.5, ge=0, lt=1)
    levels: list[str] = Field(default_factory=lambda: ["O0", "O1", "O2", "O3"])
    sizes: list[int] = Field(default_factory=lambda: [50, 100, 300])
    passages_per_cell: int = Field(250, ge=1)
    split_fractions: dict[str, float] = Field(default_factory=lambda: {"train": 0.6, "validation": 0.2, "test": 0.2})
    edit_rate: float = Field(0.15, ge=0, le=1)
    synonym_rate: float = Field(0.5, ge=0, le=1)
    min_host_words: int = Field(100, ge=1)
    seed: int = 42


@dataclass
class _Source:
    doc: Document
    spans: list[tuple[int, int]]
    sentence_starts: list[int]
    used: list[tuple[int, int]]

    @classmethod
    def build(cls, doc: Document) -> _Source:
        spans = word_spans(doc.text)
        word_starts = [sp[0] for sp in spans]
        first_words = sorted({bisect.bisect_left(word_starts, s) for s, _ in sentence_spans(doc.text)})
        return cls(doc, spans, [i for i in first_words if i < len(spans)], [])

    def free_starts(self, size: int, sentence_aligned: bool = True) -> list[int]:
        """Débuts (de phrase si `sentence_aligned`) d'un passage de `size` mots libre."""
        candidates = self.sentence_starts if sentence_aligned else range(len(self.spans))
        result = []
        for start in candidates:
            end = start + size
            if end > len(self.spans):
                break
            if all(end <= a or start >= b for a, b in self.used):
                result.append(start)
        return result


def _assign(
    items: Sequence[Document], fractions: Mapping[str, float], seed: int, key: str
) -> dict[str, list[Document]]:
    ordered = list(items)
    make_random(seed, "synthetic", "splits", key).shuffle(ordered)
    result: dict[str, list[Document]] = {}
    position = 0
    for split, count in allocate(len(ordered), fractions).items():
        result[split] = ordered[position : position + count]
        position += count
    return result


def generate_plagiarism_corpus(
    base: Sequence[Document],
    config: SyntheticConfig = SyntheticConfig(),
    synonyms: SynonymProvider | None = None,
    translator: Translator | None = None,
    paraphraser: Paraphraser | None = None,
    name: str = "synthetic",
) -> Corpus:
    """Génère le corpus de plagiat synthétique, entièrement déterminé par `config.seed`."""
    seed = config.seed
    max_size = max(config.sizes)
    documents = sorted(base, key=lambda d: d.id)
    counts = {d.id: count_words(d.text) for d in documents}
    eligible = [d for d in documents if counts[d.id] >= max_size + 20]
    make_random(seed, "synthetic", "sources").shuffle(eligible)
    sources = eligible[: config.n_sources]
    taken = {d.id for d in sources}
    hosts = [d for d in documents if d.id not in taken and counts[d.id] >= config.min_host_words]
    make_random(seed, "synthetic", "hosts").shuffle(hosts)
    hosts = hosts[: config.n_suspicious]
    if len(sources) < config.n_sources or len(hosts) < config.n_suspicious:
        raise PlagiaError(
            f"Corpus de base insuffisant : {len(sources)}/{config.n_sources} sources d'au moins "
            f"{max_size + 20} mots et {len(hosts)}/{config.n_suspicious} hôtes disponibles."
        )

    source_splits = _assign(sources, config.split_fractions, seed, "sources")
    host_splits = _assign(hosts, config.split_fractions, seed, "hosts")
    cells = [(level, size) for level in config.levels for size in config.sizes]
    host_cell: dict[str, tuple[str, int]] = {}
    for split in sorted(host_splits):
        split_hosts = host_splits[split]
        n_plagiarized = round(len(split_hosts) * (1 - config.no_plagiarism_fraction))
        offset = make_random(seed, "synthetic", "cells", split).randrange(len(cells))
        for i, host in enumerate(split_hosts[:n_plagiarized]):
            host_cell[host.id] = cells[(i + offset) % len(cells)]
    hosts_per_cell = Counter(host_cell.values())
    missing = [c for c in cells if hosts_per_cell[c] == 0]
    if missing:
        raise PlagiaError(f"Trop peu d'hôtes plagiés pour couvrir les cellules {missing}.")
    per_host = {c: math.ceil(config.passages_per_cell / hosts_per_cell[c]) for c in cells}

    pools = {split: [_Source.build(d) for d in docs] for split, docs in source_splits.items()}
    out_docs: list[Document] = []
    cases: list[PassagePair] = []
    host_index = 0
    for split in sorted(host_splits):
        for host in host_splits[split]:
            susp_id = f"susp-{host_index:05d}"
            host_index += 1
            cell = host_cell.get(host.id)
            if cell is None:
                out_docs.append(_suspicious_doc(host, susp_id, host.text, split, None, None, 0))
                continue
            level, size = cell
            rng = make_random(seed, "synthetic", "host", host.id)
            obfuscator = build_obfuscator(
                level,
                synonyms=synonyms,
                edit_rate=config.edit_rate,
                synonym_rate=config.synonym_rate,
                translator=translator,
                paraphraser=paraphraser,
            )
            insertions: list[tuple[int, str, Passage]] = []
            boundaries = [0] + [e for _, e in sentence_spans(host.text)]
            for _ in range(per_host[cell]):
                picked = _pick_source(pools[split], size, rng)
                if picked is None:
                    log.warning("Aucune source libre de %d mots dans « %s ».", size, split)
                    break
                source, start = picked
                source.used.append((start, start + size))
                char_start = source.spans[start][0]
                char_end = source.spans[start + size - 1][1]
                original = source.doc.text[char_start:char_end]
                obfuscated = obfuscator(original, rng).strip() or original
                point = rng.choice(boundaries)
                insertions.append((point, obfuscated, Passage(source.doc.id, char_start, char_end - char_start)))
            text, positions = _insert(host.text, insertions)
            for (_, _, source_passage), (start, length) in zip(insertions, positions, strict=True):
                cases.append(
                    PassagePair(
                        suspicious=Passage(susp_id, start, length),
                        source=source_passage,
                        obfuscation=level,
                        meta={"size": size, "split": split},
                    )
                )
            out_docs.append(_suspicious_doc(host, susp_id, text, split, level, size, len(insertions)))

    source_docs = []
    for split, docs in sorted(source_splits.items()):
        for doc in docs:
            source_docs.append(
                doc.replace(
                    id=f"src-{doc.id}".replace("/", "_"),
                    work=None,
                    split=split,
                    meta={"role": "source", "origin": doc.id},
                )
            )
    rename = {doc.meta["origin"]: doc.id for doc in source_docs}
    cases = [
        PassagePair(
            suspicious=c.suspicious,
            source=Passage(rename[c.source.document_id], c.source.start, c.source.length) if c.source else None,
            obfuscation=c.obfuscation,
            meta=c.meta,
        )
        for c in cases
    ]
    pairs = sorted({(c.suspicious.document_id, c.source.document_id) for c in cases if c.source})
    produced = Counter((c.obfuscation or "", int(c.meta["size"])) for c in cases)
    shortfall = {f"{level}/{size}": produced[(level, size)] for level, size in cells}
    shortfall = {cell: n for cell, n in shortfall.items() if n < config.passages_per_cell}
    if shortfall:
        log.warning(
            "Cellules sous l'objectif de %d passages : %s. Augmentez n_sources (sources épuisées) "
            "ou n_suspicious (moins de passages par hôte).",
            config.passages_per_cell,
            ", ".join(f"{cell}={n}" for cell, n in shortfall.items()),
        )
    corpus = Corpus(
        [*source_docs, *out_docs],
        name=name,
        cases=cases,
        pairs=pairs,
        meta={
            "generator": "synthetic-plagiarism",
            "config": config.model_dump(mode="json"),
            "base": getattr(base, "name", "base"),
            "shortfall": shortfall,
        },
    )
    log.info("Corpus synthétique : %d sources, %d suspects, %d passages.", len(source_docs), len(out_docs), len(cases))
    return corpus


def _pick_source(pool: list[_Source], size: int, rng: Any) -> tuple[_Source, int] | None:
    """Source et début de passage libres : début de phrase en priorité, sinon début de mot."""
    order = list(range(len(pool)))
    rng.shuffle(order)
    for sentence_aligned in (True, False):
        for index in order:
            starts = pool[index].free_starts(size, sentence_aligned)
            if starts:
                return pool[index], rng.choice(starts)
    return None


def _insert(host_text: str, insertions: list[tuple[int, str, Passage]]) -> tuple[str, list[tuple[int, int]]]:
    """Insère les passages aux frontières de phrases ; renvoie le texte et les décalages."""
    order = sorted(range(len(insertions)), key=lambda i: insertions[i][0])
    pieces: list[str] = []
    length = 0
    previous = 0
    positions: dict[int, tuple[int, int]] = {}
    for i in order:
        point, passage_text, _ = insertions[i]
        segment = host_text[previous:point]
        if segment and pieces and not segment[0].isspace():
            pieces.append(" ")
            length += 1
        pieces.append(segment)
        length += len(segment)
        if length > 0:
            pieces.append(" ")
            length += 1
        positions[i] = (length, len(passage_text))
        pieces.append(passage_text)
        length += len(passage_text)
        previous = point
    tail = host_text[previous:]
    if tail and pieces and not tail[0].isspace():
        pieces.append(" ")
    pieces.append(tail)
    return "".join(pieces), [positions[i] for i in range(len(insertions))]


def _suspicious_doc(
    host: Document, doc_id: str, text: str, split: str, level: str | None, size: int | None, n_cases: int
) -> Document:
    return Document(
        id=doc_id,
        text=text,
        author=host.author,
        work=None,
        language=host.language,
        licence=host.licence,
        split=split,
        meta={"role": "suspicious", "origin": host.id, "level": level, "size": size, "n_cases": n_cases},
    )


def cell_counts(corpus: Corpus) -> dict[tuple[str, int], int]:
    """Nombre de passages par (niveau, taille) : contrôle de l'exigence « 250 par cellule »."""
    counter: Counter[tuple[str, int]] = Counter()
    for case in corpus.cases:
        counter[(case.obfuscation or "", int(case.meta.get("size", 0)))] += 1
    return dict(sorted(counter.items()))
