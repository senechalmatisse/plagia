"""Adaptateurs Stanford NER et Stanford POS Tagger.

Les documents d'un lot sont séparés par un jeton sentinelle ; les jetons de sortie sont réalignés sur le texte d'origine pour retrouver les décalages de caractères.
"""

from __future__ import annotations

import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from plagia.annotation.base import ANNOTATORS, Annotator
from plagia.annotation.document import AnnotatedDocument, Entity
from plagia.annotation.schema import normalize_label, ptb_to_upos
from plagia.core.errors import MissingDependencyError, PlagiaError

SEPARATOR = "PLAGIADOCUMENTSEPARATOR"
PTB_UNESCAPE = {
    "-LRB-": "(",
    "-RRB-": ")",
    "-LSB-": "[",
    "-RSB-": "]",
    "-LCB-": "{",
    "-RCB-": "}",
    "``": '"',
    "''": '"',
    "`": "'",
}
_ALTERNATIVES = {
    '"': ["“", "”", "«", "»"],
    "'": ["‘", "’"],
    "...": ["…"],
    "--": ["—", "–"],
    "-": ["–", "—"],
}


def align_tokens(text: str, tokens: Sequence[str]) -> list[tuple[int, int]]:
    """Décalages des jetons (échappements PTB et guillemets normalisés pris en compte)."""
    offsets: list[tuple[int, int]] = []
    position = 0
    for token in tokens:
        base = PTB_UNESCAPE.get(token, token)
        candidates = [token, base, *_ALTERNATIVES.get(base, [])]
        best: tuple[int, int] | None = None
        for candidate in candidates:
            index = text.find(candidate, position)
            if index != -1 and (best is None or index < best[0]):
                best = (index, index + len(candidate))
        if best is None:
            offsets.append((position, position))
            continue
        offsets.append(best)
        position = best[1]
    return offsets


def parse_tsv(output: str) -> list[list[list[tuple[str, str]]]]:
    """Sortie ``jeton\\tétiquette`` → documents → phrases → couples (le séparateur coupe)."""
    documents: list[list[list[tuple[str, str]]]] = [[]]
    sentence: list[tuple[str, str]] = []

    def _close() -> None:
        nonlocal sentence
        if sentence:
            documents[-1].append(sentence)
            sentence = []

    for line in output.splitlines():
        if not line.strip():
            _close()
            continue
        token, _, label = line.partition("\t")
        if token == SEPARATOR:
            _close()
            documents.append([])
            continue
        sentence.append((token, label.strip() or "O"))
    _close()
    return documents


def group_entities(text: str, offsets: Sequence[tuple[int, int]], labels: Sequence[str], scheme: str) -> list[Entity]:
    """Regroupe les jetons consécutifs de même étiquette (IO, IOB1 ou IOB2) en entités."""
    entities: list[Entity] = []
    current: tuple[int, int, str] | None = None
    for i, raw in enumerate([*labels, "O"]):
        prefix, _, kind = raw.partition("-") if raw[:2] in {"B-", "I-"} else ("I", "", raw)
        continuing = current is not None and kind == current[2] and prefix != "B" and raw != "O"
        if continuing:
            assert current is not None
            current = (current[0], i, current[2])
            continue
        if current is not None:
            start, end = offsets[current[0]][0], offsets[current[1]][1]
            entities.append(Entity(text[start:end], start, end, current[2], normalize_label(current[2], scheme)))
            current = None
        if raw != "O" and i < len(labels):
            current = (i, i, kind)
    return entities


def _infer_scheme(model: str) -> str:
    name = Path(model).name.lower()
    if "7class" in name or "muc" in name:
        return "stanford7"
    if "4class" in name or "conll" in name:
        return "stanford4"
    return "stanford3"


class _StanfordTool(Annotator):
    main_class = ""

    def __init__(
        self, jar: str | Path, model: str | Path, java: str = "java", memory: str = "2g", timeout: int = 3600
    ) -> None:
        self.jar = str(jar)
        self.model = str(model)
        self.java = java
        self.memory = memory
        self.timeout = timeout

    @property
    def version(self) -> str:
        return f"{Path(self.jar).name}/{Path(self.model).name}"

    def params(self) -> dict[str, Any]:
        return {"model": Path(self.model).name}

    def _arguments(self) -> list[str]:
        raise NotImplementedError

    def run(self, texts: Sequence[str]) -> list[list[list[tuple[str, str]]]]:
        """Exécute l'outil Java sur un lot et renvoie la sortie analysée par document."""
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.txt"
            source.write_text(f"\n\n{SEPARATOR}\n\n".join(texts), encoding="utf-8")
            command = [
                self.java,
                f"-mx{self.memory}",
                "-cp",
                self.jar,
                self.main_class,
                *self._arguments(),
                "-textFile",
                str(source),
                "-outputFormat",
                "tsv",
                "-encoding",
                "utf-8",
            ]
            try:
                completed = subprocess.run(
                    command, capture_output=True, text=True, encoding="utf-8", timeout=self.timeout, check=True
                )
            except FileNotFoundError as exc:
                raise MissingDependencyError("java", self.name, "Installez Java 8 ou plus récent.") from exc
            except subprocess.CalledProcessError as exc:
                raise PlagiaError(f"{self.name} a échoué : {exc.stderr[-2000:]}") from exc
        documents = parse_tsv(completed.stdout)
        if len(documents) != len(texts):
            raise PlagiaError(f"{self.name} : {len(documents)} documents en sortie pour {len(texts)} en entrée.")
        return documents


class StanfordNer(_StanfordTool):
    """Stanford NER (modèles à 3, 4 ou 7 classes)."""

    main_class = "edu.stanford.nlp.ie.crf.CRFClassifier"

    def __init__(self, jar: str | Path, model: str | Path, scheme: str | None = None, **kwargs: Any) -> None:
        super().__init__(jar, model, **kwargs)
        self.scheme = scheme or _infer_scheme(str(model))
        self.name = f"stanford-ner:{Path(self.model).name}"

    def _arguments(self) -> list[str]:
        return ["-loadClassifier", self.model]

    def process(self, texts: Sequence[str]) -> list[AnnotatedDocument]:
        results = []
        for text, sentences in zip(texts, self.run(texts), strict=True):
            tokens = [t for sentence in sentences for t, _ in sentence]
            labels = [lab for sentence in sentences for _, lab in sentence]
            offsets = align_tokens(text, tokens)
            bounds, position = [], 0
            for sentence in sentences:
                bounds.append((position, position + len(sentence)))
                position += len(sentence)
            results.append(
                AnnotatedDocument(
                    id="",
                    text=text,
                    tokens=tuple(tokens),
                    offsets=tuple(offsets),
                    sentences=tuple(bounds),
                    entities=tuple(group_entities(text, offsets, labels, self.scheme)),
                    engine=self.name,
                )
            )
        return results


class StanfordPosTagger(_StanfordTool):
    """Stanford POS Tagger (étiquettes Penn Treebank, converties en universelles)."""

    main_class = "edu.stanford.nlp.tagger.maxent.MaxentTagger"

    def __init__(self, jar: str | Path, model: str | Path, **kwargs: Any) -> None:
        super().__init__(jar, model, **kwargs)
        self.name = f"stanford-pos:{Path(self.model).name}"

    def _arguments(self) -> list[str]:
        return ["-model", self.model]

    def process(self, texts: Sequence[str]) -> list[AnnotatedDocument]:
        results = []
        for text, sentences in zip(texts, self.run(texts), strict=True):
            tokens = [t for sentence in sentences for t, _ in sentence]
            tags = [tag for sentence in sentences for _, tag in sentence]
            bounds, position = [], 0
            for sentence in sentences:
                bounds.append((position, position + len(sentence)))
                position += len(sentence)
            results.append(
                AnnotatedDocument(
                    id="",
                    text=text,
                    tokens=tuple(tokens),
                    offsets=tuple(align_tokens(text, tokens)),
                    pos=tuple(ptb_to_upos(t) for t in tags),
                    tags=tuple(tags),
                    sentences=tuple(bounds),
                    engine=self.name,
                )
            )
        return results


class StanfordPipeline(Annotator):
    """Stanford NER, complété par les étiquettes du POS Tagger lorsque les jetons concordent."""

    def __init__(self, ner: StanfordNer, pos: StanfordPosTagger | None = None) -> None:
        self.ner = ner
        self.pos = pos
        self.scheme = ner.scheme
        self.name = ner.name + (f"+{pos.name}" if pos else "")

    @property
    def version(self) -> str:
        return self.ner.version + (f"+{self.pos.version}" if self.pos else "")

    def params(self) -> dict[str, Any]:
        return {"ner": self.ner.params(), "pos": self.pos.params() if self.pos else None}

    def process(self, texts: Sequence[str]) -> list[AnnotatedDocument]:
        ner_docs = self.ner.process(texts)
        if self.pos is None:
            return ner_docs
        merged = []
        for ner_doc, pos_doc in zip(ner_docs, self.pos.process(texts), strict=True):
            if ner_doc.offsets == pos_doc.offsets:
                merged.append(ner_doc.replace(pos=pos_doc.pos, tags=pos_doc.tags))
            else:
                merged.append(ner_doc)
        return merged


ANNOTATORS.add("stanford", StanfordPipeline)
