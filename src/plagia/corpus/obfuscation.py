"""Obfuscateurs du corpus de plagiat synthétique.

Niveaux d'intensité croissante :

* `O0` copie exacte (:class:`Identity`) ;
* `O1` suppressions, insertions et remplacements aléatoires de mots (:class:`RandomEdit`) ;
* `O2` remplacement par des synonymes WordNet (:class:`Synonym`) ;
* `O3` `O2` suivi d'une permutation des phrases (:class:`Chain` + :class:`Shuffle`) ;
* `O4` reformulation par rétro-traduction ou par modèle de langage, via un traducteur ou un paraphraseur fourni par l'utilisateur (aucun modèle n'est entraîné).
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Mapping, Sequence
from functools import lru_cache
from typing import Any, Protocol

from plagia.core.errors import ConfigurationError, MissingDependencyError
from plagia.core.lexicon import ENGLISH_FUNCTION_WORDS
from plagia.core.registry import Registry
from plagia.core.text import WORD_PATTERN, sentence_spans, words

_WORD_RE = re.compile(WORD_PATTERN)
_FUNCTION_WORDS = frozenset(ENGLISH_FUNCTION_WORDS)

Translator = Callable[[str, str, str], str]
Paraphraser = Callable[[str], str]


class Obfuscator(Protocol):
    """Transformation qui masque une reprise : `obfuscator(texte, rng) → texte`."""

    def __call__(self, text: str, rng: random.Random) -> str: ...


class SynonymProvider(Protocol):
    """Fournisseur de synonymes d'un mot (en minuscules)."""

    def synonyms(self, word: str) -> list[str]: ...


OBFUSCATORS: Registry[Obfuscator] = Registry("obfuscateur")


def match_case(word: str, model: str) -> str:
    """Reproduit la casse de `model` sur `word`."""
    if model.isupper() and len(model) > 1:
        return word.upper()
    if model[:1].isupper():
        return word[:1].upper() + word[1:]
    return word


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]+([.,;:!?)])", r"\1", text)
    text = re.sub(r"([.!?])[,;:]+", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip(" \t,;:")


class DictSynonyms:
    """Synonymes tirés d'un dictionnaire (tests, démonstrations, ressources maison)."""

    def __init__(self, mapping: Mapping[str, Sequence[str]], symmetric: bool = True) -> None:
        table: dict[str, set[str]] = {}
        for word, syns in mapping.items():
            table.setdefault(word.lower(), set()).update(s.lower() for s in syns)
            if symmetric:
                group = {word.lower(), *(s.lower() for s in syns)}
                for member in group:
                    table.setdefault(member, set()).update(group - {member})
        self._table = {w: sorted(s - {w}) for w, s in table.items()}

    def synonyms(self, word: str) -> list[str]:
        """Synonymes triés (ordre déterministe)."""
        return self._table.get(word.lower(), [])


class WordNetSynonyms:
    """Synonymes WordNet (NLTK) limités aux `max_synsets` sens les plus fréquents."""

    def __init__(self, max_synsets: int = 2, single_word: bool = True, lang: str = "eng") -> None:
        self.max_synsets = max_synsets
        self.single_word = single_word
        self.lang = lang
        try:
            from nltk.corpus import wordnet
        except ImportError as exc:
            raise MissingDependencyError("nltk", "synonymes WordNet (O2)", "Lancez « uv sync ».") from exc
        try:
            wordnet.ensure_loaded()
        except LookupError as exc:
            raise MissingDependencyError(
                "wordnet (NLTK)", "synonymes WordNet (O2)", "Lancez « plagia setup »."
            ) from exc
        self._wordnet = wordnet

    @lru_cache(maxsize=100_000)  # noqa: B019
    def synonyms(self, word: str) -> list[str]:
        """Synonymes triés du mot (lemmes des premiers synsets)."""
        found: set[str] = set()
        for synset in self._wordnet.synsets(word, lang=self.lang)[: self.max_synsets]:
            for lemma in synset.lemma_names(self.lang):
                candidate = lemma.replace("_", " ")
                if self.single_word and " " in candidate:
                    continue
                if candidate.lower() != word.lower():
                    found.add(candidate.lower())
        return sorted(found)


@OBFUSCATORS.register("identity")
class Identity:
    """O0 : copie exacte."""

    def __call__(self, text: str, rng: random.Random) -> str:
        return text


@OBFUSCATORS.register("random_edit")
class RandomEdit:
    """O1 : chaque mot est, avec la probabilité `rate`, supprimé, doublé d'une insertion
    ou remplacé par un mot tiré du vocabulaire (par défaut, celui du passage)."""

    def __init__(self, rate: float = 0.15, vocabulary: Sequence[str] | None = None) -> None:
        if not 0 <= rate <= 1:
            raise ConfigurationError("rate doit être compris entre 0 et 1.")
        self.rate = rate
        self.vocabulary = list(vocabulary) if vocabulary else None

    def __call__(self, text: str, rng: random.Random) -> str:
        vocabulary = self.vocabulary or sorted(set(words(text))) or ["the"]

        def _edit(match: re.Match[str]) -> str:
            word = match.group()
            if rng.random() >= self.rate:
                return word
            operation = rng.choice(("delete", "insert", "replace"))
            if operation == "delete":
                return ""
            new = rng.choice(vocabulary)
            if operation == "insert":
                return f"{word} {new.lower() if new[:1].isupper() and not new.isupper() else new}"
            return match_case(new.lower(), word)

        result = _tidy(_WORD_RE.sub(_edit, text))
        return result or text


@OBFUSCATORS.register("synonym")
class Synonym:
    """O2 : remplace les mots pleins par un synonyme avec la probabilité `rate`."""

    def __init__(
        self,
        provider: SynonymProvider | None = None,
        rate: float = 0.5,
        min_length: int = 3,
    ) -> None:
        self.provider = provider
        self.rate = rate
        self.min_length = min_length

    def __call__(self, text: str, rng: random.Random) -> str:
        provider = self.provider or WordNetSynonyms()

        def _replace(match: re.Match[str]) -> str:
            word = match.group()
            lower = word.lower()
            if len(word) < self.min_length or not word.isalpha() or lower in _FUNCTION_WORDS:
                return word
            candidates = provider.synonyms(lower)
            if not candidates or rng.random() >= self.rate:
                return word
            return match_case(rng.choice(candidates), word)

        return _WORD_RE.sub(_replace, text)


@OBFUSCATORS.register("shuffle")
class Shuffle:
    """Permutation aléatoire des phrases (ordre toujours modifié s'il y a deux phrases ou plus)."""

    def __call__(self, text: str, rng: random.Random) -> str:
        spans = sentence_spans(text)
        if len(spans) < 2:
            return text
        parts = [text[s:e] for s, e in spans]
        order = list(range(len(parts)))
        for _ in range(20):
            rng.shuffle(order)
            if order != sorted(order):
                break
        return " ".join(parts[i] for i in order)


@OBFUSCATORS.register("chain")
class Chain:
    """Composition d'obfuscateurs appliqués dans l'ordre."""

    def __init__(self, steps: Sequence[Obfuscator]) -> None:
        self.steps = list(steps)

    def __call__(self, text: str, rng: random.Random) -> str:
        for step in self.steps:
            text = step(text, rng)
        return text


@OBFUSCATORS.register("back_translation")
class BackTranslation:
    """O4 [S] : rétro-traduction `source → pivot → source` par un traducteur fourni."""

    def __init__(self, translator: Translator, pivot: str = "fr", source: str = "en") -> None:
        self.translator = translator
        self.pivot = pivot
        self.source = source

    def __call__(self, text: str, rng: random.Random) -> str:
        return self.translator(self.translator(text, self.source, self.pivot), self.pivot, self.source)


@OBFUSCATORS.register("paraphrase")
class Paraphrase:
    """O4 [S] : reformulation par un paraphraseur fourni (modèle de langage déjà entraîné)."""

    def __init__(self, paraphraser: Paraphraser) -> None:
        self.paraphraser = paraphraser

    def __call__(self, text: str, rng: random.Random) -> str:
        return self.paraphraser(text)


LEVELS = ("O0", "O1", "O2", "O3", "O4")


def build_obfuscator(
    level: str,
    synonyms: SynonymProvider | None = None,
    edit_rate: float = 0.15,
    synonym_rate: float = 0.5,
    translator: Translator | None = None,
    paraphraser: Paraphraser | None = None,
    **_: Any,
) -> Obfuscator:
    """Obfuscateur correspondant au niveau `O0` à `O4`."""
    if level == "O0":
        return Identity()
    if level == "O1":
        return RandomEdit(rate=edit_rate)
    if level == "O2":
        return Synonym(synonyms, rate=synonym_rate)
    if level == "O3":
        return Chain([Synonym(synonyms, rate=synonym_rate), Shuffle()])
    if level == "O4":
        if translator is not None:
            return BackTranslation(translator)
        if paraphraser is not None:
            return Paraphrase(paraphraser)
        raise ConfigurationError("Le niveau O4 nécessite un traducteur (rétro-traduction) ou un paraphraseur.")
    raise ConfigurationError(f"Niveau d'obfuscation inconnu : {level} (attendu : {', '.join(LEVELS)}).")
