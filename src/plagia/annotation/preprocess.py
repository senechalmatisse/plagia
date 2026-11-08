"""Prétraitement configurable étape par étape.

Étapes : segmentation (jetons de l'annotateur ou expression régulière), casse, ponctuation, mots vides, normalisation (lemmes spaCy ou racines Snowball de NLTK),
nombres. Chaque étape s'active par la configuration pour en mesurer l'effet ; en stylométrie, mots vides et ponctuation sont conservés par défaut.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from plagia.annotation.document import AnnotatedDocument, as_annotated
from plagia.core.errors import ConfigurationError, MissingDependencyError
from plagia.core.lexicon import function_words
from plagia.core.text import is_punctuation

Normalization = Literal["none", "lemma", "stem"]
StopwordSource = Literal["nltk", "sklearn", "builtin"]
_LANGUAGE_NAMES = {"en": "english", "fr": "french", "de": "german", "es": "spanish", "it": "italian"}


@lru_cache(maxsize=16)
def stopword_list(language: str = "en", source: StopwordSource = "nltk") -> frozenset[str]:
    """Liste de mots vides : NLTK (référence), scikit-learn (anglais) ou liste intégrée."""
    if source == "builtin":
        return function_words(language)
    if source == "sklearn":
        if language != "en":
            raise ConfigurationError("La liste scikit-learn n'existe qu'en anglais.")
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        return frozenset(ENGLISH_STOP_WORDS)
    try:
        from nltk.corpus import stopwords
    except ImportError as exc:
        raise MissingDependencyError("nltk", "mots vides NLTK", "Lancez « uv sync ».") from exc
    try:
        return frozenset(stopwords.words(_LANGUAGE_NAMES.get(language, language)))
    except LookupError as exc:
        raise MissingDependencyError("stopwords (NLTK)", "mots vides NLTK", "Lancez « plagia setup ».") from exc


@lru_cache(maxsize=8)
def _stemmer(language: str) -> Any:
    try:
        from nltk.stem.snowball import SnowballStemmer
    except ImportError as exc:
        raise MissingDependencyError("nltk", "racinisation Snowball", "Lancez « uv sync ».") from exc
    return SnowballStemmer(_LANGUAGE_NAMES.get(language, language))


class Preprocessor:
    """Transforme un document en liste de jetons normalisés."""

    def __init__(
        self,
        lowercase: bool = True,
        punctuation: bool = True,
        stopwords: bool = True,
        normalization: Normalization = "none",
        numbers: Literal["keep", "mask", "drop"] = "keep",
        language: str = "en",
        stopwords_source: StopwordSource = "nltk",
        min_length: int = 1,
    ) -> None:
        if normalization not in {"none", "lemma", "stem"}:
            raise ConfigurationError(f"Normalisation inconnue : {normalization}")
        self.lowercase = lowercase
        self.punctuation = punctuation
        self.stopwords = stopwords
        self.normalization = normalization
        self.numbers = numbers
        self.language = language
        self.stopwords_source = stopwords_source
        self.min_length = min_length

    def params(self) -> dict[str, Any]:
        """Paramètres (pour les clés de cache et les rapports)."""
        return dict(vars(self))

    def tokens(self, item: AnnotatedDocument | str) -> list[str]:
        """Jetons normalisés d'un document."""
        doc = as_annotated(item)
        if self.normalization == "lemma":
            if doc.lemmas is None:
                raise ConfigurationError("La lemmatisation nécessite des annotations spaCy (annotation.engine: spacy).")
            raw = list(doc.lemmas)
        else:
            raw = list(doc.tokens)
        stop = None if self.stopwords else stopword_list(self.language, self.stopwords_source)
        stem = _stemmer(self.language) if self.normalization == "stem" else None
        result = []
        for original, token in zip(doc.tokens, raw, strict=True):
            if is_punctuation(original):
                if self.punctuation:
                    result.append(original)
                continue
            if any(c.isdigit() for c in token) and token.replace(",", "").replace(".", "").isdigit():
                if self.numbers == "drop":
                    continue
                if self.numbers == "mask":
                    result.append("<num>")
                    continue
            lower = token.lower()
            if stop is not None and lower in stop:
                continue
            if len(token) < self.min_length:
                continue
            value = lower if self.lowercase else token
            if stem is not None:
                value = stem.stem(value)
            result.append(value)
        return result

    def sentences(self, item: AnnotatedDocument | str) -> list[list[str]]:
        """Jetons normalisés de chaque phrase."""
        doc = as_annotated(item)
        if not doc.sentences:
            return [self.tokens(doc)]
        out = []
        for start, end in doc.sentences:
            sub = doc.replace(
                tokens=doc.tokens[start:end],
                offsets=doc.offsets[start:end],
                pos=doc.pos[start:end] if doc.pos else None,
                tags=doc.tags[start:end] if doc.tags else None,
                lemmas=doc.lemmas[start:end] if doc.lemmas else None,
                sentences=(),
            )
            out.append(self.tokens(sub))
        return out
