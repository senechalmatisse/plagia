"""Classificateur de paires de passages « reprise ou non ».

Chaque paire (fenêtre de phrases suspecte, fenêtre source) est décrite par un vecteur de similarités :
lexicales (cosinus TF-IDF, recouvrement de shingles, LCS, Dice…), d'entités (Jaccard, cosinus, divergence de types) et sémantiques (LSA).
Les noms des mesures composent les jeux de caractéristiques de l'étude B (L, L + E, L + E + S).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from plagia.core.config import ModelConfig
from plagia.core.errors import ConfigurationError
from plagia.core.text import words
from plagia.features.shingles import shingle_set
from plagia.models.base import Classifier
from plagia.similarity.entity import EntitySimilarity, TypeDivergence
from plagia.similarity.semantic import SemanticSimilarity
from plagia.similarity.sequence import lcs_length, levenshtein_distance
from plagia.similarity.sets import containment, dice, jaccard
from plagia.similarity.threshold import ThresholdChoice, select_threshold
from plagia.tasks.attribution import make_classifier
from plagia.tasks.plagiarism.passages import PassageView

PassagePairView = tuple[PassageView, PassageView]

LEXICAL = (
    "cosine_tfidf",
    "containment_shingles",
    "jaccard_shingles",
    "dice_words",
    "lcs",
    "levenshtein",
    "length_ratio",
)
ENTITY = ("jaccard_entities", "cosine_entities", "type_divergence", "kb_type_divergence")
SEMANTIC = ("semantic",)
FEATURE_NAMES = LEXICAL + ENTITY + SEMANTIC


def _lower_words(text: str) -> list[str]:
    return [w.lower() for w in words(text)]


class PairFeatureExtractor:
    """Calcule un vecteur de similarités par paire de passages."""

    def __init__(
        self,
        features: Sequence[str] = ("cosine_tfidf", "containment_shingles", "lcs"),
        shingle_size: int = 3,
        semantic_components: int = 100,
    ) -> None:
        unknown = set(features) - set(FEATURE_NAMES)
        if unknown:
            raise ConfigurationError(f"Mesures de paires inconnues : {sorted(unknown)} (disponibles : {FEATURE_NAMES})")
        self.features = list(features)
        self.shingle_size = shingle_size
        self.semantic_components = semantic_components
        self._entity_jaccard = EntitySimilarity("jaccard")
        self._entity_cosine = EntitySimilarity("cosine")
        self._types = TypeDivergence()
        self._kb_types = TypeDivergence(depth=2)

    def fit(self, texts: Sequence[str]) -> PairFeatureExtractor:
        """Ajuste l'IDF (et le modèle sémantique) sur des textes d'entraînement."""
        self.tfidf_ = TfidfVectorizer(sublinear_tf=True, dtype=np.float64).fit(list(texts))
        if "semantic" in self.features:
            self.semantic_ = SemanticSimilarity("lsa", n_components=self.semantic_components, min_df=1).fit(list(texts))
        return self

    def transform(self, pairs: Sequence[PassagePairView]) -> np.ndarray:
        """Matrice (paires × mesures)."""
        n = len(pairs)
        columns: dict[str, np.ndarray] = {}
        if n == 0:
            return np.zeros((0, len(self.features)))
        if "cosine_tfidf" in self.features:
            left = normalize(self.tfidf_.transform([a.text for a, _ in pairs]))
            right = normalize(self.tfidf_.transform([b.text for _, b in pairs]))
            columns["cosine_tfidf"] = np.asarray(left.multiply(right).sum(axis=1)).ravel()
        if "semantic" in self.features:
            columns["semantic"] = self.semantic_.score_many([(a.text, b.text) for a, b in pairs])
        needs_words = bool(set(self.features) & {"dice_words", "lcs", "levenshtein", "length_ratio"})
        needs_shingles = bool(set(self.features) & {"containment_shingles", "jaccard_shingles"})
        per_pair: dict[str, list[float]] = {name: [] for name in self.features if name not in columns}
        for a, b in pairs:
            context: dict[str, Any] = {}
            if needs_words:
                context["wa"], context["wb"] = _lower_words(a.text), _lower_words(b.text)
            if needs_shingles:
                context["sa"] = shingle_set(a.text, self.shingle_size)
                context["sb"] = shingle_set(b.text, self.shingle_size)
            for name in per_pair:
                per_pair[name].append(self._single(name, a, b, context))
        for name, values in per_pair.items():
            columns[name] = np.asarray(values, dtype=float)
        return np.column_stack([columns[name] for name in self.features])

    def _single(self, name: str, a: PassageView, b: PassageView, scope: dict[str, Any]) -> float:
        if name == "containment_shingles":
            return max(containment(scope["sa"], scope["sb"]), containment(scope["sb"], scope["sa"]))
        if name == "jaccard_shingles":
            return jaccard(scope["sa"], scope["sb"])
        if name == "dice_words":
            return dice(set(scope["wa"]), set(scope["wb"]))
        if name == "lcs":
            shortest = min(len(scope["wa"]), len(scope["wb"]))
            return lcs_length(scope["wa"], scope["wb"]) / shortest if shortest else 0.0
        if name == "levenshtein":
            longest = max(len(scope["wa"]), len(scope["wb"]))
            return 1 - levenshtein_distance(scope["wa"], scope["wb"]) / longest if longest else 0.0
        if name == "length_ratio":
            la, lb = len(scope["wa"]), len(scope["wb"])
            return min(la, lb) / max(la, lb) if max(la, lb) else 0.0
        if name == "jaccard_entities":
            return self._entity_jaccard.score(a.annotated(), b.annotated())
        if name == "cosine_entities":
            return self._entity_cosine.score(a.annotated(), b.annotated())
        if name == "type_divergence":
            return self._types.score(a.annotated(), b.annotated())
        if name == "kb_type_divergence":
            return self._kb_types.score(a.annotated(), b.annotated())
        raise ConfigurationError(f"Mesure inconnue : {name}")


class PairClassifier:
    """Classificateur binaire de paires de passages avec seuil calibré sur la validation."""

    def __init__(
        self, extractor: PairFeatureExtractor, classifier: ModelConfig | Classifier | None = None, seed: int = 0
    ) -> None:
        self.extractor = extractor
        self.classifier = make_classifier(classifier or ModelConfig(type="logreg"), seed)
        self.threshold = 0.5
        self.calibration: ThresholdChoice | None = None

    def fit(self, pairs: Sequence[PassagePairView], labels: Sequence[int]) -> PairClassifier:
        """Apprend sur des paires étiquetées (l'extracteur doit être ajusté)."""
        self.classifier.fit(self.extractor.transform(pairs), list(labels))
        return self

    def predict_proba(self, pairs: Sequence[PassagePairView]) -> np.ndarray:
        """Score de reprise de chaque paire dans [0, 1]."""
        if not pairs:
            return np.zeros(0)
        scores = self.classifier.predict_scores(self.extractor.transform(pairs))
        classes = list(self.classifier.classes_)
        return scores[:, classes.index(1)] if 1 in classes else np.zeros(len(pairs))

    def calibrate(
        self,
        pairs: Sequence[PassagePairView],
        labels: Sequence[int],
        criterion: str = "f1",
        target_precision: float = 0.9,
    ) -> ThresholdChoice:
        """Choisit le seuil sur la validation."""
        self.calibration = select_threshold(labels, self.predict_proba(pairs), criterion, target_precision)
        self.threshold = self.calibration.threshold
        return self.calibration

    def predict(self, pairs: Sequence[PassagePairView]) -> np.ndarray:
        """Décisions binaires au seuil calibré."""
        return (self.predict_proba(pairs) >= self.threshold).astype(int)

    def as_verifier(self) -> Callable[[Sequence[PassagePairView]], np.ndarray]:
        """Fonction de vérification utilisable par :class:`TextAligner`."""
        return self.predict_proba
