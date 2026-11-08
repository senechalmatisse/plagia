"""Vérification d'auteur : deux textes ont-ils le même auteur ?

* :class:`AuthorVerifier` : classification binaire d'une paire à partir des différences absolues de caractéristiques et des similarités cosinus par bloc ;
* :class:`ImpostersVerifier` ([C]) : méthode des imposteurs (Koppel et Winter, 2014), sans apprentissage supervisé.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from plagia.annotation.document import AnnotatedDocument
from plagia.core.config import BlockConfig, ModelConfig, RepresentationConfig
from plagia.core.seeds import make_random, make_rng
from plagia.features.representation import block_slices, build_representation
from plagia.models.base import Classifier
from plagia.tasks.attribution import make_classifier

Pair = tuple[int, int]


def make_pairs(
    authors: Sequence[str], n_pairs: int, seed: int, positive_fraction: float = 0.5
) -> list[tuple[int, int, int]]:
    """Paires d'indices `(i, j, même_auteur)` équilibrées, tirées par graine."""
    rng = make_random(seed, "verification-pairs")
    by_author: dict[str, list[int]] = {}
    for i, author in enumerate(authors):
        by_author.setdefault(author, []).append(i)
    multi = [a for a, idx in by_author.items() if len(idx) >= 2]
    names = sorted(by_author)
    pairs: set[tuple[int, int, int]] = set()
    n_positive = round(n_pairs * positive_fraction) if multi else 0
    attempts = 0
    while len(pairs) < n_positive and attempts < 50 * n_pairs:
        attempts += 1
        author = rng.choice(sorted(multi))
        i, j = sorted(rng.sample(by_author[author], 2))
        pairs.add((i, j, 1))
    attempts = 0
    while len(pairs) < n_pairs and len(names) >= 2 and attempts < 50 * n_pairs:
        attempts += 1
        a, b = rng.sample(names, 2)
        i, j = sorted((rng.choice(by_author[a]), rng.choice(by_author[b])))
        pairs.add((i, j, 0))
    return sorted(pairs)


class PairFeaturizer:
    """Vecteur d'une paire : |f(a) − f(b)| et cosinus par bloc."""

    def __init__(self, representation: RepresentationConfig | Sequence[BlockConfig]) -> None:
        self.union = build_representation(representation)

    def fit(self, documents: Sequence[AnnotatedDocument | str]) -> PairFeaturizer:
        self.union.fit(list(documents))
        self.slices_ = block_slices(self.union)
        return self

    def matrix(self, documents: Sequence[AnnotatedDocument | str]) -> sparse.csr_matrix:
        """Représentation des documents."""
        return sparse.csr_matrix(self.union.transform(list(documents)))

    def pair_features(self, matrix: sparse.csr_matrix, pairs: Sequence[Pair]) -> sparse.csr_matrix:
        """Caractéristiques des paires d'indices `(i, j)` de `matrix`."""
        left = matrix[[i for i, _ in pairs]]
        right = matrix[[j for _, j in pairs]]
        difference = abs(left - right)
        cosines = []
        for block in self.slices_.values():
            a, b = normalize(left[:, block]), normalize(right[:, block])
            cosines.append(np.asarray(a.multiply(b).sum(axis=1)))
        return sparse.hstack([difference, sparse.csr_matrix(np.hstack(cosines))], format="csr")


class AuthorVerifier:
    """Classifieur binaire « même auteur » sur paires de textes."""

    def __init__(
        self,
        representation: RepresentationConfig | Sequence[BlockConfig],
        classifier: ModelConfig | Classifier | None = None,
        seed: int = 0,
    ) -> None:
        self.featurizer = PairFeaturizer(representation)
        self.classifier = make_classifier(classifier or ModelConfig(type="logreg"), seed)
        self.seed = seed

    def fit(
        self, documents: Sequence[AnnotatedDocument | str], pairs: Sequence[tuple[int, int, int]]
    ) -> AuthorVerifier:
        documents = list(documents)
        self.featurizer.fit(documents)
        matrix = self.featurizer.matrix(documents)
        features = self.featurizer.pair_features(matrix, [(i, j) for i, j, _ in pairs])
        self.classifier.fit(features, [label for _, _, label in pairs])
        return self

    def predict_proba(self, documents: Sequence[AnnotatedDocument | str], pairs: Sequence[Pair]) -> np.ndarray:
        """Probabilité « même auteur » de chaque paire."""
        matrix = self.featurizer.matrix(list(documents))
        scores = self.classifier.predict_scores(self.featurizer.pair_features(matrix, pairs))
        positive = list(self.classifier.classes_).index(1) if 1 in list(self.classifier.classes_) else -1
        return scores[:, positive]


def decide(probabilities: np.ndarray, margin: float = 0.0) -> list[int | None]:
    """Décisions binaires ; abstention (`None`) si |p − 0,5| ≤ `margin` (mesure c@1)."""
    return [None if abs(p - 0.5) <= margin and margin > 0 else int(p > 0.5) for p in probabilities]


def _subspace_cosine(x: Any, y: Any, columns: np.ndarray) -> np.ndarray:
    """Cosinus entre les lignes de `x` et de `y` restreintes aux colonnes tirées."""
    x, y = normalize(x[:, columns]), normalize(y[:, columns])
    return np.asarray((x @ y.T).toarray()).ravel()


class ImpostersVerifier:
    """Méthode des imposteurs : proportion d'itérations (sous-espaces aléatoires d'attributs)
    où `b` est plus proche de `a` que de tous les imposteurs (score symétrisé)."""

    def __init__(
        self,
        representation: RepresentationConfig | Sequence[BlockConfig],
        n_iterations: int = 100,
        feature_fraction: float = 0.5,
        n_imposters: int = 20,
        seed: int = 0,
    ) -> None:
        self.featurizer = PairFeaturizer(representation)
        self.n_iterations = n_iterations
        self.feature_fraction = feature_fraction
        self.n_imposters = n_imposters
        self.seed = seed

    def fit(self, documents: Sequence[AnnotatedDocument | str]) -> ImpostersVerifier:
        """Ajuste la représentation et mémorise le réservoir d'imposteurs."""
        documents = list(documents)
        self.featurizer.fit(documents)
        self.pool_ = normalize(self.featurizer.matrix(documents))
        return self

    def _score_one(self, a: sparse.csr_matrix, b: sparse.csr_matrix, rng: np.random.Generator) -> float:
        n_features = a.shape[1]
        wins = 0
        for _ in range(self.n_iterations):
            columns = rng.random(n_features) < self.feature_fraction
            imposters = self.pool_[
                rng.choice(self.pool_.shape[0], size=min(self.n_imposters, self.pool_.shape[0]), replace=False)
            ]

            target = _subspace_cosine(b, a, columns)[0]
            if target > _subspace_cosine(b, imposters, columns).max(initial=-1.0):
                wins += 1
        return wins / self.n_iterations

    def score(self, documents: Sequence[AnnotatedDocument | str], pairs: Sequence[Pair]) -> np.ndarray:
        """Score dans [0, 1] de chaque paire (moyenne des deux sens)."""
        matrix = normalize(self.featurizer.matrix(list(documents)))
        rng = make_rng(self.seed, "imposters")
        scores = []
        for i, j in pairs:
            a, b = matrix[i], matrix[j]
            scores.append((self._score_one(a, b, rng) + self._score_one(b, a, rng)) / 2)
        return np.array(scores)
