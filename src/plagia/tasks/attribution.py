"""Attribution d'auteur en ensemble fermé (T1) : représentation par blocs + classificateur.

Le réglage des hyperparamètres (α du lissage, C du SVM, bornes de fréquence, plage de n-grammes…) se fait par validation croisée stratifiée à 5 plis sur l'entraînement,
groupée par œuvre pour le corpus littéraire. Les blocs de caractéristiques s'ajustent à l'intérieur de chaque pli : aucune fuite.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from plagia.annotation.document import AnnotatedDocument
from plagia.core.config import BlockConfig, ModelConfig, RepresentationConfig
from plagia.core.errors import PlagiaError
from plagia.corpus.splits import cv_splitter, effective_folds
from plagia.features.representation import block_slices, build_representation
from plagia.models.base import Classifier, Explanation
from plagia.models.naive_bayes import NaiveBayes
from plagia.models.sklearn_models import SklearnClassifier, build_classifier, param_prefix, set_random_state


def make_classifier(config: ModelConfig | Classifier, seed: int = 0) -> Classifier:
    """Classificateur à partir d'une configuration (graine fixée si applicable)."""
    if isinstance(config, Classifier):
        return config
    return set_random_state(build_classifier(config.type, **config.params), seed)


def expand_grid(grid: Mapping[str, Sequence[Any]], classifier: Classifier) -> dict[str, list[Any]]:
    """Noms de grille : `C` → `clf__estimator__C` ; `features__…` conservés tels quels."""
    expanded: dict[str, list[Any]] = {}
    for key, values in grid.items():
        if key.startswith(("features__", "clf__")):
            expanded[key] = list(values)
        else:
            expanded[f"clf__{param_prefix(classifier)}{key}"] = list(values)
    return expanded


class AuthorshipAttributor:
    """Chaîne d'attribution : `Pipeline([("features", FeatureUnion), ("clf", Classifier)])`."""

    def __init__(
        self,
        representation: RepresentationConfig | Sequence[BlockConfig],
        classifier: ModelConfig | Classifier,
        seed: int = 0,
    ) -> None:
        self.representation = representation
        self.seed = seed
        self.pipeline = Pipeline(
            [("features", build_representation(representation)), ("clf", make_classifier(classifier, seed))]
        )
        self.search_summary: dict[str, Any] = {}

    @property
    def classifier(self) -> Classifier:
        """Classificateur ajusté."""
        clf: Classifier = self.pipeline.named_steps["clf"]
        return clf

    def fit(self, documents: Sequence[AnnotatedDocument | str], authors: Sequence[str]) -> AuthorshipAttributor:
        """Apprend la représentation et le classificateur sur l'entraînement."""
        self.pipeline.fit(list(documents), list(authors))
        return self

    def tune(
        self,
        documents: Sequence[AnnotatedDocument | str],
        authors: Sequence[str],
        grid: Mapping[str, Sequence[Any]] | None = None,
        groups: Sequence[str] | None = None,
        cv: int = 5,
        scoring: str = "f1_macro",
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """Validation croisée sur l'entraînement puis réapprentissage avec les meilleurs paramètres."""
        documents, authors = list(documents), list(authors)
        param_grid = expand_grid(grid or {}, self.classifier)
        folds = effective_folds(authors, groups, cv)
        if not param_grid or folds < 2:
            self.fit(documents, authors)
            self.search_summary = {"best_params": {}, "best_score": None, "folds": folds}
            return self.search_summary
        search = GridSearchCV(
            self.pipeline,
            param_grid,
            scoring=scoring,
            cv=cv_splitter(folds, groups is not None, self.seed),
            n_jobs=n_jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(documents, authors, groups=groups)
        self.pipeline = search.best_estimator_
        self.search_summary = {
            "best_params": dict(search.best_params_),
            "best_score": float(search.best_score_),
            "folds": folds,
            "cv_mean": [float(x) for x in search.cv_results_["mean_test_score"]],
        }
        return self.search_summary

    def predict(self, documents: Sequence[AnnotatedDocument | str]) -> list[str]:
        """Auteur prédit pour chaque document."""
        return [str(p) for p in self.pipeline.predict(list(documents))]

    def predict_scores(self, documents: Sequence[AnnotatedDocument | str]) -> np.ndarray:
        """Scores de confiance par auteur."""
        features = self.pipeline.named_steps["features"].transform(list(documents))
        return self.classifier.predict_scores(features)

    def rank(self, documents: Sequence[AnnotatedDocument | str], k: int = 3) -> list[list[tuple[str, float]]]:
        """Les `k` auteurs les plus probables avec un score de confiance."""
        features = self.pipeline.named_steps["features"].transform(list(documents))
        return self.classifier.rank(features, k)

    def feature_names(self) -> list[str]:
        """Noms des attributs (préfixés par le nom du bloc)."""
        return [str(n) for n in self.pipeline.named_steps["features"].get_feature_names_out()]

    def explain(self, top_k: int = 20) -> Explanation:
        """Attributs les plus discriminants par auteur."""
        return self.classifier.explain(self.feature_names(), top_k)

    def block_contributions(self) -> dict[str, float]:
        """Part de la masse des poids (|coef| ou |log-rapport|) portée par chaque bloc."""
        clf = self.classifier
        weights: np.ndarray | None = None
        if isinstance(clf, NaiveBayes):
            weights = np.abs(clf.log_likelihood_ratios())
        elif isinstance(clf, SklearnClassifier):
            coef = clf.coefficients()
            weights = np.abs(coef) if coef is not None else None
            if weights is None and hasattr(clf.estimator_, "feature_importances_"):
                weights = np.asarray(clf.estimator_.feature_importances_, dtype=float)[None, :]
        if weights is None:
            return {}
        total = float(weights.sum()) or 1.0
        slices = block_slices(self.pipeline.named_steps["features"])
        return {name: float(weights[:, s].sum()) / total for name, s in slices.items()}

    def save(self, path: str | Path) -> Path:
        """Sérialise le modèle (joblib)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target, compress=3)
        return target

    @staticmethod
    def load(path: str | Path) -> AuthorshipAttributor:
        """Recharge un modèle sérialisé."""
        if not Path(path).is_file():
            raise PlagiaError(
                f"Modèle introuvable : {path}. Entraînez-le avec « plagia attribute --train <corpus> --save {path} »."
            )
        model: AuthorshipAttributor = joblib.load(path)
        return model
