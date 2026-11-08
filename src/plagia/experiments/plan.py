"""Plan d'expérience : conditions (produit des facteurs) et graines.

Chaque condition reçoit un identifiant lisible et stable (`rep=R1|clf=nb|authors=5…`).
Les lignes de base s'ajoutent comme conditions marquées `baseline`. Les filtres `include` / `exclude` de la configuration permettent un plan fractionnaire.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from plagia.core.config import ExperimentConfig, RepresentationConfig
from plagia.core.errors import ConfigurationError
from plagia.features.representation import ablation_representations

DETERMINISTIC_CLUSTERERS = {"agglomerative", "dbscan", "hdbscan"}


def _fmt(value: Any) -> str:
    return "all" if value is None else str(value)


@dataclass(frozen=True)
class Condition:
    """Combinaison de facteurs d'une expérience."""

    id: str
    params: Mapping[str, Any]
    seeds: tuple[int, ...]
    baseline: bool = False


@dataclass(frozen=True)
class Run:
    """Exécution d'une condition avec une graine."""

    condition: Condition
    seed: int

    @property
    def key(self) -> str:
        """Clé unique de l'exécution (reprise des campagnes interrompues)."""
        return f"{self.condition.id}#{self.seed}"


def _condition(factors: Sequence[tuple[str, str, Any]], seeds: Sequence[int], baseline: bool = False) -> Condition:
    identifier = "|".join(f"{short}={_fmt(value)}" for short, _, value in factors)
    return Condition(identifier, {name: value for _, name, value in factors}, tuple(seeds), baseline)


def _matches(params: Mapping[str, Any], selector: Mapping[str, Any]) -> bool:
    for key, wanted in selector.items():
        values = wanted if isinstance(wanted, list) else [wanted]
        if params.get(key) not in values:
            return False
    return True


@dataclass
class Plan:
    """Conditions et exécutions d'une expérience."""

    config: ExperimentConfig
    conditions: list[Condition] = field(default_factory=list)
    representations: dict[str, RepresentationConfig] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> Plan:
        """Développe le plan complet de la configuration (filtres appliqués)."""
        seeds = config.protocol.seed_list(config.seed)
        builder = {
            "attribution": _attribution_conditions,
            "verification": _verification_conditions,
            "plagiarism": _plagiarism_conditions,
            "clustering": _clustering_conditions,
        }[config.task]
        conditions, representations = builder(config, seeds)
        if config.include:
            conditions = [c for c in conditions if c.baseline or any(_matches(c.params, s) for s in config.include)]
        if config.exclude:
            conditions = [c for c in conditions if not any(_matches(c.params, s) for s in config.exclude)]
        identifiers = [c.id for c in conditions]
        if len(set(identifiers)) != len(identifiers):
            raise ConfigurationError("Identifiants de conditions dupliqués : vérifiez les noms des facteurs.")
        return cls(config, conditions, representations)

    @property
    def runs(self) -> list[Run]:
        """Toutes les exécutions (condition × graine)."""
        return [Run(c, s) for c in self.conditions for s in c.seeds]

    def summary(self) -> dict[str, Any]:
        """Effectifs du plan."""
        main = [c for c in self.conditions if not c.baseline]
        return {
            "task": self.config.task,
            "conditions": len(main),
            "baseline_conditions": len(self.conditions) - len(main),
            "runs": len(self.runs),
            "seeds": sorted({s for c in self.conditions for s in c.seeds}),
        }


def _attribution_conditions(
    config: ExperimentConfig, seeds: list[int]
) -> tuple[list[Condition], dict[str, RepresentationConfig]]:
    section = config.attribution
    assert section is not None
    representations = dict(section.representations)
    if section.ablation is not None:
        base = representations.get(section.ablation.base)
        if base is None:
            raise ConfigurationError(f"Représentation d'ablation inconnue : {section.ablation.base}")
        representations.update(
            ablation_representations(base, section.ablation.families, prefix=f"{section.ablation.base}:")
        )
    conditions = []
    grid = itertools.product(section.n_authors, section.n_train, section.text_length)
    for n_authors, n_train, length in grid:
        for rep, clf in itertools.product(representations, section.classifiers):
            conditions.append(
                _condition(
                    [
                        ("rep", "representation", rep),
                        ("clf", "classifier", clf),
                        ("authors", "n_authors", n_authors),
                        ("train", "n_train", n_train),
                        ("len", "text_length", length),
                    ],
                    seeds,
                )
            )
        if section.majority_baseline:
            conditions.append(
                _condition(
                    [
                        ("rep", "representation", "-"),
                        ("clf", "classifier", "majority"),
                        ("authors", "n_authors", n_authors),
                        ("train", "n_train", n_train),
                        ("len", "text_length", length),
                    ],
                    seeds,
                    baseline=True,
                )
            )
    return conditions, representations


def _verification_conditions(
    config: ExperimentConfig, seeds: list[int]
) -> tuple[list[Condition], dict[str, RepresentationConfig]]:
    section = config.verification
    assert section is not None
    methods = list(section.classifiers) + (["imposters"] if section.imposters else [])
    if not methods:
        raise ConfigurationError("verification : au moins un classificateur ou imposters: true.")
    conditions = [
        _condition([("rep", "representation", rep), ("method", "method", method)], seeds)
        for rep, method in itertools.product(section.representations, methods)
    ]
    return conditions, dict(section.representations)


def _plagiarism_conditions(
    config: ExperimentConfig, seeds: list[int]
) -> tuple[list[Condition], dict[str, RepresentationConfig]]:
    section = config.plagiarism
    assert section is not None
    conditions = []
    for level, size in itertools.product(section.levels, section.sizes):
        for features, clf in itertools.product(section.feature_sets, section.classifiers):
            conditions.append(
                _condition(
                    [
                        ("level", "level", level),
                        ("size", "size", size),
                        ("features", "feature_set", features),
                        ("clf", "classifier", clf),
                    ],
                    seeds,
                )
            )
        if section.baselines:
            for baseline in ("baseline_cosine", "baseline_containment"):
                conditions.append(
                    _condition(
                        [
                            ("level", "level", level),
                            ("size", "size", size),
                            ("features", "feature_set", "-"),
                            ("clf", "classifier", baseline),
                        ],
                        seeds,
                        baseline=True,
                    )
                )
    return conditions, {}


def _clustering_conditions(
    config: ExperimentConfig, seeds: list[int]
) -> tuple[list[Condition], dict[str, RepresentationConfig]]:
    section = config.clustering
    assert section is not None
    conditions = []
    for corpus, rep, dim, algo in itertools.product(
        section.corpora, section.representations, section.dimensions, section.algorithms
    ):
        algo_type = section.algorithms[algo].type
        run_seeds = seeds[:1] if algo_type in DETERMINISTIC_CLUSTERERS else seeds
        conditions.append(
            _condition(
                [
                    ("corpus", "corpus", corpus),
                    ("rep", "representation", rep),
                    ("dim", "dimensions", dim),
                    ("algo", "algorithm", algo),
                ],
                run_seeds,
            )
        )
    return conditions, dict(section.representations)
