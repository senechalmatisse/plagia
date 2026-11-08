"""Assemblage d'une représentation par blocs combinables pondérés et ablations.

Une représentation est une `FeatureUnion` scikit-learn : chaque bloc (éventuellement précédé d'un masquage) produit une matrice creuse,
multipliée par le poids du bloc, puis concaténée aux autres.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion, Pipeline

from plagia.core.config import BlockConfig, RepresentationConfig
from plagia.core.errors import ConfigurationError
from plagia.features.base import FEATURE_BLOCKS, block_family
from plagia.features.masking import Masker


def build_block(config: BlockConfig) -> BaseEstimator:
    """Bloc décrit par la configuration (masquage éventuel compris)."""
    block = FEATURE_BLOCKS.create(config.type, **config.params)
    if config.mask and config.mask != "none":
        return Pipeline([("mask", Masker(mode=config.mask)), ("block", block)])
    return block


def build_representation(blocks: Sequence[BlockConfig] | RepresentationConfig) -> FeatureUnion:
    """`FeatureUnion` pondérée des blocs (noms uniques, poids par bloc)."""
    configs = blocks.blocks if isinstance(blocks, RepresentationConfig) else list(blocks)
    if not configs:
        raise ConfigurationError("Une représentation doit contenir au moins un bloc.")
    names: list[str] = []
    for config in configs:
        name = config.block_name
        suffix = 2
        while name in names:
            name = f"{config.block_name}_{suffix}"
            suffix += 1
        names.append(name)
    return FeatureUnion(
        [(name, build_block(config)) for name, config in zip(names, configs, strict=True)],
        transformer_weights={name: config.weight for name, config in zip(names, configs, strict=True)},
    )


def representation_families(representation: RepresentationConfig) -> list[str]:
    """Familles présentes dans une représentation, dans l'ordre d'apparition."""
    seen: list[str] = []
    for block in representation.blocks:
        family = block_family(block.type)
        if family not in seen:
            seen.append(family)
    return seen


def ablate(representation: RepresentationConfig, family: str, mode: str) -> RepresentationConfig:
    """Représentation réduite à `family` (`mode='only'`) ou privée de `family` (`'without'`)."""
    if mode == "only":
        blocks = [b for b in representation.blocks if block_family(b.type) == family]
    elif mode == "without":
        blocks = [b for b in representation.blocks if block_family(b.type) != family]
    else:
        raise ConfigurationError(f"Mode d'ablation inconnu : {mode}")
    if not blocks:
        raise ConfigurationError(f"L'ablation « {mode} {family} » ne laisse aucun bloc.")
    return RepresentationConfig(blocks=blocks, description=f"{mode} {family}")


def ablation_representations(
    representation: RepresentationConfig, families: Sequence[str] | None = None, prefix: str = ""
) -> dict[str, RepresentationConfig]:
    """Représentations d'ablation : chaque famille isolée puis retirée de l'ensemble."""
    present = representation_families(representation)
    chosen = [f for f in (families or present) if f in present]
    result: dict[str, RepresentationConfig] = {}
    for family in chosen:
        result[f"{prefix}only_{family}"] = ablate(representation, family, "only")
        if len(present) > 1:
            result[f"{prefix}without_{family}"] = ablate(representation, family, "without")
    return result


def block_slices(union: FeatureUnion) -> dict[str, slice]:
    """Colonnes occupées par chaque bloc dans la matrice concaténée (union ajustée)."""
    slices: dict[str, slice] = {}
    start = 0
    for name, transformer in union.transformer_list:
        width = len(transformer.get_feature_names_out())
        slices[name] = slice(start, start + width)
        start += width
    return slices
