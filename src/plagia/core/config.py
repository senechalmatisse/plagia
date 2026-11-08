"""Configuration immuable et validée (pydantic) des expériences.

Une expérience est décrite par un fichier YAML : tâche, corpus, prétraitement, blocs de caractéristiques, algorithmes et hyperparamètres, protocole et graine.
Le fichier est validé au chargement ; toute clé inconnue est refusée. Les chemins relatifs sont interprétés depuis le répertoire courant (racine du dépôt).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from plagia.core.errors import ConfigurationError
from plagia.core.hashing import stable_hash
from plagia.core.io import read_yaml

MaskMode = Literal["none", "entities", "propn", "content"]
TaskName = Literal["attribution", "verification", "plagiarism", "clustering"]


def _all() -> list[int | None]:
    return [None]


def _dimensions() -> list[int | None]:
    return [100, 300]


def _author_label() -> list[Literal["author", "work"]]:
    return ["author"]


class FrozenModel(BaseModel):
    """Modèle immuable qui refuse les clés inconnues."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class BlockConfig(FrozenModel):
    """Bloc de caractéristiques : type enregistré, poids et paramètres."""

    type: str
    name: str | None = None
    weight: float = Field(1.0, gt=0)
    mask: MaskMode | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def block_name(self) -> str:
        """Nom du bloc dans la représentation (défaut : son type)."""
        return self.name or self.type


class RepresentationConfig(FrozenModel):
    """Représentation : concaténation pondérée de blocs."""

    blocks: list[BlockConfig] = Field(min_length=1)
    description: str = ""


class ModelConfig(FrozenModel):
    """Algorithme enregistré, paramètres fixes et grille d'hyperparamètres."""

    type: str
    params: dict[str, Any] = Field(default_factory=dict)
    grid: dict[str, list[Any]] = Field(default_factory=dict)


class ProtocolConfig(FrozenModel):
    """Protocole d'évaluation."""

    cv_folds: int = Field(5, ge=2)
    seeds: int | list[int] = 5
    bootstrap: int = Field(1000, ge=0)
    confidence: float = Field(0.95, gt=0, lt=1)
    scoring: str = "f1_macro"
    group_by: Literal["none", "work"] = "none"
    validation_size: float = Field(0.2, gt=0, lt=1)
    tune: bool = True

    def seed_list(self, master: int) -> list[int]:
        """Graines des exécutions : `master + i` ou liste explicite."""
        if isinstance(self.seeds, int):
            if self.seeds < 1:
                raise ConfigurationError("protocol.seeds doit valoir au moins 1.")
            return [master + i for i in range(self.seeds)]
        return list(self.seeds)


class AnnotationConfig(FrozenModel):
    """Annotateurs (étiquettes, entités, liage, typage) et cache."""

    engine: Literal["spacy", "stanford", "heuristic", "none"] = "spacy"
    model: str = "en_core_web_sm"
    batch_size: int = Field(64, ge=1)
    n_process: int = Field(1, ge=1)
    java: str = "java"
    stanford_ner_jar: Path | None = None
    stanford_ner_model: Path | None = None
    stanford_pos_jar: Path | None = None
    stanford_pos_model: Path | None = None
    linker: Literal["none", "spotlight"] = "none"
    spotlight_url: str = "http://localhost:2222/rest"
    spotlight_confidence: float = Field(0.5, ge=0, le=1)
    spotlight_support: int = Field(20, ge=0)
    knowledge_base: Literal["none", "dbpedia", "yago"] = "none"
    taxonomy_path: Path | None = None
    sparql_endpoint: str | None = None
    cache_path: Path = Path("data/cache/annotations.sqlite")


class OpenSetConfig(FrozenModel):
    """Ensemble ouvert : auteurs retirés de l'entraînement et seuil τ."""

    unknown_authors: int = Field(2, ge=1)
    threshold: float | Literal["auto"] = "auto"


class AblationConfig(FrozenModel):
    """Ablations : familles isolées puis retirées d'une représentation complète."""

    base: str
    families: list[str] | None = None


class AttributionConfig(FrozenModel):
    """Étude A : attribution d'auteur."""

    corpus: Path
    representations: dict[str, RepresentationConfig] = Field(min_length=1)
    classifiers: dict[str, ModelConfig] = Field(min_length=1)
    n_authors: list[int | None] = Field(default_factory=_all)
    n_train: list[int | None] = Field(default_factory=_all)
    text_length: list[int | None] = Field(default_factory=_all)
    top_k: int = Field(3, ge=1)
    explain_top_k: int = Field(20, ge=1)
    error_sample: int = Field(30, ge=0)
    majority_baseline: bool = True
    open_set: OpenSetConfig | None = None
    ablation: AblationConfig | None = None


class VerificationConfig(FrozenModel):
    """Vérification d'auteur : paires de textes, même auteur ou non."""

    corpus: Path
    representations: dict[str, RepresentationConfig] = Field(min_length=1)
    classifiers: dict[str, ModelConfig] = Field(default_factory=dict)
    imposters: bool = False
    pairs_per_split: int = Field(400, ge=10)
    abstain_margin: float = Field(0.0, ge=0, lt=0.5)


class RetrievalConfig(FrozenModel):
    """Recherche des sources candidates."""

    method: Literal["tfidf", "shingles", "minhash", "pairs"] = "tfidf"
    k: int = Field(10, ge=1)
    chunk_sentences: int = Field(4, ge=1)
    shingle_size: int = Field(5, ge=2)
    num_perm: int = Field(128, ge=16)
    bands: int = Field(32, ge=1)


class AlignmentConfig(FrozenModel):
    """Alignement et filtrage."""

    seeds: Literal["ngrams", "sentences", "both"] = "both"
    ngram_size: int = Field(4, ge=2)
    sentence_threshold: float = Field(0.5, ge=0, le=1)
    dice_threshold: float = Field(0.5, ge=0, le=1)
    max_gap: int = Field(250, ge=0)
    min_seeds: int = Field(2, ge=1)
    min_density: float = Field(1.0, ge=0)
    min_similarity: float = Field(0.5, ge=0, le=1)
    confirmation: Literal["dice", "containment", "edit"] = "dice"
    min_length: int = Field(150, ge=0)
    mode: Literal["seeds", "windows", "both"] = "seeds"
    verify: bool = True


class PlagiarismConfig(FrozenModel):
    """Étude B : détection de plagiat."""

    corpus: Path
    levels: list[str] = Field(default_factory=lambda: ["O0", "O1", "O2", "O3"])
    sizes: list[int] = Field(default_factory=lambda: [50, 100, 300])
    feature_sets: dict[str, list[str]] = Field(min_length=1)
    classifiers: dict[str, ModelConfig] = Field(min_length=1)
    window_sentences: int = Field(3, ge=1)
    negatives_per_positive: int = Field(3, ge=1)
    test_prevalence: list[int] = Field(default_factory=lambda: [1, 10])
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    threshold_criterion: Literal["f1", "precision"] = "f1"
    target_precision: float = Field(0.9, gt=0, le=1)
    baselines: bool = True
    passage_level: bool = True
    origin_k: int = Field(5, ge=1)
    max_suspicious: int | None = None


class ClusteringCorpusConfig(FrozenModel):
    """Corpus à regrouper et étiquettes de référence."""

    path: Path
    split: str | None = None
    labels: list[Literal["author", "work"]] = Field(default_factory=_author_label)


class ClusteringConfig(FrozenModel):
    """Étude C : regroupement."""

    corpora: dict[str, ClusteringCorpusConfig] = Field(min_length=1)
    representations: dict[str, RepresentationConfig] = Field(min_length=1)
    dimensions: list[int | None] = Field(default_factory=_dimensions)
    algorithms: dict[str, ModelConfig] = Field(min_length=1)
    k: int | Literal["true", "silhouette", "elbow"] = "true"
    k_range: list[int] = Field(default_factory=lambda: list(range(2, 21)))
    families_threshold: float | None = None


class ExperimentConfig(FrozenModel):
    """Configuration complète d'une expérience."""

    name: str = Field(pattern=r"^[A-Za-z0-9_.\-]+$")
    task: TaskName
    description: str = ""
    seed: int = 42
    output_dir: Path = Path("results")
    n_jobs: int = 1
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    attribution: AttributionConfig | None = None
    verification: VerificationConfig | None = None
    plagiarism: PlagiarismConfig | None = None
    clustering: ClusteringConfig | None = None
    include: list[dict[str, Any]] | None = None
    exclude: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def _check_task_section(self) -> ExperimentConfig:
        if getattr(self, self.task) is None:
            raise ValueError(f"La section « {self.task} » est obligatoire pour task={self.task}.")
        for other in ("attribution", "verification", "plagiarism", "clustering"):
            if other != self.task and getattr(self, other) is not None:
                raise ValueError(f"Section « {other} » inattendue pour task={self.task}.")
        return self

    @property
    def section(self) -> Any:
        """Section propre à la tâche."""
        return getattr(self, self.task)

    @property
    def run_dir(self) -> Path:
        """Répertoire des résultats de l'expérience."""
        return self.output_dir / self.name

    def fingerprint(self) -> str:
        """Empreinte stable de la configuration (sans `n_jobs` ni `output_dir`)."""
        data = self.model_dump(mode="json", exclude={"n_jobs", "output_dir"})
        return stable_hash(data)

    def with_overrides(self, **changes: Any) -> ExperimentConfig:
        """Copie validée avec des champs de premier niveau modifiés."""
        data = self.model_dump()
        data.update(changes)
        return ExperimentConfig.model_validate(data)


def load_config(path: str | Path) -> ExperimentConfig:
    """Charge et valide une configuration YAML ; lève `ConfigurationError` si invalide."""
    data = read_yaml(path)
    if not isinstance(data, dict):
        raise ConfigurationError(f"{path} : le document YAML doit être un dictionnaire.")
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(f"{path} : configuration invalide\n{exc}") from exc
