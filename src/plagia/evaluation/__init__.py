"""Couche `evaluation` (ne dépend que de `core`) : mesures de classification, de
détection, de PAN et de regroupement ; bootstrap ; tests statistiques ; ablations ; explications."""

from plagia.evaluation.ablation import paired_comparisons
from plagia.evaluation.bootstrap import bootstrap_ci, paired_bootstrap
from plagia.evaluation.classification import accuracy, classification_metrics, f1_macro, topk_accuracy
from plagia.evaluation.clustering import clustering_metrics, purity
from plagia.evaluation.detection import (
    accuracy_at_k,
    binary_metrics,
    c_at_1,
    recall_at_k,
    verification_metrics,
)
from plagia.evaluation.explain import ERROR_CATEGORIES, confusion_pairs, error_sample, family_shares
from plagia.evaluation.pan import case_detection_rates, granularity, pan_measures, plagdet
from plagia.evaluation.stats import cohen_kappa, holm, mcnemar, summarize, wilcoxon_seeds

__all__ = [
    "ERROR_CATEGORIES",
    "accuracy",
    "accuracy_at_k",
    "binary_metrics",
    "bootstrap_ci",
    "c_at_1",
    "case_detection_rates",
    "classification_metrics",
    "clustering_metrics",
    "cohen_kappa",
    "confusion_pairs",
    "error_sample",
    "f1_macro",
    "family_shares",
    "granularity",
    "holm",
    "mcnemar",
    "paired_bootstrap",
    "paired_comparisons",
    "pan_measures",
    "plagdet",
    "purity",
    "recall_at_k",
    "summarize",
    "topk_accuracy",
    "verification_metrics",
    "wilcoxon_seeds",
]
