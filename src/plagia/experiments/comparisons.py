"""Synthèse d'une campagne : résumé par condition (moyenne, écart-type, intervalles) et
comparaisons appariées (McNemar, bootstrap de la différence, Holm), ablations comprises."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from plagia.core.config import ExperimentConfig
from plagia.evaluation.ablation import paired_comparisons
from plagia.evaluation.stats import summarize
from plagia.experiments.store import ResultStore

MAIN_METRICS = {
    "attribution": ["f1_macro", "accuracy", "f1_micro"],
    "verification": ["auc", "f1", "c_at_1"],
    "plagiarism": [
        "pairs_1to1_f1",
        "pairs_1to10_f1",
        "pairs_1to10_average_precision",
        "plagdet",
        "origin_accuracy_at_1",
    ],
    "clustering": ["author_ari", "author_nmi", "author_v_measure", "author_purity", "work_ari", "author_silhouette"],
}
PRIMARY_METRIC = {
    "attribution": "f1_macro",
    "verification": "auc",
    "plagiarism": "pairs_1to10_f1",
    "clustering": "author_ari",
}


def summary_rows(frame: pd.DataFrame, task: str, confidence: float = 0.95) -> list[dict[str, Any]]:
    """Moyenne, écart-type et intervalle (Student, sur les graines) des mesures principales."""
    if frame.empty:
        return []
    metrics = [m for m in MAIN_METRICS.get(task, []) if m in frame.columns]
    param_columns = [c for c in frame.columns if c.startswith("param_")]
    rows = []
    for condition, group in frame.groupby("condition", sort=True):
        row: dict[str, Any] = {"condition": condition, "seeds": len(group), "baseline": bool(group["baseline"].iloc[0])}
        row.update({c: group[c].iloc[0] for c in param_columns})
        for metric in metrics:
            stats = summarize(group[metric].astype(float).tolist(), confidence)
            row[f"{metric}_mean"] = stats["mean"]
            row[f"{metric}_std"] = stats["std"]
            row[f"{metric}_ci_low"] = stats["ci_low"]
            row[f"{metric}_ci_high"] = stats["ci_high"]
            if f"{metric}_low" in group.columns:
                row[f"{metric}_boot_low"] = float(group[f"{metric}_low"].mean())
                row[f"{metric}_boot_high"] = float(group[f"{metric}_high"].mean())
        row["duration_s_mean"] = float(group["duration_s"].mean())
        rows.append(row)
    return rows


def _attribution_predictions(
    store: ResultStore, records: list[dict[str, Any]]
) -> dict[tuple[Any, ...], dict[str, dict[str, tuple[str, str]]]]:
    grouped: dict[tuple[Any, ...], dict[str, dict[str, tuple[str, str]]]] = defaultdict(lambda: defaultdict(dict))
    for record in records:
        params = record["params"]
        group = (params.get("classifier"), params.get("n_authors"), params.get("n_train"), params.get("text_length"))
        for prediction in store.predictions(record):
            grouped[group][str(params.get("representation"))][f"{record['seed']}:{prediction['id']}"] = (
                prediction["true"],
                prediction["pred"],
            )
    return grouped


def comparison_rows(store: ResultStore, config: ExperimentConfig) -> list[dict[str, Any]]:
    """Comparaisons appariées à une condition de référence, par groupe de facteurs.

    * attribution : chaque représentation contre la première listée (et, pour les
      ablations, chaque variante contre la représentation complète) ;
    * plagiat : chaque jeu de caractéristiques contre le premier listé, par cellule.
    """
    records = [r for r in store.records("ok") if not r.get("baseline")]
    protocol = config.protocol
    rows: list[dict[str, Any]] = []
    if config.task == "attribution" and config.attribution is not None:
        reference = next(iter(config.attribution.representations))
        ablation_base = config.attribution.ablation.base if config.attribution.ablation else None
        for group, by_rep in sorted(_attribution_predictions(store, records).items(), key=lambda kv: str(kv[0])):
            references = {reference: [k for k in by_rep if ":" not in k]}
            if ablation_base:
                references[ablation_base] = [k for k in by_rep if k.startswith(f"{ablation_base}:")]
            for ref, members in references.items():
                if ref not in by_rep or not members:
                    continue
                subset = {k: by_rep[k] for k in [ref, *members] if k in by_rep}
                for row in paired_comparisons(
                    ref, subset, n_resamples=protocol.bootstrap, confidence=protocol.confidence, seed=config.seed
                ):
                    row.update(
                        {"classifier": group[0], "n_authors": group[1], "n_train": group[2], "text_length": group[3]}
                    )
                    rows.append(row)
    elif config.task == "plagiarism" and config.plagiarism is not None:
        reference = next(iter(config.plagiarism.feature_sets))
        grouped: dict[tuple[Any, ...], dict[str, dict[str, tuple[str, str]]]] = defaultdict(lambda: defaultdict(dict))
        thresholds = {r["key"]: r["metrics"].get("threshold", 0.5) for r in records}
        for record in records:
            params = record["params"]
            group = (params["level"], params["size"], params["classifier"])
            for p in store.predictions(record):
                key = f"{record['seed']}:{p['susp']}:{p['s'][0]}:{p['src']}:{p['t'][0]}"
                grouped[group][str(params["feature_set"])][key] = (
                    str(p["label"]),
                    str(int(p["score"] >= thresholds[record["key"]])),
                )
        for group, by_features in sorted(grouped.items(), key=lambda kv: str(kv[0])):
            if reference not in by_features or len(by_features) < 2:
                continue
            for row in paired_comparisons(
                reference, by_features, n_resamples=protocol.bootstrap, confidence=protocol.confidence, seed=config.seed
            ):
                row.update({"level": group[0], "size": group[1], "classifier": group[2]})
                rows.append(row)
    return rows
