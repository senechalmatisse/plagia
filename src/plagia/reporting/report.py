"""Rapport d'évaluation régénéré par script à partir des résultats d'une expérience.

Contenu : description et traçabilité, tableau des métriques avec intervalles de confiance, effet des facteurs, matrices de confusion, courbes précision-rappel,
attributs discriminants, comparaisons appariées, échantillon d'erreurs et limites. Formats : HTML autonome (figures intégrées) ou Markdown (figures en PNG à côté).
"""

from __future__ import annotations

import base64
import html as html_lib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from plagia.core.errors import PlagiaError
from plagia.core.io import atomic_write_text, ensure_dir, read_json, read_jsonl
from plagia.evaluation.explain import ERROR_CATEGORIES
from plagia.reporting import figures, html
from plagia.reporting.tables import fmt, metric_table, to_html, to_markdown

PRIMARY = {"attribution": "f1_macro", "verification": "auc", "plagiarism": "pairs_1to10_f1", "clustering": "author_ari"}
METRICS = {
    "attribution": ["f1_macro", "accuracy"],
    "verification": ["auc", "f1", "c_at_1"],
    "plagiarism": [
        "pairs_1to1_f1",
        "pairs_1to10_f1",
        "pairs_1to10_average_precision",
        "plagdet",
        "origin_accuracy_at_1",
    ],
    "clustering": ["author_ari", "author_nmi", "author_purity", "work_ari"],
}
# Libellés lisibles des mesures dans les figures (les tableaux gardent les noms des CSV).
METRIC_LABELS = {
    "f1_macro": "F1 macro",
    "accuracy": "exactitude",
    "auc": "AUC",
    "c_at_1": "c@1",
    "pairs_1to1_f1": "F1 des paires (prévalence 1:1)",
    "pairs_1to10_f1": "F1 des paires (prévalence 1:10)",
    "plagdet": "plagdet",
    "author_ari": "ARI (référence : auteur)",
    "author_nmi": "NMI (référence : auteur)",
}

FACTORS = {
    "attribution": [
        ("param_n_train", "Textes d'entraînement par auteur"),
        ("param_n_authors", "Nombre d'auteurs"),
        ("param_text_length", "Longueur du texte à classer (mots)"),
    ],
    "plagiarism": [("param_level", "Niveau d'obfuscation"), ("param_size", "Taille du passage (mots)")],
    "clustering": [("param_dimensions", "Dimensions après SVD")],
    "verification": [],
}
SERIES = {
    "attribution": "param_representation",
    "plagiarism": "param_feature_set",
    "clustering": "param_representation",
    "verification": "param_method",
}


@dataclass
class _Document:
    """Suite d'éléments rendus en HTML ou en Markdown."""

    elements: list[tuple[str, Any]] = field(default_factory=list)

    def heading(self, text: str) -> None:
        self.elements.append(("h2", text))

    def subheading(self, text: str) -> None:
        self.elements.append(("h3", text))

    def paragraph(self, text: str) -> None:
        self.elements.append(("p", text))

    def table(self, frame: pd.DataFrame, highlight: str | None = None, numeric: bool = False) -> None:
        self.elements.append(("table", (frame, highlight, numeric)))

    def figure(self, png: str, alt: str) -> None:
        self.elements.append(("figure", (png, alt)))

    def kpis(self, values: Sequence[tuple[str, str]]) -> None:
        self.elements.append(("kpis", list(values)))

    def render_html(self, title: str, subtitle: str) -> str:
        body: list[str] = []
        open_section = False
        for kind, value in self.elements:
            if kind == "h2":
                if open_section:
                    body.append("</section>")
                body.append(f"<section><h2>{html_lib.escape(value)}</h2>")
                open_section = True
            elif kind == "h3":
                body.append(f"<h3>{html_lib.escape(value)}</h3>")
            elif kind == "p":
                body.append(f"<p>{html_lib.escape(value)}</p>")
            elif kind == "table":
                body.append(to_html(value[0], value[1], value[2]))
            elif kind == "figure":
                body.append(html.figure(*value))
            elif kind == "kpis":
                body.append(html.kpis(value))
        if open_section:
            body.append("</section>")
        return html.page(title, "".join(body), subtitle)

    def render_markdown(self, title: str, subtitle: str, assets: Path) -> str:
        lines = [f"# {title}", "", f"_{subtitle}_", ""]
        counter = 0
        for kind, value in self.elements:
            if kind == "h2":
                lines += [f"## {value}", ""]
            elif kind == "h3":
                lines += [f"### {value}", ""]
            elif kind == "p":
                lines += [value, ""]
            elif kind == "table":
                lines += [to_markdown(value[0])]
            elif kind == "figure":
                counter += 1
                name = f"figure_{counter:02d}.png"
                ensure_dir(assets)
                (assets / name).write_bytes(base64.b64decode(value[0]))
                lines += [f"![{value[1]}]({assets.name}/{name})", ""]
            elif kind == "kpis":
                lines += [
                    "| " + " | ".join(k for k, _ in value) + " |",
                    "| " + " | ".join("---" for _ in value) + " |",
                    "| " + " | ".join(v for _, v in value) + " |",
                    "",
                ]
        return "\n".join(lines) + "\n"


def _records(directory: Path) -> list[dict[str, Any]]:
    path = directory / "runs.jsonl"
    if not path.exists():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        latest[record["key"]] = record
    return list(latest.values())


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size > 1 else pd.DataFrame()


def _artifacts(directory: Path, record: dict[str, Any]) -> dict[str, Any]:
    path = record.get("artifacts")
    data: dict[str, Any] = read_json(directory / path) if path and (directory / path).exists() else {}
    return data


def _best(summary: pd.DataFrame, metric: str) -> pd.Series | None:
    column = f"{metric}_mean"
    if summary.empty or column not in summary:
        return None
    candidates = summary[~summary["baseline"].astype(bool)] if "baseline" in summary else summary
    candidates = candidates.dropna(subset=[column])
    if candidates.empty:
        return None
    # Ex æquo : on montre le réglage le plus difficile (plus d'auteurs, de niveaux…).
    keys, ascending = [column], [False]
    for extra in ("param_n_authors", "param_level", "param_size"):
        if extra in candidates:
            keys.append(extra)
            ascending.append(False)
    ordered = candidates.assign(**{k: candidates[k].astype(str).str.zfill(6) for k in keys[1:]})
    return candidates.loc[ordered.sort_values(keys, ascending=ascending, kind="stable").index[0]]


def generate_report(directory: str | Path, fmt_name: str = "html", output: str | Path | None = None) -> Path:
    """Génère le rapport d'une expérience à partir de son répertoire de résultats."""
    directory = Path(directory)
    if not directory.is_dir():
        raise PlagiaError(f"Répertoire de résultats introuvable : {directory}.")
    if not _records(directory):
        raise PlagiaError(f"Aucune exécution dans {directory} : lancez d'abord « plagia run <configuration.yaml> ».")
    config_data = read_json(directory / "config.json")["config"] if (directory / "config.json").exists() else {}
    environment = read_json(directory / "environment.json") if (directory / "environment.json").exists() else {}
    plan = read_json(directory / "plan.json") if (directory / "plan.json").exists() else {}
    task = config_data.get("task", "attribution")
    records = _records(directory)
    ok = [r for r in records if r.get("status") == "ok"]
    summary = _read_csv(directory / "summary.csv")
    results = _read_csv(directory / "results.csv")
    comparisons = _read_csv(directory / "comparisons.csv")
    primary = PRIMARY.get(task, "f1_macro")
    doc = _Document()

    doc.heading("Expérience")
    if config_data.get("description"):
        doc.paragraph(config_data["description"])
    plan_summary = plan.get("summary", {})
    doc.kpis(
        [
            ("Conditions", str(plan_summary.get("conditions", "—"))),
            ("Lignes de base", str(plan_summary.get("baseline_conditions", "—"))),
            ("Exécutions réussies", f"{len(ok)} / {plan_summary.get('runs', len(records))}"),
            ("Graines", ", ".join(map(str, plan_summary.get("seeds", []))) or "—"),
        ]
    )
    packages = environment.get("packages", {})
    trace = [
        ("plagia", environment.get("plagia")),
        ("Python", environment.get("python")),
        ("Commit Git", _commit_label(environment.get("git") or {})),
        ("Annotation", json.dumps(environment.get("annotation", {}), ensure_ascii=False)),
        *[(name, version) for name, version in packages.items() if version],
    ]
    doc.subheading("Traçabilité")
    doc.table(pd.DataFrame(trace, columns=["élément", "version"]))
    protocol = config_data.get("protocol", {})
    if task == "attribution":
        grouping = "groupés par œuvre" if protocol.get("group_by") == "work" else "stratifiés"
        folds = protocol.get("cv_folds", "—")
        tuning = f"réglage par validation croisée à {folds} plis sur l'entraînement ({grouping})"
    elif task == "plagiarism":
        tuning = "classificateur appris sur l'entraînement, seuil choisi sur la validation"
    elif task == "verification":
        tuning = "paires d'entraînement et de test tirées dans des jeux disjoints"
    else:
        tuning = "regroupement non supervisé, évaluation externe contre les étiquettes"
    doc.paragraph(
        f"Protocole : {tuning} ; test évalué une seule fois par configuration ; "
        f"{protocol.get('bootstrap', '—')} rééchantillonnages bootstrap ; "
        f"intervalles à {int(100 * protocol.get('confidence', 0.95))} %."
    )
    if any(r.get("status") == "error" for r in records):
        doc.paragraph(
            f"{sum(r.get('status') == 'error' for r in records)} exécution(s) en erreur : "
            "relancez « plagia run » pour les reprendre."
        )

    doc.heading("Résultats principaux")
    if summary.empty:
        doc.paragraph("Aucun résultat disponible.")
    else:
        params = [c for c in summary.columns if c.startswith("param_")]
        ordered = summary.sort_values(f"{primary}_mean", ascending=False) if f"{primary}_mean" in summary else summary
        table = metric_table(ordered, METRICS.get(task, []), params)
        doc.paragraph(
            "Moyenne sur les graines (intervalle de confiance à 95 % : bootstrap sur le test quand il est disponible, "
            "sinon Student sur les graines). Écarts-types dans summary.csv. Lignes de base en italique."
        )
        doc.table(table, highlight="référence", numeric=True)

    if not results.empty:
        _factor_section(doc, results, task, primary)
    best = _best(summary, primary)
    if best is not None:
        best_records = sorted((r for r in ok if r["condition"] == best["condition"]), key=lambda r: r["seed"])
        if task == "attribution":
            _attribution_details(doc, directory, best, best_records, ok)
        elif task == "plagiarism":
            _plagiarism_details(doc, directory, results, ok)
        elif task == "clustering":
            _clustering_details(doc, directory, best, best_records)
    if not comparisons.empty:
        _comparison_section(doc, comparisons)
    _limits(doc, config_data, environment)

    title = f"Rapport d'évaluation — {config_data.get('name', directory.name)}"
    subtitle = f"Tâche : {task} · généré le {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}"
    if fmt_name == "md":
        target = Path(output) if output else directory / "report.md"
        atomic_write_text(target, doc.render_markdown(title, subtitle, target.parent / f"{target.stem}_files"))
    else:
        target = Path(output) if output else directory / "report.html"
        atomic_write_text(target, doc.render_html(title, subtitle))
    return target


def _factor_section(doc: _Document, results: pd.DataFrame, task: str, metric: str) -> None:
    if metric not in results:
        return
    series_column = SERIES.get(task)
    for factor, label in FACTORS.get(task, []):
        if factor not in results or results[factor].nunique() < 2 or series_column not in results:
            continue
        data = results[~results["baseline"].astype(bool)] if "baseline" in results else results
        others = [c for c in data.columns if c.startswith("param_") and c not in {factor, series_column}]
        # Valeur la plus favorable des autres facteurs : la plus fréquente (vue en coupe).
        for column in others:
            mode = data[column].mode()
            if not mode.empty:
                data = data[data[column] == mode.iloc[0]]
        series: dict[str, list[tuple[Any, float, float, float]]] = defaultdict(list)
        rows = []
        for (name, value), group in data.groupby([series_column, factor]):
            values = group[metric].astype(float)
            mean = float(values.mean())
            half = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            series[str(name)].append((value, mean, mean - half, mean + half))
            rows.append(
                {
                    "série": name,
                    label: value,
                    metric: mean,
                    "écart-type": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "graines": len(values),
                }
            )
        if not series:
            continue
        doc.subheading(f"Effet du facteur : {label}")
        plotted = dict(series)
        if len(plotted) > len(figures.CATEGORICAL):
            # Trop de séries pour la palette : la figure garde les représentations principales,
            # les variantes d'ablation (« base:sans_x ») restent dans le tableau.
            plotted = {name: points for name, points in series.items() if ":" not in name}
        if 0 < len(plotted) <= len(figures.CATEGORICAL):
            readable = METRIC_LABELS.get(metric, metric)
            doc.figure(
                figures.factor_figure(plotted, label, readable, f"{readable} selon {label.lower()}"),
                f"{readable} selon {label}",
            )
        doc.table(pd.DataFrame(rows))


def _attribution_details(
    doc: _Document, directory: Path, best: pd.Series, records: list[dict[str, Any]], ok: list[dict[str, Any]]
) -> None:
    doc.heading("Meilleure condition : explication")
    doc.paragraph(f"Condition : {best['condition']}.")
    if not records:
        return
    first = _artifacts(directory, records[0])
    confusion = first.get("confusion")
    if confusion:
        doc.figure(figures.confusion_figure(confusion["labels"], confusion["matrix"]), "Matrice de confusion")
    contributions: dict[str, list[float]] = defaultdict(list)
    for record in records:
        for block, share in _artifacts(directory, record).get("block_contributions", {}).items():
            contributions[block].append(share)
    if len(contributions) >= 2:
        means = {block: float(np.mean(v)) for block, v in contributions.items()}
        doc.subheading("Contribution de chaque bloc (part de la masse des poids)")
        doc.figure(
            figures.bar_figure(means, "Contribution des blocs de caractéristiques", "part des poids"),
            "Contribution des blocs",
        )
        doc.table(pd.DataFrame(sorted(means.items(), key=lambda kv: -kv[1]), columns=["bloc", "part"]))
    explanation = first.get("explanation", {})
    if explanation:
        doc.subheading("Attributs les plus discriminants par auteur")
        rows = []
        for author in sorted(explanation)[:8]:
            rows.append(
                {
                    "auteur": author,
                    "attributs (poids)": " · ".join(f"{f} ({w:.2f})" for f, w in explanation[author][:8]),
                }
            )
        doc.table(pd.DataFrame(rows))
    confusions = first.get("confusions", [])
    if confusions:
        doc.subheading("Confusions les plus fréquentes")
        doc.table(pd.DataFrame(confusions, columns=["auteur réel", "auteur prédit", "nombre"]))
    errors = first.get("errors", [])
    if errors:
        doc.subheading(f"Échantillon d'erreurs à catégoriser à la main ({len(errors)})")
        doc.paragraph(
            "Catégories suggérées : "
            + " ; ".join(ERROR_CATEGORIES)
            + ". Le fichier d'artefacts JSON contient la colonne « category » à remplir."
        )
        doc.table(pd.DataFrame(errors)[["id", "true", "predicted", "excerpt"]].head(30))
    open_set = [r["metrics"] for r in ok if "open_set_f1_macro" in r.get("metrics", {})]
    if open_set:
        doc.subheading("Ensemble ouvert (auteur inconnu)")
        doc.table(
            pd.DataFrame(open_set)[["open_set_f1_macro", "open_set_unknown_recall", "open_set_threshold"]]
            .describe()
            .loc[["mean", "std"]]
            .reset_index()
        )


def _plagiarism_details(doc: _Document, directory: Path, results: pd.DataFrame, ok: list[dict[str, Any]]) -> None:
    doc.heading("Détection de plagiat : détail")
    if "plagdet" in results and "param_level" in results:
        table = (
            results.groupby(["param_level", "param_feature_set", "param_classifier"], dropna=False)[
                [
                    c
                    for c in (
                        "plagdet",
                        "pan_precision",
                        "pan_recall",
                        "pan_granularity",
                        "passage_precision",
                        "passage_recall",
                    )
                    if c in results
                ]
            ]
            .mean()
            .reset_index()
        )
        table.columns = [str(c).removeprefix("param_") for c in table.columns]
        doc.subheading("Mesures de PAN au niveau des passages (moyennes sur les tailles et les graines)")
        doc.table(table)
    levels = sorted({r["params"]["level"] for r in ok})
    if not levels:
        return
    hardest = levels[-1]
    curves: dict[str, dict[str, list[float]]] = {}
    for record in sorted(ok, key=lambda r: (str(r["params"].get("feature_set")), r["params"]["classifier"], r["seed"])):
        params = record["params"]
        name = f"{params.get('feature_set') or 'base'} / {params['classifier']}"
        if params["level"] != hardest or name in curves:
            continue
        curve = _artifacts(directory, record).get("pr_curve")
        if curve:
            curves[name] = curve
        if len(curves) >= len(figures.CATEGORICAL):
            break
    if curves:
        doc.subheading(f"Courbes précision-rappel au niveau {hardest} (prévalence 1:10)")
        doc.figure(figures.pr_figure(curves, f"Paires de passages, niveau {hardest}"), "Courbes précision-rappel")
    coefficients = []
    for record in ok:
        explanation = _artifacts(directory, record).get("explanation", {})
        for label, weights in explanation.items():
            if label in {"1", "*"}:
                coefficients += [
                    {
                        "jeu": record["params"].get("feature_set"),
                        "classificateur": record["params"]["classifier"],
                        "mesure": f,
                        "poids": w,
                    }
                    for f, w in weights
                ]
    if coefficients:
        frame = pd.DataFrame(coefficients).groupby(["jeu", "classificateur", "mesure"])["poids"].mean().reset_index()
        doc.subheading("Poids des mesures de similarité dans le classificateur de paires")
        doc.table(frame.sort_values(["jeu", "classificateur", "poids"], ascending=[True, True, False]))


def _clustering_details(doc: _Document, directory: Path, best: pd.Series, records: list[dict[str, Any]]) -> None:
    doc.heading("Meilleur regroupement : interprétation")
    doc.paragraph(f"Condition : {best['condition']}.")
    if not records:
        return
    artifacts = _artifacts(directory, records[0])
    projection = artifacts.get("projection")
    if projection:
        doc.figure(
            figures.projection_figure(
                projection["x"],
                projection["y"],
                [str(a) for a in projection["author"]],
                "Projection 2D colorée par auteur",
            ),
            "Projection par auteur",
        )
        works = [str(w) for w in projection.get("work", [])]
        if works and len(set(works)) > len(set(projection["author"])):
            doc.figure(
                figures.projection_figure(projection["x"], projection["y"], works, "Projection 2D colorée par œuvre"),
                "Projection par œuvre",
            )
        doc.figure(
            figures.projection_figure(
                projection["x"],
                projection["y"],
                [f"groupe {c}" for c in projection["cluster"]],
                "Projection 2D colorée par groupe",
            ),
            "Projection par groupe",
        )
    characteristic = artifacts.get("characteristic_features", {})
    if characteristic:
        rows = [
            {"groupe": k, "attributs caractéristiques": " · ".join(f for f, _ in v[:8])}
            for k, v in list(characteristic.items())[:12]
        ]
        doc.subheading("Attributs caractéristiques des groupes")
        doc.table(pd.DataFrame(rows))
    profiles = artifacts.get("entity_profiles", {})
    if profiles:
        rows = [
            {
                "groupe": k,
                "entités": " · ".join(f"{e} ({n})" for e, n in v.get("entities", [])),
                "types": " · ".join(f"{t} ({n})" for t, n in v.get("types", [])),
            }
            for k, v in list(profiles.items())[:12]
        ]
        doc.subheading("Entités et types les plus fréquents par groupe")
        doc.table(pd.DataFrame(rows))
    selection = artifacts.get("k_selection")
    if selection:
        doc.subheading("Choix du nombre de groupes")
        doc.table(
            pd.DataFrame({"k": selection["k"], "inertie": selection["inertia"], "silhouette": selection["silhouette"]})
        )


def _commit_label(git: dict[str, Any]) -> str | None:
    """Hash du commit, signalé quand le répertoire de travail contenait des modifications."""
    commit = git.get("commit")
    if commit and git.get("dirty"):
        return f"{commit} (avec modifications non validées)"
    return commit


def _factor_value(value: Any) -> str:
    """Valeur d'un facteur dans un tableau : entier, ou « all » (pas de restriction) si absente."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "all"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _comparison_section(doc: _Document, comparisons: pd.DataFrame) -> None:
    doc.heading("Comparaisons appariées")
    doc.paragraph(
        "Chaque condition est comparée à la référence sur les mêmes documents et les mêmes graines : test de McNemar, "
        "intervalle bootstrap de la différence de F1, p-valeurs corrigées par la méthode de Holm."
    )
    columns = [
        c
        for c in (
            "reference",
            "condition",
            "classifier",
            "n_authors",
            "n_train",
            "text_length",
            "level",
            "size",
            "n",
            "diff",
            "diff_low",
            "diff_high",
            "mcnemar_p_holm",
            "mcnemar_p_significant",
            "bootstrap_p_holm",
        )
        if c in comparisons
    ]
    frame = comparisons[columns].copy()
    for column in ("n_authors", "n_train", "text_length", "size"):
        if column in frame:
            frame[column] = frame[column].map(_factor_value)
    for column in ("diff", "diff_low", "diff_high", "mcnemar_p_holm", "bootstrap_p_holm"):
        if column in frame:
            frame[column] = frame[column].map(lambda v: fmt(float(v), 4))
    doc.table(frame)


def _limits(doc: _Document, config: dict[str, Any], environment: dict[str, Any]) -> None:
    doc.heading("Limites et précautions")
    doc.paragraph(
        "Un score de similarité n'est pas une preuve de plagiat : citations, formules et textes standardisés "
        "produisent des correspondances légitimes ; l'outil propose des passages candidats et ne rend aucun verdict."
    )
    engine = (config.get("annotation") or {}).get("engine")
    if engine in {"heuristic", "none"}:
        doc.paragraph(
            "Annotations produites par l'annotateur heuristique (ou absentes) : les résultats sur les entités et les "
            "étiquettes grammaticales sont indicatifs ; utilisez spaCy pour l'étude."
        )
    section = config.get(config.get("task", ""), {}) or {}
    corpus = str(section.get("corpus", "")) + " ".join(
        str(c.get("path", "")) for c in (section.get("corpora") or {}).values()
    )
    if "toy" in corpus:
        doc.paragraph(
            "Corpus jouet généré hors ligne : il valide la chaîne de traitement, pas les hypothèses scientifiques. "
            "Les conclusions de l'étude reposent sur C50, le corpus littéraire et PAN."
        )
