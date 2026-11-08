"""Tableaux des rapports : mise en forme des métriques avec intervalles, export HTML et Markdown."""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from typing import Any

import pandas as pd


def fmt(value: Any, digits: int = 3) -> str:
    """Nombre arrondi, ou texte tel quel."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "oui" if value else "non"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        return f"{value:.{digits}f}"
    return str(value)


def with_interval(mean: float, low: float | None, high: float | None, digits: int = 3) -> str:
    """`0.812 (0.790–0.834)` : moyenne et intervalle de confiance."""
    if low is None or high is None or any(isinstance(v, float) and math.isnan(v) for v in (low, high)):
        return fmt(mean, digits)
    return f"{fmt(mean, digits)} ({fmt(low, digits)}–{fmt(high, digits)})"


def to_html(frame: pd.DataFrame, highlight_column: str | None = None, numeric: bool = False) -> str:
    """Table HTML simple (lignes de base en italique ; `numeric` : cellules sans retour à la ligne)."""
    if frame.empty:
        return "<p class='muted'>Aucune donnée.</p>"
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    rows = []
    for _, row in frame.iterrows():
        css = " class='baseline'" if highlight_column and bool(row.get(highlight_column)) else ""
        cells = "".join(f"<td>{html.escape(fmt(v))}</td>" for v in row.tolist())
        rows.append(f"<tr{css}>{cells}</tr>")
    css_class = " class='numeric'" if numeric else ""
    body = "".join(rows)
    return f"<div class='table'><table{css_class}><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def to_markdown(frame: pd.DataFrame) -> str:
    """Table Markdown (barres verticales échappées)."""
    if frame.empty:
        return "_Aucune donnée._\n"

    def cell(value: Any) -> str:
        return fmt(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(cell(c) for c in frame.columns) + " |"
    separator = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, separator, *body]) + "\n"


def metric_table(summary: pd.DataFrame, metrics: Sequence[str], params: Sequence[str]) -> pd.DataFrame:
    """Table « condition × métriques » : moyenne (intervalle de confiance bootstrap ou Student)."""
    columns: dict[str, list[Any]] = {p.removeprefix("param_"): summary[p].tolist() for p in params if p in summary}
    columns["graines"] = summary["seeds"].tolist()
    for metric in metrics:
        if f"{metric}_mean" not in summary:
            continue
        low_col = f"{metric}_boot_low" if f"{metric}_boot_low" in summary else f"{metric}_ci_low"
        high_col = f"{metric}_boot_high" if f"{metric}_boot_high" in summary else f"{metric}_ci_high"
        columns[metric] = [
            with_interval(m, lo, hi)
            for m, lo, hi in zip(summary[f"{metric}_mean"], summary[low_col], summary[high_col], strict=True)
        ]
    columns["référence"] = ["ligne de base" if b else "" for b in summary["baseline"]]
    return pd.DataFrame(columns)
