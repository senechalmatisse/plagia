"""Figures des rapports (matplotlib, rendu sans affichage).

Palette de référence validée (ordre catégoriel fixe, jamais cyclé ; rampe bleue pour les grandeurs ; encres et surfaces dédiées).
Les graphiques en nuage n'utilisent que les trois premières teintes (validées toutes paires) et regroupent le reste dans « autres ».
Chaque figure est accompagnée dans le rapport d'un tableau des valeurs tracées.
"""

from __future__ import annotations

import base64
import io
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SEQUENTIAL = ("#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b")
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
OTHER = "#c3c2b7"

STYLE: Any = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False,
    "legend.labelcolor": INK_SECONDARY,
    "lines.linewidth": 2.0,
}


def to_base64(figure: Any) -> str:
    """Image PNG encodée en base 64 (incluse dans le HTML)."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(figure)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def series_colors(names: Sequence[str]) -> dict[str, str]:
    """Couleur fixe par série dans l'ordre d'apparition (au plus 8 séries)."""
    if len(names) > len(CATEGORICAL):
        raise ValueError("Plus de 8 séries : regroupez-les ou utilisez des petits multiples.")
    return {name: CATEGORICAL[i] for i, name in enumerate(names)}


def confusion_figure(
    labels: Sequence[str], matrix: Sequence[Sequence[int]], title: str = "Matrice de confusion (normalisée par ligne)"
) -> str:
    """Matrice de confusion (rampe séquentielle bleue, valeurs affichées jusqu'à 12 classes)."""
    data = np.asarray(matrix, dtype=float)
    totals = data.sum(axis=1, keepdims=True)
    normalized = np.divide(data, totals, out=np.zeros_like(data), where=totals > 0)
    n = len(labels)
    size = min(10.0, 2.5 + 0.28 * n)
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(size, size * 0.9))
        axis.grid(False)
        cmap = LinearSegmentedColormap.from_list("plagia-seq", SEQUENTIAL)
        image = axis.imshow(normalized, cmap=cmap, vmin=0, vmax=1)
        axis.set_xticks(range(n), labels=labels, rotation=90 if n > 6 else 0, fontsize=8 if n > 12 else 9)
        axis.set_yticks(range(n), labels=labels, fontsize=8 if n > 12 else 9)
        axis.set_xlabel("Auteur prédit")
        axis.set_ylabel("Auteur réel")
        axis.set_title(title, loc="left")
        if n <= 12:
            for i in range(n):
                for j in range(n):
                    if data[i, j]:
                        axis.text(
                            j,
                            i,
                            f"{int(data[i, j])}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="#ffffff" if normalized[i, j] > 0.55 else INK,
                        )
        bar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        bar.outline.set_visible(False)
        bar.ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY)
        return to_base64(figure)


def pr_figure(curves: Mapping[str, Mapping[str, Sequence[float]]], title: str = "Courbes précision-rappel") -> str:
    """Courbes précision-rappel (une couleur fixe par série, légende)."""
    names = list(curves)[: len(CATEGORICAL)]
    colors = series_colors(names)
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(6.4, 4.4))
        for name in names:
            curve = curves[name]
            axis.step(curve["recall"], curve["precision"], where="post", color=colors[name], label=name)
        axis.set_xlim(0, 1.01)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("Rappel")
        axis.set_ylabel("Précision")
        axis.set_title(title, loc="left")
        if len(names) > 1:
            axis.legend(loc="lower left", fontsize=8)
        return to_base64(figure)


def factor_figure(
    series: Mapping[str, Sequence[tuple[Any, float, float, float]]],
    x_label: str,
    y_label: str,
    title: str,
) -> str:
    """Effet d'un facteur : une ligne par série, points (≥ 8 px) et intervalle en bande.

    `series[nom] = [(x, moyenne, borne basse, borne haute), …]`.
    """
    names = list(series)[: len(CATEGORICAL)]
    means = [mean for name in names for _, mean, *_ in series[name]]
    bounded = bool(means) and min(means) >= -1e-9 and max(means) <= 1 + 1e-9
    if bounded:  # mesure bornée (F1, exactitude, ARI…) : intervalles ramenés dans [0, 1]
        series = {name: [(x, m, max(0.0, low), min(1.0, high)) for x, m, low, high in series[name]] for name in names}
    colors = series_colors(names)
    categories = sorted({x for name in names for x, *_ in series[name]}, key=lambda v: (isinstance(v, str), v))
    position = {x: i for i, x in enumerate(categories)}
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(6.6, 4.2))
        for name in names:
            points = sorted(series[name], key=lambda p: position[p[0]])
            xs = [position[p[0]] for p in points]
            means = [p[1] for p in points]
            axis.fill_between(
                xs, [p[2] for p in points], [p[3] for p in points], color=colors[name], alpha=0.14, linewidth=0
            )
            axis.plot(
                xs,
                means,
                color=colors[name],
                marker="o",
                markersize=6,
                markeredgecolor=SURFACE,
                markeredgewidth=1.5,
                label=name,
            )
        axis.set_xticks(range(len(categories)), labels=[str(c) for c in categories])
        if bounded:
            axis.set_ylim(0, 1.02)  # mesure bornée : échelle complète, aucun écart exagéré
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(title, loc="left")
        if len(names) > 1:
            axis.legend(fontsize=8, loc="best")
        return to_base64(figure)


def bar_figure(values: Mapping[str, float], title: str, x_label: str = "") -> str:
    """Barres horizontales d'une seule série (valeurs affichées en bout de barre)."""
    items = sorted(values.items(), key=lambda kv: kv[1])
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(6.4, 0.45 * len(items) + 1.2))
        axis.grid(axis="y", visible=False)
        axis.set_axisbelow(True)
        positions = range(len(items))
        axis.barh(positions, [v for _, v in items], color=CATEGORICAL[0], height=0.6)
        axis.set_yticks(list(positions), labels=[k for k, _ in items])
        for i, (_, value) in enumerate(items):
            axis.text(value, i, f" {value:.2f}", va="center", fontsize=8, color=INK_SECONDARY)
        axis.set_xlabel(x_label)
        axis.set_title(title, loc="left")
        return to_base64(figure)


def projection_figure(
    x: Sequence[float], y: Sequence[float], categories: Sequence[str], title: str, focus: int = 3
) -> str:
    """Projection 2D : les `focus` catégories les plus fréquentes en couleur, les autres en gris."""
    counts = Counter(categories)
    highlighted = [c for c, _ in counts.most_common(focus)]
    colors = series_colors(highlighted)
    with plt.rc_context(STYLE):
        figure, axis = plt.subplots(figsize=(6.0, 5.0))
        axis.grid(False)
        xs, ys = np.asarray(x), np.asarray(y)
        cats = np.asarray(categories)
        others = ~np.isin(cats, highlighted)
        if others.any():
            axis.scatter(
                xs[others],
                ys[others],
                s=16,
                color=OTHER,
                edgecolors=SURFACE,
                linewidths=0.6,
                label=f"autres ({len(counts) - len(highlighted)})",
            )
        for name in highlighted:
            mask = cats == name
            axis.scatter(xs[mask], ys[mask], s=22, color=colors[name], edgecolors=SURFACE, linewidths=0.8, label=name)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(title, loc="left")
        axis.legend(fontsize=8, loc="best")
        return to_base64(figure)
