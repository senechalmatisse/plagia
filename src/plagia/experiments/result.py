"""Résultat d'une exécution : métriques (avec intervalles), durée, prédictions, artefacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RunResult:
    """Résultat d'une exécution (condition × graine)."""

    key: str
    condition: str
    seed: int
    params: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0
    status: str = "ok"
    error: str | None = None
    baseline: bool = False
    predictions: list[dict[str, Any]] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        """Enregistrement JSON (sans prédictions ni artefacts, stockés à part)."""
        return {
            "key": self.key,
            "condition": self.condition,
            "seed": self.seed,
            "params": self.params,
            "metrics": self.metrics,
            "duration": self.duration,
            "status": self.status,
            "error": self.error,
            "baseline": self.baseline,
        }


def add_ci(metrics: dict[str, Any], name: str, interval: dict[str, float]) -> None:
    """Ajoute `name`, `name_low` et `name_high` aux métriques."""
    metrics[name] = interval["value"]
    metrics[f"{name}_low"] = interval["low"]
    metrics[f"{name}_high"] = interval["high"]
