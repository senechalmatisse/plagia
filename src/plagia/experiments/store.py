"""Stockage des résultats : une ligne par exécution.

Disposition du répertoire d'une expérience :

* `config.yaml` et `config.json` : configuration archivée ; `plan.json` : plan ;
  `environment.json` : versions (bibliothèques, modèles, bases, hash Git) ;
* `runs.jsonl` : journal d'exécutions (écriture incrémentale, reprise) ;
* `results.csv` (et `results.parquet` si pyarrow est installé) : une ligne par
  exécution — condition, graine, paramètres, métriques, durée, version du code ;
* `predictions/<exécution>.jsonl` et `artifacts/<exécution>.json`.
"""

from __future__ import annotations

import contextlib
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from plagia.core.config import ExperimentConfig
from plagia.core.hashing import sha256_text
from plagia.core.io import append_jsonl, ensure_dir, read_json, read_jsonl, write_json, write_jsonl, write_yaml
from plagia.experiments.result import RunResult


def safe_name(key: str) -> str:
    """Nom de fichier sûr et unique dérivé d'une clé d'exécution."""
    slug = re.sub(r"[^A-Za-z0-9_.=-]+", "_", key).strip("_")[:120]
    return f"{slug}-{sha256_text(key)[:8]}"


class ResultStore:
    """Journal des exécutions et fichiers dérivés d'une expérience."""

    RUNS = "runs.jsonl"
    RESULTS = "results.csv"
    SUMMARY = "summary.csv"

    def __init__(self, directory: str | Path) -> None:
        self.directory = ensure_dir(directory)

    @property
    def runs_path(self) -> Path:
        """Chemin du journal `runs.jsonl`."""
        return self.directory / self.RUNS

    def archive(
        self,
        config: ExperimentConfig,
        plan_summary: dict[str, Any],
        conditions: list[dict[str, Any]],
        environment: dict[str, Any],
    ) -> None:
        """Archive configuration, plan et environnement."""
        data = config.model_dump(mode="json")
        write_yaml(self.directory / "config.yaml", data)
        write_json(self.directory / "config.json", {"fingerprint": config.fingerprint(), "config": data})
        write_json(self.directory / "plan.json", {"summary": plan_summary, "conditions": conditions})
        write_json(self.directory / "environment.json", environment)

    def append(self, result: RunResult, experiment: str, task: str, code_version: str) -> None:
        """Ajoute une exécution au journal et écrit ses prédictions et artefacts."""
        record = result.record()
        record.update(
            {
                "experiment": experiment,
                "task": task,
                "code_version": code_version,
                "finished": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        name = safe_name(result.key)
        if result.predictions:
            write_jsonl(self.directory / "predictions" / f"{name}.jsonl", result.predictions)
            record["predictions"] = f"predictions/{name}.jsonl"
        if result.artifacts:
            write_json(self.directory / "artifacts" / f"{name}.json", result.artifacts)
            record["artifacts"] = f"artifacts/{name}.json"
        append_jsonl(self.runs_path, record)

    def records(self, status: str | None = "ok") -> list[dict[str, Any]]:
        """Dernier enregistrement de chaque exécution (filtré par statut)."""
        if not self.runs_path.exists():
            return []
        latest: dict[str, dict[str, Any]] = {}
        for record in read_jsonl(self.runs_path):
            latest[record["key"]] = record
        return [r for r in latest.values() if status is None or r.get("status") == status]

    def completed(self) -> set[str]:
        """Clés des exécutions terminées avec succès (reprise)."""
        return {r["key"] for r in self.records("ok")}

    def predictions(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        """Prédictions d'une exécution."""
        path = record.get("predictions")
        return list(read_jsonl(self.directory / path)) if path else []

    def artifacts(self, record: dict[str, Any]) -> dict[str, Any]:
        """Artefacts d'une exécution."""
        path = record.get("artifacts")
        data: dict[str, Any] = read_json(self.directory / path) if path else {}
        return data

    def dataframe(self, status: str | None = "ok") -> pd.DataFrame:
        """Une ligne par exécution : paramètres (`param_*`), métriques, durée, version."""
        rows = []
        for record in self.records(status):
            row: dict[str, Any] = {
                "experiment": record.get("experiment"),
                "task": record.get("task"),
                "condition": record["condition"],
                "seed": record["seed"],
                "baseline": record.get("baseline", False),
                "status": record.get("status"),
                "duration_s": round(float(record.get("duration", 0.0)), 3),
                "code_version": record.get("code_version"),
            }
            row.update({f"param_{k}": ("all" if v is None else v) for k, v in record.get("params", {}).items()})
            row.update({k: v for k, v in record.get("metrics", {}).items() if not isinstance(v, (dict, list))})
            rows.append(row)
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values(["condition", "seed"], kind="stable").reset_index(drop=True)
        return frame

    def write_csv(self, parquet: bool = True) -> Path:
        """Écrit `results.csv` (et `results.parquet` si possible)."""
        frame = self.dataframe()
        path = self.directory / self.RESULTS
        frame.to_csv(path, index=False, lineterminator="\n")
        if parquet:
            with contextlib.suppress(ImportError, ValueError):
                frame.to_parquet(self.directory / "results.parquet", index=False)
        return path

    def write_table(self, name: str, rows: list[dict[str, Any]]) -> Path:
        """Écrit une table auxiliaire (comparaisons, résumé…) au format CSV."""
        path = self.directory / name
        pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")
        return path
