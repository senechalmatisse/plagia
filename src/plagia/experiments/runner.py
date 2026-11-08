"""Exécution d'une campagne : parallèle, avec reprise et écriture incrémentale.

1. le plan est développé et archivé avec la configuration et l'environnement ;
2. les annotations de tous les corpus sont calculées une fois (ou lues dans le cache) ;
3. chaque exécution (condition × graine) tourne dans un processus de travail, en mode hors ligne ; son résultat est ajouté au journal dès qu'il est disponible ;
4. les résultats, le résumé et les comparaisons appariées sont écrits en CSV.

Une exécution en erreur est journalisée sans interrompre la campagne et sera rejouée à la reprise suivante.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from plagia.annotation.pipeline import annotation_versions
from plagia.core.config import ExperimentConfig
from plagia.core.errors import CorpusNotFoundError
from plagia.core.logging import get_logger
from plagia.core.progress import progress
from plagia.core.versions import code_version, environment_info
from plagia.experiments.attribution import run_attribution
from plagia.experiments.clustering import run_clustering
from plagia.experiments.comparisons import comparison_rows, summary_rows
from plagia.experiments.context import ExperimentContext, corpus_paths, get_context
from plagia.experiments.plagiarism import run_plagiarism
from plagia.experiments.plan import Plan, Run
from plagia.experiments.result import RunResult
from plagia.experiments.store import ResultStore
from plagia.experiments.verification import run_verification

log = get_logger(__name__)

RUNNERS: dict[str, Callable[[ExperimentContext, Any, int], RunResult]] = {
    "attribution": run_attribution,
    "verification": run_verification,
    "plagiarism": run_plagiarism,
    "clustering": run_clustering,
}


def execute(config_data: dict[str, Any], run: Run) -> RunResult:
    """Exécute une condition avec une graine (fonction appelée dans les processus de travail)."""
    config = ExperimentConfig.model_validate(config_data)
    context = get_context(config)
    start = time.perf_counter()
    try:
        result = RUNNERS[config.task](context, run.condition, run.seed)
        result.key = run.key
    except Exception:
        result = RunResult(
            run.key,
            run.condition.id,
            run.seed,
            dict(run.condition.params),
            status="error",
            error=traceback.format_exc(limit=25),
            baseline=run.condition.baseline,
        )
    result.duration = time.perf_counter() - start
    return result


class Runner:
    """Orchestration d'une expérience décrite par une configuration."""

    def __init__(
        self, config: ExperimentConfig, n_jobs: int | None = None, resume: bool = True, limit: int | None = None
    ) -> None:
        self.config = config
        self.n_jobs = n_jobs or config.n_jobs
        self.resume = resume
        self.limit = limit
        self.plan = Plan.from_config(config)
        for path in corpus_paths(config):
            if not (Path(path) / "documents.jsonl").exists():
                raise CorpusNotFoundError(path)
        self.store = ResultStore(config.run_dir)

    def pending(self) -> list[Run]:
        """Exécutions restant à faire."""
        done = self.store.completed() if self.resume else set()
        todo = [r for r in self.plan.runs if r.key not in done]
        return todo[: self.limit] if self.limit else todo

    def run(self) -> ResultStore:
        """Exécute les exécutions en attente puis écrit les tables de synthèse."""
        config = self.config
        environment = environment_info(
            {"annotation": annotation_versions(config.annotation), "config_fingerprint": config.fingerprint()}
        )
        conditions = [
            {"id": c.id, "params": dict(c.params), "seeds": list(c.seeds), "baseline": c.baseline}
            for c in self.plan.conditions
        ]
        self.store.archive(config, self.plan.summary(), conditions, environment)
        todo = self.pending()
        log.info("Plan : %s ; %d exécution(s) à faire.", self.plan.summary(), len(todo))
        if todo:
            context = get_context(config)
            context.prepare()
            version = code_version()
            data = config.model_dump(mode="python")
            if self.n_jobs == 1:
                results = (execute(data, run) for run in todo)
            else:
                from joblib import Parallel, delayed

                results = Parallel(n_jobs=self.n_jobs, return_as="generator_unordered")(
                    delayed(execute)(data, run) for run in todo
                )
            failures = 0
            for result in progress(results, desc=config.name, total=len(todo)):
                self.store.append(result, config.name, config.task, version)
                if result.status != "ok":
                    failures += 1
                    log.error("Échec de %s :\n%s", result.key, result.error)
            if failures:
                log.warning("%d exécution(s) en erreur (rejouées à la prochaine reprise).", failures)
        self.finalize()
        return self.store

    def finalize(self) -> None:
        """Écrit `results.csv`, `summary.csv` et `comparisons.csv`."""
        self.store.write_csv()
        frame = self.store.dataframe()
        self.store.write_table(
            ResultStore.SUMMARY, summary_rows(frame, self.config.task, self.config.protocol.confidence)
        )
        rows = comparison_rows(self.store, self.config)
        if rows:
            self.store.write_table("comparisons.csv", rows)
