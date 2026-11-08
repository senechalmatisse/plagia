"""Couche `experiments` : plan, exécution parallèle avec reprise, stockage des résultats."""

from plagia.experiments.context import ExperimentContext, get_context
from plagia.experiments.plan import Condition, Plan, Run
from plagia.experiments.result import RunResult
from plagia.experiments.runner import Runner, execute
from plagia.experiments.store import ResultStore

__all__ = [
    "Condition",
    "ExperimentContext",
    "Plan",
    "ResultStore",
    "Run",
    "RunResult",
    "Runner",
    "execute",
    "get_context",
]
