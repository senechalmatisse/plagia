"""Architecture en couches : chaque paquet n'importe que les couches du dessous."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "plagia"

ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "corpus": {"core"},
    "annotation": {"core", "corpus"},
    "features": {"core", "corpus", "annotation"},
    "similarity": {"core", "corpus", "annotation", "features"},
    "models": {"core", "corpus", "annotation", "features"},
    "tasks": {"core", "corpus", "annotation", "features", "similarity", "models"},
    "evaluation": {"core"},
    "experiments": {"core", "corpus", "annotation", "features", "similarity", "models", "tasks", "evaluation"},
    "reporting": {"core", "corpus", "annotation", "features", "similarity", "models", "tasks", "evaluation"},
    "cli": {
        "core",
        "corpus",
        "annotation",
        "features",
        "similarity",
        "models",
        "tasks",
        "evaluation",
        "experiments",
        "reporting",
    },
}


def _imported_layers(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    layers: set[str] = set()
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if parts[0] == "plagia" and len(parts) > 1:
                layers.add(parts[1])
    return layers


def test_layers_only_depend_on_lower_layers() -> None:
    violations = []
    for layer, allowed in ALLOWED.items():
        for path in sorted((PACKAGE / layer).rglob("*.py")):
            for imported in _imported_layers(path) - allowed - {layer}:
                violations.append(f"{path.relative_to(PACKAGE)} → {imported}")
    assert not violations, "Dépendances interdites :\n" + "\n".join(violations)


def test_every_layer_is_a_package() -> None:
    assert {p.name for p in PACKAGE.iterdir() if p.is_dir() and (p / "__init__.py").exists()} == set(ALLOWED)
