"""Application Streamlit du démonstrateur, lancée par `plagia demo --serve`.

`streamlit run` exécute ce fichier comme un script ; ses options suivent `--` (`--model`, `--sources`, `--engine`, `--cache`, `--top-k`).
Toute la logique est dans :class:`plagia.reporting.demo.DemoAnalyzer` : ce module ne fait que l'interface.
Dépendance facultative : `uv sync --extra demo`.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from plagia.cli.main import _load_sources
from plagia.core.errors import PlagiaError
from plagia.reporting.demo import DemoAnalyzer


def _options(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="plagia demo --serve")
    parser.add_argument("--model", default=None)
    parser.add_argument("--sources", default=None)
    parser.add_argument("--engine", default=None)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--top-k", dest="top_k", type=int, default=5)
    return parser.parse_args(argv)


@st.cache_resource(show_spinner="Chargement du modèle et indexation des sources…")
def _analyzer(
    model: str | None, sources: str | None, engine: str | None, cache: str | None, top_k: int
) -> DemoAnalyzer:
    documents = _load_sources(sources) if sources else []
    return DemoAnalyzer.from_paths(model, documents, engine=engine, cache_path=cache, top_k=top_k)


def main() -> None:
    """Page unique : saisie, auteurs les plus probables, passages repris surlignés."""
    options = _options(sys.argv[1:])
    st.set_page_config(page_title="plagia — démonstrateur", layout="wide")
    st.title("Démonstrateur plagia")
    st.caption(
        "Attribution d'auteur et détection de passages repris. Un score de similarité n'est pas une "
        "preuve de plagiat : examinez chaque passage et sa source."
    )
    try:
        analyzer = _analyzer(options.model, options.sources, options.engine, options.cache, options.top_k)
    except (PlagiaError, OSError) as exc:
        st.error(f"Impossible de préparer le démonstrateur : {exc}")
        st.stop()
    uploaded = st.file_uploader("Fichier texte (UTF-8)", type=["txt"])
    default = uploaded.read().decode("utf-8") if uploaded is not None else ""
    text = st.text_area("Texte à analyser", value=default, height=260)
    if not st.button("Analyser", type="primary") or not text.strip():
        return
    with st.spinner("Analyse…"):
        result = analyzer.analyze(text)
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Auteurs les plus probables")
        if result.ranking:
            frame = pd.DataFrame(result.ranking, columns=["Auteur", "Score"])
            frame.index = range(1, len(frame) + 1)
            st.dataframe(frame, use_container_width=True)
        else:
            st.info("Aucun modèle d'attribution chargé (--model).")
    with right:
        st.subheader("Passages repris")
        if result.report_html is not None:
            components.html(result.report_html, height=760, scrolling=True)
        else:
            st.info("Aucune collection de sources chargée (--sources).")


main()
