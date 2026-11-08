"""Démonstrateur : auteurs les plus probables et passages repris d'un texte saisi.

Toute la logique est ici, sans interface : `plagia demo --text` en tire une page HTML autonome et l'application Streamlit (`plagia demo --serve`) appelle :meth:`DemoAnalyzer.analyze` à chaque saisie.
Aucun verdict n'est rendu : la page montre des scores, des passages et leurs sources, à examiner par un lecteur.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from plagia.annotation.cache import AnnotationCache
from plagia.annotation.document import AnnotatedDocument
from plagia.annotation.pipeline import annotate_documents
from plagia.core.config import AnnotationConfig
from plagia.core.types import Document
from plagia.reporting.html import kpis, page, render_similarity_report, section
from plagia.tasks.attribution import AuthorshipAttributor
from plagia.tasks.plagiarism import (
    CandidateRetriever,
    Extender,
    ExtrinsicDetector,
    Filter,
    SeedMatcher,
    SimilarityReport,
    TextAligner,
    build_report,
)


@dataclass
class DemoResult:
    """Résultat de l'analyse d'un texte : classement des auteurs et rapport de similarité."""

    name: str
    text: str
    ranking: list[tuple[str, float]] = field(default_factory=list)
    report: SimilarityReport | None = None
    report_html: str | None = None

    def ranking_html(self) -> str:
        """Table des auteurs les plus probables (score de confiance du modèle)."""
        if not self.ranking:
            return "<p class='muted'>Aucun modèle d'attribution chargé.</p>"
        rows = "".join(
            f"<tr><td>{rank}</td><td>{html.escape(author)}</td><td>{score:.3f}</td></tr>"
            for rank, (author, score) in enumerate(self.ranking, start=1)
        )
        return (
            "<div class='table'><table class='numeric'><thead><tr><th>Rang</th><th>Auteur</th>"
            f"<th>Score</th></tr></thead><tbody>{rows}</tbody></table></div>"
            "<p class='muted'>Score : confiance du modèle parmi les auteurs connus de l'entraînement "
            "(un texte d'un autre auteur reçoit quand même un classement).</p>"
        )

    def to_html(self) -> str:
        """Page autonome : auteurs probables puis passages repris surlignés."""
        values = [("Mots", str(len(self.text.split())))]
        if self.ranking:
            values.append(("Auteur le plus probable", self.ranking[0][0]))
        body = kpis(values) + section("Auteurs les plus probables", self.ranking_html())
        if self.report is not None:
            body += f"<h2>Passages repris</h2>{self._report_body()}"
        else:
            body += section("Passages repris", "<p class='muted'>Aucune collection de sources chargée.</p>")
        return page(f"Démonstrateur — {self.name}", body, "Attribution d'auteur et détection de passages repris")

    def _report_body(self) -> str:
        """Corps du rapport de similarité (sans son propre en-tête de page)."""
        assert self.report_html is not None
        start = self.report_html.index("<div class='kpis'>")
        end = self.report_html.rindex("</main>")
        return self.report_html[start:end]


class DemoAnalyzer:
    """Analyse un texte avec un modèle d'attribution enregistré et une collection de sources.

    Les deux volets sont facultatifs mais l'un au moins est requis. Les sources sont
    annotées et indexées une fois, à la construction ; chaque appel à :meth:`analyze`
    n'annote que le texte saisi (cache SQLite consulté si un moteur est choisi).
    """

    def __init__(
        self,
        model: AuthorshipAttributor | None = None,
        sources: Sequence[Document] = (),
        engine: str | None = None,
        cache_path: str | Path | None = None,
        top_k: int = 5,
        retrieval: str = "tfidf",
        candidates: int = 10,
        ngram_size: int = 4,
        min_density: float = 1.0,
        min_length: int = 150,
    ) -> None:
        if model is None and not sources:
            raise ValueError("Le démonstrateur demande un modèle d'attribution, des sources, ou les deux.")
        self.model = model
        self.engine = engine or getattr(model, "engine", "none")
        self.cache_path = Path(cache_path) if cache_path else Path("data/cache/annotations.sqlite")
        self.top_k = top_k
        self.sources = {d.id: d.text for d in sources}
        self.detector: ExtrinsicDetector | None = None
        if sources:
            aligner = TextAligner(
                SeedMatcher(ngram_size=ngram_size), Extender(min_density=min_density), Filter(min_length)
            )
            retriever = CandidateRetriever(retrieval, k=min(candidates, len(sources)))
            self.detector = ExtrinsicDetector(retriever, aligner).fit(self._annotate(list(sources)))

    @classmethod
    def from_paths(
        cls, model_path: str | Path | None, sources: Sequence[Document] = (), **options: object
    ) -> DemoAnalyzer:
        """Construit l'analyseur à partir d'un modèle enregistré (`plagia attribute --save`)."""
        model = AuthorshipAttributor.load(model_path) if model_path else None
        return cls(model, sources, **options)  # type: ignore[arg-type]

    def _annotate(self, documents: list[Document]) -> list[AnnotatedDocument]:
        config = AnnotationConfig.model_validate({"engine": self.engine, "cache_path": str(self.cache_path)})
        cache = AnnotationCache(config.cache_path) if self.engine != "none" else None
        try:
            return annotate_documents(documents, config, cache=cache)
        finally:
            if cache is not None:
                cache.close()

    def analyze(self, text: str, name: str = "texte saisi") -> DemoResult:
        """Classement des auteurs (si modèle) et passages repris avec leurs sources (si sources)."""
        if not text.strip():
            raise ValueError("Texte vide.")
        document = self._annotate([Document("demo", text)])[0]
        result = DemoResult(name, text)
        if self.model is not None:
            result.ranking = self.model.rank([document], self.top_k)[0]
        if self.detector is not None:
            report = build_report(name, text, self.detector.detect(document))
            result.report = report
            result.report_html = render_similarity_report(report, text, self.sources)
        return result
