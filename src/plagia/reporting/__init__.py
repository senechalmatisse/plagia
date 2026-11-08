"""Couche `reporting` : tableaux, figures et rapports HTML ou Markdown."""

from plagia.reporting.html import render_similarity_report
from plagia.reporting.report import generate_report

__all__ = ["generate_report", "render_similarity_report"]
