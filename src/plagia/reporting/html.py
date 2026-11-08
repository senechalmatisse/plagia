"""Gabarit HTML des rapports et rendu du rapport de similarité d'un document."""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence

from plagia.reporting.figures import CATEGORICAL
from plagia.tasks.plagiarism.report import SimilarityReport

CSS = """
:root { color-scheme: light; --surface:#fcfcfb; --page:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e;
  --muted:#898781; --grid:#e1e0d9; --border:rgba(11,11,11,0.10); }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; line-height:1.5; }
main { max-width: 1100px; margin: 0 auto; padding: 32px 24px 64px; }
h1 { font-size: 1.7rem; margin: 0 0 4px; } h2 { font-size: 1.25rem; margin-top: 2.2rem; }
h3 { font-size: 1.02rem; margin-top: 1.4rem; color: var(--ink2); }
p, li { color: var(--ink2); } .muted { color: var(--muted); }
section { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 18px 22px; margin: 18px 0; }
.table { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: .86rem; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--ink2); font-weight: 600; background: var(--page); position: sticky; top: 0; }
tr.baseline td { font-style: italic; color: var(--ink2); }
table.numeric td { white-space: nowrap; }
img { max-width: 100%; height: auto; display: block; margin: 8px 0; }
.kpis { display: flex; flex-wrap: wrap; gap: 14px; }
.kpi { border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; min-width: 150px;
  background: var(--surface); }
.kpi .value { font-size: 1.4rem; color: var(--ink); } .kpi .label { font-size: .8rem; color: var(--muted); }
mark { border-radius: 3px; padding: 0 1px; color: var(--ink); }
.text { white-space: pre-wrap; background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px; font-size: .92rem; }
.legend span { display: inline-block; margin-right: 14px; font-size: .85rem; color: var(--ink2); }
.swatch { display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:-1px; }
.warning { border-left: 4px solid var(--muted); padding-left: 12px; }
"""


def page(title: str, body: str, subtitle: str = "") -> str:
    """Document HTML autonome."""
    sub = f"<p class='muted'>{html.escape(subtitle)}</p>" if subtitle else ""
    return (
        "<!DOCTYPE html>\n<html lang='fr'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
        f"<body><main><h1>{html.escape(title)}</h1>{sub}{body}</main></body></html>\n"
    )


def section(title: str, content: str) -> str:
    """Section titrée."""
    return f"<section><h2>{html.escape(title)}</h2>{content}</section>"


def figure(png_base64: str, alt: str) -> str:
    """Figure PNG intégrée."""
    return f"<img alt='{html.escape(alt)}' src='data:image/png;base64,{png_base64}'>"


def kpis(values: Sequence[tuple[str, str]]) -> str:
    """Bandeau de valeurs clés."""
    items = "".join(
        f"<div class='kpi'><div class='value'>{html.escape(v)}</div><div class='label'>{html.escape(k)}</div></div>"
        for k, v in values
    )
    return f"<div class='kpis'>{items}</div>"


def _tint(hex_color: str, alpha: float = 0.28) -> str:
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def render_similarity_report(
    report: SimilarityReport, text: str, sources: Mapping[str, str], title: str | None = None
) -> str:
    """Rapport HTML : score global, couverture, sources, texte surligné et passages alignés.

    Chaque source reçoit une couleur fixe (8 au plus, les suivantes regroupées) ; le
    surlignage porte aussi l'identifiant de la source en infobulle (pas de couleur seule).
    """
    ranked = sorted(report.sources, key=lambda s: -report.sources[s]["characters"])
    colors = {s: CATEGORICAL[i] if i < 7 else "#c3c2b7" for i, s in enumerate(ranked)}
    pieces, cursor = [], 0
    for det in sorted(report.detections, key=lambda d: d.suspicious.start):
        start, end = max(det.suspicious.start, cursor), det.suspicious.end
        if end <= start:
            continue
        pieces.append(html.escape(text[cursor:start]))
        source = det.source.document_id if det.source else "?"
        color = colors.get(source, "#c3c2b7")
        tip = f"source : {source} — score {det.score:.2f}"
        pieces.append(
            f"<mark style='background:{_tint(color)}' title='{html.escape(tip)}'>{html.escape(text[start:end])}</mark>"
        )
        cursor = end
    pieces.append(html.escape(text[cursor:]))
    legend = "".join(
        f"<span><i class='swatch' style='background:{colors[s]}'></i>{html.escape(s)} — "
        f"{report.sources[s]['passages']} passage(s), {report.sources[s]['characters']} caractères</span>"
        for s in ranked
    )
    rows = []
    for det in report.detections:
        source_text = ""
        if det.source is not None and det.source.document_id in sources:
            source_text = sources[det.source.document_id][det.source.start : det.source.end]
        rows.append(
            f"<tr><td>{det.suspicious.start}–{det.suspicious.end}</td>"
            f"<td>{html.escape(det.source.document_id if det.source else '—')}</td>"
            f"<td>{det.score:.2f}</td><td>{html.escape(text[det.suspicious.start : det.suspicious.end][:400])}</td>"
            f"<td>{html.escape(source_text[:400])}</td></tr>"
        )
    table = (
        "<div class='table'><table><thead><tr><th>Empan suspect</th><th>Source</th><th>Score</th>"
        f"<th>Passage suspect</th><th>Passage source</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )
    body = kpis(
        [
            ("Score global", f"{report.score:.2f}"),
            ("Couverture", f"{100 * report.coverage:.1f} %"),
            ("Passages candidats", str(len(report.detections))),
            ("Sources", str(len(report.sources))),
        ]
    )
    body += (
        "<p class='warning'>Un score de similarité n'est pas une preuve de plagiat : citations, formules "
        "et textes standardisés produisent des correspondances légitimes. Examinez chaque passage.</p>"
    )
    body += section("Texte analysé", f"<div class='legend'>{legend}</div><div class='text'>{''.join(pieces)}</div>")
    body += section("Passages alignés", table)
    return page(title or f"Rapport de similarité — {report.document_id}", body, f"{report.length} caractères analysés")
