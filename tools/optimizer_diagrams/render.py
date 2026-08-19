"""Standalone HTML renderer for the optimizer systems maps.

Constraints the output has to satisfy, all of them checked by
``tests/test_optimizer_diagrams.py``:

* no external CDN, font or runtime dependency — the saved file works offline;
* no JavaScript — every affordance is HTML plus CSS;
* the systems map itself is selectable, searchable monospace text in ``<pre>``;
* light and dark themes, following the reader's system preference;
* wide maps scroll inside their own container instead of clipping or forcing
  the page to scroll sideways;
* keyboard-reachable section navigation and focusable map regions;
* prints without losing boundaries or labels.
"""

from __future__ import annotations

import html
from typing import Iterable

from .model import DiagramPage

CSS = """
:root {
  color-scheme: light dark;
  --paper: #fbfaf9;
  --paper-raised: #ffffff;
  --ink: #16150f;
  --ink-soft: #5c5850;
  --ink-faint: #8a857b;
  --rule: #e0dcd4;
  --rule-strong: #c8c2b7;
  --accent: #c2560a;
  --accent-soft: #fdf1e6;
  --map-bg: #ffffff;
  --map-ink: #23211a;
  --ok: #1f6d43;
  --absent: #8a857b;
  --measure: 74ch;
  --radius: 10px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --paper: #0e0e0d;
    --paper-raised: #171716;
    --ink: #f2efe9;
    --ink-soft: #b3ada2;
    --ink-faint: #837e75;
    --rule: #2a2926;
    --rule-strong: #3d3b36;
    --accent: #ff8a2a;
    --accent-soft: #2a1a0c;
    --map-bg: #121211;
    --map-ink: #e8e4dc;
    --ok: #57c98c;
    --absent: #837e75;
  }
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: ui-sans-serif, -apple-system, "Segoe UI", Inter, Helvetica, Arial, sans-serif;
  font-size: 16px;
  line-height: 1.6;
  text-rendering: optimizeLegibility;
}
.skip {
  position: absolute; left: -9999px; top: 0;
  background: var(--accent); color: #fff; padding: 0.6rem 1rem; z-index: 10;
}
.skip:focus { left: 0; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 2.5rem 1.5rem 5rem; }
header.page { border-bottom: 1px solid var(--rule); padding-bottom: 1.75rem; margin-bottom: 2rem; }
.eyebrow {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 0.75rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--accent); margin: 0 0 0.6rem;
}
h1 { font-size: clamp(1.7rem, 1.2rem + 1.6vw, 2.5rem); line-height: 1.15; margin: 0 0 0.5rem; letter-spacing: -0.015em; }
.subtitle { color: var(--ink-soft); max-width: var(--measure); margin: 0 0 1.5rem; font-size: 1.05rem; }
.summary {
  background: var(--paper-raised); border: 1px solid var(--rule); border-left: 3px solid var(--accent);
  border-radius: var(--radius); padding: 1.1rem 1.3rem; max-width: var(--measure);
}
.summary dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.35rem 1rem; margin: 0; }
.summary dt {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-faint);
  padding-top: 0.22rem;
}
.summary dd { margin: 0; }
nav.toc { margin: 2rem 0; }
nav.toc h2 {
  font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--ink-faint); margin: 0 0 0.6rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
}
nav.toc ol { list-style: none; margin: 0; padding: 0; display: flex; flex-wrap: wrap; gap: 0.4rem; counter-reset: toc; }
nav.toc a {
  display: inline-block; padding: 0.35rem 0.7rem; border: 1px solid var(--rule);
  border-radius: 999px; text-decoration: none; color: var(--ink-soft); font-size: 0.85rem;
}
nav.toc a:hover, nav.toc a:focus-visible { border-color: var(--accent); color: var(--accent); }
section.panel { margin: 3rem 0 0; scroll-margin-top: 1rem; }
section.panel > h2 { font-size: 1.25rem; margin: 0 0 0.35rem; letter-spacing: -0.01em; }
section.panel > p.caption { color: var(--ink-soft); max-width: var(--measure); margin: 0 0 1rem; }
figure.map { margin: 0; }
.map-scroll {
  overflow-x: auto; overflow-y: hidden;
  background: var(--map-bg); border: 1px solid var(--rule-strong); border-radius: var(--radius);
  padding: 1.25rem 1.4rem;
}
.map-scroll:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
pre.ascii {
  margin: 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", "DejaVu Sans Mono", monospace;
  font-variant-ligatures: none;
  font-feature-settings: "liga" 0, "calt" 0;
  font-size: 12.5px;
  line-height: 1.45;
  white-space: pre;
  tab-size: 4;
  color: var(--map-ink);
}
figcaption { color: var(--ink-faint); font-size: 0.85rem; margin-top: 0.6rem; }
ul.notes { max-width: var(--measure); color: var(--ink-soft); padding-left: 1.1rem; }
ul.notes li { margin: 0.3rem 0; }
table { border-collapse: collapse; width: 100%; margin: 0.5rem 0 0; font-size: 0.92rem; }
th, td { text-align: left; padding: 0.5rem 0.7rem; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { color: var(--ink-faint); font-weight: 600; font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; }
td.glyph, td.mono, code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  font-variant-ligatures: none;
  white-space: pre;
}
td.glyph { color: var(--accent); }
.table-scroll { overflow-x: auto; }
.tag {
  display: inline-block; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase;
  padding: 0.1rem 0.5rem; border-radius: 999px; border: 1px solid currentColor;
}
.tag.present { color: var(--ok); }
.tag.absent { color: var(--absent); }
footer.page { margin-top: 4rem; padding-top: 1.5rem; border-top: 1px solid var(--rule); color: var(--ink-faint); font-size: 0.85rem; }
footer.page dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.25rem 1rem; margin: 0; }
footer.page dt { color: var(--ink-faint); }
footer.page dd { margin: 0; color: var(--ink-soft); }
a { color: var(--accent); }
@media print {
  :root { --paper: #fff; --paper-raised: #fff; --map-bg: #fff; --ink: #000; --map-ink: #000; --rule: #999; --rule-strong: #555; }
  body { font-size: 11pt; }
  nav.toc, .skip { display: none; }
  .map-scroll { overflow: visible; border: 1px solid #555; page-break-inside: avoid; }
  pre.ascii { font-size: 8pt; }
  section.panel { page-break-inside: avoid; }
}
"""


def _e(value: str) -> str:
    return html.escape(str(value), quote=True)


def _rows(pairs: Iterable[tuple[str, str]], glyph_header: str, meaning_header: str) -> str:
    body = "\n".join(
        f"      <tr><td class=\"glyph\">{_e(glyph)}</td><td>{_e(meaning)}</td></tr>"
        for glyph, meaning in pairs
    )
    return (
        '  <div class="table-scroll">\n    <table>\n'
        f"      <thead><tr><th scope=\"col\">{_e(glyph_header)}</th>"
        f"<th scope=\"col\">{_e(meaning_header)}</th></tr></thead>\n"
        f"      <tbody>\n{body}\n      </tbody>\n    </table>\n  </div>"
    )


def render_page(page: DiagramPage) -> str:
    meta_tags = "\n".join(
        f'  <meta name="synth:{_e(field_.label.lower().replace(" ", "_"))}" content="{_e(field_.value)}">'
        for field_ in page.metadata
    )
    toc = "\n".join(
        f'      <li><a href="#{_e(panel.slug)}">{_e(panel.title)}</a></li>' for panel in page.panels
    )
    sections = []
    for panel in page.panels:
        notes = ""
        if panel.notes:
            items = "\n".join(f"      <li>{_e(note)}</li>" for note in panel.notes)
            notes = f'\n    <ul class="notes">\n{items}\n    </ul>'
        sections.append(
            f'  <section class="panel" id="{_e(panel.slug)}" aria-labelledby="{_e(panel.slug)}-h">\n'
            f'    <h2 id="{_e(panel.slug)}-h">{_e(panel.title)}</h2>\n'
            f'    <p class="caption">{_e(panel.caption)}</p>\n'
            f"    <figure class=\"map\">\n"
            f'      <div class="map-scroll" tabindex="0" role="group" '
            f'aria-label="{_e(panel.title)} systems map, monospace text">\n'
            f'        <pre class="ascii">{_e(panel.ascii_map)}</pre>\n'
            f"      </div>\n"
            f"      <figcaption>Selectable monospace map — search this page to find any node label."
            f"</figcaption>\n"
            f"    </figure>{notes}\n"
            f"  </section>"
        )

    boundary_table = (
        f'  <section class="panel" id="boundaries" aria-labelledby="boundaries-h">\n'
        f'    <h2 id="boundaries-h">Boundaries and who may read what</h2>\n'
        f'    <p class="caption">Each row is a boundary drawn in the maps above. Crossing it '
        f"changes ownership, visibility, or both.</p>\n"
        f"{_rows(page.boundaries, 'Boundary', 'Rule')}\n  </section>"
        if page.boundaries
        else ""
    )

    receipt_rows = "\n".join(
        "      <tr>"
        f"<td>{_e(receipt.label)}</td>"
        f"<td class=\"mono\">{_e(receipt.identifier) if receipt.present else '—'}</td>"
        f"<td>{_e(receipt.detail)}</td>"
        f"<td><span class=\"tag {'present' if receipt.present else 'absent'}\">"
        f"{'recorded' if receipt.present else 'not recorded'}</span></td>"
        "</tr>"
        for receipt in page.receipts
    )
    receipts_section = (
        f'  <section class="panel" id="receipts" aria-labelledby="receipts-h">\n'
        f'    <h2 id="receipts-h">Receipts</h2>\n'
        f'    <p class="caption">Identifiers from real runs. A row marked <em>not recorded</em> '
        f"means the corresponding run has not been executed yet — the map still renders, it just "
        f"cannot point at evidence.</p>\n"
        f'    <div class="table-scroll">\n      <table>\n'
        f'      <thead><tr><th scope="col">Receipt</th><th scope="col">Identifier</th>'
        f'<th scope="col">Detail</th><th scope="col">Status</th></tr></thead>\n'
        f"      <tbody>\n{receipt_rows}\n      </tbody>\n      </table>\n    </div>\n  </section>"
        if page.receipts
        else ""
    )

    legend_section = (
        f'  <section class="panel" id="legend" aria-labelledby="legend-h">\n'
        f'    <h2 id="legend-h">Legend</h2>\n'
        f'    <p class="caption">One visual language across all seven maps.</p>\n'
        f"{_rows(page.legend, 'Glyph', 'Meaning')}\n  </section>"
    )

    footer_rows = "\n".join(
        f"      <dt>{_e(field_.label)}</dt><dd>{_e(field_.value)}"
        + (f" <span class=\"tag\">{_e(field_.note)}</span>" if field_.note else "")
        + "</dd>"
        for field_ in page.metadata
    )

    nav_entries = toc + (
        '\n      <li><a href="#boundaries">Boundaries</a></li>' if page.boundaries else ""
    ) + (
        '\n      <li><a href="#receipts">Receipts</a></li>' if page.receipts else ""
    ) + '\n      <li><a href="#legend">Legend</a></li>'

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(page.title)}</title>
  <meta name="description" content="{_e(page.subtitle)}">
  <meta name="generator" content="{_e(page.schema)}">
{meta_tags}
  <style>{CSS}</style>
</head>
<body>
<a class="skip" href="#main">Skip to the systems maps</a>
<div class="wrap">
  <header class="page">
    <p class="eyebrow">{_e(page.optimizer)} · {_e(page.schema_version)}</p>
    <h1>{_e(page.title)}</h1>
    <p class="subtitle">{_e(page.subtitle)}</p>
    <div class="summary">
      <dl>
        <dt>Hypothesis</dt><dd>{_e(page.hypothesis)}</dd>
        <dt>Verdict</dt><dd>{_e(page.verdict)}</dd>
        <dt>Confidence</dt><dd>{_e(page.confidence)}</dd>
        <dt>Why</dt><dd>{_e(page.why)}</dd>
      </dl>
    </div>
  </header>
  <nav class="toc" aria-label="Sections">
    <h2>Sections</h2>
    <ol>
{nav_entries}
    </ol>
  </nav>
  <main id="main">
{chr(10).join(sections)}
{boundary_table}
{receipts_section}
{legend_section}
  </main>
  <footer class="page">
    <dl>
{footer_rows}
    </dl>
  </footer>
</div>
</body>
</html>
"""


__all__ = ["CSS", "render_page"]
