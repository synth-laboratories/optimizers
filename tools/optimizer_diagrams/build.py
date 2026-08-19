"""Build the seven v0.7 optimizer systems maps.

    python -m optimizer_diagrams.build --out docs/diagrams

Writes one standalone ``.html`` per page plus the generator input as ``.json``,
so the data model that produced a page is archived next to the page itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .diagrams import containers, gelo, gepa, mapo, ohco, optimizer_env, vzpo
from .model import DiagramPage
from .render import render_page

#: Ordered so the two shared-platform pages come first.
BUILDERS = (
    containers.build,
    optimizer_env.build,
    gepa.build,
    gelo.build,
    ohco.build,
    vzpo.build,
    mapo.build,
)


def all_pages() -> list[DiagramPage]:
    return [builder() for builder in BUILDERS]


def render_index(pages: list[DiagramPage]) -> str:
    """A plain index page. Same house style, no maps."""

    from .render import CSS

    rows = "\n".join(
        f"      <tr><td><a href=\"{page.slug}.html\">{page.title}</a></td>"
        f"<td>{page.optimizer}</td><td>{page.verdict}</td></tr>"
        for page in pages
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v0.7 optimizer systems maps</title>
  <meta name="description" content="Seven ASCII systems maps for the Synth optimizer platform.">
  <style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <header class="page">
    <p class="eyebrow">Synth · v0.7</p>
    <h1>Optimizer systems maps</h1>
    <p class="subtitle">Seven standalone pages. Each one is a two-dimensional ASCII map of a real
      system boundary, rendered as selectable monospace text with no scripts and no external
      dependencies.</p>
  </header>
  <main id="main">
    <div class="table-scroll">
      <table>
      <thead><tr><th scope="col">Page</th><th scope="col">Owner</th><th scope="col">Verdict</th></tr></thead>
      <tbody>
{rows}
      </tbody>
      </table>
    </div>
  </main>
</div>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="optimizer_diagrams.build", description=__doc__)
    parser.add_argument("--out", default="docs/diagrams", help="output directory")
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pages = all_pages()
    for page in pages:
        (out / f"{page.slug}.html").write_text(render_page(page), encoding="utf-8")
        (out / f"{page.slug}.json").write_text(
            json.dumps(page.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"wrote {out / (page.slug + '.html')}")
    (out / "index.html").write_text(render_index(pages), encoding="utf-8")
    print(f"wrote {out / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
