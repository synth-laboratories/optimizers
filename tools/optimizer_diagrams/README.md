# Optimizer systems maps

Seven standalone HTML pages whose systems maps are two-dimensional ASCII
drawings, not Mermaid, SVG or screenshots. One generator, seven data models.

```bash
PYTHONPATH=tools python3 -m optimizer_diagrams.build --out docs/diagrams
PYTHONPATH=tools python3 -m pytest tests/test_optimizer_diagrams.py
```

Pages: portable container system, shared optimizer environment and training
loop, GEPA, GELO, OHCO, VZPO, and MAPO on multi-agent DungeonGrid.

## Why a canvas instead of hand-typed box art

`canvas.py` places boxes and connectors by coordinate on a fixed character
grid. Hand-typed art drifts the moment a label changes length; the canvas
raises rather than silently overwriting a box, so a map cannot be committed
in a corrupted state.

## What the output guarantees

The build writes both the rendered `.html` and the generator input `.json`, so
the data model that produced a page is archived beside it. Each page:

* is fully self-contained — no scripts, no CDN, no external fonts;
* renders the map as selectable, searchable monospace text in a `<pre>`;
* scrolls wide maps inside their own container rather than clipping, and never
  makes the page body scroll sideways;
* supports light and dark via `prefers-color-scheme`, and prints without losing
  boundaries or labels;
* offers a skip link, a keyboard-reachable section nav and focusable map
  regions;
* embeds optimizer, schema and status metadata as `<meta name="synth:…">`;
* renders a receipt row as **not recorded** when the corresponding run has not
  happened, so a page degrades honestly instead of implying evidence.

`tests/test_optimizer_diagrams.py` asserts every one of those properties plus
the required nodes of each map, and enforces the honesty gate: a page with no
recorded receipt may not claim proof.
