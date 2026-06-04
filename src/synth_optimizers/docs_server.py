#!/usr/bin/env python3
"""Centralized local docs + console server for synth-optimizers.

This is the docs counterpart to :mod:`synth_optimizers.board_server`. Where the
board projects *run evidence* into a live HTML board, ``DocsSource`` projects a
tree of markdown files into a navigable HTML docs site, and the *console*
handler stitches BOTH behind a single port as two tabs (Dashboard + Docs).

Layout (one origin, one port):

- ``GET /``                       -- the console shell: header (Synth logo +
                                     title) and a tab bar that swaps between two
                                     same-origin iframes.
- ``GET /board/``                 -- the existing run board HTML, verbatim
                                     (``o11y.render_board_html``); its live
                                     endpoints below are served by this handler.
- ``GET /api/runs``               -- board snapshot JSON (one-shot).
- ``GET /api/stream``             -- board SSE stream.
- ``GET /api/runs/{id}/{events,timings,limits}`` -- per-run drill-down.
- ``GET /docs/``                  -- the docs viewer HTML (nav sidebar + pane).
- ``GET /api/docs``               -- the docs nav tree as JSON.
- ``GET /api/docs/page?path=...`` -- one rendered markdown page as HTML.
- ``GET /assets/synth-logo.svg``  -- the header logo.

The docs renderer is a dependency-free markdown subset (headings, fenced and
inline code, bold/italic, links, lists, tables, blockquotes, rules) so the
published wheel keeps the same pure-stdlib footprint as the board.
"""
from __future__ import annotations

import html
import json
import re
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Sequence
from urllib.parse import parse_qs, unquote

from .board_server import EVENTS_BASE, STREAM_PATH, _Hub

# Docs ship as package data under ``synth_optimizers/docs/<name>/`` so a
# pip-installed console can find them without a repo checkout.
_DOCS_DIR = Path(__file__).resolve().parent / "docs"


def bundled_docs_root(name: str = "gepa") -> Path:
    """Path to a docs set bundled in the package (default: the GEPA docs)."""
    root = _DOCS_DIR / name
    if not root.is_dir():
        raise FileNotFoundError(f"no bundled docs named {name!r} at {root}")
    return root


def bundled_logo() -> Path | None:
    """The Synth logo shipped with the GEPA docs, if present."""
    logo = _DOCS_DIR / "gepa" / "assets" / "synth-logo.png"
    return logo if logo.exists() else None


# ---------------------------------------------------------------------------
# Docs model
# ---------------------------------------------------------------------------

# Files may carry a leading ``NN-`` ordering prefix; it sets nav order and is
# stripped from both the URL key and the display title.
_ORDER_PREFIX = re.compile(r"^(\d+)[-_]")


@dataclass(frozen=True)
class DocPage:
    """One markdown file projected into the docs tree."""

    key: str          # url-stable slug, e.g. "cli" or "guides/auth"
    title: str        # display title (first H1, else humanized filename)
    source: Path      # absolute path on disk
    order: int        # sort key within its section
    section: str      # top-level folder ("" = root)


class DocsSource:
    """ServiceBoardSource-shaped source backed by a tree of markdown files.

    Mirrors the board sources' contract: a ``title`` plus pure projection
    methods (``index`` / ``render_page``) the handler can call per request, so
    edits on disk show up on the next reload with no restart.
    """

    def __init__(self, roots: Sequence[Path | str], *, title: str) -> None:
        self._roots = [Path(r).resolve() for r in roots]
        self._title = title

    @property
    def title(self) -> str:
        return self._title

    # -- projection ---------------------------------------------------------

    def _pages(self) -> dict[str, DocPage]:
        pages: dict[str, DocPage] = {}
        for root in self._roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.md")):
                if any(part.startswith(".") for part in path.relative_to(root).parts):
                    continue
                page = self._project(root, path)
                pages.setdefault(page.key, page)
        return pages

    @staticmethod
    def _project(root: Path, path: Path) -> DocPage:
        rel = path.relative_to(root)
        parts = list(rel.parts)
        stem = path.stem
        match = _ORDER_PREFIX.match(stem)
        order = int(match.group(1)) if match else 1_000
        clean_stem = _ORDER_PREFIX.sub("", stem)
        slug_parts = [_ORDER_PREFIX.sub("", p) for p in parts[:-1]] + [clean_stem]
        key = "/".join(slug_parts)
        section = slug_parts[0] if len(slug_parts) > 1 else ""
        title = _first_heading(path) or _humanize(clean_stem)
        return DocPage(key=key, title=title, source=path, order=order, section=section)

    def index(self) -> dict:
        """Nav tree: ordered sections, each with ordered pages."""
        pages = sorted(self._pages().values(), key=lambda p: (p.section, p.order, p.title))
        sections: dict[str, list[dict]] = {}
        for page in pages:
            sections.setdefault(page.section, []).append(
                {"key": page.key, "title": page.title}
            )
        # Root pages first, then named sections alphabetically.
        ordered = [""] + sorted(s for s in sections if s)
        nav = [
            {"section": _humanize(s) if s else "", "pages": sections[s]}
            for s in ordered
            if s in sections
        ]
        default = pages[0].key if pages else None
        return {"title": self._title, "nav": nav, "default": default}

    def render_page(self, key: str) -> str:
        page = self._pages().get(key)
        if page is None:
            raise KeyError(key)
        return markdown_to_html(page.source.read_text(encoding="utf-8"))


def _first_heading(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _humanize(slug: str) -> str:
    return slug.replace("-", " ").replace("_", " ").strip().title()


# ---------------------------------------------------------------------------
# Markdown -> HTML (dependency-free subset)
# ---------------------------------------------------------------------------

_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)([^*]+)\*(?!\*)")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _inline(text: str) -> str:
    """Escape, then re-introduce inline markup. Code spans are protected."""
    spans: list[str] = []

    def stash(match: re.Match) -> str:
        spans.append(f"<code>{html.escape(match.group(1), quote=False)}</code>")
        return f"\x00{len(spans) - 1}\x00"

    text = _INLINE_CODE.sub(stash, text)
    text = html.escape(text, quote=False)
    text = _LINK.sub(
        lambda m: f'<a href="{html.escape(m.group(2))}" target="_blank" rel="noopener">{m.group(1)}</a>',
        text,
    )
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    text = _ITALIC.sub(r"<em>\1</em>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: spans[int(m.group(1))], text)
    return text


def markdown_to_html(md: str) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    list_stack: list[str] = []  # "ul" / "ol"

    def close_lists() -> None:
        while list_stack:
            out.append(f"</{list_stack.pop()}>")

    while i < n:
        line = lines[i]

        # Fenced code block
        if line.lstrip().startswith("```"):
            close_lists()
            lang = line.lstrip()[3:].strip()
            body: list[str] = []
            i += 1
            while i < n and not lines[i].lstrip().startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1  # consume closing fence
            cls = f' class="lang-{html.escape(lang)}"' if lang else ""
            out.append(f"<pre><code{cls}>{html.escape(chr(10).join(body), quote=False)}</code></pre>")
            continue

        # Blank line
        if not line.strip():
            close_lists()
            i += 1
            continue

        # Heading
        m = _HEADING.match(line)
        if m:
            close_lists()
            level = len(m.group(1))
            text = m.group(2).strip()
            anchor = _slug(text)
            out.append(f'<h{level} id="{anchor}">{_inline(text)}</h{level}>')
            i += 1
            continue

        # Horizontal rule
        if re.match(r"^\s*(---|\*\*\*|___)\s*$", line):
            close_lists()
            out.append("<hr>")
            i += 1
            continue

        # Table (header row followed by a separator row)
        if "|" in line and i + 1 < n and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i + 1]) and "-" in lines[i + 1]:
            close_lists()
            i = _emit_table(lines, i, out)
            continue

        # Blockquote
        if line.lstrip().startswith(">"):
            close_lists()
            quote: list[str] = []
            while i < n and lines[i].lstrip().startswith(">"):
                quote.append(lines[i].lstrip()[1:].lstrip())
                i += 1
            out.append(f"<blockquote>{_inline(' '.join(quote))}</blockquote>")
            continue

        # Lists
        ul = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
        ol = re.match(r"^(\s*)\d+\.\s+(.*)$", line)
        if ul or ol:
            kind = "ul" if ul else "ol"
            content = (ul or ol).group(2)
            if not list_stack or list_stack[-1] != kind:
                if list_stack:
                    out.append(f"</{list_stack.pop()}>")
                out.append(f"<{kind}>")
                list_stack.append(kind)
            out.append(f"<li>{_inline(content)}</li>")
            i += 1
            continue

        # Paragraph (gather consecutive plain lines)
        close_lists()
        para: list[str] = [line]
        i += 1
        while i < n and lines[i].strip() and not _is_block_start(lines[i]):
            para.append(lines[i])
            i += 1
        out.append(f"<p>{_inline(' '.join(s.strip() for s in para))}</p>")

    close_lists()
    return "\n".join(out)


def _is_block_start(line: str) -> bool:
    stripped = line.lstrip()
    return (
        stripped.startswith("```")
        or stripped.startswith(">")
        or bool(_HEADING.match(line))
        or bool(re.match(r"^(\s*)[-*+]\s+", line))
        or bool(re.match(r"^(\s*)\d+\.\s+", line))
        or "|" in line
    )


def _emit_table(lines: list[str], start: int, out: list[str]) -> int:
    def cells(row: str) -> list[str]:
        row = row.strip().strip("|")
        return [c.strip() for c in row.split("|")]

    header = cells(lines[start])
    i = start + 2  # skip header + separator
    out.append("<table><thead><tr>")
    out.extend(f"<th>{_inline(c)}</th>" for c in header)
    out.append("</tr></thead><tbody>")
    while i < len(lines) and "|" in lines[i] and lines[i].strip():
        out.append("<tr>")
        out.extend(f"<td>{_inline(c)}</td>" for c in cells(lines[i]))
        out.append("</tr>")
        i += 1
    out.append("</tbody></table>")
    return i


# ---------------------------------------------------------------------------
# HTML shells
# ---------------------------------------------------------------------------

# Shared dark palette, matched to the board (o11y) tokens.
_PALETTE = """
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3;
    --muted:#8b949e; --run:#58a6ff; --accent:#58a6ff; }
"""


def render_console_html(*, title: str, board_path: str = "/board/", docs_path: str = "/docs/") -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
{_PALETTE}
  * {{ box-sizing:border-box; }}
  html, body {{ margin:0; height:100%; background:var(--bg); color:var(--text);
    font:14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }}
  body {{ display:flex; flex-direction:column; }}
  header {{ display:flex; align-items:center; gap:18px; padding:12px 20px;
    border-bottom:1px solid var(--border); background:var(--panel); }}
  .brand {{ display:flex; align-items:center; gap:12px; }}
  .brand img {{ height:30px; display:block; border-radius:7px; }}
  .brand .t {{ font-size:15px; font-weight:600; }}
  .switch {{ display:flex; gap:3px; margin-left:6px; padding:3px; background:var(--bg);
    border:1px solid var(--border); border-radius:9px; }}
  .switch button {{ background:transparent; color:var(--muted); border:0; border-radius:6px;
    padding:7px 18px; font:inherit; font-weight:600; cursor:pointer; transition:color .1s, background .1s; }}
  .switch button:hover {{ color:var(--text); }}
  .switch button.active {{ color:var(--text); background:var(--panel);
    box-shadow:inset 0 0 0 1px var(--border), 0 1px 2px rgba(0,0,0,.4); }}
  .spacer {{ flex:1; }}
  .hint {{ color:var(--muted); font-size:11px; display:flex; align-items:center; gap:6px; }}
  kbd {{ font:inherit; font-size:10px; line-height:1; background:var(--bg); color:var(--text);
    border:1px solid var(--border); border-bottom-width:2px; border-radius:4px; padding:3px 5px; }}
  main {{ flex:1; position:relative; }}
  iframe {{ position:absolute; inset:0; width:100%; height:100%; border:0; }}
  iframe[hidden] {{ display:none; }}
</style></head>
<body>
<header>
  <div class="brand">
    <img src="/assets/synth-logo" alt="Synth">
    <span class="t">{html.escape(title)}</span>
  </div>
  <div class="switch" role="tablist">
    <button id="tab-dashboard" class="active">Dashboard</button>
    <button id="tab-docs">Docs</button>
  </div>
  <span class="spacer"></span>
  <span class="hint"><kbd>1</kbd> dashboard <kbd>2</kbd> docs <kbd>T</kbd> toggle</span>
</header>
<main>
  <iframe id="view-dashboard" src="{board_path}"></iframe>
  <iframe id="view-docs" src="{docs_path}" hidden></iframe>
</main>
<script>
  const order = ['dashboard', 'docs'];
  let current = 'dashboard';

  function show(name) {{
    if (!order.includes(name)) name = 'dashboard';
    current = name;
    for (const t of order) {{
      document.getElementById('view-' + t).hidden = (t !== name);
      document.getElementById('tab-' + t).classList.toggle('active', t === name);
    }}
    if (location.hash.slice(1) !== name) history.replaceState(null, '', '#' + name);
  }}
  function toggle() {{ show(current === 'dashboard' ? 'docs' : 'dashboard'); }}

  // Key handling: 1=dashboard, 2=docs, t/`=toggle. Ignored while typing in a field.
  function onKey(e) {{
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const el = e.target, tag = (el && el.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select' || (el && el.isContentEditable)) return;
    if (e.key === '1') {{ show('dashboard'); e.preventDefault(); }}
    else if (e.key === '2') {{ show('docs'); e.preventDefault(); }}
    else if (e.key === 't' || e.key === 'T' || e.key === '`') {{ toggle(); e.preventDefault(); }}
  }}

  for (const name of order) {{
    document.getElementById('tab-' + name).addEventListener('click', () => show(name));
  }}
  document.addEventListener('keydown', onKey);

  // The views are same-origin iframes, so attach the same key handler inside each
  // — otherwise shortcuts would be swallowed once focus moves into a view.
  for (const name of order) {{
    const frame = document.getElementById('view-' + name);
    const attach = () => {{ try {{ frame.contentDocument.addEventListener('keydown', onKey); }} catch (_) {{}} }};
    frame.addEventListener('load', attach);
    attach();
  }}

  window.addEventListener('hashchange', () => show(location.hash.slice(1)));
  show(location.hash.slice(1) || 'dashboard');
</script>
</body></html>"""


def render_docs_html(*, title: str, api_base: str = "/api/docs") -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — docs</title>
<style>
{_PALETTE}
  * {{ box-sizing:border-box; }}
  html, body {{ margin:0; height:100%; background:var(--bg); color:var(--text);
    font:14px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; }}
  body {{ display:flex; }}
  aside {{ width:248px; flex:none; height:100vh; overflow:auto; padding:18px 14px;
    border-right:1px solid var(--border); background:var(--panel); }}
  aside .sec {{ color:var(--muted); font-size:10px; text-transform:uppercase;
    letter-spacing:.06em; margin:16px 6px 6px; }}
  aside a {{ display:block; padding:5px 8px; border-radius:6px; color:var(--muted);
    text-decoration:none; font-size:13px; }}
  aside a:hover {{ color:var(--text); background:var(--bg); }}
  aside a.active {{ color:var(--text); background:var(--bg); border-left:2px solid var(--accent); }}
  main {{ flex:1; height:100vh; overflow:auto; }}
  article {{ max-width:820px; margin:0 auto; padding:34px 40px 80px; }}
  article h1 {{ font-size:26px; margin:.2em 0 .6em; }}
  article h2 {{ font-size:20px; margin:1.6em 0 .5em; padding-bottom:.3em; border-bottom:1px solid var(--border); }}
  article h3 {{ font-size:16px; margin:1.3em 0 .4em; }}
  article p, article li {{ font-family:-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
  article a {{ color:var(--accent); }}
  article code {{ background:#1c2230; border:1px solid var(--border); border-radius:4px;
    padding:1px 5px; font-size:12.5px; }}
  article pre {{ background:#010409; border:1px solid var(--border); border-radius:8px;
    padding:14px 16px; overflow:auto; }}
  article pre code {{ background:none; border:0; padding:0; font-size:12.5px; line-height:1.55; }}
  article table {{ border-collapse:collapse; width:100%; margin:1em 0; font-size:13px; }}
  article th, article td {{ border:1px solid var(--border); padding:7px 10px; text-align:left; }}
  article th {{ background:var(--panel); }}
  article blockquote {{ margin:1em 0; padding:.4em 1em; border-left:3px solid var(--accent);
    color:var(--muted); background:var(--panel); border-radius:0 6px 6px 0; }}
  article hr {{ border:0; border-top:1px solid var(--border); margin:2em 0; }}
  .empty {{ color:var(--muted); padding:40px; }}
</style></head>
<body>
<aside id="nav"></aside>
<main><article id="content"><div class="empty">Loading…</div></article></main>
<script>
  const API = "{api_base}";
  const nav = document.getElementById('nav');
  const content = document.getElementById('content');
  let index = null;

  function pageFromHash() {{ return decodeURIComponent(location.hash.replace(/^#\\/?/, '')); }}

  async function loadPage(key) {{
    const res = await fetch(API + '/page?path=' + encodeURIComponent(key));
    if (!res.ok) {{ content.innerHTML = '<div class="empty">Not found: ' + key + '</div>'; return; }}
    const data = await res.json();
    content.innerHTML = data.html;
    content.parentElement.scrollTop = 0;
    for (const a of nav.querySelectorAll('a')) a.classList.toggle('active', a.dataset.key === key);
  }}

  function renderNav() {{
    nav.innerHTML = '';
    for (const section of index.nav) {{
      if (section.section) {{
        const h = document.createElement('div'); h.className = 'sec'; h.textContent = section.section;
        nav.appendChild(h);
      }}
      for (const page of section.pages) {{
        const a = document.createElement('a');
        a.href = '#/' + page.key; a.dataset.key = page.key; a.textContent = page.title;
        nav.appendChild(a);
      }}
    }}
  }}

  async function boot() {{
    index = await (await fetch(API)).json();
    renderNav();
    const target = pageFromHash() || index.default;
    if (target) loadPage(target);
    else content.innerHTML = '<div class="empty">No docs found.</div>';
  }}
  window.addEventListener('hashchange', () => {{ const k = pageFromHash(); if (k) loadPage(k); }});
  boot();
</script>
</body></html>"""


# ---------------------------------------------------------------------------
# Console server (board + docs behind one port)
# ---------------------------------------------------------------------------


@dataclass
class _Logo:
    body: bytes
    content_type: str = "image/svg+xml"


def _load_logo(logo_path: Path | str | None) -> _Logo:
    if logo_path is None:
        # Minimal fallback wordmark so the header is never empty.
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="92" height="30">'
            '<rect width="92" height="30" rx="7" fill="#e6edf3"/>'
            '<text x="46" y="20" font-family="monospace" font-size="15" '
            'font-weight="700" fill="#0d1117" text-anchor="middle">Synth</text></svg>'
        )
        return _Logo(svg.encode())
    path = Path(logo_path)
    ctype = "image/png" if path.suffix.lower() == ".png" else "image/svg+xml"
    return _Logo(path.read_bytes(), ctype)


def _console_handler_factory(hub: _Hub, board_source, docs_source: DocsSource, logo: _Logo):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args) -> None:
            pass

        def do_GET(self) -> None:
            raw = self.path.split("?", 1)[0]
            query = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}

            # Console shell + assets
            if raw == "/":
                self._html(render_console_html(title=docs_source.title))
            elif raw == "/assets/synth-logo":
                self._bytes(logo.body, logo.content_type)

            # Board surface (reuses o11y.render_board_html + the board hub/source)
            elif raw in ("/board", "/board/"):
                from .o11y import render_board_html

                self._html(
                    render_board_html(
                        hub.latest,
                        title=board_source.title,
                        live_endpoint=STREAM_PATH,
                        service_url=getattr(board_source, "service_url", None),
                        events_base=EVENTS_BASE,
                    )
                )
            elif raw == "/api/runs":
                self._json(hub.latest)
            elif raw == STREAM_PATH:
                self._stream()
            elif raw.startswith(EVENTS_BASE + "/") and raw.endswith("/events"):
                run_id = unquote(raw[len(EVENTS_BASE) + 1 : -len("/events")])
                since = int(query.get("since", ["0"])[0])
                self._json({"run_id": run_id, "events": board_source.run_events(run_id, since=since)})
            elif raw.startswith(EVENTS_BASE + "/") and raw.endswith("/timings"):
                run_id = unquote(raw[len(EVENTS_BASE) + 1 : -len("/timings")])
                self._json(board_source.run_timings(run_id))
            elif raw.startswith(EVENTS_BASE + "/") and raw.endswith("/limits"):
                run_id = unquote(raw[len(EVENTS_BASE) + 1 : -len("/limits")])
                self._json(board_source.run_limits(run_id))

            # Docs surface
            elif raw in ("/docs", "/docs/"):
                self._html(render_docs_html(title=docs_source.title))
            elif raw == "/api/docs":
                self._json(docs_source.index())
            elif raw == "/api/docs/page":
                key = unquote(query.get("path", [""])[0])
                try:
                    self._json({"key": key, "html": docs_source.render_page(key)})
                except KeyError:
                    self.send_error(404, "doc not found")
            else:
                self.send_error(404, "not found")

        # -- writers --------------------------------------------------------

        def _html(self, text: str) -> None:
            self._bytes(text.encode(), "text/html; charset=utf-8")

        def _json(self, data: dict) -> None:
            self._bytes(json.dumps(data).encode(), "application/json")

        def _bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _stream(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            event = hub.subscribe()
            try:
                while True:
                    if event.wait(timeout=15.0):
                        event.clear()
                        payload = json.dumps(hub.latest)
                        self.wfile.write(f"event: board\ndata: {payload}\n\n".encode())
                    else:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                hub.unsubscribe(event)

    return Handler


def serve_console(
    board_source,
    docs_source: DocsSource,
    *,
    host: str = "127.0.0.1",
    port: int = 8766,
    interval: float = 2.0,
    logo_path: Path | str | None = None,
) -> None:
    """Serve the board and docs behind one port until interrupted.

    ``board_source`` is any board source (e.g. ``AggregateSource`` or a
    file-backed source) exposing ``title``/``snapshot``/``run_events`` etc.;
    ``docs_source`` is a :class:`DocsSource`. When ``logo_path`` is omitted the
    Synth logo bundled with the GEPA docs is used.
    """
    logo = _load_logo(logo_path if logo_path is not None else bundled_logo())
    hub = _Hub(board_source, interval=interval)
    threading.Thread(target=hub.run_forever, name="console-board-poller", daemon=True).start()

    httpd = ThreadingHTTPServer((host, port), _console_handler_factory(hub, board_source, docs_source, logo))
    print(f"[console] dashboard + docs: http://{host}:{port}/  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[console] shutting down...")
    finally:
        hub.stop()
        httpd.shutdown()
