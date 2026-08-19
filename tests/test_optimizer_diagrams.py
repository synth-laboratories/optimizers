"""Structure gates for the v0.7 optimizer systems maps.

These assert the properties §10 of the v0.7 scope asks for and that a human
cannot reliably eyeball across seven pages: alignment safety, required nodes,
offline self-containment, theming, accessibility affordances and the graceful
absent-receipt state.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from optimizer_diagrams.build import BUILDERS, all_pages, render_index  # noqa: E402
from optimizer_diagrams.canvas import Canvas, CanvasError  # noqa: E402
from optimizer_diagrams.model import BASE_LEGEND, DiagramPage  # noqa: E402
from optimizer_diagrams.render import render_page  # noqa: E402

PAGES = all_pages()
BY_SLUG = {page.slug: page for page in PAGES}

REQUIRED_SLUGS = {
    "portable-container-system",
    "optimizer-environment",
    "gepa",
    "gelo",
    "ohco",
    "vzpo",
    "mapo-dungeongrid",
}

#: Nodes each page must actually draw, not merely mention in prose.
REQUIRED_NODES = {
    "portable-container-system": ("REGISTRY", "S3 SOURCE BUNDLE", "synth.container_bundle.v1", "PROVENANCE RECORD"),
    "optimizer-environment": ("OptimizerObservation", "OptimizerAction", "OptimizerTransition", "OptimizerCheckpoint", "FIXED SPLITS", "LIFECYCLE"),
    "gepa": ("CANDIDATE POOL", "PARENT SELECTION", "REFLECTION", "FRONTIER UPDATE", "SEALED HELDOUT", "ONE PROPOSER STEP"),
    "gelo": ("ARCHIVE / BOARD", "RETURN", "MUTATE", "ADMISSION", "CAPABILITY GATE", "distillation"),
    "ohco": ("BASE + SCOPE", "ISOLATED WORKSPACE", "ADMISSION GATE", "PROTECTED", "BASELINE vs CANDIDATE"),
    "vzpo": ("OPTIMIZER SIDE", "VERIFIER PROXY", "VERIFIER (sealed)", "FROZEN HELDOUT GRADER", "REB-063"),
    "mapo-dungeongrid": (
        "HERO_1", "HERO_2", "SHARED / TEAM STATE", "HARNESS", "DUNGEONGRID ENVIRONMENT",
        "TEAM REWARD VECTOR", "REWARD → HERO_1", "REWARD → HERO_2", "BRANCH CHECKPOINT",
        "FORK A", "FORK B", "ONE PROPOSER STEP", "prompt_protocol", "harness", "rlvr",
        "claim:vault", "SEALED HELDOUT", "OPTIMIZER CHECKPOINT",
    ),
}


@pytest.fixture(scope="module")
def rendered() -> dict[str, str]:
    return {page.slug: render_page(page) for page in PAGES}


def test_exactly_the_seven_required_pages_are_built():
    assert len(BUILDERS) == 7
    assert {page.slug for page in PAGES} == REQUIRED_SLUGS


def test_every_page_has_at_least_one_map_and_a_legend():
    for page in PAGES:
        assert page.panels, page.slug
        assert page.legend[: len(BASE_LEGEND)] == BASE_LEGEND
        assert all(panel.ascii_map.strip() for panel in page.panels)


@pytest.mark.parametrize("slug", sorted(REQUIRED_SLUGS))
def test_required_nodes_are_drawn_in_the_maps(slug: str):
    maps = "\n".join(panel.ascii_map for panel in BY_SLUG[slug].panels)
    for node in REQUIRED_NODES[slug]:
        assert node in maps, f"{slug} map is missing node {node!r}"


def test_mapo_map_is_genuinely_two_dimensional():
    """Heroes side by side on the same rows, not stacked in a list."""

    panel = next(p for p in BY_SLUG["mapo-dungeongrid"].panels if p.slug == "multi-agent-step")
    rows_with_both = [
        line for line in panel.ascii_map.splitlines() if "local observation" in line
    ]
    assert rows_with_both, "heroes are not drawn on shared rows"
    assert all(line.count("local observation") == 2 for line in rows_with_both)


def test_every_map_line_fits_a_rectangular_grid():
    """No stray control characters or tabs that would break column alignment."""

    for page in PAGES:
        for panel in page.panels:
            for line in panel.ascii_map.splitlines():
                assert "\t" not in line, f"{page.slug}/{panel.slug} contains a tab"
                assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", line)


def test_maps_stay_within_a_scrollable_width_budget():
    for page in PAGES:
        for panel in page.panels:
            widest = max(len(line) for line in panel.ascii_map.splitlines())
            assert widest <= 150, f"{page.slug}/{panel.slug} is {widest} columns wide"


def test_pages_are_self_contained(rendered: dict[str, str]):
    for slug, html in rendered.items():
        assert "<script" not in html.lower(), f"{slug} contains a script"
        assert "http://" not in html.replace("http://www.w3.org", ""), slug
        assert "https://" not in html, slug
        assert "cdn" not in html.lower(), slug


def test_pages_declare_both_themes_and_print_rules(rendered: dict[str, str]):
    for slug, html in rendered.items():
        assert "prefers-color-scheme: dark" in html, slug
        assert "@media print" in html, slug
        assert "color-scheme: light dark" in html, slug


def test_pages_are_keyboard_navigable(rendered: dict[str, str]):
    for slug, html in rendered.items():
        assert 'class="skip"' in html, slug
        assert '<nav class="toc"' in html, slug
        assert 'tabindex="0"' in html, slug
        assert 'aria-label=' in html, slug


def test_maps_scroll_instead_of_clipping(rendered: dict[str, str]):
    for slug, html in rendered.items():
        assert "overflow-x: auto" in html, slug
        assert "white-space: pre" in html, slug
        assert "font-variant-ligatures: none" in html, slug


def test_absent_receipts_render_as_not_recorded(rendered: dict[str, str]):
    ohco = rendered["ohco"]
    assert "not recorded" in ohco
    mapo = rendered["mapo-dungeongrid"]
    assert "recorded" in mapo
    assert all(receipt.present for receipt in BY_SLUG["mapo-dungeongrid"].receipts)
    assert not any(receipt.present for receipt in BY_SLUG["portable-container-system"].receipts)


def test_metadata_is_embedded_for_every_page(rendered: dict[str, str]):
    for page in PAGES:
        html = rendered[page.slug]
        assert f"<title>{page.title}</title>" in html
        assert 'name="generator" content="synth.optimizer_diagram.v1"' in html
        for field_ in page.metadata:
            assert field_.value in html


def test_no_raw_absolute_paths_are_used_as_labels(rendered: dict[str, str]):
    for slug, html in rendered.items():
        assert "/Users/" not in html, slug
        assert "/private/tmp" not in html, slug


def test_every_page_states_a_verdict_and_a_confidence():
    for page in PAGES:
        assert page.hypothesis and page.verdict and page.confidence and page.why
        assert len(page.verdict) > 10


def test_a_page_with_no_recorded_receipt_does_not_claim_proof():
    """The honesty gate: no evidence recorded ⇒ no confidence asserted."""

    hedges = ("not proven", "not yet proven", "specified", "outstanding", "not built", "not validated")
    for page in PAGES:
        if any(receipt.present for receipt in page.receipts):
            continue
        verdict = page.verdict.lower()
        assert any(hedge in verdict for hedge in hedges), f"{page.slug}: {page.verdict}"
        assert page.confidence.lower().startswith("none"), f"{page.slug}: {page.confidence}"


def test_mapo_is_the_only_page_claiming_proof():
    proving = [page.slug for page in PAGES if page.verdict.lower().startswith("proven")]
    assert proving == ["mapo-dungeongrid"]


def test_page_json_round_trips():
    for page in PAGES:
        payload = json.loads(json.dumps(page.to_dict()))
        assert payload["slug"] == page.slug
        assert len(payload["panels"]) == len(page.panels)
        assert payload["schema"] == "synth.optimizer_diagram.v1"


def test_index_links_every_page():
    index = render_index(PAGES)
    for page in PAGES:
        assert f'href="{page.slug}.html"' in index


def test_canvas_refuses_a_label_that_would_overwrite_a_box():
    c = Canvas(40, 6)
    left = c.box(0, 0, 12, 4)
    right = c.box(0, 16, 12, 4)
    with pytest.raises(CanvasError):
        c.connect_h(left, right, label="far too long a label")
    c.connect_h(left, right, label="ok")
    assert "▶" in c.render()


def test_canvas_refuses_to_draw_outside_its_bounds():
    c = Canvas(10, 3)
    with pytest.raises(CanvasError):
        c.text(0, 5, "far too long")


def test_render_escapes_untrusted_text():
    page = DiagramPage(
        slug="x", title="<script>alert(1)</script>", subtitle="s", optimizer="o",
        schema_version="v", verdict="verdict text", confidence="c", why="w", hypothesis="h",
    )
    html = render_page(page)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
