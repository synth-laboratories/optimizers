"""The portable container bundle standard shared by every task runtime."""

from __future__ import annotations

from ..canvas import Canvas
from ..model import DiagramPage, MetaField, Panel, Receipt


def _binding() -> str:
    c = Canvas(128, 26)
    c.text(0, 0, "ONE MANIFEST BINDS AN IMMUTABLE IMAGE TO A CONTENT-ADDRESSED SOURCE BUNDLE")

    build = c.box(3, 0, 34, 8, title="BUILD (clean revision)", lines=(
        "repository · full commit sha",
        "build recipe digest",
        "builder identity",
        "excludes .git, creds, caches,",
        "venvs, weights, local outputs",
    ))
    registry = c.box(3, 50, 34, 8, style="double", title="REGISTRY (immutable)", lines=(
        "image reference",
        "digest sha256:…",
        "platforms: amd64, arm64",
        "",
        "mutable tags are display-only",
    ))
    store = c.box(13, 50, 34, 8, style="double", title="S3 SOURCE BUNDLE", lines=(
        "content-addressed key",
        "digest sha256:…",
        "media type + size bytes",
        "",
        "digest computed before upload",
    ))
    c.connect_h(build, registry, label="push image")
    c.connect_h(build, store, row=build.mid_row + 10, label="upload src")
    c.vline(build.mid_col, build.bottom + 1, store.mid_row)
    c.hline(store.mid_row, build.mid_col + 1, store.left - 1)
    c.put(store.mid_row, build.mid_col, "└")
    c.put(store.mid_row, store.left - 1, "▶")

    manifest = c.box(8, 90, 36, 12, title="synth.container_bundle.v1", lines=(
        "bundleId · revision",
        "image{reference, digest}",
        "source{uri, digest, mediaType}",
        "contracts{taskFamily,",
        "  capabilities, optimizer}",
        "runtime{command, ports,",
        "  healthcheck, timeout}",
        "security · provenance · license",
        "manifest digest sha256:…",
    ))
    c.connect_h(registry, manifest, row=registry.mid_row, label="bind")
    c.connect_h(store, manifest, row=store.mid_row, label="bind")
    c.text(23, 0, "Generic orchestration dispatches on declared capabilities — never on a benchmark name.")
    c.text(24, 0, "A mismatch is a refusal to launch, never a pin quietly rewritten to match a local tag.")
    return c.render()


def _launcher() -> str:
    c = Canvas(130, 24)
    c.text(0, 0, "ONE DOWNLOADER / LAUNCHER — the same eleven steps for HealthBench, Craftax, EssayBench and DungeonGrid")

    resolve = c.box(3, 0, 30, 9, title="1–4  RESOLVE + VERIFY", lines=(
        "resolve manifest",
        "check authorization",
        "download / cache source",
        "verify digest",
        "safe extract:",
        "  no traversal,",
        "  no symlink escape",
    ))
    pull = c.box(3, 44, 30, 9, title="5–6  IMAGE + PREFLIGHT", lines=(
        "pull exact image digest",
        "validate platform",
        "validate resources, ports",
        "validate network policy",
        "resolve secret references",
        "  (never logged)",
    ))
    start = c.box(3, 88, 30, 9, title="7–9  START + RECORD", lines=(
        "isolated run root",
        "wait on declared health",
        "and capability contract",
        "record container id, ports,",
        "image / source / manifest",
        "digests, log paths",
    ))
    c.connect_h(resolve, pull, label="verified")
    c.connect_h(pull, start, label="admitted")

    stop = c.box(15, 46, 30, 6, title="10–11  STOP + RESTART", lines=(
        "stop descendants",
        "clean temporary state",
        "restart from same bundle",
    ))
    c.connect_v(pull, stop, col=pull.mid_col)

    conf = c.box(15, 0, 34, 6, title="CONFORMANCE SUITE", lines=(
        "403 · 404 · timeout · cut body",
        "arch mismatch · no capability",
        "cache corruption · concurrent",
    ))
    c.connect_h(conf, stop, reverse=True)

    prov = c.box(15, 84, 36, 6, title="PROVENANCE RECORD §", lines=(
        "every run cites the manifest",
        "digest it actually launched",
        "round-trips back to the build",
    ))
    c.connect_h(stop, prov)
    c.text(22, 0, "Refuse to launch when the image, source, architecture, task contract or optimizer capability is unavailable.")
    return c.render()


def build() -> DiagramPage:
    return DiagramPage(
        slug="portable-container-system",
        title="Portable container bundle standard",
        subtitle=(
            "One immutable image, one content-addressed source bundle, one manifest that binds "
            "them, and one launcher that verifies and records everything it did."
        ),
        optimizer="Shared platform",
        schema_version="synth.container_bundle.v1",
        hypothesis=(
            "Every supported task runtime can be launched, verified and restarted through one "
            "capability-dispatched path, with no benchmark-name conditionals in generic code."
        ),
        verdict="Specified. Not yet proven end to end in this campaign.",
        confidence="None — no bundle has been built, uploaded or launched under this schema yet.",
        why=(
            "The v0.7 MAPO lanes ran against the in-process DungeonGrid environment, not through "
            "a published bundle, so every receipt row below is honestly empty."
        ),
        metadata=(
            MetaField("Manifest schema", "synth.container_bundle.v1"),
            MetaField("Status", "design", "no bundle materialised"),
            MetaField("Task families", "healthbench · craftax · essaybench · dungeongrid"),
            MetaField("Dispatch rule", "declared capabilities, never task names"),
        ),
        panels=(
            Panel(
                slug="binding",
                title="Image and source binding",
                caption=(
                    "The manifest is the only place the image digest, the source digest, the task "
                    "contract and the build revision meet. Everything downstream cites its digest."
                ),
                ascii_map=_binding(),
                notes=(
                    "Mutable tags are display-only; an immutable registry digest is required.",
                    "Credentials never enter the source bundle or the manifest — only secret "
                    "references, resolved at launch and never logged.",
                ),
            ),
            Panel(
                slug="launcher",
                title="Downloader and launcher",
                caption=(
                    "One CLI path, eleven ordered steps, and a conformance suite that exercises "
                    "the failure half of each one."
                ),
                ascii_map=_launcher(),
                notes=(
                    "Restart uses the same immutable bundle, so a restarted container is the same "
                    "container by construction rather than by convention.",
                ),
            ),
        ),
        boundaries=(
            ("Registry", "Immutable. A digest is required; a tag is a label for humans."),
            ("Object store", "Content-addressed. Bytes are verified before extraction, not after."),
            ("Secrets", "Referenced by name in the manifest, resolved at launch, never written to logs or bundles."),
            ("Generic orchestration", "May read declared capabilities only. A benchmark-name branch here is a defect."),
        ),
        receipts=(
            Receipt("HealthBench bundle", "", "manifest, image digest, source digest"),
            Receipt("Craftax bundle", "", "manifest, image digest, source digest"),
            Receipt("EssayBench bundle", "", "manifest, image digest, source digest"),
            Receipt("Conformance suite", "", "schema, security, failure-injection and restart tests"),
        ),
    )
