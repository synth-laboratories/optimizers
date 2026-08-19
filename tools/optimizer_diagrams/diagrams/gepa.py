"""GEPA — candidate, frontier, reflection and merge."""

from __future__ import annotations

from ..canvas import Canvas
from ..model import DiagramPage, MetaField, Panel, Receipt


def _search() -> str:
    c = Canvas(132, 26)
    c.text(0, 0, "GEPA SEARCH — parent selection, reflective proposal, evaluation, frontier update")

    pool = c.box(3, 0, 34, 9, title="CANDIDATE POOL §", lines=(
        "seed instruction modules",
        "per-candidate lineage",
        "per-example scores and",
        "aggregate scores",
        "frontier membership",
        "archive of rejected arms",
    ))
    select = c.box(3, 48, 30, 9, title="PARENT SELECTION", lines=(
        "frontier-aware sampling",
        "per-example dominance",
        "not a global argmax:",
        "a candidate that wins one",
        "example keeps its place",
    ))
    reflect = c.box(3, 92, 36, 9, title="REFLECTION / ASI", lines=(
        "failure traces for the",
        "selected module only",
        "proposer rewrites the",
        "instruction text",
        "merge combines two parents",
        "module-by-module",
    ))
    c.connect_h(pool, select, label="sample")
    c.connect_h(select, reflect, label="target")

    evalb = c.box(15, 48, 30, 8, style="double", title="EVALUATION (trust)", lines=(
        "minibatch on train split",
        "full train on promotion",
        "evaluator-owned scoring",
        "cache keyed by candidate",
    ))
    lane = 85  # clear of the sealed region on the right
    row = reflect.mid_row + 2
    c.hline(row, lane, reflect.left - 1)
    c.put(row, lane, "┐")
    c.vline(lane, row + 1, evalb.mid_row)
    c.hline(evalb.mid_row, evalb.right + 1, lane - 1)
    c.put(evalb.mid_row, lane, "┘")
    c.put(evalb.mid_row, evalb.right + 1, "◀")
    c.text(row - 1, lane - 9, "candidate")

    front = c.box(15, 0, 34, 8, title="FRONTIER UPDATE", lines=(
        "accept · reject · archive",
        "per-example Pareto retained",
        "regression on any example is",
        "recorded, not averaged away",
    ))
    c.connect_h(front, evalb, reverse=True, label="scores")
    c.connect_v(pool, front, col=front.left + 4, reverse=True)
    c.text(pool.bottom + 2, front.left + 6, "accepted §")

    seal = c.box(15, 92, 36, 8, style="heavy", title="SEALED HELDOUT", lines=(
        "never used for selection",
        "read once on the frozen",
        "winner, for the verdict only",
    ))
    c.text(24, 0, "Selection reads train evidence only; heldout is opened after the arms are frozen.")
    return c.render()


def _durability() -> str:
    c = Canvas(128, 22)
    c.text(0, 0, "GEPA DURABILITY — the v0.7 target mapping onto synth.optimizer_env.v1")

    svc = c.box(3, 0, 34, 8, title="GEPA SERVICE (shipped)", lines=(
        "local engine + hosted runs",
        "state slices: candidates,",
        "  frontier",
        "algorithm event stream",
        "tunnel or container pool",
    ))
    proto = c.box(3, 48, 36, 8, title="SHARED PROTOCOL", lines=(
        "observation ← pool, frontier,",
        "  evidence, budgets",
        "action → propose, select,",
        "  evaluate, accept, merge",
        "reward ← decomposed vector",
    ))
    ck = c.box(3, 94, 32, 8, title="CHECKPOINT §", lines=(
        "cursor · pool · frontier",
        "evaluator + cache identity",
        "model / prompt / decode pins",
        "container manifest digests",
    ))
    c.connect_h(svc, proto, label="map")
    c.connect_h(proto, ck, label="bind")

    gap = c.box(14, 0, 34, 6, title="KNOWN GAPS", lines=(
        "lossy pause / resume paths",
        "one-step not yet uniform",
        "reward not yet decomposed",
    ))
    step = c.box(14, 48, 36, 6, title="ONE PROPOSER STEP", lines=(
        "load checkpoint → one",
        "reflective proposal, no loop",
        "replayable vs a frozen evaluator",
    ))
    rl = c.box(14, 94, 32, 6, title="RLVR-COMPATIBLE EPISODE", lines=(
        "a real observation / action /",
        "reward transition — not a",
        "renamed prompt search",
    ))
    c.connect_h(gap, step, label="close")
    c.connect_h(step, rl, label="enables")
    c.text(20, 0, "Async and flash modes may advertise different capabilities, but the sync path stays the correctness baseline.")
    return c.render()


def build() -> DiagramPage:
    return DiagramPage(
        slug="gepa",
        title="GEPA",
        subtitle=(
            "Reflective prompt evolution over a per-example frontier, and what it takes to make "
            "it durable enough to be the reference optimizer environment."
        ),
        optimizer="GEPA",
        schema_version="synth.optimizer_env.v1 (target)",
        hypothesis=(
            "GEPA's existing candidate, frontier, reflection and merge behaviour already fits the "
            "shared optimizer-environment contract, and the remaining work is durability, not "
            "algorithm change."
        ),
        verdict="Search behaviour shipped. Protocol mapping and durability work outstanding.",
        confidence="None for the v0.7 claims — no GEPA lane was run in this campaign.",
        why=(
            "The hosted and local GEPA surfaces exist and expose candidate and frontier state, "
            "but nothing here has been mapped onto synth.optimizer_env.v1 or given a one-step "
            "operation yet."
        ),
        metadata=(
            MetaField("Repository", "optimizers", "branch v0.7/better-optimizer-visuals"),
            MetaField("Shipped surface", "local engine + hosted runs"),
            MetaField("State slices", "candidates · frontier"),
            MetaField("Protocol mapping", "not started"),
        ),
        panels=(
            Panel(
                slug="search",
                title="Search loop",
                caption=(
                    "Parent selection is frontier-aware rather than a global argmax, which is what "
                    "keeps a candidate that wins a single example alive."
                ),
                ascii_map=_search(),
                notes=(
                    "Per-example scores are retained, so a regression on one example is visible "
                    "instead of averaged away.",
                    "Reflection sees failure traces for the targeted module only.",
                ),
            ),
            Panel(
                slug="durability",
                title="Durability and protocol mapping",
                caption=(
                    "The v0.7 work: map the shipped service state onto the shared protocol, close "
                    "the lossy resume paths, and add the uniform one-step operation."
                ),
                ascii_map=_durability(),
                notes=(
                    "'RLVR-compatible' means a real observation/action/reward transition suitable "
                    "for policy training. It does not mean relabelling prompt optimization.",
                ),
            ),
        ),
        boundaries=(
            ("Policy-visible", "Pool, frontier, train evidence, budgets, allowed actions."),
            ("Sealed evaluation", "Heldout examples. Excluded from both observation and selection."),
            ("Operator-only", "Evaluator internals, cache keys, provider routing."),
        ),
        receipts=(
            Receipt("HealthBench hello-world", "", "bounded eval receipt"),
            Receipt("Craftax hello-world", "", "bounded eval receipt"),
            Receipt("pause → restart → continue", "", "reconciliation proof"),
            Receipt("one proposer step", "", "single reflective proposal from a checkpoint"),
            Receipt("RLVR-compatible episode", "", "one real transition with a decomposed reward"),
        ),
    )
