"""OHCO — harness-candidate code optimization."""

from __future__ import annotations

from ..canvas import Canvas
from ..model import DiagramPage, MetaField, Panel, Receipt


def _candidate_contract() -> str:
    c = Canvas(130, 26)
    c.text(0, 0, "OHCO HARNESS CANDIDATE — a real code patch, admitted before it costs environment spend")

    base = c.box(3, 0, 34, 10, title="BASE + SCOPE", lines=(
        "repository · full commit sha",
        "allowed subtree (allowlist)",
        "parent candidate id",
        "patch digest sha256:…",
        "code size / file / path caps",
        "explicit harness objective",
        "immutable container + source",
        "bundle digests",
    ))
    ws = c.box(3, 50, 32, 10, title="ISOLATED WORKSPACE", lines=(
        "candidate-private checkout",
        "declared sandbox",
        "declared network policy",
        "apply patch",
        "",
        "rollback proof and",
        "workspace-integrity check",
    ))
    gate = c.box(3, 94, 34, 10, style="double", title="ADMISSION GATE", lines=(
        "build",
        "typecheck",
        "unit tests",
        "integration tests",
        "environment QA",
        "",
        "a broken candidate is",
        "rejected here, not scored",
    ))
    c.connect_h(base, ws, label="materialise")
    c.connect_h(ws, gate, label="build+test")

    run = c.box(16, 50, 32, 7, title="PINNED CRAFTAX CONTAINER", lines=(
        "fixed seeds",
        "evidence gates",
        "traces · rewards · steps",
        "achievements · errors",
    ))
    c.connect_h(run, gate, row=run.mid_row, reverse=True, label="admitted")

    verdict = c.box(16, 0, 34, 7, title="BASELINE vs CANDIDATE", lines=(
        "matched seeds only",
        "patch provenance retained",
        "improvement hypothesis is",
        "resolved honestly",
    ))
    c.connect_h(verdict, run, reverse=True, label="evidence")
    c.text(24, 0, "The candidate never sees the evaluator, the heldout seeds, the reward implementation or the evidence gates.")
    return c.render()


def _protected() -> str:
    c = Canvas(120, 20)
    c.text(0, 0, "WHAT OHCO MAY AND MAY NOT TOUCH")

    may = c.box(3, 0, 44, 9, title="WRITABLE (allowlisted subtree)", lines=(
        "CodeAct harness / library surface",
        "observation encoding",
        "tool and action wiring",
        "planning and retry logic",
        "prompt scaffolding inside the harness",
        "configuration the harness owns",
    ))
    never = c.box(3, 60, 46, 9, style="heavy", title="PROTECTED (refusal, not warning)", lines=(
        "evaluator implementation",
        "heldout seeds and splits",
        "reward implementation",
        "evidence gates",
        "protected task files",
        "the benchmark itself",
    ))
    c.connect_h(may, never, label="refused")

    note = c.box(14, 0, 44, 5, title="WHY THIS IS THE WHOLE POINT", lines=(
        "a candidate that can edit the benchmark",
        "will edit the benchmark",
    ))
    c.text(14, 60, "A patch that touches a protected path is rejected")
    c.text(15, 60, "at admission, before any container is launched, so")
    c.text(16, 60, "the refusal costs nothing and leaves a receipt.")
    return c.render()


def build() -> DiagramPage:
    return DiagramPage(
        slug="ohco",
        title="OHCO",
        subtitle=(
            "Optimizing the harness rather than the prompt: real code patches, admitted by build "
            "and test before they are allowed to cost environment spend."
        ),
        optimizer="OHCO",
        schema_version="synth.optimizer_env.v1 (target)",
        hypothesis=(
            "Measured Craftax performance is limited by the CodeAct harness — observation "
            "encoding, tool wiring, retry logic — more than by the instruction text, so a real "
            "code patch should beat a prompt rewrite."
        ),
        verdict="Specified. Not proven: no OHCO Craftax lane was run in this campaign.",
        confidence="None.",
        why=(
            "OHCO is a private hosted compatibility lane in the public package today. The "
            "candidate contract and the protected-surface rules below are design, and the one "
            "place the pattern has actually been executed is MAPO's harness lane on DungeonGrid."
        ),
        metadata=(
            MetaField("Repository", "optimizers", "branch v0.7/better-optimizer-visuals"),
            MetaField("Public status", "future hosted lane", "no public local executor"),
            MetaField("Proof target", "Craftax CodeAct harness"),
            MetaField("Executed analogue", "MAPO harness lane", "DungeonGrid, heldout +0.1486"),
        ),
        panels=(
            Panel(
                slug="candidate-contract",
                title="Harness candidate contract",
                caption=(
                    "Every stage a patch must survive before a single rollout is spent on it, and "
                    "the provenance that follows it through."
                ),
                ascii_map=_candidate_contract(),
                notes=(
                    "Rejecting broken candidates before environment spend is what makes a code-"
                    "editing optimizer affordable at all.",
                    "The MAPO harness lane implements exactly this shape end to end: patch ops → "
                    "workspace → py_compile → contract test → rollout, with a unified diff and a "
                    "patch digest in the receipt.",
                ),
            ),
            Panel(
                slug="protected",
                title="Writable and protected surfaces",
                caption=(
                    "The allowlist is the safety property. A candidate that can edit the "
                    "benchmark will eventually edit the benchmark."
                ),
                ascii_map=_protected(),
                notes=(
                    "Refusal happens at admission, so it costs nothing and still leaves a receipt.",
                ),
            ),
        ),
        boundaries=(
            ("Writable subtree", "The declared harness allowlist and nothing else."),
            ("Protected", "Evaluator, heldout seeds, reward implementation, evidence gates, protected task files."),
            ("Sandbox", "Candidate code runs under a declared sandbox and network policy, in a candidate-private workspace."),
            ("Sealed evaluation", "Heldout seeds, opened only for the frozen winner."),
        ),
        receipts=(
            Receipt("Craftax baseline", "", "fixed seeds, frozen harness"),
            Receipt("real patch candidate", "", "unified diff + patch digest"),
            Receipt("build / test admission", "", "rejected candidates cost zero rollouts"),
            Receipt("pause / restart", "", "durability proof"),
            Receipt("one proposer step", "", "single patch proposal from a checkpoint"),
        ),
    )
