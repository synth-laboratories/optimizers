"""VZPO — verifier-driven optimization behind a Chinese wall."""

from __future__ import annotations

from ..canvas import Canvas
from ..model import DiagramPage, MetaField, Panel, Receipt


def _wall() -> str:
    c = Canvas(128, 27)
    c.text(0, 0, "VZPO — THE CHINESE WALL: what the optimizer may read from the verifier, and nothing more")

    opt = c.box(4, 0, 34, 10, title="OPTIMIZER SIDE", lines=(
        "candidate prompts,",
        "preference pairs or",
        "reward-model configs",
        "lineage and budgets",
        "",
        "sees only the four fields",
        "the proxy is allowed to",
        "return",
    ))
    proxy = c.box(4, 50, 30, 10, style="double", title="VERIFIER PROXY", lines=(
        "capability-scoped",
        "returns:",
        "  score",
        "  rank",
        "  weakest dimension",
        "  allowed hints",
        "",
        "nothing else crosses",
    ))
    verifier = c.box(4, 90, 34, 10, style="heavy", title="VERIFIER (sealed)", lines=(
        "rubric internals",
        "per-criterion detail",
        "heldout essays",
        "grader prompts",
        "",
        "unreachable from the",
        "optimizer side by",
        "construction",
    ))
    c.connect_h(opt, proxy, label="candidate")
    c.connect_h(proxy, verifier, label="scoped")

    reward = c.box(17, 50, 30, 6, title="VERIFIER REWARD", lines=(
        "the training signal",
        "may saturate",
        "fidelity is reported",
    ))
    c.connect_v(proxy, reward, col=proxy.mid_col)

    grader = c.box(17, 90, 34, 6, style="heavy", title="FROZEN HELDOUT GRADER", lines=(
        "the outcome measure",
        "never the training signal",
        "read once, after freezing",
    ))
    c.connect_h(reward, grader, label="distinct")

    pins = c.box(17, 0, 34, 6, title="PINS AND CONTROLS", lines=(
        "model · decode · repeats",
        "sample size · null control",
        "blinded comparison",
    ))
    c.connect_h(pins, reward, label="carried")
    c.text(24, 0, "Verifier reward and frozen heldout outcome are different quantities; reporting")
    c.text(25, 0, "one as the other is the failure this wall exists to prevent.")
    return c.render()


def _lanes() -> str:
    c = Canvas(132, 28)
    c.text(0, 0, "THREE VZPO LANES ON ESSAYBENCH — substrate proven, lift still open")

    prompt = c.box(3, 0, 36, 6, title="REB-063 · prompt / GEPA", lines=(
        "verifier-scored prompt search",
        "positive blinded result exists",
        "historical evidence, not proof",
        "of the new hosted adapter",
    ))
    dpo = c.box(11, 0, 36, 6, title="REB-064 · Zeno-ranked DPO", lines=(
        "preference pairs from ranks",
        "substrate proven",
        "measured lift: open",
    ))
    rlvr = c.box(19, 0, 36, 6, title="REB-065 · Zeno-reward RLVR", lines=(
        "verifier reward for GRPO",
        "substrate proven",
        "measured lift: open",
    ))

    adapter = c.box(11, 52, 32, 6, title="HOSTED ADAPTER", lines=(
        "one optimizer with modes,",
        "or a family with distinct",
        "algorithm ids — decide and",
        "write it down",
    ))
    for box in (prompt, dpo, rlvr):
        c.hline(box.mid_row, box.right + 1, adapter.left - 3)
        c.put(box.mid_row, adapter.left - 3, "┤")
    c.vline(adapter.left - 3, prompt.mid_row, rlvr.mid_row)
    c.put(prompt.mid_row, adapter.left - 3, "┐")
    c.put(rlvr.mid_row, adapter.left - 3, "┘")
    c.hline(adapter.mid_row, adapter.left - 3, adapter.left - 1)
    c.put(adapter.mid_row, adapter.left - 1, "▶")

    risk = c.box(11, 94, 34, 6, title="KNOWN BLOCKER", lines=(
        "provider TPM retry and",
        "cooldown defect: bound it",
        "or stop cleanly and record",
        "the exact resume point",
    ))
    c.connect_h(adapter, risk, label="risk")
    c.text(26, 0, "Do not claim DPO or RLVR lift. Their substrates are proven; the measured effect is not.")
    return c.render()


def build() -> DiagramPage:
    return DiagramPage(
        slug="vzpo",
        title="VZPO",
        subtitle=(
            "Verifier-driven prompt, preference and reward optimization on EssayBench, with a "
            "hard wall between the training signal and the outcome measure."
        ),
        optimizer="VZPO",
        schema_version="synth.optimizer_env.v1 (target)",
        hypothesis=(
            "A capability-scoped verifier proxy can drive prompt, preference and reward "
            "optimization without ever exposing rubric internals or heldout essays."
        ),
        verdict="Measurement spine exists as REB evidence. The hosted adapter is not built.",
        confidence="None for the v0.7 adapter; the REB-063 prompt result is historical evidence, not proof of this adapter.",
        why=(
            "REB-063/064/065 establish the wall, the pins and the substrates. Extracting them "
            "into a reusable hosted adapter, and bounding the provider TPM retry defect, is "
            "outstanding work."
        ),
        metadata=(
            MetaField("Reference tasks", "REB-063 · REB-064 · REB-065"),
            MetaField("Benchmark", "EssayBench"),
            MetaField("Adapter status", "not built"),
            MetaField("Known blocker", "provider TPM retry / cooldown"),
        ),
        panels=(
            Panel(
                slug="wall",
                title="The verifier wall",
                caption=(
                    "Four fields cross the proxy: score, rank, weakest dimension, and the hints "
                    "explicitly allowed. Everything else stays sealed."
                ),
                ascii_map=_wall(),
                notes=(
                    "Verifier reward may saturate; reward-model fidelity is reported alongside it "
                    "rather than assumed.",
                    "Selection never reads the frozen heldout grader.",
                ),
            ),
            Panel(
                slug="lanes",
                title="Three lanes, one adapter",
                caption=(
                    "Whether VZPO is one optimizer with modes or a family with distinct algorithm "
                    "ids is an open decision that has to be written down before the adapter ships."
                ),
                ascii_map=_lanes(),
                notes=(
                    "Start with the smallest methodologically valid prompt-lane test; escalate "
                    "only once the wall audit passes.",
                ),
            ),
        ),
        boundaries=(
            ("Optimizer side", "Candidates, lineage, budgets, and the four proxy fields."),
            ("Verifier proxy", "Capability-scoped. Enumerates exactly what may cross."),
            ("Sealed verifier", "Rubric internals, per-criterion detail, grader prompts."),
            ("Sealed heldout", "Heldout essays and the frozen grader outcome."),
        ),
        receipts=(
            Receipt("REB-063 prompt result", "historical", "positive blinded result; predates this adapter"),
            Receipt("hosted adapter run", "", "wall audit, calls, tokens, lineage, checkpoint"),
            Receipt("heldout evaluation", "", "frozen grader, read once"),
            Receipt("pause / restart / one-step", "", "durability proof"),
        ),
    )
