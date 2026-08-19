"""GELO — Go-Explore over a prompt space, extended to five candidate targets."""

from __future__ import annotations

from ..canvas import Canvas
from ..model import DiagramPage, MetaField, Panel, Receipt


def _archive() -> str:
    c = Canvas(128, 24)
    c.text(0, 0, "GELO — GO-EXPLORE ARCHIVE: cells, returns, mutations, admission")

    archive = c.box(3, 0, 36, 9, title="ARCHIVE / BOARD §", lines=(
        "cells keyed by a behaviour",
        "descriptor, not by score",
        "one incumbent per cell",
        "visit counts and returns",
        "lineage back to the seed",
        "public board rows only",
    ))
    ret = c.box(3, 50, 30, 9, title="RETURN", lines=(
        "pick a promising cell",
        "restore its state",
        "explore from there",
        "",
        "coverage beats greed:",
        "an unvisited cell beats",
        "a marginal score gain",
    ))
    mutate = c.box(3, 90, 34, 9, title="MUTATE", lines=(
        "proposer role rewrites the",
        "candidate for that cell",
        "mutation vocabulary is",
        "declared per target mode",
        "cache mode controls reuse",
    ))
    c.connect_h(archive, ret, label="select")
    c.connect_h(ret, mutate, label="explore")

    admit = c.box(15, 50, 30, 7, style="double", title="ADMISSION (trust)", lines=(
        "evaluate on train split",
        "reward mode declared",
        "cell reassignment allowed",
        "invalid evidence ≠ zero",
    ))
    lane = mutate.left + 6
    c.vline(lane, mutate.bottom + 1, admit.mid_row)
    c.hline(admit.mid_row, admit.right + 1, lane - 1)
    c.put(admit.mid_row, lane, "┘")
    c.put(admit.mid_row, admit.right + 1, "◀")
    c.text(mutate.bottom + 2, lane + 2, "candidate")
    c.connect_h(archive, admit, row=admit.mid_row, reverse=True, label="update §")
    c.text(22, 0, "A candidate that lands in a new cell is kept for coverage even when its score is lower than the incumbent's.")
    return c.render()


def _modes() -> str:
    c = Canvas(128, 27)
    c.text(0, 0, "FIVE TARGET MODES BEHIND ONE HONEST CAPABILITY MODEL")

    env = c.box(10, 0, 30, 7, title="CANDIDATE ENVELOPE", lines=(
        "common identity, lineage,",
        "cell descriptor, reward",
        "",
        "mode-specific payload below",
    ))

    modes = (
        (2, "prompt", "instruction modules", "SHIPPED (hosted)"),
        (6, "harness", "reviewed code patch", "sandboxed build + test"),
        (10, "sft", "training config → adapter", "capability preflight"),
        (14, "rlvr", "policy state + rollouts", "capability preflight"),
        (18, "distillation", "student + teacher traces", "fidelity evaluation"),
    )
    boxes = []
    for row, name, payload, status in modes:
        boxes.append(c.box(row, 44, 40, 4, title=name, lines=(f"{payload}", f"{status}")))
    bus = env.right + 3
    c.hline(env.mid_row, env.right + 1, bus)
    c.vline(bus, boxes[0].mid_row, boxes[-1].mid_row)
    for box in boxes:
        c.hline(box.mid_row, bus + 1, box.left - 1)
        c.put(box.mid_row, box.left - 1, "▶")
        c.put(box.mid_row, bus, "├")
    c.put(boxes[0].mid_row, bus, "┌")
    c.put(boxes[-1].mid_row, bus, "└")

    gate = c.box(10, 94, 32, 7, style="heavy", title="CAPABILITY GATE", lines=(
        "a mode is 'validated' only",
        "with a real run behind it",
        "",
        "preflight ≠ proof",
    ))
    for box in boxes:
        c.hline(box.mid_row, box.right + 1, gate.left - 3)
    c.vline(gate.left - 3, boxes[0].mid_row, boxes[-1].mid_row)
    for box in boxes:
        c.put(box.mid_row, gate.left - 3, "┤")
    c.hline(gate.mid_row, gate.left - 3, gate.left - 1)
    c.put(gate.mid_row, gate.left - 1, "▶")
    c.put(boxes[0].mid_row, gate.left - 3, "┐")
    c.put(boxes[-1].mid_row, gate.left - 3, "┘")
    c.text(25, 0, "One untyped payload for all five modes would hide exactly the differences that matter; each mode declares its own schema.")
    return c.render()


def build() -> DiagramPage:
    return DiagramPage(
        slug="gelo",
        title="GELO",
        subtitle=(
            "Go-Explore over a prompt space today, and the five-target capability model that "
            "turns it into a general exploration optimizer."
        ),
        optimizer="GELO",
        schema_version="synth.optimizer_env.v1 (target)",
        hypothesis=(
            "Coverage-first exploration over an archive of behaviour cells generalises from "
            "prompts to harness patches, training configs, policies and student models — behind "
            "one candidate envelope with per-mode schemas."
        ),
        verdict="Prompt mode shipped hosted. The other four modes are specified, not validated.",
        confidence="None for the four unvalidated modes; no v0.7 GELO lane was run.",
        why=(
            "The public package exposes hosted GELO with a board state slice and Go-Explore "
            "prompt-space configuration. Harness, SFT, RLVR and distillation exist as design "
            "targets with no run behind them."
        ),
        metadata=(
            MetaField("Repository", "optimizers", "branch v0.7/better-optimizer-visuals"),
            MetaField("Shipped surface", "hosted only", "no public local executor"),
            MetaField("State slice", "board"),
            MetaField("Validated modes", "1 of 5", "prompt"),
        ),
        panels=(
            Panel(
                slug="archive",
                title="Archive, return and mutate",
                caption=(
                    "The Go-Explore loop: cells are keyed by behaviour rather than score, so "
                    "coverage survives contact with a greedy objective."
                ),
                ascii_map=_archive(),
                notes=(
                    "Cache mode, reward mode and checkpoint semantics are declared configuration, "
                    "not implicit behaviour.",
                    "Public board rows must not leak internal paths.",
                ),
            ),
            Panel(
                slug="modes",
                title="Five target modes",
                caption=(
                    "One candidate envelope, five payload schemas, and a capability gate that "
                    "refuses to call a mode validated without a real run."
                ),
                ascii_map=_modes(),
                notes=(
                    "If a task does not expose a methodologically sound RLVR action/reward "
                    "contract, the honest move is to record that limitation, not to invent one.",
                ),
            ),
        ),
        boundaries=(
            ("Policy-visible", "Board rows, cell descriptors, train evidence, budgets."),
            ("Operator-only", "Cache keys, workspace paths, proposer routing and billing mode."),
            ("Sealed evaluation", "Heldout evidence for the frozen winner."),
            ("Sandbox", "Harness and SFT modes execute candidate code or training under a declared sandbox and network policy."),
        ),
        receipts=(
            Receipt("HealthBench hello-world", "", "bounded eval receipt"),
            Receipt("Craftax hello-world", "", "bounded eval receipt"),
            Receipt("harness mode", "", "real bounded patch → test → eval loop"),
            Receipt("RLVR mode", "", "one bounded run with a real reward contract"),
            Receipt("pause / restart / fork", "", "durability proof"),
        ),
    )
