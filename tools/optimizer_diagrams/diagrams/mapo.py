"""MAPO on multi-agent DungeonGrid."""

from __future__ import annotations

from ..canvas import Canvas
from ..model import DiagramPage, MetaField, Panel, Receipt


def _control_plane() -> str:
    c = Canvas(146, 36)
    c.text(0, 0, "OPTIMIZER CONTROL PLANE — one episode model, three candidate varieties")
    c.text(1, 0, "policy-visible surface on the left, sealed and protected regions fenced on the right")

    obs = c.box(4, 0, 40, 10, title="OBSERVATION §", lines=(
        "sequence · digest",
        "incumbent · candidates · frontier",
        "train evidence + uncertainty",
        "failure clusters · pending work",
        "budgets · allowed actions",
        "target modules · versions",
        "policy-visible fields only",
    ))
    pol = c.box(4, 54, 36, 10, title="PROPOSER POLICY", lines=(
        "deterministic grid search",
        "  (spend-free default)",
        "or a learned / LLM proposer",
        "",
        "emits exactly ONE",
        "typed action per step",
    ))
    act = c.box(4, 100, 40, 10, title="ACTION", lines=(
        "propose_candidate",
        "allocate_evaluation",
        "accept · reject · defer",
        "select_parent · annotate",
        "checkpoint · pause · resume",
        "fork · stop",
    ))
    c.connect_h(obs, pol, label="observe")
    c.connect_h(pol, act, label="decide")

    adm = c.box(17, 100, 40, 8, title="ADMISSION GATE", lines=(
        "schema validation",
        "harness: patch → build → contract",
        "protected-surface refusal",
        "rejected ⇒ zero rollout spend",
    ))
    c.connect_v(act, adm, label="admit")

    ev = c.box(17, 54, 36, 8, style="double", title="EVALUATOR (trust)", lines=(
        "owns seeds and reward",
        "train / selection / heldout",
        "paired (quest, seed) compare",
        "evidence cache, no re-charge",
    ))
    c.connect_h(ev, adm, label="admitted", reverse=True)

    env = c.box(17, 0, 40, 8, title="DUNGEONGRID ROLLOUTS", lines=(
        "bounded multi-agent episodes",
        "per-agent obs · actions · messages",
        "decomposed reward vector",
        "failure signals",
    ))
    c.connect_h(env, ev, label="allocate", reverse=True)

    ck = c.box(28, 0, 40, 7, title="OPTIMIZER CHECKPOINT §", lines=(
        "cursor · candidates · frontier",
        "budgets · RNG · evaluator id",
        "actor + harness + container ids",
        "self-verifying sha256 digest",
    ))
    c.connect_v(env, ck, label="evidence §")

    seal = c.box(28, 54, 36, 7, style="heavy", title="SEALED HELDOUT", lines=(
        "heldout seeds: digest + count",
        "generalization reward",
        "read once, after arms freeze",
        "never enters OBSERVATION",
    ))
    c.connect_v(ev, seal)

    fork = c.box(28, 100, 40, 7, title="FORK · RESTART · ONE-STEP", lines=(
        "restore ⇒ identical state digest",
        "fork ⇒ new episode id + lineage",
        "step --max-actions 1",
        "no double scoring or charging",
    ))
    c.connect_h(seal, fork, label="frozen")
    c.connect_v(adm, fork, col=adm.left + 4)

    # feedback lane on the far right, clear of every box
    lane = 144
    entry = obs.left + 4
    c.vline(lane, 3, fork.mid_row)
    c.hline(3, entry, lane)
    c.put(3, lane, "┐")
    c.put(fork.mid_row, lane, "┘")
    c.hline(fork.mid_row, fork.right + 1, lane - 1)
    c.put(3, entry, "┌")
    c.put(4, entry, "▼")
    c.text(2, 60, "next observation · new transition sequence linked to the checkpoint")
    return c.render()


def _multi_agent_step() -> str:
    c = Canvas(132, 38)
    c.text(0, 0, "ONE DUNGEONGRID TRANSITION — two heroes, one Warden, one shared world")

    h1 = c.box(3, 0, 34, 9, title="HERO_1 · scout", lines=(
        "local observation:",
        "  visible tiles, objects,",
        "  teammates, AP, messages",
        "brief directive: claim",
        "policy v2 · sha256:df83…",
    ))
    board = c.box(3, 48, 36, 9, title="SHARED / TEAM STATE §", lines=(
        "claims{target → hero}",
        "reported tiles · threats",
        "notes · routing events",
        "harness-owned;",
        "not environment state",
    ))
    h2 = c.box(3, 98, 34, 9, title="HERO_2 · rear guard", lines=(
        "local observation:",
        "  visible tiles, objects,",
        "  teammates, AP, messages",
        "brief directive: claim",
        "policy v2 · sha256:df83…",
    ))
    c.connect_h(h1, board, row=6, label="claim:vault")
    c.connect_h(board, h2, row=8, label="reads claim")
    c.text(1, 0, "communication edge ──▶ delivered by the environment message bus and counted in the reward vector")

    harness = c.box(15, 16, 100, 8, title="HARNESS — the only patchable surface", lines=(
        "encode_observation     route_message        ingest_message",
        "arbitrate              note_step            recover",
        "should_branch          extract_failure_signals",
        "assign_roles           snapshot / restore",
    ))
    c.connect_v(h1, harness, col=h1.mid_col, label="action")
    c.connect_v(h2, harness, col=h2.mid_col, label="action")
    c.connect_v(board, harness, col=board.mid_col)

    env = c.box(25, 16, 100, 5, style="double", title="DUNGEONGRID ENVIRONMENT (protected)", lines=(
        "rules_engine · grid_engine · achievements · message bus · Warden policy",
        "transition: world state, RNG, reward, per-hero ledger, trace record",
    ))
    c.connect_v(harness, env, col=harness.mid_col)
    c.text(24, harness.mid_col + 2, "admitted actions")

    r1 = c.box(32, 0, 34, 5, title="REWARD → HERO_1", lines=("env per-hero ledger", "counterfactual estimate"))
    rt = c.box(32, 48, 36, 5, title="TEAM REWARD VECTOR", lines=("env_return · outcome · progress", "cooperation · comms · penalties"))
    r2 = c.box(32, 98, 34, 5, title="REWARD → HERO_2", lines=("env per-hero ledger", "counterfactual estimate"))
    c.vline(env.left - 2, env.mid_row, r1.top - 1)
    c.hline(env.mid_row, env.left - 2, env.left - 1)
    c.put(env.mid_row, env.left - 2, "┌")
    c.hline(r1.top - 1, r1.mid_col, env.left - 2)
    c.put(r1.top - 1, env.left - 2, "└")
    c.put(r1.top - 1, r1.mid_col, "▼")
    c.vline(env.right + 2, env.mid_row, r2.top - 1)
    c.hline(env.mid_row, env.right + 1, env.right + 2)
    c.put(env.mid_row, env.right + 2, "┐")
    c.hline(r2.top - 1, env.right + 2, r2.mid_col)
    c.put(r2.top - 1, env.right + 2, "┘")
    c.put(r2.top - 1, r2.mid_col, "▼")
    c.connect_v(env, rt, col=rt.mid_col)
    c.text(37, 0, "team reward, per-agent attribution and counterfactual contribution stay three "
                  "separate quantities")
    return c.render()


def _branch_lattice() -> str:
    c = Canvas(128, 27)
    c.text(0, 0, "BRANCH CHECKPOINTS — the environment-level pause, restart and fork lattice")

    root = c.box(6, 0, 32, 7, title="ROLLOUT (parent)", lines=(
        "quest · seed · env revision",
        "harness + policy snapshots",
        "step index 0 … n",
        "branch policy decides where",
    ))
    br = c.box(6, 48, 36, 7, style="heavy", title="BRANCH CHECKPOINT §", lines=(
        "world state + RNG state",
        "agent states · roles · versions",
        "messages + pending queue",
        "parent rollout id · step index",
    ))
    c.connect_h(root, br, label="should_branch")

    f1 = c.box(2, 94, 34, 5, title="FORK A", lines=("resume → identical digest", "team reward recorded"))
    f2 = c.box(10, 94, 34, 5, title="FORK B", lines=("resume → identical digest", "compared, never re-scored"))
    c.hline(f1.mid_row, br.right + 3, f1.left - 1)
    c.hline(f2.mid_row, br.right + 3, f2.left - 1)
    c.vline(br.right + 3, f1.mid_row, f2.mid_row)
    c.hline(br.mid_row, br.right + 1, br.right + 3)
    c.put(f1.mid_row, br.right + 3, "┌")
    c.put(f2.mid_row, br.right + 3, "└")
    c.put(br.mid_row, br.right + 3, "┤")
    c.put(f1.mid_row, f1.left - 1, "▶")
    c.put(f2.mid_row, f2.left - 1, "▶")
    c.text(1, 90, "two forks, one immutable parent")

    loo = c.box(18, 48, 34, 6, title="LEAVE-ONE-OUT CREDIT", lines=(
        "one fork per hero,",
        "that hero forced passive",
        "Δ team reward = contribution",
    ))
    c.connect_v(br, loo)

    guard = c.box(18, 0, 32, 6, title="IMMUTABILITY GUARD", lines=(
        "digest before forks",
        "== digest after forks",
        "restore deep-copies state",
    ))
    c.connect_h(guard, loo, reverse=True)

    step = c.box(18, 94, 34, 6, title="ONE PROPOSER STEP", lines=(
        "load checkpoint → 1 action",
        "optionally 1 bounded eval",
        "the loop never continues",
    ))
    c.connect_h(loo, step)
    c.text(25, 0, "A fork is a new episode id carrying parentEpisodeId and parentCheckpointDigest; "
                  "the parent history is never rewritten.")
    return c.render()


def _varieties() -> str:
    c = Canvas(124, 26)
    c.text(0, 0, "THREE CANDIDATE VARIETIES — one envelope, one checkpoint, one evaluator")

    env = c.box(9, 0, 32, 7, title="CANDIDATE ENVELOPE", lines=(
        "id · variety · generation",
        "parent · proposer · digest",
        "admission · scores · status",
        "target modules",
    ))
    p = c.box(3, 52, 34, 5, title="prompt_protocol", lines=(
        "role briefs · protocol mode",
        "target: what heroes say",
    ))
    h = c.box(10, 52, 34, 5, title="harness", lines=(
        "unified diff to harness.py",
        "target: how the team is run",
    ))
    r = c.box(17, 52, 34, 5, title="rlvr", lines=(
        "policy weight vector",
        "target: what heroes choose",
    ))
    bus_out = env.right + 4
    c.hline(env.mid_row, env.right + 1, bus_out)
    c.vline(bus_out, p.mid_row, r.mid_row)
    for box in (p, h, r):
        c.hline(box.mid_row, bus_out, box.left - 1)
        c.put(box.mid_row, box.left - 1, "▶")
    c.put(p.mid_row, bus_out, "┌")
    c.put(r.mid_row, bus_out, "└")
    c.text(1, bus_out - 4, "dispatch")

    ev = c.box(10, 92, 32, 5, title="ONE EVALUATOR", lines=(
        "same seeds and reward,",
        "same sealed heldout",
    ))
    bus_in = ev.left - 4
    c.vline(bus_in, p.mid_row, r.mid_row)
    for box in (p, h, r):
        c.hline(box.mid_row, box.right + 1, bus_in)
    c.put(p.mid_row, bus_in, "┐")
    c.put(r.mid_row, bus_in, "┘")
    c.put(h.mid_row, bus_in, "┤")
    c.text(1, bus_in - 3, "collect")
    c.hline(ev.mid_row, bus_in, ev.left - 1)
    c.put(ev.mid_row, ev.left - 1, "▶")

    c.text(24, 0, "The episode state machine never branches on these names; it dispatches through "
                  "the variety registry.")
    return c.render()


def build() -> DiagramPage:
    return DiagramPage(
        slug="mapo-dungeongrid",
        title="MAPO on multi-agent DungeonGrid",
        subtitle=(
            "Multi-agent prompt/protocol, harness and RLVR optimization behind one episode "
            "model, one checkpoint format and one evaluator."
        ),
        optimizer="MAPO",
        schema_version="dungeongrid.mapo.episode.v1 · synth.optimizer_env.v1",
        hypothesis=(
            "Coordination structure — what heroes say, how the team is run, and what they "
            "choose — is the binding constraint on DungeonGrid team reward, not individual "
            "hero skill."
        ),
        verdict="Proven in all three varieties on bounded paired-seed runs.",
        confidence=(
            "Low-to-moderate. The direction is consistent — 6 of 6 matched heldout seeds win in "
            "every lane — but the prompt/protocol and RLVR lanes show zero cross-seed variance, "
            "so those six seeds replicate a near-deterministic outcome rather than sampling a "
            "noisy one. Only the harness lane has real seed spread (sd 0.027). One quest per "
            "lane; scripted and linear policies only."
        ),
        why=(
            "Each lane's champion was selected on train seeds and then improved on disjoint "
            "sealed heldout seeds: +1.13 (prompt/protocol), +0.15 (harness), +0.08 (RLVR)."
        ),
        metadata=(
            MetaField("Repository", "DungeonGrid", "branch v0.7/mapo-dungeongrid"),
            MetaField("Optimizer schema", "dungeongrid.mapo.episode.v1"),
            MetaField("Environment schema", "synth.optimizer_env.v1"),
            MetaField("Harness API", "dungeongrid.mapo.harness.v1"),
            MetaField("Reward schema", "dungeongrid.mapo.reward.v1"),
            MetaField("Provider spend", "none", "scripted and linear policies only"),
        ),
        panels=(
            Panel(
                slug="control-plane",
                title="Optimizer control plane",
                caption=(
                    "The observation → proposer → action → admission → evaluation → checkpoint "
                    "loop. Everything a candidate may read is on the left; everything it may "
                    "never read is fenced on the right."
                ),
                ascii_map=_control_plane(),
                notes=(
                    "A rejected candidate costs zero rollout budget: admission runs build and "
                    "contract tests before the evaluator is ever called.",
                    "The evaluator owns the seeds and the reward definition, and a harness patch "
                    "that tried to import it would be refused by the protected-surface gate.",
                    "Restarting rehydrates the evaluator cache, so resumed work is never scored "
                    "or charged twice.",
                ),
            ),
            Panel(
                slug="multi-agent-step",
                title="One multi-agent transition",
                caption=(
                    "Two heroes with genuinely local observations, a communication edge carrying "
                    "a claim, the harness-owned shared state, the protected environment, and "
                    "reward flowing back along three separately-labelled paths."
                ),
                ascii_map=_multi_agent_step(),
                notes=(
                    "Messages are only useful because a teammate reads them: a claim announcement "
                    "lands in shared state and steers the other hero off the same target.",
                    "The harness sits between agents and environment and is the only patchable "
                    "surface; rules, reward and achievements are protected.",
                    "Team reward, per-agent attribution and counterfactual contribution are three "
                    "different quantities and are never collapsed into one number.",
                ),
            ),
            Panel(
                slug="branch-lattice",
                title="Branch checkpoints, forks and one-step",
                caption=(
                    "The environment-level lattice. A branch checkpoint binds world state and RNG "
                    "together with agent, harness and policy state, so two forks from the same "
                    "point reproduce identically and the parent never moves."
                ),
                ascii_map=_branch_lattice(),
                notes=(
                    "The state digest is computed over the decoded environment state, not the "
                    "pickle bytes — pickle output is not stable across processes, so a byte "
                    "comparison would prove nothing.",
                    "Leave-one-out credit is the only causal estimator here: it costs one bounded "
                    "fork per hero and is labelled as an estimate, not an accounting split.",
                ),
            ),
            Panel(
                slug="varieties",
                title="Three varieties, one envelope",
                caption=(
                    "Prompt/protocol, harness and RLVR candidates differ only in their payload "
                    "schema and their admission path. Everything downstream is shared."
                ),
                ascii_map=_varieties(),
                notes=(
                    "Adding a fourth variety means adding one entry to the proposer registry, not "
                    "editing the state machine.",
                    "Candidates are single-patch: lineage is recorded but harness patches are not "
                    "yet composed across generations.",
                    "Two of the three lanes are seed-invariant under a deterministic hero policy, "
                    "so their six matched seeds are replications, not independent samples. The "
                    "paired verdict is sound; the implied power is not.",
                ),
            ),
        ),
        boundaries=(
            ("Policy-visible", "Incumbent, candidate summaries, train evidence, failure clusters, budgets, allowed actions."),
            ("Operator-only", "Reward source and weighting, evaluator internals, workspace paths."),
            ("Sealed evaluation", "Heldout seeds and heldout evidence. Exposed as a digest and a count; a leak check fails the observation closed."),
            ("Protected surfaces", "env.py, rules_engine.py, grid_engine.py, engine_runtime.py, achievements.py, rewards.py, evaluator.py, protocol.py — a patch touching any of them is refused."),
            ("Secret", "Provider credentials. No provider is called in any lane, so none are loaded."),
        ),
        receipts=(
            Receipt("prompt/protocol lane", "mapo_prompt_6f1ea6e815", "moonblade_vault_lite, heldout +1.1313, 6/6 seeds, sd 0.000"),
            Receipt("harness lane", "mapo_harness_224dcfcc98", "glass_maze_lite, heldout +0.1486, 6/6 seeds, sd 0.027"),
            Receipt("RLVR lane", "mapo_rlvr_6f48cb973a", "moonblade_vault_lite, heldout +0.0800, 6/6 seeds, sd 0.000"),
            Receipt("pause / restart", "state_digest_identical", "all three lanes; rollouts never double-charged"),
            Receipt("fork", "parent_state_digest_unchanged", "all three lanes; new episode id with parent checkpoint digest"),
            Receipt("one proposer step", "actions_taken = 1", "bounded; --execute adds exactly one evaluation transition"),
        ),
        extra_legend=(
            ("claim:<id>", "a hero announcing the target it is about to take"),
            ("§", "durable state bound into a checkpoint"),
        ),
    )
