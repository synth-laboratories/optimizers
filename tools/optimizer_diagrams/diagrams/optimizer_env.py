"""The shared optimizer-as-environment protocol and the policy training loop."""

from __future__ import annotations

from ..canvas import Canvas
from ..model import DiagramPage, MetaField, Panel, Receipt


def _protocol() -> str:
    c = Canvas(134, 27)
    c.text(0, 0, "synth.optimizer_env.v1 — five optimizers, one state machine, no algorithm names in generic code")

    obs = c.box(3, 0, 36, 10, title="OptimizerObservation", lines=(
        "objective · hypotheses",
        "incumbent · candidates",
        "frontier · lineage",
        "evidence + uncertainty",
        "failure clusters · signals",
        "pending / in-flight work",
        "budgets · allowed actions",
        "versions · digest",
    ))
    act = c.box(3, 50, 32, 10, title="OptimizerAction", lines=(
        "propose · select_parent",
        "allocate_evaluation",
        "accept · reject · defer",
        "merge · patch · discard",
        "adjust_setting · annotate",
        "checkpoint · pause · resume",
        "fork · stop",
    ))
    trans = c.box(3, 96, 36, 10, title="OptimizerTransition", lines=(
        "pre-observation digest",
        "pre-state digest",
        "action + actor identity",
        "executed work + evidence",
        "post-state digest",
        "RewardVector",
        "usage · violations",
        "continuation / terminal",
    ))
    c.connect_h(obs, act, label="policy")
    c.connect_h(act, trans, label="execute")

    ck = c.box(16, 0, 36, 9, title="OptimizerCheckpoint §", lines=(
        "schema + algorithm versions",
        "state and event cursor",
        "candidates · lineage · frontier",
        "pending work dispositions",
        "RNG · budgets · usage ledger",
        "evaluator + cache identity",
        "container manifest digests",
    ))
    c.connect_v(obs, ck, col=obs.left + 4, label="")
    c.text(14, obs.left + 6, "checkpoint §")

    ext = c.box(16, 50, 32, 9, title="ALGORITHM EXTENSIONS", lines=(
        "extensions{gepa: …}",
        "extensions{gelo: …}",
        "extensions{ohco: …}",
        "extensions{vzpo: …}",
        "extensions{mapo: …}",
        "",
        "unknown keys survive",
        "a round trip untouched",
    ))
    c.connect_h(ck, ext, row=ck.mid_row, label="carried")

    life = c.box(16, 96, 36, 9, title="LIFECYCLE", lines=(
        "pause: stop new work, resolve",
        "  in-flight, persist cursor",
        "restart: verify, reconcile,",
        "  never re-score or re-charge",
        "fork: new episode id with",
        "  parentEpisodeId + digest",
        "step: exactly one action",
    ))
    c.connect_h(ext, life, row=ext.mid_row, label="uniform")
    return c.render()


def _training_loop() -> str:
    c = Canvas(128, 25)
    c.text(0, 0, "OPTIMIZER-AS-RL-ENVIRONMENT — the surface a proposer policy would be trained on")

    ep = c.box(3, 0, 34, 7, title="OPTIMIZER EPISODES", lines=(
        "GEPA · GELO · OHCO",
        "VZPO · MAPO",
        "observation / action /",
        "transition / reward records",
    ))
    export = c.box(3, 48, 32, 7, title="TRAJECTORY EXPORT", lines=(
        "versioned training records",
        "large evidence by reference",
        "behaviour-policy identity",
        "logprobs where claimed",
    ))
    splits = c.box(3, 92, 34, 7, style="heavy", title="FIXED SPLITS", lines=(
        "train / development /",
        "sealed evaluation",
        "anti-leakage tests guard",
        "heldout and verifier state",
    ))
    c.connect_h(ep, export, label="export")
    c.connect_h(export, splits, label="partition")

    policy = c.box(14, 48, 32, 8, title="PROPOSER POLICY", lines=(
        "deterministic baseline",
        "  (the control arm)",
        "Nemotron 3.5 Lightning",
        "  adapter + preflight",
        "offline imitation first,",
        "then preference, then RL",
    ))
    c.connect_v(export, policy, col=export.mid_col, label="")

    replay = c.box(14, 0, 34, 8, title="REPLAY", lines=(
        "deterministic or recorded",
        "evaluator where supported",
        "one-step operation is the",
        "unit of comparison:",
        "fixed policy vs proposer",
    ))
    c.connect_h(replay, policy, reverse=True)

    claim = c.box(14, 92, 34, 8, title="WHAT MAY BE CLAIMED", lines=(
        "environment contract: yes,",
        "  once a real transition runs",
        "policy improvement: only with",
        "  a real optimizer-policy",
        "  update AND a sealed",
        "  evaluation receipt",
    ))
    c.connect_h(policy, claim)
    c.text(23, 0, "Renaming prompt optimization to RLVR is not an RL result. A scripted replay is evidence about the environment, not the policy.")
    return c.render()


def build() -> DiagramPage:
    return DiagramPage(
        slug="optimizer-environment",
        title="Shared optimizer environment and training loop",
        subtitle=(
            "One versioned contract for observation, action, transition, reward, checkpoint, "
            "pause, fork and single-step — plus the surface a proposer policy would train on."
        ),
        optimizer="Shared platform",
        schema_version="synth.optimizer_env.v1",
        hypothesis=(
            "GEPA, GELO, OHCO, VZPO and MAPO can share one resumable environment contract "
            "without pushing algorithm names into generic state-machine code."
        ),
        verdict="Implemented and exercised by MAPO; not yet mapped onto the other four.",
        confidence="Moderate for the contract, none for cross-optimizer portability.",
        why=(
            "MAPO implements the full contract — actions, digests, checkpoint, pause/restart, "
            "fork, one-step and trajectory export — and its gates pass. No other optimizer has "
            "been mapped onto it yet, so portability is asserted, not shown."
        ),
        metadata=(
            MetaField("Protocol schema", "synth.optimizer_env.v1"),
            MetaField("Reference implementation", "MAPO", "dungeongrid.mapo.episode.v1"),
            MetaField("Mapped optimizers", "1 of 5", "GEPA, GELO, OHCO, VZPO outstanding"),
            MetaField("Policy training", "surface only", "no policy update run"),
        ),
        panels=(
            Panel(
                slug="protocol",
                title="Core types and lifecycle",
                caption=(
                    "The nine core types and the four lifecycle operations every optimizer must "
                    "support identically. Algorithm-specific data rides in extension maps."
                ),
                ascii_map=_protocol(),
                notes=(
                    "A checkpoint validates its own digest on load; a mismatch is an error, not a "
                    "warning.",
                    "An optimizer advertises only the actions it genuinely supports — MAPO, for "
                    "example, declares merge and discard unsupported rather than faking them.",
                ),
            ),
            Panel(
                slug="training-loop",
                title="Trajectory export and the policy loop",
                caption=(
                    "What has to exist before anyone may say a proposer policy was trained, and "
                    "the line between environment evidence and policy evidence."
                ),
                ascii_map=_training_loop(),
                notes=(
                    "Sealed evaluation splits are fixed before any policy sees a trajectory.",
                    "The one-step operation is deliberately the unit of comparison: it makes a "
                    "fixed policy and a candidate proposer directly comparable on the same "
                    "checkpoint.",
                ),
            ),
        ),
        boundaries=(
            ("Policy-visible", "Observation fields explicitly classified as policy-visible, and nothing else."),
            ("Operator-only", "Reward weighting, evaluator internals, workspace and artifact paths."),
            ("Sealed evaluation", "Heldout data, hidden rubric state and generalization reward."),
            ("Secret", "Provider credentials and prohibited trace content. Never reachable from an observation."),
        ),
        receipts=(
            Receipt("MAPO contract conformance", "50 gates passing", "protocol, candidates, rollout and episode suites"),
            Receipt("GEPA mapping", "", "map service state onto the shared protocol"),
            Receipt("GELO mapping", "", "map board and candidate state onto the shared protocol"),
            Receipt("Nemotron preflight", "", "adapter, capability preflight and one bounded inference-only step"),
        ),
    )
