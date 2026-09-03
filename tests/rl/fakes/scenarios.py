"""Container configurations covering the conformance case list.

Every scenario is a :class:`~fakes.container.ContainerConfig`. There
is no task name, harness name, or environment name anywhere in this module,
and nothing downstream may select a fake by task id: a scenario is a *capability
configuration*, and the only difference between the conformant and the
non-conformant halves is which :class:`EvidenceDefects` flag is set.

Conformant scenarios, keyed in :data:`CONFORMANT`:

===================================  ===============================================
``one_call_classification``          one call, one reward, sequential, direct action
``multi_turn_environment_reward``    three turns stitching under the strict prefix
``multi_turn_declared_compaction``   a fork with declared compaction provenance
``joint_episode_two_groups``         four instances over two parameter groups
``deferred_verifier``                reward settles only after finalize
``rubric_scored_judge``              judge spans recorded, never trainable
``competitive_realtime``             two teams, one pinned, rank reward channel
``deferred_program_quiesced``        policy-authored loop killed at the horizon
``clipped_no_quiescence``            cannot quiesce, serves a clipped snapshot
``tito_classification``              declares tokens-in/tokens-out
``zero_reward_classification``       zero is a score, not an absence
``artifact_by_reference``            trace by reference plus digest
``prompt_budget_truncate``           overlong prompt truncated under a stated rule
``prompt_budget_compact``            overlong prompt compacted under a stated rule
``degraded_concurrency``             advertises fewer leases than the plan wants
===================================  ===============================================

Non-conformant scenarios, keyed in :data:`NON_CONFORMANT` with the typed error
each one must produce:

=====================================  ==================  ================================
``missing_logprobs``                   ``EvidenceError``   logprob vector omitted
``sentinel_logprobs``                  ``EvidenceError``   ``-9999.0`` in the vector
``zero_logprobs``                      ``EvidenceError``   identically-zero vector
``short_logprobs``                     ``EvidenceError``   length disagrees with tokens
``absent_reward``                      ``EvidenceError``   no channel; absent is not zero
``dropped_cross_team_channel``         ``EvidenceError``   declared channel, no messages
``rerendering_multi_turn``             ``EvidenceError``   unexplained prefix divergence
``flattened_wire``                     ``EvidenceError``   responses persisted as chat
``opponent_alias_resolution``          ``TopologyError``   opponent resolves ``latest``
``missing_instance_trajectory``        ``TopologyError``   one instance has no trajectory
``probe_indistinguishable``            ``EvidenceError``   probe looks like real evidence
``deferred_program_unquiesced``        ``EvidenceError``   effects outlive the horizon
``competitive_match_set_drift``        ``MixedGroupError`` pin drifts inside one group
``prompt_budget_refuse``               container refusal    overlong prompt refused
``rejected_mandatory_clause``          handshake refusal    run stops before any spend
=====================================  ==================  ================================
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from synth_optimizers.contracts.rl_identity import (
    AgentInstance,
    CommunicationChannel,
    Horizon,
    MixedGroupError,
    Team,
    Topology,
    TopologyError,
)
from synth_optimizers.contracts.rl_records import EvidenceError, RendererProfile

from .container import ContainerConfig, EvidenceDefects

TASK_ROWS: tuple[str, ...] = ("row_0001", "row_0002", "row_0003", "row_0004")

PINNED_OPPONENT = "checkpoint::frozen-0007"


def renderer_profile(**overrides: object) -> RendererProfile:
    payload: dict[str, object] = {
        "profile_id": "renderers.family-a.low.v1",
        "package": "renderers",
        "package_version": "0.1.11",
        "config_digest": "sha256:cfg-a",
        "tokenizer_id": "vendor/model-20b",
        "tokenizer_digest": "sha256:tok-a",
        "stop_token_ids": [200002, 199999],
    }
    payload.update(overrides)
    return RendererProfile.from_payload(payload)


# --------------------------------------------------------------------------- #
# Declared topologies
# --------------------------------------------------------------------------- #


def solo_topology(
    *, actuation_model: str = "direct_action", horizon: Horizon | None = None
) -> Topology:
    return Topology(
        topology_id="topo-solo-1",
        turn_model="sequential",
        actuation_model=actuation_model,
        reward_relation="cooperative",
        agent_instances=(
            AgentInstance(
                agent_instance_id="inst_a",
                role_id="role_primary",
                policy_type_id="type_primary",
                team_id="team_solo",
                trainable=True,
            ),
        ),
        teams=(Team(team_id="team_solo", trainable=True, minimum_viable_roster=1),),
        horizon=horizon,
        parameter_groups={"type_primary": "pg_primary"},
    )


def party_topology() -> Topology:
    """Four agent instances mapped onto two shared policy parameter groups."""

    roles = (
        ("inst_a1", "role_alpha", "type_alpha"),
        ("inst_a2", "role_alpha", "type_alpha"),
        ("inst_b1", "role_beta", "type_beta"),
        ("inst_b2", "role_beta", "type_beta"),
    )
    return Topology(
        topology_id="topo-party-4x2",
        turn_model="sequential",
        actuation_model="direct_action",
        reward_relation="cooperative",
        agent_instances=tuple(
            AgentInstance(
                agent_instance_id=instance_id,
                role_id=role_id,
                policy_type_id=policy_type_id,
                team_id="team_party",
                trainable=True,
            )
            for instance_id, role_id, policy_type_id in roles
        ),
        teams=(Team(team_id="team_party", trainable=True, minimum_viable_roster=3),),
        communication_channels=(
            CommunicationChannel(channel_id="party_chat", scope="intra_team"),
        ),
        horizon=Horizon(horizon_kind="steps", value=3.0),
        parameter_groups={"type_alpha": "pg_alpha", "type_beta": "pg_beta"},
    )


def competitive_topology() -> Topology:
    """Two teams: one trainable, one pinned non-trainable. Concurrent realtime."""

    return Topology(
        topology_id="topo-versus-2x2",
        turn_model="concurrent_realtime",
        actuation_model="direct_action",
        reward_relation="competitive_rank",
        agent_instances=(
            AgentInstance(
                agent_instance_id="home_1",
                role_id="role_alpha",
                policy_type_id="type_alpha",
                team_id="team_home",
                trainable=True,
            ),
            AgentInstance(
                agent_instance_id="home_2",
                role_id="role_beta",
                policy_type_id="type_beta",
                team_id="team_home",
                trainable=True,
            ),
            AgentInstance(
                agent_instance_id="away_1",
                role_id="role_alpha",
                policy_type_id="type_opponent",
                team_id="team_away",
                trainable=False,
                pinned_identity=PINNED_OPPONENT,
            ),
            AgentInstance(
                agent_instance_id="away_2",
                role_id="role_beta",
                policy_type_id="type_opponent",
                team_id="team_away",
                trainable=False,
                pinned_identity=PINNED_OPPONENT,
            ),
        ),
        teams=(
            Team(team_id="team_home", trainable=True, minimum_viable_roster=2),
            Team(team_id="team_away", trainable=False, minimum_viable_roster=2),
        ),
        communication_channels=(
            CommunicationChannel(channel_id="team_pm", scope="intra_team"),
            CommunicationChannel(channel_id="public", scope="cross_team"),
        ),
        horizon=Horizon(
            horizon_kind="wall_clock", value=5400.0, time_dilation=4.0, grace_seconds=120.0
        ),
        parameter_groups={"type_alpha": "pg_alpha", "type_beta": "pg_beta"},
    )


def _base(**overrides: object) -> ContainerConfig:
    payload: dict[str, object] = {
        "container_id": "fake-container",
        "image_digest": "sha256:image-a",
        "renderer_profile": renderer_profile(),
        "topology": solo_topology(),
        "taskset_id": "taskset-a",
        "task_ids": TASK_ROWS,
        "splits": {"train": TASK_ROWS[:3], "eval": TASK_ROWS[3:]},
    }
    payload.update(overrides)
    return ContainerConfig(**payload)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Conformant scenarios
# --------------------------------------------------------------------------- #


def one_call_classification() -> ContainerConfig:
    """(a) One call in, one environment reward out. The simplest conformant case."""

    return _base(container_id="fake-oneshot", turns=1, reward_value=0.75, seed=11)


def multi_turn_environment_reward() -> ContainerConfig:
    """(b) Three turns whose prompts stitch under the strict-prefix rule."""

    return _base(container_id="fake-multiturn", turns=3, reward_value=0.5, seed=12)


def multi_turn_declared_compaction() -> ContainerConfig:
    """A fork: turn 1 re-renders but declares compaction and opens a branch."""

    return _base(
        container_id="fake-compaction",
        turns=3,
        declared_compaction_turn=1,
        compaction_authored_by_policy=True,
        seed=13,
    )


def joint_episode_two_groups() -> ContainerConfig:
    """(c) Four agent instances over two shared policy parameter groups."""

    return _base(
        container_id="fake-party",
        topology=party_topology(),
        turns=2,
        seed=14,
        partial_roster_disposition="drop_instance",
    )


def deferred_verifier() -> ContainerConfig:
    """(d) Scoring is staged: the reward route stays pending until finalize."""

    return _base(
        container_id="fake-deferred",
        turns=2,
        deferred_scoring=True,
        reward_kind="deferred_verifier",
        settlement_window_seconds=150.0,
        seed=15,
    )


def rubric_scored_judge() -> ContainerConfig:
    """(e) A container-owned rubric judge: its spans are recorded, not trained."""

    return _base(
        container_id="fake-rubric",
        turns=1,
        judge_spans=True,
        reward_kind="rubric_judge",
        seed=16,
    )


def competitive_realtime() -> ContainerConfig:
    """(f) Concurrent real-time, two teams, one pinned, rank reward channel."""

    return _base(
        container_id="fake-versus",
        topology=competitive_topology(),
        turns=2,
        reward_kind="rank",
        reward_value=1.0,
        optimized_team_id="team_home",
        advertised_concurrency=30,
        lease_ttl_seconds=900.0,
        settlement_window_seconds=150.0,
        seed=17,
    )


def deferred_program_quiesced() -> ContainerConfig:
    """(g) A policy-authored loop, killed at the horizon and scored correctly."""

    return _base(
        container_id="fake-program-quiesced",
        topology=solo_topology(
            actuation_model="deferred_program",
            horizon=Horizon(horizon_kind="env_ticks", value=100.0),
        ),
        turns=3,
        quiescence_supported=True,
        seed=18,
    )


def clipped_no_quiescence() -> ContainerConfig:
    """Cannot quiesce; declares it and serves a horizon-clipped snapshot."""

    return _base(
        container_id="fake-clipped",
        topology=solo_topology(
            actuation_model="deferred_program",
            horizon=Horizon(horizon_kind="env_ticks", value=100.0),
        ),
        turns=3,
        quiescence_supported=False,
        settlement_window_seconds=30.0,
        seed=19,
    )


def tito_classification() -> ContainerConfig:
    """Declares tokens-in/tokens-out. Prompt ids must match the message-in fake."""

    return _base(
        container_id="fake-oneshot",
        turns=1,
        reward_value=0.75,
        seed=11,
        tito_supported=True,
        sampling_transport="tokens_in_tokens_out",
    )


def zero_reward_classification() -> ContainerConfig:
    """Zero is a score. It must stay distinguishable from an absent reward."""

    return _base(container_id="fake-zero", turns=1, reward_value=0.0, seed=20)


def artifact_by_reference() -> ContainerConfig:
    """A trace too large to inline: stored by reference with a digest."""

    return _base(container_id="fake-byref", turns=2, artifact_by_reference=True, seed=21)


def prompt_budget_truncate() -> ContainerConfig:
    return _base(
        container_id="fake-truncate",
        turns=1,
        max_prompt_tokens=10,
        prompt_budget_policy="truncate",
        seed=22,
    )


def prompt_budget_compact() -> ContainerConfig:
    return _base(
        container_id="fake-compact",
        turns=1,
        max_prompt_tokens=10,
        prompt_budget_policy="compact",
        seed=23,
    )


def degraded_concurrency() -> ContainerConfig:
    """Advertises one lease. A plan asking for more must be re-handshaked."""

    return _base(container_id="fake-narrow", turns=1, advertised_concurrency=1, seed=24)


# --------------------------------------------------------------------------- #
# Deliberately non-conformant scenarios
# --------------------------------------------------------------------------- #


def missing_logprobs() -> ContainerConfig:
    return _base(
        container_id="fake-nologprobs",
        turns=1,
        seed=31,
        defects=EvidenceDefects(omit_logprobs=True),
    )


def sentinel_logprobs() -> ContainerConfig:
    return _base(
        container_id="fake-sentinel",
        turns=1,
        seed=32,
        defects=EvidenceDefects(sentinel_logprobs=True),
    )


def zero_logprobs() -> ContainerConfig:
    return _base(
        container_id="fake-zerologprobs",
        turns=1,
        seed=33,
        defects=EvidenceDefects(zero_logprobs=True),
    )


def short_logprobs() -> ContainerConfig:
    return _base(
        container_id="fake-shortlogprobs",
        turns=1,
        seed=34,
        defects=EvidenceDefects(logprob_length_delta=-2),
    )


def absent_reward() -> ContainerConfig:
    return _base(
        container_id="fake-noreward",
        turns=1,
        seed=35,
        defects=EvidenceDefects(absent_reward=True),
    )


def dropped_cross_team_channel() -> ContainerConfig:
    return _base(
        container_id="fake-dropchannel",
        topology=competitive_topology(),
        turns=2,
        reward_kind="rank",
        optimized_team_id="team_home",
        advertised_concurrency=30,
        seed=36,
        defects=EvidenceDefects(dropped_channel_id="public"),
    )


def rerendering_multi_turn() -> ContainerConfig:
    return _base(
        container_id="fake-rerender",
        turns=3,
        seed=37,
        defects=EvidenceDefects(rerender_turn=1),
    )


def flattened_wire() -> ContainerConfig:
    return _base(
        container_id="fake-flatwire",
        turns=1,
        wire_api="responses",
        seed=38,
        defects=EvidenceDefects(flatten_wire=True),
    )


def opponent_alias_resolution() -> ContainerConfig:
    return _base(
        container_id="fake-aliasopponent",
        topology=competitive_topology(),
        turns=1,
        reward_kind="rank",
        optimized_team_id="team_home",
        advertised_concurrency=30,
        seed=39,
        defects=EvidenceDefects(opponent_alias="latest"),
    )


def missing_instance_trajectory() -> ContainerConfig:
    return _base(
        container_id="fake-missinginstance",
        topology=party_topology(),
        turns=2,
        seed=40,
        partial_roster_disposition="refuse",
        defects=EvidenceDefects(missing_instance_id="inst_b2"),
    )


def missing_instance_dropped() -> ContainerConfig:
    """Same absence, but the run declares ``drop_instance`` and records it."""

    return _base(
        container_id="fake-droppedinstance",
        topology=party_topology(),
        turns=2,
        seed=40,
        partial_roster_disposition="drop_instance",
        defects=EvidenceDefects(missing_instance_id="inst_b2"),
    )


def probe_indistinguishable() -> ContainerConfig:
    return _base(
        container_id="fake-probeblend",
        turns=1,
        seed=41,
        defects=EvidenceDefects(probe_indistinguishable=True),
    )


def deferred_program_unquiesced() -> ContainerConfig:
    return _base(
        container_id="fake-program-loose",
        topology=solo_topology(
            actuation_model="deferred_program",
            horizon=Horizon(horizon_kind="env_ticks", value=100.0),
        ),
        turns=3,
        quiescence_supported=True,
        seed=18,
        defects=EvidenceDefects(unquiesced_deferred_program=True),
    )


def competitive_match_set_drift() -> ContainerConfig:
    """Identical to :func:`competitive_realtime` but for the resolved match set.

    Two attempts, one from each, differ only in ``match_set_revision_id`` --
    which is exactly the case the note requires be rejected as one group.
    """

    return _base(
        container_id="fake-versus",
        topology=competitive_topology(),
        turns=2,
        reward_kind="rank",
        reward_value=1.0,
        optimized_team_id="team_home",
        advertised_concurrency=30,
        lease_ttl_seconds=900.0,
        settlement_window_seconds=150.0,
        seed=17,
        defects=EvidenceDefects(match_set_drift="match-set-0099"),
    )


def prompt_budget_refuse() -> ContainerConfig:
    return _base(
        container_id="fake-refuse",
        turns=1,
        max_prompt_tokens=10,
        prompt_budget_policy="refuse",
        seed=42,
    )


def rejected_mandatory_clause() -> ContainerConfig:
    """A container that cannot honor a mandatory clause for this run."""

    return _base(
        container_id="fake-rejects",
        turns=1,
        seed=43,
        clause_overrides={
            "evidence.behavior_logprobs": (
                "rejected",
                "sampler does not return per-token behavior logprobs",
            )
        },
    )


def skewed_clock() -> ContainerConfig:
    """Wall-clock horizon with a skew past tolerance: a rejected clause."""

    return _base(
        container_id="fake-skewed",
        topology=competitive_topology(),
        turns=1,
        reward_kind="rank",
        optimized_team_id="team_home",
        advertised_concurrency=30,
        clock_skew_seconds=9.5,
        skew_tolerance_seconds=1.0,
        seed=44,
    )


# --------------------------------------------------------------------------- #
# Registries -- selection is by capability configuration, never by task name
# --------------------------------------------------------------------------- #

CONFORMANT: Mapping[str, Callable[[], ContainerConfig]] = {
    "one_call_classification": one_call_classification,
    "multi_turn_environment_reward": multi_turn_environment_reward,
    "multi_turn_declared_compaction": multi_turn_declared_compaction,
    "joint_episode_two_groups": joint_episode_two_groups,
    "deferred_verifier": deferred_verifier,
    "rubric_scored_judge": rubric_scored_judge,
    "competitive_realtime": competitive_realtime,
    "deferred_program_quiesced": deferred_program_quiesced,
    "clipped_no_quiescence": clipped_no_quiescence,
    "tito_classification": tito_classification,
    "zero_reward_classification": zero_reward_classification,
    "artifact_by_reference": artifact_by_reference,
    "prompt_budget_truncate": prompt_budget_truncate,
    "prompt_budget_compact": prompt_budget_compact,
    "degraded_concurrency": degraded_concurrency,
}

NON_CONFORMANT: Mapping[str, tuple[Callable[[], ContainerConfig], type[Exception] | None]] = {
    "missing_logprobs": (missing_logprobs, EvidenceError),
    "sentinel_logprobs": (sentinel_logprobs, EvidenceError),
    "zero_logprobs": (zero_logprobs, EvidenceError),
    "short_logprobs": (short_logprobs, EvidenceError),
    "absent_reward": (absent_reward, EvidenceError),
    "dropped_cross_team_channel": (dropped_cross_team_channel, EvidenceError),
    "rerendering_multi_turn": (rerendering_multi_turn, EvidenceError),
    "flattened_wire": (flattened_wire, EvidenceError),
    "opponent_alias_resolution": (opponent_alias_resolution, TopologyError),
    "missing_instance_trajectory": (missing_instance_trajectory, TopologyError),
    "probe_indistinguishable": (probe_indistinguishable, EvidenceError),
    "deferred_program_unquiesced": (deferred_program_unquiesced, EvidenceError),
    "competitive_match_set_drift": (competitive_match_set_drift, MixedGroupError),
    "prompt_budget_refuse": (prompt_budget_refuse, None),
    "rejected_mandatory_clause": (rejected_mandatory_clause, None),
    "skewed_clock": (skewed_clock, None),
}
