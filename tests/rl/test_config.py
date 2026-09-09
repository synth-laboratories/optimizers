"""The run configuration: what it accepts, and everything it refuses.

A configuration loader that ignores a key it does not know is a run that
quietly does something else. Every test here is about a refusal or about the
plan a document expands to.
"""

from __future__ import annotations

import pytest

from synth_optimizers.rl import config as config_module
from synth_optimizers.rl.config import ConfigError, RunConfig

NOTE_SURFACE = """
schema_version = "cispo.container.v1"
run_id = "run_note"

[container]
url = "http://127.0.0.1:8080"
headers = {}
auth_bearer_env = "CONTAINER_TOKEN"

[taskset]
train_split = "train"
evaluation_split = "heldout"
train_ids = ["row_0001", "row_0002"]
evaluation_ids = ["row_0004"]

[model]
provider = "tinker"
id = "openai/gpt-oss-20b"
family = "gpt_oss"
rank = 8
resume_from_checkpoint = "ckpt_parent_immutable"

[plan]
preset = "cispo"
group_size = 8
groups_per_step = 1
target_train_updates = 1
maximum_sampled_groups = 5
credit = "length_weighted_leave_one_out_standardized"
correction = {kind = "staleness_drop", max_weight_staleness = 1}
schedule = {weight_mode = "async_lag"}
reducer = "branch_aware_root_mean"

[plan.objective]
eps_low = 1.0
eps_high = 4.0

[pipeline]
mode = "async_queued"
max_execution_slots = 8
rollout_queue_capacity = 16
score_queue_capacity = 8
train_ready_capacity = 2
maximum_policy_lag = 1
rollout_retries = 2
score_retries = 1

[topology]
expected_topology_id = "topo-declared-4x6"
trainable_teams = ["team_a"]
partial_roster = "drop_instance"
same_policy_reduction = "token_weighted_mean"

[topology.policy_types]
miner = "miner_policy"
scout = "scout_policy"

[opponents]
match_set_revision = "match-set-0007"
allow_alias_resolution = false

[reward]
optimized_channel = "team_rank"
horizon_grace_seconds = 120

[evaluation]
paired = true
baseline_samples = 4
trained_samples = 4
fixed_match_set = true

[lifecycle]
resume_requires_rehandshake = true

[offline]
mode = "off"

[artifacts]
checkpoint_every_published_update = true
retain_training_state = true
catalog = "runs/checkpoints.jsonl"
"""

MINIMAL = """
schema_version = "cispo.container.v1"

[container]
url = "http://127.0.0.1:8080"

[taskset]
train_ids = ["row_0001"]

[model]
provider = "fake"
id = "vendor/model-20b"
family = "family_a"

[plan]
preset = "{preset}"
group_size = 2
groups_per_step = 1
target_train_updates = 1
maximum_sampled_groups = 2

[reward]
optimized_channel = "score"
"""


def minimal(preset: str = "cispo", **extra: str) -> str:
    """The smallest valid document, with extra lines folded into a section."""

    text = MINIMAL.format(preset=preset)
    for section, body in extra.items():
        header = f"[{section}]"
        if header in text:
            text = text.replace(header, f"{header}\n{body}", 1)
        else:
            text += f"\n{header}\n{body}\n"
    return text


def test_pipeline_lag_must_fit_algorithm_assembly_bound():
    with pytest.raises(ConfigError, match='max_weight_staleness'):
        config_module.loads(minimal(pipeline='maximum_policy_lag = 1'))
    configured = config_module.loads(minimal(
        pipeline='maximum_policy_lag = 1',
        plan='correction = {kind="staleness_drop", enabled=true, max_weight_staleness=1}\nschedule = {weight_mode="async_lag"}',
    ))
    assert configured.expanded_plan().correction.max_weight_staleness == 1


# --------------------------------------------------------------------------- #
# What it accepts
# --------------------------------------------------------------------------- #


def test_the_notes_configuration_surface_loads_and_expands() -> None:
    config = config_module.loads(NOTE_SURFACE)

    assert isinstance(config, RunConfig)
    assert config.run_id == "run_note"
    assert config.container.url == "http://127.0.0.1:8080"
    assert config.taskset.train_ids == ("row_0001", "row_0002")
    assert config.topology.policy_types == {"miner": "miner_policy", "scout": "scout_policy"}
    assert config.opponents.match_set_revision == "match-set-0007"
    assert config.reward.optimized_channel == "team_rank"
    assert config.evaluation.paired is True
    assert config.model.resume_from_checkpoint == "ckpt_parent_immutable"

    plan = config.expanded_plan()
    assert plan.preset == "cispo"
    assert plan.rollout.cardinality == 8
    assert plan.groups_per_step == 1
    assert plan.credit.kind == "length_weighted_leave_one_out_standardized"
    assert plan.objective.eps_low == 1.0 and plan.objective.eps_high == 4.0
    assert plan.plan_hash


def test_a_second_preset_loads_through_the_identical_path() -> None:
    cispo = config_module.loads(minimal("cispo")).expanded_plan()
    gspo = config_module.loads(minimal("gspo")).expanded_plan()

    assert cispo.objective.kind == "cispo"
    assert gspo.objective.kind == "gspo"
    assert cispo.plan_hash != gspo.plan_hash
    # Same loader, same fields, no algorithm branch anywhere between them.
    assert set(cispo.to_dict()) == set(gspo.to_dict())


def test_a_dimension_override_may_be_a_kind_or_a_table() -> None:
    bare = config_module.loads(minimal(plan='credit = "group_mean"'))
    assert bare.expanded_plan().credit.kind == "group_mean"

    table = config_module.loads(
        MINIMAL.format(preset="cispo") + '\n[plan.credit]\nkind = "group_mean"\n'
    )
    assert table.expanded_plan().credit.kind == "group_mean"


def test_the_redacted_payload_never_carries_a_secret(monkeypatch) -> None:
    monkeypatch.setenv("CONTAINER_TOKEN", "s3cret")
    config = config_module.loads(NOTE_SURFACE)

    headers = config.container.resolved_headers()
    assert headers["Authorization"] == "Bearer s3cret"

    payload = config.redacted_payload()
    assert "s3cret" not in str(payload)
    assert payload["plan_hash"] == config.expanded_plan().plan_hash
    assert payload["expanded_plan"]["preset"] == "cispo"


def test_a_missing_bearer_variable_is_a_refusal(monkeypatch) -> None:
    monkeypatch.delenv("CONTAINER_TOKEN", raising=False)
    config = config_module.loads(NOTE_SURFACE)

    with pytest.raises(ConfigError, match="CONTAINER_TOKEN"):
        config.container.resolved_headers()


def test_maximum_sampled_groups_defaults_to_the_packing_the_plan_asks_for() -> None:
    text = MINIMAL.format(preset="cispo").replace("maximum_sampled_groups = 2\n", "")
    config = config_module.loads(text)

    assert config.maximum_sampled_groups == 1


# --------------------------------------------------------------------------- #
# What it refuses
# --------------------------------------------------------------------------- #


def test_an_unknown_key_is_refused_rather_than_ignored() -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        config_module.loads(minimal(pipeline="max_execution_slots = 2\nrollout_burst = 4"))


def test_an_unknown_section_is_refused() -> None:
    with pytest.raises(ConfigError, match="unknown top-level sections"):
        config_module.loads(minimal(cispo="group_size = 8"))


def test_an_unknown_preset_is_refused_by_name() -> None:
    with pytest.raises(ConfigError, match="unknown preset"):
        config_module.loads(minimal("not_an_algorithm"))


@pytest.mark.parametrize(
    ("section", "body"),
    [
        ("model", 'renderer = "renderers.gpt-oss.low.v1"'),
        ("reward", 'reward_mode = "exact_match"'),
        ("taskset", 'harness = "a_harness_name"'),
        ("container", 'environment = "an_environment_name"'),
    ],
)
def test_a_container_concern_is_refused_by_name(section: str, body: str) -> None:
    with pytest.raises(ConfigError, match="container concern"):
        config_module.loads(minimal(**{section: body}))


def test_a_top_level_container_concern_section_is_refused() -> None:
    with pytest.raises(ConfigError, match="container concern"):
        config_module.loads(minimal(environment='id = "anything"'))


def test_queue_depth_minus_one_may_not_exceed_the_staleness_bound() -> None:
    with pytest.raises(ConfigError, match="maximum_policy_lag"):
        config_module.loads(
            minimal(pipeline="train_ready_capacity = 3\nmaximum_policy_lag = 1")
        )


def test_train_calls_may_not_exceed_the_plans_step_ceiling() -> None:
    text = MINIMAL.format(preset="cispo").replace(
        "target_train_updates = 1",
        "target_train_updates = 40\nsteps_per_round = 15",
    ).replace("maximum_sampled_groups = 2", "maximum_sampled_groups = 80")
    with pytest.raises(ConfigError, match="step ceiling"):
        config_module.loads(text)


def test_the_group_budget_must_cover_the_updates_it_promises() -> None:
    text = MINIMAL.format(preset="cispo").replace(
        "target_train_updates = 1", "target_train_updates = 3"
    )
    with pytest.raises(ConfigError, match="maximum_sampled_groups"):
        config_module.loads(text)


def test_replay_mode_needs_a_source_run() -> None:
    with pytest.raises(ConfigError, match="source_run_id"):
        config_module.loads(minimal(offline='mode = "replay"'))


def test_source_runs_outside_replay_mode_are_refused() -> None:
    with pytest.raises(ConfigError, match="only meaningful in replay"):
        config_module.loads(minimal(offline='mode = "off"\nsource_run_ids = ["run_a"]'))


def test_an_unsupported_schema_version_is_refused() -> None:
    with pytest.raises(ConfigError, match="schema_version"):
        config_module.loads(minimal().replace("cispo.container.v1", "cispo.container.v2"))


def test_a_paired_evaluation_needs_both_arms() -> None:
    with pytest.raises(ConfigError, match="paired evaluation"):
        config_module.loads(minimal(evaluation="paired = true\nbaseline_samples = 4"))


def test_load_reads_a_file_and_names_it(tmp_path) -> None:
    path = tmp_path / "run_alpha.toml"
    path.write_text(minimal(), encoding="utf-8")

    config = config_module.load(path)

    assert config.run_id == "run_alpha"
