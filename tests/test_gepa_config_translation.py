from pathlib import Path

from synth_optimizers.gepa import GepaConfig


def test_minibatch_acceptance_criterion_survives_sdk_translation(tmp_path: Path) -> None:
    source = tmp_path / "recipe.toml"
    source.write_text(
        """
[run]
run_id = "criterion_roundtrip"
output_dir = "."

[container]
url = "http://127.0.0.1:9999"

[taskset]
train_split = "train"
heldout_split = "test"
train_ids = ["train:0"]
heldout_ids = ["test:0"]

[candidate]
target_modules = ["prompt"]

[seed_candidate]
prompt = "classify"

[policy]
enabled = true
provider = "openai"
model = "gpt-4.1-nano"
proxy_mode = "proxy_only"

[proposer]
timeout_seconds = 600
message_stall_timeout_seconds = 300

[gepa]
minibatch_acceptance_criterion = "improvement_or_equal"
acceptance_criterion = "primary_improvement"

[gepa.task_pools]
pareto = ["train:0"]
minibatch = ["train:0"]
reflection = ["train:0"]
heldout = ["test:0"]
"""
    )

    config = GepaConfig.from_toml(source)
    translated = config.to_toml_dict()["gepa"]

    assert translated["minibatch_acceptance_criterion"] == "improvement_or_equal"
    assert translated["acceptance_criterion"] == "primary_improvement"
    assert config.to_toml_dict()["proposer"]["message_stall_timeout_seconds"] == 300


def test_adaptive_rollout_settings_survive_translation() -> None:
    from synth_optimizers.gepa import GepaTomlSection

    settings = {
        "enabled": False, "initial": 3, "min": 2, "max": 9,
        "increase_step": 2, "decrease_step": 3, "increase_after_successes": 7,
        "overload_status_codes": [429, 503],
    }
    section = GepaTomlSection.model_validate({
        "pipeline": {"workers": {"rollout": 30}, "adaptive_rollout_concurrency": settings}
    })
    output = {}
    section.pipeline_config().apply_to_gepa(output)
    assert output["pipeline"]["adaptive_rollout_concurrency"] == settings


def test_unknown_adaptive_rollout_setting_is_rejected() -> None:
    import pytest
    from pydantic import ValidationError
    from synth_optimizers.gepa import GepaTomlSection

    with pytest.raises(ValidationError):
        GepaTomlSection.model_validate({
            "pipeline": {"adaptive_rollout_concurrency": {"inital": 3}}
        })
