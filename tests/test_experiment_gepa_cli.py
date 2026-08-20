"""The `gepa.cli` adapter: rendering, refusals, and the correlation round trip.

GEPA itself is not run here — that needs a container and proposer credentials —
so the engine is replaced by a runner that seals the manifest GEPA would seal.
What is under test is everything the experiment layer is responsible for: which
knobs may vary, that a rendered arm is a config GEPA would accept, that each
trial's bookkeeping is derived rather than declared, and that a manifest whose
envelope does not match the plan is refused rather than joined to the wrong arm.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from synth_optimizers.experiment import (
    ExperimentContractError,
    ExperimentRunner,
    GepaCliAdapter,
    compile_plan,
    load_spec,
    parse_spec,
)
from synth_optimizers.experiment.adapters.gepa_cli import RESERVED_PATHS

BASE_CONFIG = """
[run]
run_id = "base"
output_dir = ".out/base"
seed = 0

[container]
url = "http://127.0.0.1:8875"

[taskset]
train_split = "train"
heldout_split = "test"
train_ids = ["0", "1", "2"]
heldout_ids = ["100"]

[candidate]
target_modules = ["stage2_system"]

[seed_candidate]
stage2_system = "Classify the query."

[policy]
provider = "openai"
model = "gpt-4.1-nano"
api_key_env = "OPENAI_API_KEY"

[proposer]
backend = "codex_app_server"
provider = "openai"
auth_mode = "chatgpt"
model = "gpt-5.6-luna"
reasoning_effort = "low"

[gepa]
max_generations = 2
proposals_per_generation = 1
minibatch_size = 3

[gepa.task_pools]
pareto = ["0", "1", "2"]
minibatch = ["0", "1", "2"]
reflection = ["0", "1", "2"]
heldout = ["100"]

[cache]
mode = "off"
"""

SPEC = """
schema = "synth.experiment.v1"
experiment_id = "luna-effort-v1"
executor = "gepa.cli"
base = "{base}"

[design]
primary_metric = "heldout_reward"
secondary_metrics = ["cost_usd"]
pairing = "block"
counterbalance = true
missing_policy = "fail"
min_blocks_for_claim = 3
bootstrap_resamples = 2000

[blocks]
kind = "seed"
values = [11, 12, 13, 14]

[factors]
"proposer.reasoning_effort" = ["low", "medium"]

[budget]
max_trials = 16

[isolation]
cache_namespace = "per_trial"
container = "fresh_per_trial"

[executor_options]
output_dir = "{output_dir}"
"""


class FakeGepa:
    """Seals the terminal manifest GEPA seals, and nothing else.

    Reward is a deterministic function of seed and effort so the paired
    comparison has an unambiguous answer: `medium` is worth +0.10 on every seed.
    """

    def __init__(self, *, effort_bonus=None, fail_efforts=(), mangle_correlation=False):
        self.effort_bonus = effort_bonus or {"low": 0.0, "medium": 0.10}
        self.fail_efforts = set(fail_efforts)
        self.mangle_correlation = mangle_correlation
        self.configs: list[Path] = []

    def __call__(self, config_path: Path):
        self.configs.append(config_path)
        document = tomllib.loads(config_path.read_text(encoding="utf-8"))
        effort = document["proposer"]["reasoning_effort"]
        if effort in self.fail_efforts:
            raise RuntimeError(f"proposer refused at effort {effort}")
        run_dir = Path(document["run"]["output_dir"]) / document["run"]["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        correlation = dict(document["run"]["correlation"])
        if self.mangle_correlation:
            correlation["arm_id"] = "arm_somebody_else"
        (run_dir / "result_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "gepa.result-manifest.v1",
                    "run_id": document["run"]["run_id"],
                    "status": "completed",
                    "correlation": correlation,
                    "best_candidate": {
                        "candidate_id": "cand_1",
                        "heldout_reward": 0.5
                        + self.effort_bonus[effort]
                        + (document["run"]["seed"] % 3) * 0.01,
                        "train_reward": 0.4,
                    },
                    "cost_usd": 1.25,
                    "usage": {"rollouts": 12},
                }
            ),
            encoding="utf-8",
        )
        return {"manifest_path": str(run_dir / "result_manifest.json")}


@pytest.fixture
def rig(tmp_path):
    base = tmp_path / "banking77.toml"
    base.write_text(BASE_CONFIG, encoding="utf-8")
    spec_path = tmp_path / "experiment.toml"
    spec_path.write_text(SPEC.format(base=base, output_dir=tmp_path / "runs"), encoding="utf-8")
    return base, spec_path


def adapter_for(spec, runner=None):
    return GepaCliAdapter.from_spec(spec, runner=runner)


def test_the_catalog_offers_how_the_proposer_thinks_not_how_much_it_runs(rig):
    _, spec_path = rig
    spec = load_spec(spec_path)
    catalog = adapter_for(spec).factor_catalog(spec)
    paths = {factor.path for factor in catalog.factors}

    assert "proposer.reasoning_effort" in paths
    assert "gepa.pipeline.mode" in paths
    # Varying these changes how much work an arm does, which voids any
    # wall-clock or cost comparison between arms.
    assert "gepa.max_generations" not in paths
    assert "gepa.minibatch_size" not in paths
    assert "gepa.max_total_rollouts" not in paths


def test_a_reserved_bookkeeping_path_cannot_be_a_treatment(tmp_path, rig):
    _, spec_path = rig
    payload = load_spec(spec_path).to_json()
    payload["factors"] = {"run.seed": [1, 2]}
    spec = parse_spec(payload)
    with pytest.raises(ExperimentContractError, match="derived per trial"):
        compile_plan(spec, adapter_for(spec))
    assert "run.seed" in RESERVED_PATHS


def test_every_arm_is_proven_valid_before_anything_runs(tmp_path, rig):
    _, spec_path = rig
    payload = load_spec(spec_path).to_json()
    payload["factors"] = {"proposer.reasoning_effort": ["low", "extreme"]}
    spec = parse_spec(payload)
    with pytest.raises(ExperimentContractError, match="does not accept"):
        compile_plan(spec, adapter_for(spec))


def test_expansion_derives_per_trial_bookkeeping(rig):
    _, spec_path = rig
    spec = load_spec(spec_path)
    plan = compile_plan(spec, adapter_for(spec))

    assert len(plan.trials) == 8
    assert len({trial.trial_derived["run_id"] for trial in plan.trials}) == 8
    assert len({trial.trial_derived["cache_namespace"] for trial in plan.trials}) == 8
    assert {trial.trial_derived["seed"] for trial in plan.trials} == {11, 12, 13, 14}
    assert {arm.subject.subject_id for arm in plan.arms} == {
        "gpt-5.6-luna@low",
        "gpt-5.6-luna@medium",
    }
    assert plan.primary_metric_direction == "maximize"
    assert plan.secondary_metric_directions["cost_usd"] == "minimize"


def test_the_rendered_config_carries_the_envelope_and_only_the_treatment(rig, tmp_path):
    _, spec_path = rig
    spec = load_spec(spec_path)
    gepa = FakeGepa()
    runner = ExperimentRunner(spec, adapter_for(spec, gepa), tmp_path / "exp")
    runner.run(limit=1)

    document = tomllib.loads(gepa.configs[0].read_text(encoding="utf-8"))
    base = tomllib.loads(Path(spec.base).read_text(encoding="utf-8"))
    assert document["run"]["correlation"]["experiment_id"] == "luna-effort-v1"
    assert document["run"]["correlation"]["trial_id"].startswith("t")
    assert "run_id" not in document["run"]["correlation"]
    assert document["run"]["run_id"] != base["run"]["run_id"]
    assert document["cache"]["namespace"] == document["run"]["run_id"]
    # Everything outside the treatment and the per-trial block is the base.
    assert document["gepa"] == base["gepa"]
    assert document["taskset"] == base["taskset"]
    assert document["policy"] == base["policy"]


def test_end_to_end_paired_comparison(rig, tmp_path):
    _, spec_path = rig
    spec = load_spec(spec_path)
    runner = ExperimentRunner(spec, adapter_for(spec, FakeGepa()), tmp_path / "exp")
    summary = runner.run()
    report = runner.report()

    assert summary.dispatched == 8
    comparison = report.comparisons[0]
    assert comparison.blocks_paired == 4
    assert comparison.mean_delta == pytest.approx(0.10)
    assert comparison.ci_low > 0
    assert report.claim.allowed, report.claim.blockers
    assert report.totals["cost_usd"] == pytest.approx(10.0)

    cost = report.secondary["cost_usd"][0]
    assert cost.direction == "minimize"
    assert cost.mean_delta == pytest.approx(0.0)


def test_a_manifest_echoing_the_wrong_envelope_is_refused(rig, tmp_path):
    _, spec_path = rig
    spec = load_spec(spec_path)
    runner = ExperimentRunner(
        spec, adapter_for(spec, FakeGepa(mangle_correlation=True)), tmp_path / "exp"
    )
    with pytest.raises(ExperimentContractError, match="echoed a different"):
        runner.run(limit=1)
    # Nothing was sealed, so the trial stays pending rather than joining an arm
    # it may not belong to.
    assert runner.outcomes.sealed_trial_ids() == set()


def test_an_arm_that_cannot_run_is_retained_as_failure_not_absence(rig, tmp_path):
    _, spec_path = rig
    spec = load_spec(spec_path)
    runner = ExperimentRunner(
        spec, adapter_for(spec, FakeGepa(fail_efforts=["medium"])), tmp_path / "exp"
    )
    runner.run()
    report = runner.report()

    failing = next(arm for arm in report.arms if "medium" in arm.label)
    healthy = next(arm for arm in report.arms if "low" in arm.label)
    assert failing.completed_trials == 0
    assert failing.failed_trials == 4
    assert failing.failure_classes == {"infra": 4}
    assert healthy.completed_trials == 4
    assert not report.claim.allowed
    assert any("missing_policy" in blocker for blocker in report.claim.blockers)


def test_resume_reruns_nothing(rig, tmp_path):
    _, spec_path = rig
    spec = load_spec(spec_path)
    gepa = FakeGepa()
    root = tmp_path / "exp"
    ExperimentRunner(spec, adapter_for(spec, gepa), root).run()
    dispatched = len(gepa.configs)

    again = ExperimentRunner(spec, adapter_for(spec, gepa), root)
    assert again.run().dispatched == 0
    assert len(gepa.configs) == dispatched


def test_a_moved_base_config_refuses_to_resume(rig, tmp_path):
    base, spec_path = rig
    spec = load_spec(spec_path)
    root = tmp_path / "exp"
    ExperimentRunner(spec, adapter_for(spec, FakeGepa()), root).run(limit=2)

    base.write_text(BASE_CONFIG.replace("minibatch_size = 3", "minibatch_size = 5"), "utf-8")
    moved = ExperimentRunner(load_spec(spec_path), adapter_for(load_spec(spec_path)), root)
    with pytest.raises(ExperimentContractError, match="different experiment"):
        moved.run()


def test_the_envelope_survives_a_toml_round_trip(rig, tmp_path):
    """The failure this guards against is silent: TOML has no null.

    An optional field emitted as `null` would vanish on the way through an
    executor's own config format, and the envelope would come back with a
    different digest than the one the plan minted.
    """

    from synth_optimizers.gepa import _toml_dumps

    _, spec_path = rig
    spec = load_spec(spec_path)
    plan = compile_plan(spec, adapter_for(spec))
    envelope = plan.correlation_for(plan.trials[0])

    path = tmp_path / "round-trip.toml"
    path.write_text(_toml_dumps({"run": {"correlation": envelope.to_json()}}), encoding="utf-8")
    echoed = tomllib.loads(path.read_text(encoding="utf-8"))["run"]["correlation"]

    assert echoed == envelope.to_json()
    assert not _has_null(envelope.to_json())


def _has_null(value):
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_has_null(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_null(item) for item in value)
    return False
