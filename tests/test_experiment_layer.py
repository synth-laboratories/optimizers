"""The experiment layer: expansion, refusals, correlation, resume, and claims.

These drive the real `eval` runtime through the real adapter, with the same fake
trial executor `test_eval_runner.py` uses, so what is under test is the
experiment layer's arithmetic and its refusals — not Docker, and not a mock of
the thing the layer is supposed to be integrating with.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_eval_runner import FIXTURE_RECIPE, PINNED, FakeExecutor, make_home, stage

from synth_optimizers.eval import EvalHome
from synth_optimizers.experiment import (
    EvalRuntimeAdapter,
    ExperimentContractError,
    ExperimentRunner,
    compile_plan,
    load_spec,
    mint_trial_id,
    parse_spec,
)
from synth_optimizers.experiment.plan import assert_only_treatment_differs, diff_projections

EXPERIMENT_ID = "fixture-policy-ablation-v1"


def write_spec(
    tmp_path: Path,
    home: EvalHome,
    candidate_set_path: Path,
    *,
    levels=("baseline", "champion"),
    blocks=(101, 102, 201, 202),
    replicates: int = 1,
    counterbalance: bool = True,
    missing_policy: str = "fail",
    min_blocks: int = 2,
    extra: str = "",
    experiment_id: str = EXPERIMENT_ID,
) -> Path:
    path = tmp_path / "experiment.toml"
    path.write_text(
        f'''
schema = "synth.experiment.v1"
experiment_id = "{experiment_id}"
executor = "eval.runtime"
base = "{FIXTURE_RECIPE}"

[design]
primary_metric = "reward"
secondary_metrics = ["steps"]
pairing = "block"
counterbalance = {str(counterbalance).lower()}
missing_policy = "{missing_policy}"
min_blocks_for_claim = {min_blocks}
bootstrap_resamples = 2000

[blocks]
kind = "seed"
values = {list(blocks)}
replicates = {replicates}

[factors]
"policy.candidate" = {list(levels)}

[budget]
max_trials = 64

[isolation]
cache_namespace = "per_trial"
container = "fresh_per_trial"

[executor_options]
home = "{home.root}"
candidate_set = "{candidate_set_path}"
{extra}
'''.strip()
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def rig(tmp_path):
    home = make_home(tmp_path, capacity=2)
    candidate_set = stage(home, tmp_path)
    spec_path = write_spec(tmp_path, home, candidate_set.root / "candidate_set.json")
    return home, candidate_set, spec_path


def adapter_for(spec, executor=None):
    return EvalRuntimeAdapter.from_spec(spec, executor=executor)


# --------------------------------------------------------------------- expand


def test_expansion_is_deterministic_and_exact(rig):
    _, _, spec_path = rig
    spec = load_spec(spec_path)
    first = compile_plan(spec, adapter_for(spec))
    second = compile_plan(load_spec(spec_path), adapter_for(spec))

    assert len(first.trials) == 8  # 2 arms x 4 blocks x 1 replicate
    assert first.plan_digest == second.plan_digest
    assert [t.trial_id for t in first.trials] == [t.trial_id for t in second.trials]
    assert len({t.trial_id for t in first.trials}) == 8
    assert first.trials[0].trial_id == mint_trial_id(
        experiment_id=EXPERIMENT_ID,
        arm_id=first.trials[0].arm_id,
        block_id=first.trials[0].block_id,
        replicate=0,
    )


def test_counterbalance_alternates_who_goes_first(rig):
    _, _, spec_path = rig
    spec = load_spec(spec_path)
    plan = compile_plan(spec, adapter_for(spec))
    firsts = [plan.trials[index].arm_id for index in range(0, len(plan.trials), len(plan.arms))]
    assert len(set(firsts)) == 2, "each arm must lead at least one block"
    assert firsts[0] != firsts[1]

    fixed_order = compile_plan(
        parse_spec(
            {**spec.to_json(), "design": {**spec.to_json()["design"], "counterbalance": False}}
        ),
        adapter_for(spec),
    )
    leaders = {
        fixed_order.trials[index].arm_id
        for index in range(0, len(fixed_order.trials), len(fixed_order.arms))
    }
    assert len(leaders) == 1, "without counterbalancing one arm always leads"


def test_subject_is_the_content_digest_not_the_label(rig):
    _, candidate_set, spec_path = rig
    spec = load_spec(spec_path)
    plan = compile_plan(spec, adapter_for(spec))
    digests = {arm.subject.subject_content_digest for arm in plan.arms}
    assert digests == {
        candidate_set.candidate(c.id).artifact_digest
        for c in candidate_set.candidates
        if c.label in ("baseline", "champion")
    }
    assert all(arm.subject.subject_kind == "policy-candidate" for arm in plan.arms)
    assert_only_treatment_differs(plan)
    diff = diff_projections(plan, plan.arms[0].arm_id, plan.arms[1].arm_id)
    assert list(diff["treatment"]) == ["policy.candidate"]


def test_correlation_envelope_is_minted_per_trial(rig):
    _, _, spec_path = rig
    spec = load_spec(spec_path)
    plan = compile_plan(spec, adapter_for(spec))
    envelopes = [plan.correlation_for(trial) for trial in plan.trials]
    assert len({envelope.digest for envelope in envelopes}) == len(plan.trials)
    for envelope in envelopes:
        assert envelope.plan_digest == plan.plan_digest
        assert set(envelope.aliases()) == {"experiment_id", "trial_id", "candidate_id"}
        assert envelope.candidate_id.startswith("policy_")


# -------------------------------------------------------------------- refuse


def test_unknown_factor_is_refused(tmp_path, rig):
    home, candidate_set, _ = rig
    path = write_spec(
        tmp_path,
        home,
        candidate_set.root / "candidate_set.json",
        extra="",
    )
    payload = load_spec(path).to_json()
    payload["factors"] = {"limits.timeout_seconds": [60, 120]}
    spec = parse_spec(payload)
    with pytest.raises(ExperimentContractError, match="ablatable factor"):
        compile_plan(spec, adapter_for(spec))


def test_seed_outside_the_recipe_is_refused(tmp_path, rig):
    home, candidate_set, _ = rig
    path = write_spec(tmp_path, home, candidate_set.root / "candidate_set.json", blocks=(101, 999))
    spec = load_spec(path)
    with pytest.raises(ExperimentContractError, match="does not declare seeds"):
        compile_plan(spec, adapter_for(spec))


def test_over_budget_matrix_is_refused(tmp_path, rig):
    home, candidate_set, _ = rig
    path = write_spec(tmp_path, home, candidate_set.root / "candidate_set.json")
    payload = load_spec(path).to_json()
    payload["budget"] = {"max_trials": 4}
    with pytest.raises(ExperimentContractError, match="over the declared"):
        parse_spec(payload)


def test_single_level_factor_is_refused(tmp_path, rig):
    home, candidate_set, _ = rig
    path = write_spec(
        tmp_path, home, candidate_set.root / "candidate_set.json", levels=("baseline",)
    )
    with pytest.raises(ExperimentContractError, match="at least two levels"):
        load_spec(path)


def test_claim_floor_above_the_block_count_is_refused(tmp_path, rig):
    home, candidate_set, _ = rig
    path = write_spec(
        tmp_path,
        home,
        candidate_set.root / "candidate_set.json",
        blocks=(101, 102),
        min_blocks=5,
    )
    with pytest.raises(ExperimentContractError, match="exceeds the"):
        load_spec(path)


def test_unknown_top_level_key_is_refused(tmp_path, rig):
    home, candidate_set, _ = rig
    path = write_spec(tmp_path, home, candidate_set.root / "candidate_set.json")
    payload = load_spec(path).to_json()
    payload["couterbalance"] = True  # a typo of a real control
    with pytest.raises(ExperimentContractError, match="unknown top-level keys"):
        parse_spec(payload)


# ------------------------------------------------------------------- execute


def run_experiment(spec_path: Path, root: Path, executor: FakeExecutor, **kwargs):
    spec = load_spec(spec_path)
    runner = ExperimentRunner(spec, adapter_for(spec, executor), root, **kwargs)
    summary = runner.run()
    return runner, summary, runner.report()


def test_end_to_end_paired_comparison(tmp_path, rig):
    _, _, spec_path = rig
    executor = FakeExecutor()
    _, summary, report = run_experiment(spec_path, tmp_path / "exp", executor)

    assert summary.dispatched == 8
    assert report.totals["completed_trials"] == 8
    assert report.totals["completion_rate"] == 1.0

    comparison = report.comparisons[0]
    assert comparison.blocks_paired == 4
    # champion scores 2.0 against baseline's 1.0 on every seed, by construction.
    assert comparison.mean_delta == pytest.approx(1.0)
    assert comparison.wins == 4 and comparison.losses == 0
    assert comparison.ci_low > 0
    assert comparison.p_method == "exact-permutation"
    assert comparison.p_value == pytest.approx(2 / 16)
    assert report.claim.allowed, report.claim.blockers
    assert report.direction == "maximize"

    secondary = report.secondary["steps"][0]
    assert secondary.direction == "minimize"
    assert secondary.mean_delta == pytest.approx(0.0)


def test_each_trial_is_its_own_sealed_receipt(tmp_path, rig):
    _, _, spec_path = rig
    runner, _, report = run_experiment(spec_path, tmp_path / "exp", FakeExecutor())
    rows = runner.outcomes.load()
    assert len(rows) == 8
    assert len({row.receipt_digest for row in rows}) == 8
    assert len({row.executor_run_id for row in rows}) == 8
    for row in rows:
        manifest = json.loads(Path(row.evidence["result_manifest"]).read_text(encoding="utf-8"))
        assert manifest["correlation"]["trial_id"] == row.trial_id
        assert manifest["correlation"]["experiment_id"] == EXPERIMENT_ID
        assert row.infra["image_reference"].endswith(PINNED)
        # Every trial keeps the rollout it was scored on, digest and all, so a
        # number can be re-derived later rather than trusted.
        traces = row.evidence["trace_refs"]
        assert len(traces) == 1
        assert traces[0]["relative_path"] == "trace.jsonl"
        assert traces[0]["digest"].startswith("sha256:")
    # Per-trial isolation is a fact the report can check, not a policy string.
    assert report.fairness.distinct_cache_namespaces == 8
    assert report.fairness.trials_observed == 8


def test_resume_is_idempotent(tmp_path, rig):
    _, _, spec_path = rig
    root = tmp_path / "exp"
    executor = FakeExecutor()
    _, first, _ = run_experiment(spec_path, root, executor)
    calls_after_first = len(executor.calls)

    spec = load_spec(spec_path)
    again = ExperimentRunner(spec, adapter_for(spec, executor), root)
    second = again.run()

    assert first.dispatched == 8
    assert second.dispatched == 0
    assert second.skipped == 8
    assert len(executor.calls) == calls_after_first
    assert len(again.outcomes.load()) == 8
    assert again.outcomes.load().conflicts == ()


def test_partial_run_then_resume_completes_the_matrix(tmp_path, rig):
    _, _, spec_path = rig
    root = tmp_path / "exp"
    executor = FakeExecutor()
    spec = load_spec(spec_path)
    first = ExperimentRunner(spec, adapter_for(spec, executor), root)
    assert first.run(limit=3).dispatched == 3
    assert not first.report().claim.allowed

    second = ExperimentRunner(spec, adapter_for(spec, executor), root)
    summary = second.run()
    assert summary.dispatched == 5
    assert second.report().totals["completed_trials"] == 8


def test_failed_trials_are_retained_and_block_the_claim(tmp_path, rig):
    _, _, spec_path = rig
    executor = FakeExecutor(fail_labels={"champion"})
    _, _, report = run_experiment(spec_path, tmp_path / "exp", executor)

    by_arm = {arm.label: arm for arm in report.arms}
    champion = next(arm for label, arm in by_arm.items() if "champion" in label)
    baseline = next(arm for label, arm in by_arm.items() if "baseline" in label)
    assert champion.completed_trials == 0
    assert champion.failed_trials == 4
    assert champion.failure_classes == {"rig": 4}
    assert len(champion.missing_blocks) == 4
    assert baseline.completed_trials == 4

    assert not report.claim.allowed
    assert any("missing_policy" in blocker for blocker in report.claim.blockers)
    assert report.comparisons[0].blocks_paired == 0


def test_differential_missingness_blocks_a_pairwise_complete_claim(tmp_path, rig):
    home, candidate_set, _ = rig
    spec_path = write_spec(
        tmp_path,
        home,
        candidate_set.root / "candidate_set.json",
        missing_policy="pairwise_complete",
        min_blocks=2,
    )
    executor = FakeExecutor(fail_labels={"champion"}, fail_stages={"screen"})
    _, _, report = run_experiment(spec_path, tmp_path / "exp", executor)

    assert not report.claim.allowed
    assert any("missingness differs" in blocker for blocker in report.claim.blockers)


def test_aa_run_never_yields_a_headline(tmp_path, rig):
    _, _, spec_path = rig
    spec = load_spec(spec_path)
    runner = ExperimentRunner(spec, adapter_for(spec, FakeExecutor()), tmp_path / "aa", mode="aa")
    runner.run()
    report = runner.report()

    assert len(report.arms) == 2
    assert report.arms[0].mean == report.arms[1].mean
    assert report.comparisons[0].mean_delta == pytest.approx(0.0)
    assert not report.claim.allowed
    assert any("A/A run" in blocker for blocker in report.claim.blockers)


def test_a_moved_pin_refuses_to_resume(tmp_path, rig):
    home, _, spec_path = rig
    root = tmp_path / "exp"
    executor = FakeExecutor()
    spec = load_spec(spec_path)
    ExperimentRunner(spec, adapter_for(spec, executor), root).run(limit=2)

    home.write_pin(FIXTURE_RECIPE, "sha256:" + "cd" * 32)
    moved = ExperimentRunner(load_spec(spec_path), adapter_for(load_spec(spec_path)), root)
    with pytest.raises(ExperimentContractError, match="different experiment"):
        moved.run()


def test_an_edited_plan_file_is_detected(tmp_path, rig):
    _, _, spec_path = rig
    root = tmp_path / "exp"
    spec = load_spec(spec_path)
    runner = ExperimentRunner(spec, adapter_for(spec, FakeExecutor()), root)
    runner.prepare()
    payload = json.loads(runner.plan_path.read_text(encoding="utf-8"))
    payload["blocks"]["ids"] = payload["blocks"]["ids"][:2]
    runner.plan_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExperimentContractError, match="edited after it was written"):
        runner.prepare()


def test_a_contradictory_outcome_row_blocks_the_claim(tmp_path, rig):
    _, _, spec_path = rig
    root = tmp_path / "exp"
    runner, _, _ = run_experiment(spec_path, root, FakeExecutor())
    rows = runner.outcomes.load()
    duplicate = rows.rows[0].to_json()
    duplicate["metrics"] = {"reward": 99.0}
    with open(runner.outcomes.path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(duplicate) + "\n")

    reloaded = runner.outcomes.load()
    assert len(reloaded.conflicts) == 1
    report = runner.report()
    assert not report.claim.allowed
    assert any("sealed twice" in blocker for blocker in report.claim.blockers)
    # The first row still wins; a later write cannot overwrite sealed evidence.
    assert report.arms[0].mean != 99.0


def test_reports_are_reproducible(tmp_path, rig):
    _, _, spec_path = rig
    root = tmp_path / "exp"
    runner, _, first = run_experiment(spec_path, root, FakeExecutor())
    second = runner.report()
    assert first.to_json() == second.to_json()


class TiedExecutor(FakeExecutor):
    """Both arms score identically, so the paired difference is exactly zero."""

    def run(self, request, *, on_event, should_cancel, heartbeat):
        trial = json.loads((request.input_dir / "trial.json").read_text(encoding="utf-8"))
        original = trial["candidate"]["label"]
        trial["candidate"]["label"] = "baseline"
        (request.input_dir / "trial.json").write_text(json.dumps(trial), encoding="utf-8")
        try:
            return super().run(
                request, on_event=on_event, should_cancel=should_cancel, heartbeat=heartbeat
            )
        finally:
            trial["candidate"]["label"] = original
            (request.input_dir / "trial.json").write_text(json.dumps(trial), encoding="utf-8")


def test_an_interval_containing_zero_blocks_the_claim(tmp_path, rig):
    _, _, spec_path = rig
    _, _, report = run_experiment(spec_path, tmp_path / "exp", TiedExecutor())

    comparison = report.comparisons[0]
    assert comparison.blocks_paired == 4
    assert comparison.mean_delta == pytest.approx(0.0)
    assert comparison.ci_low <= 0.0 <= comparison.ci_high
    assert not report.claim.allowed
    assert any("contains zero" in blocker for blocker in report.claim.blockers)
    # The null result is still fully reported; only the headline is refused.
    assert report.totals["completed_trials"] == 8


#: The exact bytes the Rust `synth_optimizer_platform::correlation` tests pin.
#: Both sides assert this literal, so a field added to one and not the other
#: fails a test here rather than a run in the field.
RUST_WIRE_FORM = '{"arm_id":"arm_6edf53cf5835","block_id":"seed:104","experiment_id":"luna-effort-v1","plan_digest":"sha256:abababababababababababababababababababababababababababababababab","replicate":0,"schema_version":"synth.correlation.v1","subject":{"subject_content_digest":"sha256:cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd","subject_id":"gpt-5.6-luna@low","subject_kind":"proposer-policy"},"trial_id":"t676b09f94e51e2f3"}'


def test_the_envelope_wire_form_matches_what_the_rust_side_pins():
    from synth_optimizers.experiment.models import (
        CorrelationEnvelope,
        SubjectRef,
        canonical_json,
    )

    envelope = CorrelationEnvelope(
        experiment_id="luna-effort-v1",
        arm_id="arm_6edf53cf5835",
        block_id="seed:104",
        replicate=0,
        trial_id="t676b09f94e51e2f3",
        plan_digest="sha256:" + "ab" * 32,
        subject=SubjectRef(
            subject_kind="proposer-policy",
            subject_id="gpt-5.6-luna@low",
            subject_content_digest="sha256:" + "cd" * 32,
        ),
    )
    assert canonical_json(envelope.to_json()) == RUST_WIRE_FORM
    # Parsing it back is lossless, which is what makes the digest a join key.
    assert CorrelationEnvelope.from_mapping(json.loads(RUST_WIRE_FORM)) == envelope


def test_an_out_of_repo_adapter_can_register_itself():
    from synth_optimizers.experiment import REGISTRY, register_adapter

    class MatrixAdapter:
        executor_id = "test.matrix"

        @classmethod
        def from_spec(cls, spec, **overrides):
            return cls()

    try:
        register_adapter(MatrixAdapter)
        assert REGISTRY["test.matrix"] is MatrixAdapter

        class Impostor(MatrixAdapter):
            pass

        # Two adapters answering to one executor id would make which executor a
        # sealed plan actually ran a function of import order.
        with pytest.raises(ExperimentContractError, match="already registered"):
            register_adapter(Impostor)
        register_adapter(Impostor, replace=True)
        assert REGISTRY["test.matrix"] is Impostor
    finally:
        REGISTRY.pop("test.matrix", None)


def test_a_registration_missing_the_protocol_is_refused():
    from synth_optimizers.experiment import register_adapter

    class NoId:
        @classmethod
        def from_spec(cls, spec, **overrides):
            return cls()

    class NoFactory:
        executor_id = "test.nofactory"

    with pytest.raises(ExperimentContractError, match="executor_id"):
        register_adapter(NoId)
    with pytest.raises(ExperimentContractError, match="from_spec"):
        register_adapter(NoFactory)


class FlakyOnceExecutor(FakeExecutor):
    """Fails every trial on the first pass, succeeds on any re-dispatch."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seen: set[str] = set()

    def run(self, request, *, on_event, should_cancel, heartbeat):
        first = request.trial_id not in self.seen
        self.seen.add(request.trial_id)
        if first:
            import time as _t

            from synth_optimizers.eval.executor import TrialExecution

            return TrialExecution(
                exit_code=1,
                timed_out=False,
                cancelled=False,
                started_at=_t.time(),
                finished_at=_t.time(),
                stderr_tail="container vanished",
            )
        return super().run(
            request, on_event=on_event, should_cancel=should_cancel, heartbeat=heartbeat
        )


def test_a_rig_failure_can_be_retried_and_the_retry_is_never_silent(tmp_path, rig):
    _, _, spec_path = rig
    root = tmp_path / "exp"
    spec = load_spec(spec_path)
    executor = FlakyOnceExecutor()

    first = ExperimentRunner(spec, adapter_for(spec, executor), root)
    first.run()
    assert first.report().totals["completed_trials"] == 0
    assert not first.report().claim.allowed

    second = ExperimentRunner(spec, adapter_for(spec, executor), root)
    assert second.run(retry_rig_failures=True).dispatched == 8
    report = second.report()

    assert report.totals["completed_trials"] == 8
    assert report.totals["retried_trials"] == 8
    # The superseded failures are still in the log; nothing was edited away.
    rows = second.outcomes.load()
    assert len(rows.superseded) == 8
    assert all(row.attempt == 0 for row in rows.superseded)
    assert all(row.attempt == 1 and row.supersedes for row in rows.rows)
    assert any("re-dispatched after a rig failure" in note for note in report.claim.notes)
    assert report.claim.allowed, report.claim.blockers


def test_only_the_rigs_failures_are_retryable(tmp_path, rig):
    from synth_optimizers.experiment.models import RETRYABLE_FAILURE_CLASSES

    _, _, spec_path = rig
    root = tmp_path / "exp"
    spec = load_spec(spec_path)
    runner = ExperimentRunner(spec, adapter_for(spec, FakeExecutor()), root)
    runner.run(limit=1)
    row = runner.outcomes.load().rows[0]

    # A completed trial is never re-dispatched, whatever the flag says.
    assert not row.retryable
    from dataclasses import replace

    assert replace(row, status="failed", failure_class="rig").retryable
    assert replace(row, status="failed", failure_class="infra").retryable
    # The thing under test, and the ceilings that may *be* the arm difference,
    # stay sealed: retrying them would be selecting for the result you wanted.
    assert not replace(row, status="failed", failure_class="policy").retryable
    assert not replace(row, status="failed", failure_class="budget").retryable
    assert not replace(row, status="timeout", failure_class="timeout").retryable
    assert RETRYABLE_FAILURE_CLASSES == frozenset({"rig", "infra"})


def test_two_rows_for_one_attempt_are_still_a_contradiction(tmp_path, rig):
    _, _, spec_path = rig
    root = tmp_path / "exp"
    runner, _, _ = run_experiment(spec_path, root, FakeExecutor())
    duplicate = runner.outcomes.load().rows[0].to_json()
    duplicate["metrics"] = {"reward": 99.0}
    with open(runner.outcomes.path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(duplicate) + "\n")

    reloaded = runner.outcomes.load()
    assert len(reloaded.conflicts) == 1
    assert reloaded.superseded == ()
    assert not runner.report().claim.allowed
