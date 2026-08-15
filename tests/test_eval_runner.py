"""Local `eval` runner behaviour: evidence, scoring, semaphore, resume, cancel.

These use a fake trial executor rather than a real container so the invariants
under test are the runner's, not Docker's. `docker/eval-fixture-target` is the
matching real image for the end-to-end check.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from synth_optimizers.eval import (
    ContainerResult,
    EvalContractError,
    EvalHome,
    EvalRunner,
    SeedLedger,
    TrialSemaphore,
    WorkerManifest,
    catalog,
)
from synth_optimizers.eval.executor import TrialExecution, TrialRunRequest
from synth_optimizers.eval.staging import CandidateSource, stage_candidate_set

FIXTURE_RECIPE = "eval.fixture.policy-smoke.v1"
PINNED = "sha256:" + "ab" * 32

# Deterministic per-label scores. `champion` beats `baseline` on every seed,
# `laggard` loses on every seed, so paired lift has an unambiguous answer.
SCORES = {"baseline": 1.0, "champion": 2.0, "laggard": 0.25}


class ConcurrencyGauge:
    """Shared across executors so the assertion is about *global* concurrency."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.live = 0
        self.peak = 0

    def enter(self) -> None:
        with self._lock:
            self.live += 1
            self.peak = max(self.peak, self.live)

    def leave(self) -> None:
        with self._lock:
            self.live -= 1


class FakeExecutor:
    """Writes a conforming `/output/result.json`, like a real target would."""

    def __init__(
        self,
        *,
        fail_labels: set[str] | None = None,
        fail_stages: set[str] | None = None,
        delay: float = 0.0,
        gauge: ConcurrencyGauge | None = None,
        omit_trace: bool = False,
    ) -> None:
        self.omit_trace = omit_trace
        self.fail_labels = fail_labels or set()
        self.fail_stages = fail_stages
        self.delay = delay
        self.gauge = gauge
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def resolve_reference(self, image: str, digest: str) -> str:
        assert digest == PINNED
        return f"{image}@{digest}"

    def run(self, request: TrialRunRequest, *, on_event, should_cancel, heartbeat):
        with self._lock:
            self.calls.append(request.trial_id)
        if self.gauge:
            self.gauge.enter()
        try:
            started = time.time()
            trial = json.loads((request.input_dir / "trial.json").read_text(encoding="utf-8"))
            label = trial["candidate"]["label"]
            on_event({"event": "progress", "step": 0})
            if self.delay:
                time.sleep(self.delay)
            stage_matches = self.fail_stages is None or trial["stage"] in self.fail_stages
            if label in self.fail_labels and stage_matches:
                # Rig failure: no result.json at all.
                return TrialExecution(
                    exit_code=1,
                    timed_out=False,
                    cancelled=False,
                    started_at=started,
                    finished_at=time.time(),
                    stderr_tail="target crashed",
                )
            reward = SCORES[label] + (trial["seed"] % 3) * 0.01
            if self.omit_trace:
                (request.output_dir / "trace.jsonl").unlink(missing_ok=True)
            else:
                (request.output_dir / "trace.jsonl").write_text(
                    json.dumps({"trial_id": trial["trial_id"], "step": 0, "action": "right"})
                    + "\n",
                    encoding="utf-8",
                )
            (request.output_dir / "result.json").write_text(
                json.dumps(
                    {
                        "schema_version": "eval.container-result.v1",
                        "trial_id": trial["trial_id"],
                        "status": "evaluated",
                        "benchmark_status": "passed",
                        "metrics": {"reward": reward, "steps": 10.0},
                        "gates": [
                            {"id": "policy_loaded", "passed": True},
                            {"id": "verifier_completed", "passed": True},
                        ],
                        "usage": {"cost_usd": None, "rollouts": 1, "wall_time_ms": 5},
                        "artifacts": [{"role": "trace", "path": "trace.jsonl"}],
                    }
                ),
                encoding="utf-8",
            )
            return TrialExecution(
                exit_code=0,
                timed_out=False,
                cancelled=False,
                started_at=started,
                finished_at=time.time(),
                stderr_tail="",
            )
        finally:
            if self.gauge:
                self.gauge.leave()


def make_home(tmp_path: Path, *, capacity: int = 2) -> EvalHome:
    root = tmp_path / "evalhome"
    home = EvalHome.open(root)
    (root / "runtime.toml").write_text(
        f'container_runtime = "docker"\nmax_concurrent_trials = {capacity}\n'
        f"lease_ttl_seconds = 30\n",
        encoding="utf-8",
    )
    home = EvalHome.open(root)
    home.write_pin(FIXTURE_RECIPE, PINNED)
    return home


def stage(home: EvalHome, tmp_path: Path, labels=("baseline", "champion", "laggard")):
    sources = []
    for label in labels:
        directory = tmp_path / "src" / label
        directory.mkdir(parents=True)
        (directory / "policy.py").write_text(
            f'class Policy:\n    label = "{label}"\n\n'
            '    def act(self, observation):\n        return "right"\n',
            encoding="utf-8",
        )
        sources.append(
            CandidateSource(
                label=label,
                path=directory,
                entrypoint="policy:Policy",
                is_baseline=label == "baseline",
            )
        )
    return stage_candidate_set(home, sources)


def write_manifest(home: EvalHome, tmp_path: Path, candidate_set, run_id: str) -> Path:
    path = tmp_path / f"{run_id}.worker.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "eval.worker-manifest.v1",
                "run_id": run_id,
                "recipe_id": FIXTURE_RECIPE,
                "home": str(home.root),
                "candidate_set_path": str(
                    home.candidates_dir / candidate_set.id / "candidate_set.json"
                ),
                "session_ref": "session_test",
            }
        ),
        encoding="utf-8",
    )
    return path


def read_events(home: EvalHome, run_id: str) -> list[dict]:
    path = home.run_dir(run_id) / "events.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# --------------------------------------------------------------- contracts


def test_eval_is_not_a_hosted_algorithm():
    from synth_optimizers import future_algorithms, hosted

    slugs = {slug.value for slug in hosted.OptimizerAlgorithmSlug}
    assert "eval" not in slugs
    assert "eval" not in future_algorithms.__dict__.get("FUTURE_ALGORITHM_SLUGS", ())


def test_evaluated_result_must_report_benchmark_status():
    with pytest.raises(EvalContractError):
        ContainerResult.from_mapping(
            {
                "schema_version": "eval.container-result.v1",
                "trial_id": "trial_1",
                "status": "evaluated",
                "metrics": {"reward": 1.0},
            }
        )


def test_missing_metric_is_absent_not_zero():
    result = ContainerResult.from_mapping(
        {
            "schema_version": "eval.container-result.v1",
            "trial_id": "trial_1",
            "status": "evaluated",
            "benchmark_status": "failed",
            "metrics": {"reward": None},
        }
    )
    assert result.metrics == {}


def test_confirmation_seeds_must_be_disjoint_from_screening():
    with pytest.raises(EvalContractError):
        SeedLedger(
            screening=(101, 102),
            confirmation=(102,),
            scenarios=("default",),
            sealed_at="2026-08-15T00:00:00Z",
        )


def test_catalog_recipes_are_pinned_or_honestly_unavailable():
    for recipe in catalog():
        assert recipe.available == (recipe.image_digest is not None)
        if not recipe.available:
            assert recipe.unavailable_reason


# ------------------------------------------------------------------- runs


def test_full_run_scores_every_candidate_and_promotes_the_winner(tmp_path):
    home = make_home(tmp_path)
    candidate_set = stage(home, tmp_path)
    manifest = write_manifest(home, tmp_path, candidate_set, "run_promote")
    executor = FakeExecutor()

    code = EvalRunner(WorkerManifest.load(manifest), executor=executor).execute()
    assert code == 0

    run_dir = home.run_dir("run_promote")
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert selection["status"] == "promoted"
    scorecards = json.loads((run_dir / "scorecards.json").read_text(encoding="utf-8"))
    assert {card["label"] for card in scorecards["screen"]} == {
        "baseline",
        "champion",
        "laggard",
    }
    winner = next(
        card for card in scorecards["confirm"] if card["candidate_id"] == selection["winner_id"]
    )
    assert winner["label"] == "champion"
    assert winner["paired_lift"] == pytest.approx(1.0)
    assert winner["paired_trials"] == 2

    # Every candidate x seed x scenario has its own terminal record.
    result_manifest = json.loads((run_dir / "result_manifest.json").read_text(encoding="utf-8"))
    # 3 candidates x 2 screening seeds, then baseline + the one survivor x 2 confirmation seeds
    assert len(result_manifest["trials"]) == 3 * 2 + 2 * 2
    for trial in result_manifest["trials"]:
        assert Path(trial["evidence"]).is_file()


def test_elimination_is_recorded_and_keeps_screening_evidence(tmp_path):
    home = make_home(tmp_path)
    candidate_set = stage(home, tmp_path)
    manifest = write_manifest(home, tmp_path, candidate_set, "run_eliminate")
    EvalRunner(WorkerManifest.load(manifest), executor=FakeExecutor()).execute()

    events = read_events(home, "run_eliminate")
    eliminated = [event for event in events if event["event"] == "eval.candidate.eliminated"]
    assert [event["label"] for event in eliminated] == ["laggard"]
    assert "keep_top_k" in eliminated[0]["reason"]
    scorecards = json.loads(
        (home.run_dir("run_eliminate") / "scorecards.json").read_text(encoding="utf-8")
    )
    laggard = next(card for card in scorecards["screen"] if card["label"] == "laggard")
    assert laggard["trials"]["valid"] == 2  # screening evidence retained, not discarded


def test_failed_container_is_failed_evidence_not_a_zero(tmp_path):
    home = make_home(tmp_path)
    candidate_set = stage(home, tmp_path)
    manifest = write_manifest(home, tmp_path, candidate_set, "run_failed")
    EvalRunner(
        WorkerManifest.load(manifest), executor=FakeExecutor(fail_labels={"laggard"})
    ).execute()

    run_dir = home.run_dir("run_failed")
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("trials/*/job_result.json"))
    ]
    failed = [record for record in records if record["status"] == "failed"]
    assert failed and all(record["metrics"] == {} for record in failed)
    assert all(record["valid"] is False for record in failed)
    scorecards = json.loads((run_dir / "scorecards.json").read_text(encoding="utf-8"))
    laggard = next(card for card in scorecards["screen"] if card["label"] == "laggard")
    reward = next(metric for metric in laggard["metrics"] if metric["metric"] == "reward")
    assert reward["mean"] is None  # never coerced to 0.0
    assert laggard["trials"]["failed"] == 2
    # A screening failure removes that candidate; it does not invalidate a
    # confirmation stage that ran cleanly on the survivors.
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert selection["status"] == "promoted"


def test_a_failed_confirmation_trial_blocks_promotion(tmp_path):
    home = make_home(tmp_path)
    candidate_set = stage(home, tmp_path)
    manifest = write_manifest(home, tmp_path, candidate_set, "run_confirm_failed")
    EvalRunner(
        WorkerManifest.load(manifest),
        executor=FakeExecutor(fail_labels={"champion"}, fail_stages={"confirm"}),
    ).execute()

    selection = json.loads(
        (home.run_dir("run_confirm_failed") / "selection.json").read_text(encoding="utf-8")
    )
    assert selection["status"] == "invalid_evidence"
    assert selection["winner_id"] is None
    assert "not scored as zero" in selection["reason"]


def test_restart_resumes_without_rerunning_terminal_trials(tmp_path):
    home = make_home(tmp_path)
    candidate_set = stage(home, tmp_path)
    manifest = write_manifest(home, tmp_path, candidate_set, "run_resume")
    first = FakeExecutor()
    EvalRunner(WorkerManifest.load(manifest), executor=first).execute()
    ledger_before = (home.run_dir("run_resume") / "seed_ledger.json").read_text(encoding="utf-8")

    second = FakeExecutor()
    EvalRunner(WorkerManifest.load(manifest), executor=second).execute()

    assert first.calls and not second.calls
    ledger_after = (home.run_dir("run_resume") / "seed_ledger.json").read_text(encoding="utf-8")
    assert ledger_before == ledger_after


def test_global_semaphore_bounds_trials_across_concurrent_runs(tmp_path):
    home = make_home(tmp_path, capacity=2)
    gauge = ConcurrencyGauge()
    threads = []
    for index in range(3):
        candidate_set = stage(home, tmp_path / f"stage{index}")
        manifest = write_manifest(home, tmp_path, candidate_set, f"run_parallel_{index}")
        runner = EvalRunner(
            WorkerManifest.load(manifest),
            executor=FakeExecutor(delay=0.1, gauge=gauge),
        )
        threads.append(threading.Thread(target=runner.execute))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)

    # Three runs, each willing to use 2 workers, still never exceed the machine
    # ceiling: the semaphore is global, not one per run.
    assert gauge.peak <= 2
    assert gauge.peak == 2  # and it is actually saturated, so the bound is real
    semaphore = TrialSemaphore(home.semaphore_dir, capacity=2, ttl_seconds=30)
    assert semaphore.snapshot()["leased"] == 0  # every lease released


def test_cancellation_seals_evidence_and_refuses_to_promote(tmp_path):
    home = make_home(tmp_path)
    candidate_set = stage(home, tmp_path)
    manifest = write_manifest(home, tmp_path, candidate_set, "run_cancel")
    runner = EvalRunner(WorkerManifest.load(manifest), executor=FakeExecutor(delay=0.4))
    runner.cancel.cancel()
    assert runner.execute() == 0

    run_dir = home.run_dir("run_cancel")
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert selection["status"] == "invalid_evidence"
    assert "cancelled" in selection["reason"]
    terminal = [
        event for event in read_events(home, "run_cancel") if event["event"] == "eval.run.terminal"
    ]
    assert terminal[-1]["status"] == "cancelled"
    assert TrialSemaphore(home.semaphore_dir, capacity=2, ttl_seconds=30).snapshot()["leased"] == 0


def test_sealed_run_refuses_a_different_candidate_set(tmp_path):
    home = make_home(tmp_path)
    first_set = stage(home, tmp_path / "a")
    manifest = write_manifest(home, tmp_path, first_set, "run_sealed")
    EvalRunner(WorkerManifest.load(manifest), executor=FakeExecutor()).execute()

    second_set = stage(home, tmp_path / "b")
    manifest = write_manifest(home, tmp_path, second_set, "run_sealed")
    runner = EvalRunner(WorkerManifest.load(manifest), executor=FakeExecutor())
    assert runner.execute() == 1
    terminal = [
        event for event in read_events(home, "run_sealed") if event["event"] == "eval.run.terminal"
    ]
    assert terminal[-1]["status"] == "failed"
    assert "sealed" in terminal[-1]["error"]


def test_staged_candidate_digest_mismatch_stops_the_run(tmp_path):
    home = make_home(tmp_path)
    candidate_set = stage(home, tmp_path)
    artifact = next((home.candidates_dir / candidate_set.id / "artifacts").iterdir())
    artifact.chmod(0o700)
    tampered = artifact / "policy.py"
    tampered.chmod(0o600)
    tampered.write_text("class Policy:\n    pass\n", encoding="utf-8")

    manifest = write_manifest(home, tmp_path, candidate_set, "run_tampered")
    assert EvalRunner(WorkerManifest.load(manifest), executor=FakeExecutor()).execute() == 1
    terminal = read_events(home, "run_tampered")[-1]
    assert terminal["status"] == "failed"
    assert "digest" in terminal["error"]


def test_every_retained_output_file_is_indexed_with_a_digest(tmp_path):
    home = make_home(tmp_path)
    candidate_set = stage(home, tmp_path)
    manifest = write_manifest(home, tmp_path, candidate_set, "run_traces")
    EvalRunner(WorkerManifest.load(manifest), executor=FakeExecutor()).execute()

    result_manifest = json.loads(
        (home.run_dir("run_traces") / "result_manifest.json").read_text(encoding="utf-8")
    )
    traces = [item for item in result_manifest["artifacts"] if item["role"] == "trace"]
    assert len(traces) == len(result_manifest["trials"])  # one saved trace per trial
    for trace in traces:
        assert Path(trace["path"]).is_file()
        assert trace["digest"].startswith("sha256:")
        assert trace["bytes"] > 0


def test_a_target_that_promises_a_trace_and_omits_it_is_incomplete_evidence(tmp_path):
    home = make_home(tmp_path)
    candidate_set = stage(home, tmp_path)
    manifest = write_manifest(home, tmp_path, candidate_set, "run_no_trace")
    EvalRunner(WorkerManifest.load(manifest), executor=FakeExecutor(omit_trace=True)).execute()

    run_dir = home.run_dir("run_no_trace")
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("trials/*/job_result.json"))
    ]
    assert records and all(record["missing_artifacts"] == ["trace"] for record in records)
    # The container evaluated the policy, but the run cannot be re-checked, so
    # the evidence is not usable for a decision.
    assert all(record["status"] == "evaluated" for record in records)
    assert all(record["valid"] is False for record in records)
    selection = json.loads((run_dir / "selection.json").read_text(encoding="utf-8"))
    assert selection["status"] == "invalid_evidence"


def test_pausing_holds_the_matrix_and_resuming_finishes_it(tmp_path):
    home = make_home(tmp_path)
    candidate_set = stage(home, tmp_path)
    manifest = write_manifest(home, tmp_path, candidate_set, "run_paused")
    runner = EvalRunner(WorkerManifest.load(manifest), executor=FakeExecutor())
    # Pause before the first trial is dispatched.
    (home.run_dir("run_paused") / "PAUSE").write_text("now", encoding="utf-8")

    thread = threading.Thread(target=runner.execute)
    thread.start()
    time.sleep(1.5)
    events = read_events(home, "run_paused")
    assert any(event["event"] == "eval.run.paused" for event in events)
    assert not any(event["event"] == "eval.trial.terminal" for event in events)

    (home.run_dir("run_paused") / "PAUSE").unlink()
    thread.join(timeout=90)
    assert not thread.is_alive()

    events = read_events(home, "run_paused")
    assert any(event["event"] == "eval.run.resumed" for event in events)
    selection = json.loads(
        (home.run_dir("run_paused") / "selection.json").read_text(encoding="utf-8")
    )
    assert selection["status"] == "promoted"  # a pause changes timing, not evidence
