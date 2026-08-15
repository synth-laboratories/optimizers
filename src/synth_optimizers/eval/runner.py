"""The local `eval` runner.

One run seals its inputs, expands them into a fair `candidate × seed × scenario`
matrix, leases a token from the machine-global semaphore for each container it
starts, normalizes what comes back, scores every candidate separately, and
issues a selection decision. Orchestration reaching a terminal state is not the
same claim as a candidate winning, and the two are reported separately.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import signal
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .executor import (
    ContainerRuntimeError,
    OciTrialExecutor,
    TrialExecutor,
    TrialRunRequest,
)
from .home import EvalHome
from .models import (
    CANDIDATE_SET_SCHEMA,
    EVAL_ALGORITHM_ID,
    EVAL_ALGORITHM_VERSION,
    RUN_MANIFEST_SCHEMA,
    TRIAL_MANIFEST_SCHEMA,
    WORKER_EVENT_SCHEMA,
    WORKER_MANIFEST_SCHEMA,
    CandidateScorecard,
    CandidateSet,
    ContainerResult,
    EvalContractError,
    PolicyCandidate,
    SelectionDecision,
    TrialKey,
    TrialRecord,
    append_jsonl,
    canonical_json,
    digest_of,
    digest_of_tree,
    write_json,
)
from .recipes import EvalRecipe
from .scoring import apply_elimination, decide, summarize_candidate
from .semaphore import SemaphoreTimeout, TrialSemaphore

CANCEL_SENTINEL = "CANCEL"
PAUSE_SENTINEL = "PAUSE"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class WorkerManifest:
    """The app-owned launch description. Never authored by an agent."""

    run_id: str
    recipe_id: str
    home: Path
    candidate_set_path: Path
    session_ref: str | None

    @classmethod
    def load(cls, path: Path) -> WorkerManifest:
        payload = json.loads(path.read_text(encoding="utf-8"))
        schema = payload.get("schema_version")
        if schema != WORKER_MANIFEST_SCHEMA:
            raise EvalContractError(
                f"worker manifest schema must be {WORKER_MANIFEST_SCHEMA}, got {schema!r}"
            )
        for key in ("run_id", "recipe_id", "home", "candidate_set_path"):
            if not isinstance(payload.get(key), str) or not payload[key].strip():
                raise EvalContractError(f"worker manifest requires {key}")
        return cls(
            run_id=payload["run_id"],
            recipe_id=payload["recipe_id"],
            home=Path(payload["home"]).expanduser(),
            candidate_set_path=Path(payload["candidate_set_path"]).expanduser(),
            session_ref=payload.get("session_ref"),
        )


class EventEmitter:
    """Run lifecycle events, appended durably and mirrored to stdout.

    The runner emits its own lifecycle events even for containers that provide
    no live stream, so Workshop's timeline is never a function of how chatty a
    particular target happens to be.
    """

    def __init__(self, path: Path, *, run_id: str, stream: Any = None) -> None:
        self.path = path
        self.run_id = run_id
        self.stream = stream
        self._lock = threading.Lock()
        self._sequence = 0

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            payload = {
                "schema_version": WORKER_EVENT_SCHEMA,
                "seq": self._sequence,
                "event": event,
                "occurred_at": _now(),
                "run_id": self.run_id,
                **fields,
            }
            append_jsonl(self.path, [payload])
            if self.stream is not None:
                self.stream.write(canonical_json(payload) + "\n")
                self.stream.flush()
            return payload


class CancellationToken:
    """Cancellation from a signal or from the app-written CANCEL sentinel.

    Workshop writes the sentinel and waits before killing the process, so the
    worker gets the chance to stop its containers, release its leases, and seal
    evidence rather than leaving a half-written run behind.
    """

    def __init__(self, run_dir: Path) -> None:
        self._event = threading.Event()
        self._sentinel = run_dir / CANCEL_SENTINEL
        self._stop = threading.Event()
        self._watcher = threading.Thread(target=self._watch, daemon=True)

    def start(self) -> None:
        self._watcher.start()
        for signal_number in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(signal_number, lambda *_: self.cancel())
            except ValueError:
                pass  # not on the main thread; the sentinel still works

    def _watch(self) -> None:
        while not self._stop.wait(0.5):
            if self._sentinel.exists():
                self._event.set()
                return

    def cancel(self) -> None:
        self._event.set()

    def close(self) -> None:
        self._stop.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class PauseGate:
    """Holds the matrix without abandoning what is already running.

    Workshop writes a `PAUSE` sentinel; the scheduler stops dispatching new
    trials while it exists. Trials already in a container keep going and seal
    their evidence — a pause that threw away in-flight work would make pausing
    more destructive than cancelling.
    """

    def __init__(self, run_dir: Path, events: "EventEmitter") -> None:
        self._sentinel = run_dir / PAUSE_SENTINEL
        self._events = events
        self._announced = False

    @property
    def paused(self) -> bool:
        return self._sentinel.exists()

    def wait_while_paused(self, should_abort: Callable[[], bool]) -> None:
        while self.paused and not should_abort():
            if not self._announced:
                self._announced = True
                self._events.emit("eval.run.paused", reason="paused by Workshop")
            time.sleep(0.5)
        if self._announced and not self.paused:
            self._announced = False
            self._events.emit("eval.run.resumed")


class EvalRunner:
    def __init__(
        self,
        manifest: WorkerManifest,
        *,
        executor: TrialExecutor | None = None,
        stream: Any = None,
    ) -> None:
        self.manifest = manifest
        self.home = EvalHome.open(manifest.home)
        self.recipe: EvalRecipe = self.home.recipe(manifest.recipe_id)
        self.candidate_set = CandidateSet.load(manifest.candidate_set_path)
        self.run_dir = self.home.run_dir(manifest.run_id)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events = EventEmitter(
            self.run_dir / "events.jsonl", run_id=manifest.run_id, stream=stream
        )
        self.cancel = CancellationToken(self.run_dir)
        self.pause = PauseGate(self.run_dir, self.events)
        self.semaphore = TrialSemaphore(
            self.home.semaphore_dir,
            capacity=self.home.config.max_concurrent_trials,
            ttl_seconds=self.home.config.lease_ttl_seconds,
        )
        self._executor = executor
        self._image_reference = ""
        self._resolved_secrets: dict[str, str] | None = None
        self._parallelism = min(
            self.recipe.limits.max_parallel_trials, self.home.config.max_concurrent_trials
        )

    # ---------------------------------------------------------------- inputs

    def _resolve_executor(self) -> TrialExecutor:
        if self._executor is None:
            self._executor = OciTrialExecutor(self.home.config.container_runtime)
        return self._executor

    def _validate_inputs(self) -> None:
        """Every digest checked before a single container starts."""

        if not self.recipe.available:
            raise EvalContractError(
                f"recipe {self.recipe.id} is unavailable: {self.recipe.unavailable_reason}"
            )
        for candidate in self.candidate_set.candidates:
            if candidate.kind not in self.recipe.target.policy_kinds:
                raise EvalContractError(
                    f"candidate {candidate.label} is {candidate.kind}, which target "
                    f"{self.recipe.image} does not accept"
                )
            path = self.candidate_set.artifact_path(candidate)
            actual = digest_of_tree(path)
            if actual != candidate.artifact_digest:
                raise EvalContractError(
                    f"staged artifact for candidate {candidate.label} has digest {actual}, "
                    f"but the candidate set records {candidate.artifact_digest}"
                )
        self._secrets()  # fail before a single container starts, not mid-matrix
        executor = self._resolve_executor()
        resolve = getattr(executor, "resolve_reference", None)
        self._image_reference = (
            resolve(self.recipe.image, self.recipe.image_digest)
            if callable(resolve)
            else self.recipe.pinned_reference
        )

    def _seal(self) -> dict[str, Any]:
        """Write the sealed input manifest, or adopt the one a prior run wrote.

        Resume reuses the sealed seeds rather than regenerating them: a restart
        must not silently change what the run was measuring.
        """

        path = self.run_dir / "input_manifest.json"
        ledger = self.recipe.seed_ledger(sealed_at=_now())
        manifest = {
            "schema_version": RUN_MANIFEST_SCHEMA,
            "run_id": self.manifest.run_id,
            "algorithm": EVAL_ALGORITHM_ID,
            "algorithm_version": EVAL_ALGORITHM_VERSION,
            "session_ref": self.manifest.session_ref,
            "recipe": {
                "id": self.recipe.id,
                "digest": digest_of(self.recipe.to_json()),
                "image": self.recipe.image,
                "image_digest": self.recipe.image_digest,
                "target_manifest": self.recipe.target.to_json(),
                "target_manifest_digest": digest_of(self.recipe.target.to_json()),
            },
            "candidate_set": {
                "schema_version": CANDIDATE_SET_SCHEMA,
                "id": self.candidate_set.id,
                "digest": self.candidate_set.digest(),
                "baseline_id": self.candidate_set.baseline_id,
                "candidates": [
                    {
                        "id": candidate.id,
                        "label": candidate.label,
                        "kind": candidate.kind,
                        "digest": candidate.artifact_digest,
                        "is_baseline": candidate.id == self.candidate_set.baseline_id,
                    }
                    for candidate in self.candidate_set.candidates
                ],
            },
            "seed_ledger": ledger.to_json(),
            "selection": self.recipe.selection.to_json(),
            "limits": self.recipe.limits.to_json(),
            "runtime": {
                "container_runtime": self.home.config.container_runtime,
                "global_max_concurrent_trials": self.home.config.max_concurrent_trials,
                "run_parallelism": self._parallelism,
                "network": self.recipe.target.network,
            },
            "created_at": _now(),
        }
        if path.is_file():
            existing = json.loads(path.read_text(encoding="utf-8"))
            for field_name in ("recipe", "candidate_set"):
                if digest_of(existing.get(field_name)) != digest_of(manifest[field_name]):
                    raise EvalContractError(
                        f"run {self.manifest.run_id} was sealed with a different "
                        f"{field_name}; start a new run instead of mutating a sealed one"
                    )
            return existing
        write_json(path, manifest)
        write_json(self.run_dir / "seed_ledger.json", ledger.to_json())
        return manifest

    # ----------------------------------------------------------------- plan

    def _trial_keys(self, stage: str, candidate_ids: Sequence[str], seeds: Sequence[int]):
        for candidate_id in candidate_ids:
            for scenario in self.recipe.scenarios:
                for seed in seeds:
                    yield TrialKey(
                        candidate_id=candidate_id, seed=seed, scenario=scenario, stage=stage
                    )

    # ------------------------------------------------------------- one trial

    def _trial_dir(self, key: TrialKey) -> Path:
        return self.run_dir / "trials" / key.trial_id

    def _existing_record(self, key: TrialKey) -> TrialRecord | None:
        path = self._trial_dir(key) / "job_result.json"
        if not path.is_file():
            return None
        try:
            return TrialRecord.from_mapping(json.loads(path.read_text(encoding="utf-8")))
        except (EvalContractError, json.JSONDecodeError, OSError):
            return None

    def _write_trial_manifest(self, key: TrialKey, candidate: PolicyCandidate) -> Path:
        input_dir = self._trial_dir(key) / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "policy").mkdir(exist_ok=True)  # bind-mount target
        write_json(
            input_dir / "trial.json",
            {
                "schema_version": TRIAL_MANIFEST_SCHEMA,
                "run_id": self.manifest.run_id,
                "trial_id": key.trial_id,
                "stage": key.stage,
                "seed": key.seed,
                "scenario": key.scenario,
                "candidate": {
                    "id": candidate.id,
                    "label": candidate.label,
                    "kind": candidate.kind,
                    "digest": candidate.artifact_digest,
                    "entrypoint": candidate.entrypoint,
                },
                "metrics": [metric.to_json() for metric in self.recipe.target.metrics],
                "required_gates": list(self.recipe.target.required_gates),
                "limits": {
                    "timeout_seconds": self.recipe.limits.timeout_seconds,
                    "max_output_bytes": self.recipe.limits.max_output_bytes,
                },
                "models": [model.to_json() for model in self.recipe.models],
                "budget": self.recipe.budget.to_json() if self.recipe.budget else None,
                "output": {
                    "result_path": "/output/result.json",
                    "events_path": "/output/events.jsonl",
                },
            },
        )
        return input_dir

    def _secrets(self) -> dict[str, str]:
        """Resolved once per run, from names the recipe declared and nothing else."""

        if self._resolved_secrets is None:
            self._resolved_secrets = {
                name: self.home.resolve_secret(name, declared=self.recipe.secrets)
                for name in self.recipe.secrets
            }
        return self._resolved_secrets

    def _run_trial(self, key: TrialKey) -> TrialRecord:
        existing = self._existing_record(key)
        if existing is not None:
            self.events.emit(
                "eval.trial.terminal",
                trial_id=key.trial_id,
                resumed=True,
                trial=existing.to_json(),
            )
            return existing
        candidate = self.candidate_set.candidate(key.candidate_id)
        self.events.emit("eval.trial.queued", trial_id=key.trial_id, **key.to_json())
        # Hold here rather than inside the semaphore: a paused run must not sit
        # on a token that another run could be using.
        self.pause.wait_while_paused(lambda: self.cancel.cancelled)
        input_dir = self._write_trial_manifest(key, candidate)
        output_dir = self._trial_dir(key) / "output"
        if output_dir.exists():
            shutil.rmtree(output_dir)  # a partial attempt is not evidence
        output_dir.mkdir(parents=True, exist_ok=True)
        started_at = _now()
        try:
            lease = self.semaphore.acquire(
                run_id=self.manifest.run_id,
                trial_id=key.trial_id,
                should_abort=lambda: self.cancel.cancelled,
            )
        except SemaphoreTimeout as error:
            return self._record(
                key,
                status="cancelled",
                started_at=started_at,
                error=str(error),
                container=None,
                exit_code=None,
            )
        self.events.emit(
            "eval.trial.started",
            trial_id=key.trial_id,
            lease_id=lease.id,
            **key.to_json(),
        )
        try:
            execution = self._resolve_executor().run(
                TrialRunRequest(
                    trial_id=key.trial_id,
                    image_reference=self._image_reference,
                    input_dir=input_dir,
                    policy_dir=self.candidate_set.artifact_path(candidate),
                    output_dir=output_dir,
                    limits=self.recipe.limits,
                    network=self.recipe.target.network,
                    secrets=self._secrets(),
                ),
                on_event=lambda payload: self.events.emit(
                    "eval.trial.event", trial_id=key.trial_id, container_event=payload
                ),
                should_cancel=lambda: self.cancel.cancelled,
                heartbeat=lambda: self.semaphore.heartbeat(lease),
            )
        except ContainerRuntimeError as error:
            return self._record(
                key,
                status="failed",
                started_at=started_at,
                error=str(error),
                container=None,
                exit_code=None,
            )
        finally:
            self.semaphore.release(lease)

        if execution.cancelled:
            return self._record(
                key,
                status="cancelled",
                started_at=started_at,
                error="cancelled before the container produced a result",
                container=None,
                exit_code=execution.exit_code,
            )
        if execution.timed_out:
            return self._record(
                key,
                status="timeout",
                started_at=started_at,
                error=(
                    f"container exceeded the recipe's {self.recipe.limits.timeout_seconds}s ceiling"
                ),
                container=None,
                exit_code=execution.exit_code,
            )
        overflow = _directory_bytes(output_dir)
        if overflow > self.recipe.limits.max_output_bytes:
            return self._record(
                key,
                status="failed",
                started_at=started_at,
                error=(
                    f"container wrote {overflow} bytes, over the recipe's "
                    f"{self.recipe.limits.max_output_bytes} byte ceiling"
                ),
                container=None,
                exit_code=execution.exit_code,
            )
        result_path = output_dir / "result.json"
        if not result_path.is_file():
            return self._record(
                key,
                status="failed",
                started_at=started_at,
                error=(
                    "container produced no /output/result.json"
                    + (f"; stderr: {execution.stderr_tail}" if execution.stderr_tail else "")
                ),
                container=None,
                exit_code=execution.exit_code,
            )
        try:
            container = ContainerResult.from_mapping(
                json.loads(result_path.read_text(encoding="utf-8"))
            )
        except (EvalContractError, json.JSONDecodeError) as error:
            return self._record(
                key,
                status="failed",
                started_at=started_at,
                error=f"unusable container result: {error}",
                container=None,
                exit_code=execution.exit_code,
            )
        if container.trial_id != key.trial_id:
            return self._record(
                key,
                status="failed",
                started_at=started_at,
                error=(
                    f"container reported trial_id {container.trial_id!r} for "
                    f"{key.trial_id!r}; evidence cannot be attributed"
                ),
                container=None,
                exit_code=execution.exit_code,
            )
        record = self._record(
            key,
            status="evaluated" if container.status == "evaluated" else "failed",
            started_at=started_at,
            error=container.error,
            container=container,
            exit_code=execution.exit_code,
        )
        if record.missing_artifacts and record.status == "evaluated":
            self.events.emit(
                "eval.trial.evidence_incomplete",
                trial_id=key.trial_id,
                missing_artifacts=list(record.missing_artifacts),
            )
        return record

    def _adopt_artifacts(
        self, key: TrialKey, container: ContainerResult | None
    ) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
        """Adopt everything the container wrote into the run's evidence tree.

        Traces are the point: a run that reports a number but throws away the
        rollout it came from cannot be re-checked later, so every declared
        artifact is resolved, digested, and indexed, and a target that promised
        a role but did not write it produces a missing-artifact failure rather
        than a quietly thinner result.
        """

        output_dir = (self._trial_dir(key) / "output").resolve()
        adopted: list[dict[str, Any]] = []
        seen: set[Path] = set()

        def adopt(role: str, relative: str, declared: bool) -> bool:
            path = (output_dir / relative).resolve()
            if not path.is_relative_to(output_dir) or not path.is_file():
                return False
            if path in seen:
                return True
            seen.add(path)
            adopted.append(
                {
                    "role": role,
                    "path": str(path),
                    "relative_path": str(path.relative_to(output_dir)),
                    "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size,
                    "declared": declared,
                }
            )
            return True

        present_roles: set[str] = set()
        for entry in container.artifacts if container else ():
            role = str(entry.get("role") or "artifact")
            relative = str(entry.get("path") or "")
            if relative and adopt(role, relative, True):
                present_roles.add(role)
        # Sweep the rest of /output so nothing the container wrote is lost,
        # even when it forgot to declare it.
        if output_dir.is_dir():
            for path in sorted(output_dir.rglob("*")):
                if path.is_file():
                    adopt("retained", str(path.relative_to(output_dir)), False)
        missing = tuple(
            role for role in self.recipe.target.required_artifacts if role not in present_roles
        )
        return tuple(adopted), missing

    def _record(
        self,
        key: TrialKey,
        *,
        status: str,
        started_at: str,
        error: str | None,
        container: ContainerResult | None,
        exit_code: int | None,
    ) -> TrialRecord:
        gates = container.gate_map() if container else {}
        missing = tuple(gate for gate in self.recipe.target.required_gates if gate not in gates)
        artifacts, missing_artifacts = self._adopt_artifacts(key, container)
        record = TrialRecord(
            key=key,
            trial_id=key.trial_id,
            status=status,
            benchmark_status=container.benchmark_status if container else None,
            metrics=dict(container.metrics) if container else {},
            gates=gates,
            missing_gates=missing
            if status == "evaluated"
            else tuple(self.recipe.target.required_gates),
            missing_artifacts=missing_artifacts
            if status == "evaluated"
            else tuple(self.recipe.target.required_artifacts),
            usage=dict(container.usage) if container else {},
            artifacts=artifacts,
            started_at=started_at,
            finished_at=_now(),
            exit_code=exit_code,
            error=error,
            evidence_dir=str(self._trial_dir(key)),
        )
        write_json(self._trial_dir(key) / "job_result.json", record.to_json())
        self.events.emit(
            "eval.trial.terminal",
            trial_id=key.trial_id,
            resumed=False,
            trial=record.to_json(),
        )
        return record

    # ------------------------------------------------------------ execution

    def _run_stage(self, stage: str, candidate_ids: Sequence[str], seeds: Sequence[int]):
        keys = list(self._trial_keys(stage, candidate_ids, seeds))
        if not keys:
            return []
        records: list[TrialRecord] = []
        with ThreadPoolExecutor(max_workers=max(1, self._parallelism)) as pool:
            for record in pool.map(self._run_trial, keys):
                records.append(record)
        return records

    def _score(
        self,
        stage: str,
        records: Sequence[TrialRecord],
        *,
        candidate_ids: Sequence[str],
        eliminations: dict[str, str] | None = None,
    ) -> list[CandidateScorecard]:
        baseline_id = self.candidate_set.baseline_id
        baseline_records = [record for record in records if record.key.candidate_id == baseline_id]
        cards = []
        for candidate_id in candidate_ids:
            candidate = self.candidate_set.candidate(candidate_id)
            reason = (eliminations or {}).get(candidate_id)
            card = summarize_candidate(
                candidate,
                records,
                target=self.recipe.target,
                stage=stage,
                is_baseline=candidate_id == baseline_id,
                baseline_records=baseline_records,
                primary_metric=self.recipe.selection.primary_metric,
                eliminated_at=stage if reason else None,
                elimination_reason=reason,
            )
            cards.append(card)
            self.events.emit("eval.candidate.scored", scorecard=card.to_json())
            if reason:
                self.events.emit(
                    "eval.candidate.eliminated",
                    candidate_id=candidate_id,
                    label=candidate.label,
                    stage=stage,
                    reason=reason,
                )
        return cards

    def execute(self) -> int:
        self.cancel.start()
        try:
            return self._execute()
        except EvalContractError as error:
            self.events.emit(
                "eval.run.terminal", status="failed", selection_status=None, error=str(error)
            )
            return 1
        except Exception as error:  # noqa: BLE001 - a worker must never die silently
            self.events.emit(
                "eval.run.terminal",
                status="failed",
                selection_status=None,
                error=str(error),
                traceback=traceback.format_exc(limit=8),
            )
            return 1
        finally:
            self.semaphore.release_run(self.manifest.run_id)
            self.cancel.close()

    def _execute(self) -> int:
        self._validate_inputs()
        manifest = self._seal()
        # A previous attempt may have died holding tokens for this run id.
        reclaimed = self.semaphore.release_run(self.manifest.run_id)
        ledger = manifest["seed_ledger"]
        all_ids = [candidate.id for candidate in self.candidate_set.candidates]
        self.events.emit(
            "eval.run.planned",
            recipe_id=self.recipe.id,
            candidate_set_id=self.candidate_set.id,
            candidates=[
                {
                    "id": c.id,
                    "label": c.label,
                    "is_baseline": c.id == self.candidate_set.baseline_id,
                }
                for c in self.candidate_set.candidates
            ],
            manifest_digest=digest_of(manifest),
            parallelism=self._parallelism,
            global_capacity=self.home.config.max_concurrent_trials,
            reclaimed_leases=reclaimed,
            planned_trials=len(all_ids)
            * len(self.recipe.scenarios)
            * (len(ledger["screening"]) + len(ledger["confirmation"])),
        )
        self.events.emit("eval.seed_ledger.sealed", ledger=ledger)

        screening = self._run_stage("screen", all_ids, ledger["screening"])
        screen_cards = self._score("screen", screening, candidate_ids=all_ids)
        survivors, eliminations = apply_elimination(
            self.recipe.selection,
            screen_cards,
            target=self.recipe.target,
            baseline_id=self.candidate_set.baseline_id,
        )
        if eliminations:
            screen_cards = self._score(
                "screen", screening, candidate_ids=all_ids, eliminations=eliminations
            )

        confirmation: list[TrialRecord] = []
        confirm_cards: list[CandidateScorecard] = []
        if ledger["confirmation"] and not self.cancel.cancelled:
            confirmation = self._run_stage("confirm", survivors, ledger["confirmation"])
            confirm_cards = self._score("confirm", confirmation, candidate_ids=survivors)

        decision_records = confirmation or screening
        decision_cards = confirm_cards or screen_cards
        decision = decide(
            self.recipe.selection,
            scorecards=decision_cards,
            records=decision_records,
            baseline_id=self.candidate_set.baseline_id,
            cancelled=self.cancel.cancelled,
        )
        self._seal_outputs(screen_cards, confirm_cards, screening, confirmation, decision)
        self.events.emit("eval.selection.completed", selection=decision.to_json())
        status = "cancelled" if self.cancel.cancelled else "completed"
        self.events.emit(
            "eval.run.terminal",
            status=status,
            selection_status=decision.status,
            winner_id=decision.winner_id,
            evidence_dir=str(self.run_dir),
        )
        return 0

    def _seal_outputs(
        self,
        screen_cards: Sequence[CandidateScorecard],
        confirm_cards: Sequence[CandidateScorecard],
        screening: Sequence[TrialRecord],
        confirmation: Sequence[TrialRecord],
        decision: SelectionDecision,
    ) -> None:
        write_json(
            self.run_dir / "scorecards.json",
            {
                "schema_version": "eval.scorecards.v1",
                "run_id": self.manifest.run_id,
                "screen": [card.to_json() for card in screen_cards],
                "confirm": [card.to_json() for card in confirm_cards],
            },
        )
        write_json(self.run_dir / "selection.json", decision.to_json())
        write_json(
            self.run_dir / "result_manifest.json",
            {
                "schema_version": "eval.result-manifest.v1",
                "run_id": self.manifest.run_id,
                "recipe_id": self.recipe.id,
                "candidate_set_id": self.candidate_set.id,
                "selection": decision.to_json(),
                "trials": [
                    {
                        "trial_id": record.trial_id,
                        "stage": record.key.stage,
                        "candidate_id": record.key.candidate_id,
                        "scenario": record.key.scenario,
                        "seed": record.key.seed,
                        "status": record.status,
                        "benchmark_status": record.benchmark_status,
                        "valid": record.valid,
                        "evidence": str(Path(record.evidence_dir) / "job_result.json"),
                    }
                    for record in [*screening, *confirmation]
                ],
                "artifacts": [
                    {
                        "trial_id": record.trial_id,
                        "candidate_id": record.key.candidate_id,
                        "stage": record.key.stage,
                        "seed": record.key.seed,
                        **artifact,
                    }
                    for record in [*screening, *confirmation]
                    for artifact in record.artifacts
                ],
                "evidence_dir": str(self.run_dir),
                "sealed_at": _now(),
            },
        )


def _directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_worker(
    manifest_path: Path,
    *,
    executor: TrialExecutor | None = None,
    stream: Any = None,
) -> int:
    """Entry point behind `synth-optimizers eval worker`."""

    return EvalRunner(
        WorkerManifest.load(manifest_path), executor=executor, stream=stream
    ).execute()


def request_cancel(home: Path, run_id: str) -> bool:
    """Ask a running worker to stop and seal. Returns False for unknown runs."""

    run_dir = EvalHome.open(home, create=False).run_dir(run_id)
    if not run_dir.is_dir():
        return False
    (run_dir / CANCEL_SENTINEL).write_text(_now(), encoding="utf-8")
    return True


__all__ = [
    "CancellationToken",
    "EvalRunner",
    "EventEmitter",
    "WorkerManifest",
    "request_cancel",
    "run_worker",
]
