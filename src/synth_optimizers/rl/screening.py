"""Screen declared training rows for stochastic learning signal, without training.

This is deliberately a single-arm runner.  It follows the production RL
bind/submit/declare/poll/finalize/evidence lifecycle, but never calls the
provider's train or checkpoint-save surfaces.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from synth_optimizers.contracts.rl_identity import GroupPin, TaskSpec
from synth_optimizers.contracts.rl_records import digest
from synth_optimizers.rl.config import RunConfig
from synth_optimizers.rl.contract import ContainerStatusError
from synth_optimizers.rl.ports import AttemptFacts, PolicyRevision
from synth_optimizers.rl.session import EvidenceNotReady

FINALIZABLE = frozenset({"completed", "failed", "cancelled", "scored", "awaiting_score"})
SCHEMA_VERSION = "rl.screening.v1"


def _usage_totals(attempts: list[Mapping[str, Any]]) -> dict[str, int]:
    totals = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    for attempt in attempts:
        usage = attempt.get("usage")
        if not isinstance(usage, Mapping):
            continue
        for key in totals:
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += int(value)
    totals["total_tokens"] = totals["prompt_tokens"] + totals["completion_tokens"]
    return totals


def selected_task_ids(summary: list[Mapping[str, Any]]) -> list[str]:
    """Return rows with mixed binary outcomes, preserving input order."""

    return [str(row["task_id"]) for row in summary if 0 < int(row["successes"]) < int(row["samples"])]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _pin(config: RunConfig, plane: Any, revision: PolicyRevision, task: TaskSpec, group_id: str, samples: int) -> GroupPin:
    capability = plane.session.capability
    return GroupPin(
        group_id=group_id,
        run_id=config.run_id,
        algorithm_plan_hash=config.expanded_plan().plan_hash,
        behavior_fingerprint=revision.behavior_fingerprint,
        policy_revision=revision.revision,
        wire_api=config.model.wire_api,
        sampling_transport=config.model.sampling_transport,
        policy_kind=config.model.policy_kind,
        model_family=config.model.family,
        container_image_digest=capability.container_image_digest,
        container_contract_hash=plane.session.startup.contract.contract_hash,
        handshake_agreement_digest=plane.session.agreement_digest,
        task_family=task.task_family,
        cardinality=samples,
        policy_set_revision_id=revision.policy_set_revision_id,
        match_set_revision_id=config.opponents.match_set_revision,
        topology_id=capability.topology.topology_id,
        policy_revision_id=revision.revision_id,
    )


def run_screen(
    config: RunConfig,
    plane: Any,
    *,
    selector: str,
    output: Path,
    samples: int = 8,
    concurrency: int = 8,
    selection_mode: str = "binary",
    minimum_successes: int = 1,
    maximum_successes: int | None = None,
    poll_limit: int = 240,
    poll_interval: float = 0.25,
    wall_clock: Any = time.time,
    monotonic_clock: Any = time.monotonic,
    max_infrastructure_retries: int = 0,
    completed_attempts: tuple[Mapping[str, Any], ...] = (),
) -> Mapping[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("screening evidence exists; explicit recovery required")
    if samples < 2:
        raise ValueError("screening needs at least two samples per task")
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if type(max_infrastructure_retries) is not int or max_infrastructure_retries < 0:
        raise ValueError("max_infrastructure_retries must be nonnegative")
    if selection_mode not in {"binary", "reward_variance"}:
        raise ValueError("unknown selection mode")
    maximum_successes = samples - 1 if maximum_successes is None else maximum_successes
    if not 1 <= minimum_successes <= maximum_successes < samples:
        raise ValueError("success bounds must retain mixed outcomes within sample count")
    if selection_mode != "binary" and (minimum_successes != 1 or maximum_successes != samples - 1):
        raise ValueError("success bounds require binary selection")
    revisions = dict(plane.binder.resolve(selector))
    if len(revisions) != 1:
        raise ValueError(f"screening requires one policy revision, got {sorted(revisions)}")
    parameter_group, revision = next(iter(revisions.items()))
    tasks = plane.session.tasks(split=config.taskset.train_split, task_ids=config.taskset.train_ids)
    by_id = {task.task_id: task for task in tasks}
    missing = [task_id for task_id in config.taskset.train_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"container did not resolve train task(s): {missing}")

    attempts: list[dict[str, Any]] = [dict(row) for row in completed_attempts]
    recovered_keys = set()
    for row in attempts:
        key = (row['task_id'], row['sample_index'])
        if (key in recovered_keys or key[0] not in by_id or
            type(key[1]) is not int or not 0 <= key[1] < samples or
            row.get('checkpoint_id') != revision.checkpoint_id or
            row.get('sampler_reference') != revision.sampler_reference or
            row.get('seed') != by_id[key[0]].seed or
            row.get('terminal_status') not in {'completed', 'scored'} or
            not row.get('trace_digest') or
            (selection_mode == 'binary' and row.get('reward') not in (0.0, 1.0))):
            raise ValueError('invalid completed screening outcome for recovery')
        recovered_keys.add(key)
    started = wall_clock()
    monotonic_started = monotonic_clock()
    pending = [
        (number, task_id, sample, 0)
        for number, task_id in enumerate(config.taskset.train_ids)
        for sample in range(samples)
        if (task_id, sample) not in recovered_keys
    ]
    active: dict[str, tuple[int, TaskSpec, Any, int, int]] = {}
    finalized: set[str] = set()
    infrastructure_failures = []
    try:
        while pending or active:
            while pending and len(active) < concurrency:
                task_number, task_id, sample_index, replacement = pending.pop(0)
                base_task = by_id[task_id]
                group_id = f"{config.run_id}::screen::{task_number:04d}"
                pin = _pin(config, plane, revision, base_task, group_id, samples)
                # Eight stochastic samples of one declared task instance:
                # sample_index and idempotency differ, its dataset seed does not.
                task = replace(base_task, group_id=group_id)
                attempt_id = f"{group_id}::s{sample_index}" + (f"::r{replacement}" if replacement else "")
                proxy_request_id = f"{attempt_id}::{parameter_group}"
                origin = plane.gateway.bind(
                    revision,
                    pin=pin,
                    sample_index=sample_index,
                    proxy_request_id=proxy_request_id,
                    attempt=AttemptFacts(rollout_id=attempt_id, task_id=task.task_id, seed=task.seed),
                )
                try:
                    rollout_id = plane.session.submit(
                        task,
                        origin,
                        pin=pin,
                        sample_index=sample_index,
                        idempotency_key=attempt_id,
                    )
                except BaseException:
                    plane.gateway.close(proxy_request_id)
                    raise
                active[rollout_id] = (sample_index, task, origin, 0, replacement)

            moved = False
            for rollout_id, (sample_index, task, origin, seen, replacement) in list(active.items()):
                state = plane.session.poll(rollout_id)
                name = str(state.get("state") or "")
                if not state.get("terminal") and name not in FINALIZABLE:
                    seen += 1
                    if seen >= poll_limit:
                        raise RuntimeError(f"screening attempt {rollout_id} exceeded its poll limit")
                    active[rollout_id] = (sample_index, task, origin, seen, replacement)
                    continue
                try:
                    if name in {"failed", "cancelled"}:
                        if replacement >= max_infrastructure_retries:
                            raise RuntimeError(
                                f"screening attempt {rollout_id} ended in terminal state {name!r}"
                            )
                        plane.gateway.close(origin.proxy_request_id)
                        del active[rollout_id]
                        task_number = config.taskset.train_ids.index(task.task_id)
                        pending.insert(0, (task_number, task.task_id, sample_index, replacement + 1))
                        moved = True
                        continue
                    if rollout_id not in finalized:
                        plane.session.finalize(rollout_id)
                        # Settle the provisional ID only after sampling finishes:
                        # declaration takes the same route lock as the live call.
                        plane.gateway.declare_attempt(
                            origin.proxy_request_id, rollout_id=rollout_id,
                            task_id=task.task_id, seed=task.seed,
                        )
                        finalized.add(rollout_id)
                    episode, reward = plane.session.evidence(rollout_id)
                    reward.validate()
                    if reward.terminal_status not in {"completed", "scored"}:
                        raise RuntimeError(
                            f"screening attempt {rollout_id} has reward terminal status "
                            f"{reward.terminal_status!r}"
                        )
                    channel = reward.optimized_channel
                    value = reward.value(channel)
                    if selection_mode == 'binary' and value not in (0.0, 1.0):
                        raise ValueError('binary screening requires rewards exactly zero or one')
                    attempts.append(
                        {
                            "task_id": task.task_id,
                            "base_seed": task.seed,
                            "sample_index": sample_index,
                            "seed": task.seed,
                            "reward": value,
                            "reward_channel": channel,
                            "rollout_id": rollout_id,
                            "trace_digest": episode.trace_digest,
                            "usage": dict(episode.usage),
                            "terminal_status": reward.terminal_status,
                            "checkpoint_id": revision.checkpoint_id,
                            "policy_revision_id": revision.revision_id,
                            "sampler_reference": revision.sampler_reference,
                        }
                    )
                    _write_json(output / 'attempts.partial.json', attempts)
                except ContainerStatusError as exc:
                    if exc.status < 500 or replacement >= max_infrastructure_retries:
                        raise
                    infrastructure_failures.append({'rollout_id':rollout_id,'task_id':task.task_id,
                        'sample_index':sample_index,'replacement_index':replacement,'error':str(exc)[:500]})
                    _write_json(output/'infrastructure-failures.json',infrastructure_failures)
                    try:
                        plane.session.terminate(rollout_id,reason='screening_scoring_infrastructure_failure')
                    except Exception:
                        pass
                    plane.gateway.close(origin.proxy_request_id)
                    del active[rollout_id]
                    pending.insert(0,(config.taskset.train_ids.index(task.task_id),task.task_id,sample_index,replacement+1))
                    moved=True
                    continue
                except EvidenceNotReady:
                    # Finalization can precede a remote verifier's completion.
                    # Keep the same attempt and sampler binding alive; never
                    # translate pending scoring into a zero or a fresh sample.
                    seen += 1
                    if seen >= poll_limit:
                        raise RuntimeError(f"screening verifier {rollout_id} exceeded its poll limit")
                    active[rollout_id] = (sample_index, task, origin, seen, replacement)
                    continue
                except BaseException:
                    raise
                else:
                    plane.gateway.close(origin.proxy_request_id)
                    del active[rollout_id]
                moved = True
            if moved:
                _write_json(output / "attempts.partial.json", attempts)
                _write_json(output / "progress.json", {
                    "completed": len(attempts), "total": len(config.taskset.train_ids) * samples,
                    "active": len(active), "usage_totals": _usage_totals(attempts),
                })
            if active and not moved:
                time.sleep(poll_interval)
    except BaseException:
        for rollout_id, (_, _, origin, _, _) in list(active.items()):
            try:
                plane.session.terminate(rollout_id, reason="screening_aborted")
            except Exception:
                pass
            try:
                plane.gateway.close(origin.proxy_request_id)
            except Exception:
                pass
        active.clear()
        raise

    attempts.sort(key=lambda row: (config.taskset.train_ids.index(row["task_id"]), row["sample_index"]))
    summary = []
    for task_id in config.taskset.train_ids:
        rows = [row for row in attempts if row["task_id"] == task_id]
        successes = sum(1 for row in rows if float(row["reward"]) > 0.0)
        values = [float(row["reward"]) for row in rows]
        mixed = minimum_successes <= successes <= maximum_successes if selection_mode == "binary" else max(values) - min(values) > 1e-8
        summary.append({"task_id": task_id, "samples": len(rows), "successes": successes, "selected": mixed,
                        "reward_min": min(values), "reward_max": max(values), "reward_mean": sum(values)/len(values)})
    selected = [row["task_id"] for row in summary if row["selected"]]
    output.mkdir(parents=True, exist_ok=True)
    attempts_path = output / "attempts.json"
    summary_path = output / "summary.json"
    _write_json(attempts_path, attempts)
    _write_json(summary_path, {"tasks": summary, "selected_train_ids": selected})
    finished = wall_clock()
    duration_seconds = monotonic_clock() - monotonic_started
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_mode": selection_mode,
        "minimum_successes": minimum_successes,
        "maximum_successes": maximum_successes,
        "run_id": config.run_id,
        "selector": selector,
        "checkpoint_id": revision.checkpoint_id,
        "policy_revision_id": revision.revision_id,
        "sampler_reference": revision.sampler_reference,
        "parameter_group_id": parameter_group,
        "plan_hash": config.expanded_plan().plan_hash,
        "handshake_id": plane.session.handshake_id,
        "agreement_digest": plane.session.agreement_digest,
        "samples_per_task": samples,
        "maximum_concurrency": concurrency,
        "task_count": len(summary),
        "attempt_count": len(attempts),
        "recovered_attempt_count": len(completed_attempts),
        "new_attempt_count": len(attempts) - len(completed_attempts),
        "recovered_attempts_digest": digest(list(completed_attempts)),
        "duration_seconds": duration_seconds,
        "attempts_per_second": (
            (len(attempts) - len(completed_attempts)) / duration_seconds if duration_seconds > 0 else None
        ),
        "usage_totals": _usage_totals(attempts),
        "selected_count": len(selected),
        "selected_train_ids": selected,
        "started_at_unix": started,
        "finished_at_unix": finished,
        "attempts_file": attempts_path.name,
        "attempts_digest": digest(attempts),
        "summary_file": summary_path.name,
        "summary_digest": digest({"tasks": summary, "selected_train_ids": selected}),
    }
    _write_json(output / "manifest.json", manifest)
    return manifest
