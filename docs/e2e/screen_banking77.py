"""Screen Banking77 train rows for stochastic learning signal, without training.

This is deliberately a single-arm runner.  It follows the production RL
bind/submit/declare/poll/finalize/evidence lifecycle, but never calls the
provider's train or checkpoint-save surfaces.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from synth_optimizers.contracts.rl_identity import GroupPin, TaskSpec
from synth_optimizers.contracts.rl_records import digest
from synth_optimizers.rl.config import RunConfig, load as load_run_config
from synth_optimizers.rl.ports import AttemptFacts, PolicyRevision
from synth_optimizers.rl.resolver import MappingArtifactProbe

FINALIZABLE = frozenset({"completed", "failed", "cancelled", "scored", "awaiting_score"})
SCHEMA_VERSION = "banking77.screening.v1"


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
    poll_limit: int = 240,
    poll_interval: float = 0.25,
    wall_clock: Any = time.time,
    monotonic_clock: Any = time.monotonic,
) -> Mapping[str, Any]:
    if samples < 2:
        raise ValueError("screening needs at least two samples per task")
    if concurrency < 1 or concurrency > samples:
        raise ValueError("concurrency must be between one and samples")
    revisions = dict(plane.binder.resolve(selector))
    if len(revisions) != 1:
        raise ValueError(f"Banking77 screening requires one policy revision, got {sorted(revisions)}")
    parameter_group, revision = next(iter(revisions.items()))
    tasks = plane.session.tasks(split=config.taskset.train_split, task_ids=config.taskset.train_ids)
    by_id = {task.task_id: task for task in tasks}
    missing = [task_id for task_id in config.taskset.train_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"container did not resolve train task(s): {missing}")

    attempts: list[dict[str, Any]] = []
    started = wall_clock()
    monotonic_started = monotonic_clock()
    for task_number, task_id in enumerate(config.taskset.train_ids):
        base_task = by_id[task_id]
        group_id = f"{config.run_id}::screen::{task_number:04d}"
        pin = _pin(config, plane, revision, base_task, group_id, samples)
        pending = list(range(samples))
        active: dict[str, tuple[int, TaskSpec, Any, int]] = {}
        try:
            while pending or active:
                while pending and len(active) < concurrency:
                    sample_index = pending.pop(0)
                    # Eight stochastic samples of one declared task instance:
                    # sample_index and idempotency differ, its dataset seed does not.
                    task = replace(base_task, group_id=group_id)
                    attempt_id = f"{group_id}::s{sample_index}"
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
                        plane.gateway.declare_attempt(
                            proxy_request_id, rollout_id=rollout_id, task_id=task.task_id, seed=task.seed
                        )
                    except BaseException:
                        plane.gateway.close(proxy_request_id)
                        raise
                    active[rollout_id] = (sample_index, task, origin, 0)

                moved = False
                for rollout_id, (sample_index, task, origin, seen) in list(active.items()):
                    state = plane.session.poll(rollout_id)
                    name = str(state.get("state") or "")
                    if not state.get("terminal") and name not in FINALIZABLE:
                        seen += 1
                        if seen >= poll_limit:
                            raise RuntimeError(f"screening attempt {rollout_id} exceeded its poll limit")
                        active[rollout_id] = (sample_index, task, origin, seen)
                        continue
                    try:
                        if name in {"failed", "cancelled"}:
                            raise RuntimeError(
                                f"screening attempt {rollout_id} ended in terminal state {name!r}"
                            )
                        plane.session.finalize(rollout_id)
                        episode, reward = plane.session.evidence(rollout_id)
                        reward.validate()
                        if reward.terminal_status not in {"completed", "scored"}:
                            raise RuntimeError(
                                f"screening attempt {rollout_id} has reward terminal status "
                                f"{reward.terminal_status!r}"
                            )
                        channel = reward.optimized_channel
                        value = reward.value(channel)
                        attempts.append(
                            {
                                "task_id": task.task_id,
                                "base_seed": base_task.seed,
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
                    except BaseException:
                        raise
                    else:
                        plane.gateway.close(origin.proxy_request_id)
                        del active[rollout_id]
                    moved = True
                if active and not moved:
                    time.sleep(poll_interval)
        except BaseException:
            for rollout_id, (_, _, origin, _) in list(active.items()):
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
        summary.append({"task_id": task_id, "samples": len(rows), "successes": successes, "selected": 0 < successes < len(rows)})
    selected = selected_task_ids(summary)
    output.mkdir(parents=True, exist_ok=True)
    attempts_path = output / "attempts.json"
    summary_path = output / "summary.json"
    _write_json(attempts_path, attempts)
    _write_json(summary_path, {"tasks": summary, "selected_train_ids": selected})
    finished = wall_clock()
    duration_seconds = monotonic_clock() - monotonic_started
    manifest = {
        "schema_version": SCHEMA_VERSION,
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
        "duration_seconds": duration_seconds,
        "attempts_per_second": (
            len(attempts) / duration_seconds if duration_seconds > 0 else None
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


def _factory(spec: str) -> Any:
    module_name, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("--plane expects MODULE:FACTORY")
    factory = getattr(importlib.import_module(module_name), name)
    if not callable(factory):
        raise ValueError(f"{spec} is not callable")
    return factory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--plane", default="paid_plane:paid")
    parser.add_argument(
        "--artifact-digests",
        help="JSON mapping of immutable provider references to observed digests.",
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    config = load_run_config(Path(args.config))
    factory = _factory(args.plane)
    parameters = inspect.signature(factory).parameters.values()
    options: dict[str, Any] = {}
    if args.artifact_digests:
        payload = json.loads(Path(args.artifact_digests).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("--artifact-digests must contain a JSON object")
        options["artifact_probe"] = MappingArtifactProbe(
            digests={str(reference): str(value) for reference, value in payload.items()}
        )
    plane = factory(config=config, **options) if parameters else factory()
    try:
        manifest = run_screen(
            config, plane, selector=args.selector, output=Path(args.output), samples=args.samples, concurrency=args.concurrency
        )
    finally:
        close = getattr(plane, "close", None)
        if callable(close):
            close()
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
