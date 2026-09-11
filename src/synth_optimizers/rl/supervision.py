"""Shared host supervision around the existing container-first RL worker.

The worker still owns HTTP handshake, roster binding, token trajectories,
finalization and training. Process fencing is not remote resource deletion.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _publish_receipt(path: Path, receipt: dict[str, Any]) -> None:
    """Atomically retain the first outcome; replay diagnostics stay in the journal."""
    descriptor, name = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(receipt, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            # A replay must neither replace the original outcome nor mask the
            # supervisor's original refusal with an incidental publication error.
            return
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def supervise_run(args: Any, *, run_id: str, plan_hash: str, config_source: bytes) -> int:
    from synth_containers.bounded_process import run_bounded_process
    from synth_containers.lifecycle_limits import DurableRolloutSupervisor, LifecycleLimits

    root = Path(args.receipts).resolve()
    custody = root / "execution-supervision"
    limits = LifecycleLimits(
        overall_seconds=args.supervision_timeout_seconds,
        work_seconds=args.supervision_timeout_seconds,
        cleanup_seconds=30,
    )

    async def execute() -> int:
        supervisor = DurableRolloutSupervisor(custody, run_id, limits)
        try:
            custody.chmod(0o700)
            snapshots = custody / "config"
            snapshots.mkdir(mode=0o700, exist_ok=True)
            snapshot = snapshots / Path(args.config).name
            if snapshot.exists():
                if snapshot.read_bytes() != config_source:
                    raise ValueError("Resumed RL configuration cannot change")
            else:
                descriptor = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "wb") as output:
                    output.write(config_source)
                    output.flush()
                    os.fsync(output.fileno())
        except BaseException:
            supervisor.close()
            raise
        receipt: dict[str, Any] = {
            "schema_version": "synth.rl.execution-supervision.v1", "run_id": run_id,
            "plan_hash": plan_hash, "deadline": supervisor.deadline.isoformat(),
            "config_sha256": hashlib.sha256(config_source).hexdigest(),
            "execution_error": None, "worker_returncode": None,
            "host_process_cleanup": "pending",
            "remote_resource_scope": "configured_http_target" if not args.plane else "custom_plane",
            "remote_resource_cleanup": "not_owned" if not args.plane else "pending",
            "remote_cleanup_reason": (
                "default plane connects to a pre-existing HTTP target; parent has no target deletion authority"
                if not args.plane else "custom plane resource ownership is not declared to the supervisor"
            ),
            "worker_created_session_cleanup": "pending",
            "worker_session_cleanup_reason": (
                "worker-created rollout/training sessions retain their own remote lifecycle; "
                "parent has no durable attempt-to-rollout ownership or independent absence contract"
            ),
        }
        command = [
            sys.executable, "-c",
            "import sys; from synth_optimizers.rl.cli import main; raise SystemExit(main(sys.argv[1:]))",
            "run", "--config", str(snapshot),
            "--receipts", str(root), "--max-ticks", str(args.max_ticks),
            "--supervised-worker-deadline", supervisor.deadline.isoformat(),
        ]
        if args.plane:
            command.extend(["--plane", args.plane])
        if args.json:
            command.append("--json")
        failure: BaseException | None = None
        try:
            async def work() -> int:
                return await run_bounded_process(
                    command, output=custody / "worker.log",
                    max_output_bytes=args.supervision_max_output_bytes,
                    env=dict(os.environ),
                    redact=tuple(value for key, value in os.environ.items()
                                 if any(word in key.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))),
                )
            returncode = await supervisor.run_phase("work", work)
            receipt["worker_returncode"] = returncode
            # The bounded actuator fences the process group before returning.
            receipt["host_process_cleanup"] = "confirmed"
            # Preserve the established CLI report while retaining the bounded,
            # redacted log as independent evidence if the viewer disconnects.
            sys.stdout.write((custody / "worker.log").read_text(encoding="utf-8", errors="replace"))
            return returncode
        except BaseException as error:
            failure = error
            supervisor.decide_stop(type(error).__name__)
            receipt["execution_error"] = type(error).__name__
            raise
        finally:
            receipt["recorded_at"] = datetime.now(UTC).isoformat()
            try:
                _publish_receipt(custody / "receipt.json", receipt)
            except Exception as publication_error:
                if failure is None:
                    raise
                failure.add_note("Supervision receipt publication failed: " + type(publication_error).__name__)
            finally:
                supervisor.close()
    return asyncio.run(execute())
