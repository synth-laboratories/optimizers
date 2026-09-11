"""Shared host supervision around the existing container-first RL worker.

The worker still owns HTTP handshake, roster binding, token trajectories,
finalization and training. Process fencing is not remote resource deletion.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def supervise_run(args: Any, *, run_id: str, plan_hash: str) -> int:
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
        receipt: dict[str, Any] = {
            "schema_version": "synth.rl.execution-supervision.v1", "run_id": run_id,
            "plan_hash": plan_hash, "deadline": supervisor.deadline.isoformat(),
            "execution_error": None, "worker_returncode": None,
            "host_process_cleanup": "pending", "remote_resource_cleanup": "pending",
            "remote_cleanup_reason": "worker exit does not establish remote provider absence",
        }
        command = [
            sys.executable, "-c",
            "import sys; from synth_optimizers.rl.cli import main; raise SystemExit(main(sys.argv[1:]))",
            "run", "--config", str(Path(args.config).resolve()),
            "--receipts", str(root), "--max-ticks", str(args.max_ticks),
            "--supervised-worker-deadline", supervisor.deadline.isoformat(),
        ]
        if args.plane:
            command.extend(["--plane", args.plane])
        if args.json:
            command.append("--json")
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
            supervisor.decide_stop(type(error).__name__)
            receipt["execution_error"] = type(error).__name__
            raise
        finally:
            receipt["recorded_at"] = datetime.now(UTC).isoformat()
            try:
                with (custody / "receipt.json").open("x", encoding="utf-8") as output:
                    json.dump(receipt, output, indent=2, allow_nan=False)
                    output.write("\n")
            finally:
                supervisor.close()
    return asyncio.run(execute())
