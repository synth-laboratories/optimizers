"""Live SSE server for the service-managed GEPA run board.

The board page consumes the standing ``gepa service`` API as the authority for
run discovery and run state. This bridge polls ``/runs`` for the board table,
serves Server-Sent Events to the browser, and proxies service-owned event
artifacts for terminal drill-downs when a run WebSocket is no longer open.

Endpoints:

- ``GET /``            -- the board HTML, wired to the SSE endpoint below.
- ``GET /api/runs``    -- the current board snapshot as JSON (one-shot).
- ``GET /api/stream``  -- ``text/event-stream``; emits a ``board`` event on connect
  and again whenever any service run projection changes.
- ``GET /api/runs/{run_id}/events`` -- service-owned ``events.jsonl`` projection.

Stdlib only; intended for local/loopback use.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Sequence
from urllib.error import HTTPError
from urllib.parse import parse_qs, quote, unquote, urlencode
from urllib.request import Request, urlopen

from ._synth_optimizers import (
    gepa_compact_run_storage,
    gepa_delete_run_storage,
    gepa_inspect_run_storage,
    gepa_workspace_storage_health,
)
from .discovery import (
    gepa_home,
    latest_run_index,
    pid_alive,
    read_run_index,
    read_service_heartbeats,
    registry_roots_from_index,
)
from .o11y import (
    BoardSource,
    RunBoard,
    latest_registry_records,
    project_limit_eta,
    project_run_events,
)

STREAM_PATH = "/api/stream"
EVENTS_BASE = "/api/runs"


@dataclass(frozen=True)
class ServiceRunBoard:
    runs: list[dict]
    liveness: dict | None = None
    scheduler: dict | None = None
    workspace_runs: dict | None = None
    run_status: dict | None = None
    storage_health: dict | None = None

    @property
    def total(self) -> int:
        return len(self.runs)

    def count(self, state: str) -> int:
        return sum(1 for run in self.runs if run["state"] == state)

    @property
    def total_cost_usd(self) -> float | None:
        return _sum_present(run.get("cost_usd") for run in self.runs)

    @property
    def total_tokens(self) -> int | None:
        return _sum_present((run.get("usage") or {}).get("total_tokens") for run in self.runs)

    def to_data(self) -> dict:
        liveness = self.liveness or {}
        scheduler = self.scheduler or {}
        workspace_runs = self.workspace_runs or {}
        run_status = self.run_status or {}
        run_counts = run_status.get("counts") if isinstance(run_status.get("counts"), dict) else {}
        service_status = _int_dict(
            run_status.get("projected_status_counts") or workspace_runs.get("by_status")
        )
        raw_status = _int_dict(run_status.get("raw_status_counts"))
        active_workers = _worker_slots(run_status) or _workers(scheduler)
        leased_count = _int_or(
            run_counts.get("leased"),
            sum(1 for worker in active_workers if worker.get("request_status") == "leased"),
        )
        raw_running_count = _int_or(
            run_counts.get("running"),
            sum(1 for worker in active_workers if worker.get("request_status") == "running"),
        )
        terminal_count = sum(
            service_status.get(status, 0) for status in ("succeeded", "failed", "cancelled")
        )
        return {
            "schema": "synth.gepa_run_board.v1",
            "source": "service",
            "generated_at": _iso(_now()),
            "summary": {
                "total": self.total,
                "queued": self.count("queued"),
                "running": self.count("running"),
                "paused": self.count("paused"),
                "succeeded": self.count("succeeded"),
                "failed": self.count("failed"),
                "unknown": self.count("unknown"),
                "total_cost_usd": self.total_cost_usd,
                "total_tokens": self.total_tokens,
                "service_last_progress_at": _iso(_parse_ts(liveness.get("last_progress_at"))),
                "service_oldest_queued_age_seconds": _number(
                    liveness.get("oldest_queued_age_seconds")
                ),
                "service_running_count": int(liveness.get("running_count") or 0),
                "service_worker_count": int(scheduler.get("worker_count") or 0),
                "service_active_workers": int(scheduler.get("active_workers") or 0),
                "service_idle_workers": int(scheduler.get("idle_workers") or 0),
                "service_queued_runnable": int(scheduler.get("queued_runnable") or 0),
                "service_queued_blocked": int(scheduler.get("queued_blocked") or 0),
                "service_status_counts": service_status,
                "service_raw_status_counts": raw_status,
                "service_leased_count": leased_count,
                "service_raw_running_count": raw_running_count,
                "service_terminal_count": _int_or(run_counts.get("terminal"), terminal_count),
                "service_stale_lease_count": _int_or(
                    run_counts.get("stale_leases"),
                    sum(1 for worker in active_workers if _is_stale_worker(worker)),
                ),
            },
            "scheduler": scheduler,
            "run_status": run_status,
            "storage_health": self.storage_health or {},
            "runs": self.runs,
        }


class ServiceBoardSource:
    """Projects the run board from a running GEPA service."""

    def __init__(self, service_url: str, *, title: str) -> None:
        service_url = service_url.rstrip("/")
        if not service_url:
            raise ValueError("service_url is required")
        self._service_url = service_url
        self._title = title
        self._source_id = f"service:{service_url}"
        self._storage_cache: tuple[float, dict] | None = None

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def title(self) -> str:
        return self._title

    @property
    def service_url(self) -> str:
        return self._service_url

    def snapshot(self) -> dict:
        runs: list[dict] = []
        workspace = self._workspace_projection()
        cursor: str | None = None
        while True:
            query = {"limit": "200"}
            if cursor:
                query["cursor"] = cursor
            page = self._get_json("/runs", query=query)
            runs.extend(_project_service_run(run) for run in page.get("items") or [])
            cursor = page.get("next_cursor")
            if not cursor:
                break
        for run in runs:
            run["source_id"] = self.source_id
            run["service_url"] = self.service_url
            run["events_url"] = f"{EVENTS_BASE}/{quote(run['run_id'], safe='')}/events"
            run["timings_url"] = f"{EVENTS_BASE}/{quote(run['run_id'], safe='')}/timings"
            run["limits_url"] = f"{EVENTS_BASE}/{quote(run['run_id'], safe='')}/limits"
            run["storage_url"] = f"{EVENTS_BASE}/{quote(run['run_id'], safe='')}/storage"
            if run.get("state") in {"queued", "running", "paused"}:
                self._enrich_run_eta(run)
            if run.get("state") == "running":
                run["ws_url"] = _service_ws_url(self.service_url, str(run.get("run_id") or ""))
            if run.get("state") == "running" and run.get("run_id"):
                self._enrich_run_state(run)
        self._enrich_scheduler(
            runs,
            workspace.get("scheduler") or {},
            workspace.get("run_status") or {},
        )
        storage_health = self.workspace_storage()
        runs.sort(key=_board_sort_key, reverse=True)
        return ServiceRunBoard(
            runs,
            workspace.get("liveness") or {},
            workspace.get("scheduler") or {},
            workspace.get("runs") or {},
            workspace.get("run_status") or {},
            storage_health,
        ).to_data()

    def run_events(self, run_id: str, *, since: int = 0) -> list[dict]:
        path = f"/runs/{quote(run_id, safe='')}/artifacts/events.jsonl"
        try:
            text = self._get_text(path)
        except HTTPError as exc:
            if exc.code == 404:
                return []
            raise
        return _project_event_lines(text, since=since)

    def run_timings(self, run_id: str) -> dict:
        path = f"/runs/{quote(run_id, safe='')}/timings"
        try:
            timings = self._get_json(path)
        except HTTPError as exc:
            if exc.code == 404:
                return {"run_id": run_id, "summary": {}, "timings": []}
            raise
        if isinstance(timings, dict):
            return timings
        return {"run_id": run_id, "summary": {}, "timings": []}

    def run_limits(self, run_id: str) -> dict:
        path = f"/runs/{quote(run_id, safe='')}/limits"
        try:
            limits = self._get_json(path)
        except HTTPError as exc:
            if exc.code == 404:
                return {"run_id": run_id, "limits": [], "events": [], "nearest_limit": None}
            raise
        if isinstance(limits, dict):
            return limits
        return {"run_id": run_id, "limits": [], "events": [], "nearest_limit": None}

    def run_storage(self, run_id: str) -> dict:
        path = f"/runs/{quote(run_id, safe='')}/storage"
        storage = self._get_json(path)
        return storage if isinstance(storage, dict) else {"run_id": run_id}

    def workspace_storage(self) -> dict:
        now = time.monotonic()
        if self._storage_cache and now - self._storage_cache[0] < 60.0:
            return self._storage_cache[1]
        try:
            storage = self._get_json("/workspace/storage")
        except Exception as exc:
            storage = {"error": str(exc), "alerts": [], "summary": {"alert_count": 0}}
        self._storage_cache = (now, storage)
        return storage

    def compact_run_storage(self, run_id: str, *, profile: str, dry_run: bool) -> dict:
        path = f"/runs/{quote(run_id, safe='')}/compact"
        return self._send_json(path, method="POST", body={"profile": profile, "dry_run": dry_run})

    def delete_run_storage(self, run_id: str) -> dict:
        path = f"/runs/{quote(run_id, safe='')}"
        self._send_json(path, method="DELETE", body=None)
        return {"run_id": run_id, "deleted": True}

    def _get_json(self, path: str, *, query: dict[str, str] | None = None) -> dict:
        text = self._get_text(path, query=query)
        return json.loads(text)

    def _get_text(self, path: str, *, query: dict[str, str] | None = None) -> str:
        url = self._url(path, query=query)
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=10.0) as response:
            return response.read().decode("utf-8")

    def _send_json(self, path: str, *, method: str, body: dict | None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            self._url(path),
            data=data,
            method=method,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        with urlopen(request, timeout=30.0) as response:
            text = response.read().decode("utf-8")
        if not text:
            return {}
        value = json.loads(text)
        return value if isinstance(value, dict) else {}

    def _url(self, path: str, *, query: dict[str, str] | None = None) -> str:
        url = f"{self._service_url}{path}"
        if query:
            url = f"{url}?{urlencode(query)}"
        return url

    def _workspace_projection(self) -> dict:
        try:
            workspace = self._get_json("/workspace")
        except Exception:
            return {}
        liveness = workspace.get("liveness") if isinstance(workspace, dict) else {}
        scheduler = workspace.get("scheduler") if isinstance(workspace, dict) else {}
        run_status = workspace.get("run_status") if isinstance(workspace.get("run_status"), dict) else {}
        return {
            "liveness": liveness if isinstance(liveness, dict) else {},
            "scheduler": _normalize_scheduler_projection(scheduler if isinstance(scheduler, dict) else {}),
            "runs": workspace.get("runs") if isinstance(workspace.get("runs"), dict) else {},
            "run_status": _normalize_run_status_projection(run_status),
        }

    def _enrich_scheduler(self, runs: list[dict], scheduler: dict, run_status: dict) -> None:
        by_run_id = {run.get("run_id"): run for run in runs if run.get("run_id")}
        queued = _queued_reasons(run_status) or (
            scheduler.get("queued") if isinstance(scheduler.get("queued"), list) else []
        )
        for item in queued:
            if not isinstance(item, dict):
                continue
            run = by_run_id.get(item.get("run_id"))
            if not run:
                continue
            run["blocked_reason"] = item.get("reason")
            run["blocked_by_run_id"] = item.get("blocked_by_run_id")
            run["why_not_running"] = item.get("why_not_running")
            run["scheduler_state"] = "blocked" if item.get("reason") else "runnable"
        workers = _worker_slots(run_status) or (
            scheduler.get("workers") if isinstance(scheduler.get("workers"), list) else []
        )
        for worker in workers:
            if not isinstance(worker, dict):
                continue
            run = by_run_id.get(worker.get("run_id"))
            if not run:
                continue
            run["worker_id"] = worker.get("worker_id")
            run["scheduler_state"] = worker.get("state")
            run["request_status"] = worker.get("request_status")
            run["worker_last_progress_at"] = _iso(
                _parse_ts(worker.get("last_worker_progress_at") or worker.get("last_progress_at"))
            )
            lease_expires_at = worker.get("lease_expires_at") or worker.get("last_heartbeat_at")
            run["lease_expires_at"] = _iso(_parse_ts(lease_expires_at))
            run["seconds_until_lease_expiry"] = worker.get("seconds_until_lease_expiry")
            run["last_run_event_at"] = _iso(_parse_ts(worker.get("last_run_event_at")))
            run["heartbeat_state"] = worker.get("heartbeat_state")
            run["stale_reason"] = worker.get("stale_reason")

    def _enrich_run_state(self, run: dict) -> None:
        try:
            state = self._get_json(f"/runs/{quote(run['run_id'], safe='')}/state")
        except Exception:
            return
        service_run = state.get("run") if isinstance(state.get("run"), dict) else {}
        cursor = state.get("cursor") if isinstance(state.get("cursor"), dict) else {}
        active = cursor.get("active_evaluation") if isinstance(cursor.get("active_evaluation"), dict) else {}
        queues = state.get("queues") if isinstance(state.get("queues"), dict) else {}

        if service_run.get("phase"):
            run["phase"] = service_run["phase"]
        if service_run.get("generation") is not None:
            run["generation"] = service_run["generation"]
        if service_run.get("candidate_count") is not None:
            run["candidate_count"] = service_run["candidate_count"]
        if service_run.get("best_candidate_id"):
            run["best_candidate_id"] = service_run["best_candidate_id"]

        active_rows = active.get("row_ids") if isinstance(active.get("row_ids"), list) else []
        active_scores = active.get("scores") if isinstance(active.get("scores"), list) else []
        candidate_evals = (
            active.get("candidate_evaluations")
            if isinstance(active.get("candidate_evaluations"), list)
            else []
        )
        run["active_evaluation"] = {
            "stage": active.get("stage"),
            "generation": active.get("generation"),
            "candidate_id": active.get("candidate_id"),
            "candidate_index": active.get("candidate_index"),
            "candidate_evaluation_count": len(candidate_evals),
            "row_count": len(active_rows),
            "scored_count": len(active_scores),
            "next_row_index": active.get("next_row_index"),
        }
        run["queue_counts"] = {
            name: int(queue.get("count") or 0)
            for name, queue in queues.items()
            if isinstance(queue, dict)
        }
        run["checkpoint_sequence"] = cursor.get("checkpoint_sequence")

    def _enrich_run_eta(self, run: dict) -> None:
        run_id = run.get("run_id")
        if not run_id:
            return
        try:
            run["eta"] = project_limit_eta(self.run_limits(str(run_id)))
        except Exception:
            run["eta"] = None

    @staticmethod
    def fingerprint(data: dict) -> str:
        runs = [
            (
                run["run_id"],
                run["state"],
                run.get("best_train_reward"),
                run.get("best_heldout_reward"),
                run.get("phase"),
                run.get("stage"),
                run.get("generation"),
                (run.get("usage") or {}).get("total_tokens"),
                run.get("last_activity_at"),
                run.get("ended_at"),
                run.get("worker_id"),
                run.get("scheduler_state"),
                run.get("blocked_reason"),
                run.get("blocked_by_run_id"),
                run.get("request_status"),
                run.get("lease_expires_at"),
                run.get("worker_last_progress_at"),
                run.get("last_run_event_at"),
                run.get("heartbeat_state"),
                run.get("stale_reason"),
                run.get("why_not_running"),
                run.get("eta"),
            )
            for run in data["runs"]
        ]
        liveness = (
            data.get("summary", {}).get("service_last_progress_at"),
            data.get("summary", {}).get("service_oldest_queued_age_seconds"),
            data.get("summary", {}).get("service_running_count"),
            data.get("summary", {}).get("service_worker_count"),
            data.get("summary", {}).get("service_active_workers"),
            data.get("summary", {}).get("service_queued_blocked"),
            data.get("summary", {}).get("service_queued_runnable"),
            data.get("summary", {}).get("service_leased_count"),
            data.get("summary", {}).get("service_raw_running_count"),
            data.get("summary", {}).get("service_terminal_count"),
            data.get("summary", {}).get("service_stale_lease_count"),
        )
        scheduler = data.get("scheduler") or {}
        run_status = data.get("run_status") or {}
        scheduler_fingerprint = (
            scheduler.get("worker_count"),
            scheduler.get("active_workers"),
            scheduler.get("idle_workers"),
            scheduler.get("queued_runnable"),
            scheduler.get("queued_blocked"),
            (run_status.get("counts") or {}),
            [
                (
                    worker.get("slot"),
                    worker.get("worker_id"),
                    worker.get("state"),
                    worker.get("run_id"),
                    worker.get("request_status"),
                    worker.get("last_worker_progress_at") or worker.get("last_progress_at"),
                    worker.get("last_run_event_at"),
                    worker.get("lease_expires_at") or worker.get("last_heartbeat_at"),
                    worker.get("heartbeat_state"),
                    worker.get("stale"),
                    worker.get("stale_reason"),
                )
                for worker in (_worker_slots(run_status) or _workers(scheduler))
            ],
            [
                (
                    item.get("run_id"),
                    item.get("reason"),
                    item.get("blocked_by_run_id"),
                    item.get("submitted_at"),
                    item.get("why_not_running"),
                )
                for item in (_queued_reasons(run_status) or _queued_items(scheduler))
            ],
        )
        return json.dumps(
            {"runs": runs, "liveness": liveness, "scheduler": scheduler_fingerprint},
            sort_keys=True,
        )


class RegistryDirSource:
    """Projects the board from GEPA_HOME plus explicit registry roots."""

    def __init__(
        self,
        roots: Sequence[str | Path] | None = None,
        *,
        title: str,
        home: str | Path | None = None,
        live_within_seconds: float | None = None,
    ) -> None:
        self._explicit_roots = [Path(root).expanduser() for root in roots or []]
        self._title = title
        self._home = gepa_home(home)
        self._live_within_seconds = live_within_seconds
        self._records = {}
        self._index = {}
        self._storage_roots: list[Path] = []
        self._storage_cache: tuple[float, dict] | None = None

    @property
    def source_id(self) -> str:
        return f"registry:{self._home}"

    @property
    def title(self) -> str:
        return self._title

    def snapshot(self) -> dict:
        entries = read_run_index(self._home)
        self._index = latest_run_index(entries)
        roots = [*self._explicit_roots, *registry_roots_from_index(entries)]
        self._storage_roots = roots
        records = latest_registry_records(roots)
        records = {run_id: record for run_id, record in records.items() if record.run_dir.exists()}
        self._records = records
        board = RunBoard(
            [
                _resolve_registry_record(record, self._live_within_seconds)
                for record in records.values()
            ]
        )
        data = board.to_data()
        data["source"] = "registry"
        for run in data["runs"]:
            run["source_id"] = self.source_id
            run["events_url"] = f"{EVENTS_BASE}/{quote(run['run_id'], safe='')}/events"
            run["timings_url"] = f"{EVENTS_BASE}/{quote(run['run_id'], safe='')}/timings"
            run["limits_url"] = f"{EVENTS_BASE}/{quote(run['run_id'], safe='')}/limits"
            run["storage_url"] = f"{EVENTS_BASE}/{quote(run['run_id'], safe='')}/storage"
            index = self._index.get(run["run_id"])
            if index and index.owning_service_url and run.get("state") == "running":
                run["service_url"] = index.owning_service_url
                run["ws_url"] = _service_ws_url(index.owning_service_url, run["run_id"])
            if index and index.pid is not None and run.get("state") == "running":
                run["pid"] = index.pid
                if not pid_alive(index.pid):
                    run["state"] = "unknown"
        data["runs"].sort(key=_board_sort_key, reverse=True)
        data["summary"] = _summary_from_runs(data["runs"])
        data["storage_health"] = self.workspace_storage()
        return data

    def run_events(self, run_id: str, *, since: int = 0) -> list[dict]:
        record = self._records.get(run_id)
        if record is None:
            self.snapshot()
            record = self._records.get(run_id)
        return project_run_events(record.event_feed_path if record else None, since=since)

    def run_timings(self, run_id: str) -> dict:
        return {"run_id": run_id, "summary": {}, "timings": []}

    def run_limits(self, run_id: str) -> dict:
        return {"run_id": run_id, "limits": [], "events": [], "nearest_limit": None}

    def run_storage(self, run_id: str) -> dict:
        record = self._record(run_id)
        terminal = _resolve_registry_record(record, self._live_within_seconds).state.value in {
            "succeeded",
            "failed",
        }
        return gepa_inspect_run_storage(str(record.run_dir), run_id=run_id, terminal=terminal)

    def workspace_storage(self) -> dict:
        now = time.monotonic()
        if self._storage_cache and now - self._storage_cache[0] < 60.0:
            return self._storage_cache[1]
        roots = self._storage_roots or self._explicit_roots
        storage = gepa_workspace_storage_health([str(root) for root in roots])
        self._storage_cache = (now, storage)
        return storage

    def compact_run_storage(self, run_id: str, *, profile: str, dry_run: bool) -> dict:
        report = self.run_storage(run_id)
        if not report.get("terminal"):
            raise ValueError(f"run {run_id} is not terminal")
        return gepa_compact_run_storage(
            str(self._record(run_id).run_dir),
            run_id=run_id,
            profile=profile,
            dry_run=dry_run,
        )

    def delete_run_storage(self, run_id: str) -> dict:
        report = self.run_storage(run_id)
        if not report.get("terminal"):
            raise ValueError(f"run {run_id} is not terminal")
        return gepa_delete_run_storage(str(self._record(run_id).run_dir), dry_run=False)

    def _record(self, run_id: str):
        record = self._records.get(run_id)
        if record is None:
            self.snapshot()
            record = self._records.get(run_id)
        if record is None:
            raise KeyError(f"run {run_id} not found")
        return record


class AggregateSource:
    """Discovers all local services each poll and unions them with file runs."""

    def __init__(
        self,
        roots: Sequence[str | Path] | None = None,
        *,
        title: str,
        service_url: str | None = None,
        home: str | Path | None = None,
        live_within_seconds: float | None = None,
    ) -> None:
        self._roots = list(roots or [])
        self._title = title
        self._service_url = service_url.rstrip("/") if service_url else None
        self._home = gepa_home(home)
        self._live_within_seconds = live_within_seconds
        self._sources_by_run: dict[str, BoardSource] = {}
        self._sources: list[BoardSource] = []

    @property
    def source_id(self) -> str:
        return "aggregate"

    @property
    def title(self) -> str:
        return self._title

    @property
    def service_url(self) -> str | None:
        return self._service_url

    def snapshot(self) -> dict:
        sources = self._discover_sources()
        rows_by_id: dict[str, dict] = {}
        source_by_run: dict[str, BoardSource] = {}
        snapshots: list[dict] = []
        errors: list[dict] = []
        storage_reports: list[dict] = []
        for source in sources:
            try:
                data = source.snapshot()
            except Exception as exc:
                errors.append({"source_id": source.source_id, "error": str(exc)})
                continue
            snapshots.append(data)
            if isinstance(data.get("storage_health"), dict) and data.get("storage_health"):
                storage_reports.append(data["storage_health"])
            for run in data.get("runs") or []:
                run_id = run.get("run_id")
                if not run_id:
                    continue
                existing = rows_by_id.get(run_id)
                if existing is None or _run_preferred(run, existing):
                    rows_by_id[run_id] = run
                    source_by_run[run_id] = source
        runs = list(rows_by_id.values())
        runs.sort(key=_board_sort_key, reverse=True)
        self._sources = sources
        self._sources_by_run = source_by_run
        return {
            "schema": "synth.gepa_run_board.v1",
            "source": "aggregate",
            "generated_at": _iso(_now()),
            "summary": _aggregate_summary(runs, snapshots),
            "scheduler": _first_non_empty(snapshots, "scheduler"),
            "run_status": _first_non_empty(snapshots, "run_status"),
            "storage_health": _merge_storage_health(storage_reports),
            "sources": [
                {
                    "source_id": source.source_id,
                    "title": source.title,
                    "kind": source.source_id.split(":", 1)[0],
                }
                for source in sources
            ],
            "source_errors": errors,
            "runs": runs,
        }

    def run_events(self, run_id: str, *, since: int = 0) -> list[dict]:
        source = self._sources_by_run.get(run_id)
        if source is None:
            self.snapshot()
            source = self._sources_by_run.get(run_id)
        return source.run_events(run_id, since=since) if source else []

    def run_timings(self, run_id: str) -> dict:
        source = self._sources_by_run.get(run_id)
        if source is None:
            self.snapshot()
            source = self._sources_by_run.get(run_id)
        return source.run_timings(run_id) if source else {"run_id": run_id, "summary": {}, "timings": []}

    def run_limits(self, run_id: str) -> dict:
        source = self._sources_by_run.get(run_id)
        if source is None:
            self.snapshot()
            source = self._sources_by_run.get(run_id)
        return source.run_limits(run_id) if source else {"run_id": run_id, "limits": [], "events": [], "nearest_limit": None}

    def workspace_storage(self) -> dict:
        sources = self._sources or self._discover_sources()
        reports = []
        for source in sources:
            if hasattr(source, "workspace_storage"):
                reports.append(source.workspace_storage())
        return _merge_storage_health(reports)

    def run_storage(self, run_id: str) -> dict:
        source = self._sources_by_run.get(run_id)
        if source is None:
            self.snapshot()
            source = self._sources_by_run.get(run_id)
        if not source:
            raise KeyError(f"run {run_id} not found")
        return source.run_storage(run_id)

    def compact_run_storage(self, run_id: str, *, profile: str, dry_run: bool) -> dict:
        source = self._sources_by_run.get(run_id)
        if source is None:
            self.snapshot()
            source = self._sources_by_run.get(run_id)
        if not source:
            raise KeyError(f"run {run_id} not found")
        return source.compact_run_storage(run_id, profile=profile, dry_run=dry_run)

    def delete_run_storage(self, run_id: str) -> dict:
        source = self._sources_by_run.get(run_id)
        if source is None:
            self.snapshot()
            source = self._sources_by_run.get(run_id)
        if not source:
            raise KeyError(f"run {run_id} not found")
        return source.delete_run_storage(run_id)

    def _discover_sources(self) -> list[BoardSource]:
        sources: list[BoardSource] = []
        seen_services: set[str] = set()
        if self._service_url:
            sources.append(ServiceBoardSource(self._service_url, title=self._title))
            seen_services.add(self._service_url)
        else:
            for heartbeat in read_service_heartbeats(self._home):
                if heartbeat.service_url in seen_services:
                    continue
                seen_services.add(heartbeat.service_url)
                sources.append(ServiceBoardSource(heartbeat.service_url, title=self._title))
        sources.append(
            RegistryDirSource(
                self._roots,
                title=self._title,
                home=self._home,
                live_within_seconds=self._live_within_seconds,
            )
        )
        return sources


def _resolve_registry_record(record, live_within_seconds: float | None):
    from .o11y import LIVE_WITHIN_SECONDS, RunStatus

    return RunStatus.resolve(
        record,
        live_within_seconds=(
            LIVE_WITHIN_SECONDS if live_within_seconds is None else live_within_seconds
        ),
    )


def _summary_from_runs(runs: list[dict]) -> dict:
    return {
        "total": len(runs),
        "queued": sum(1 for run in runs if run.get("state") == "queued"),
        "running": sum(1 for run in runs if run.get("state") == "running"),
        "paused": sum(1 for run in runs if run.get("state") == "paused"),
        "succeeded": sum(1 for run in runs if run.get("state") == "succeeded"),
        "failed": sum(1 for run in runs if run.get("state") == "failed"),
        "unknown": sum(1 for run in runs if run.get("state") == "unknown"),
        "total_cost_usd": _sum_present(run.get("cost_usd") for run in runs),
        "total_tokens": _sum_present((run.get("usage") or {}).get("total_tokens") for run in runs),
    }


def _aggregate_summary(runs: list[dict], snapshots: list[dict]) -> dict:
    summary = _summary_from_runs(runs)
    service_fields = (
        "service_running_count",
        "service_worker_count",
        "service_active_workers",
        "service_idle_workers",
        "service_queued_runnable",
        "service_queued_blocked",
        "service_leased_count",
        "service_raw_running_count",
        "service_terminal_count",
        "service_stale_lease_count",
    )
    for field in service_fields:
        summary[field] = sum(int((snapshot.get("summary") or {}).get(field) or 0) for snapshot in snapshots)
    progress_values = [
        (snapshot.get("summary") or {}).get("service_last_progress_at")
        for snapshot in snapshots
        if (snapshot.get("summary") or {}).get("service_last_progress_at")
    ]
    if progress_values:
        summary["service_last_progress_at"] = max(progress_values)
    oldest_values = [
        (snapshot.get("summary") or {}).get("service_oldest_queued_age_seconds")
        for snapshot in snapshots
        if (snapshot.get("summary") or {}).get("service_oldest_queued_age_seconds") is not None
    ]
    if oldest_values:
        summary["service_oldest_queued_age_seconds"] = max(float(value) for value in oldest_values)
    return summary


def _first_non_empty(snapshots: list[dict], key: str) -> dict:
    for snapshot in snapshots:
        value = snapshot.get(key)
        if isinstance(value, dict) and value:
            return value
    return {}


def _merge_storage_health(reports: list[dict]) -> dict:
    roots_by_path: dict[str, dict] = {}
    alerts: list[dict] = []
    thresholds = {}
    generated_at = None
    for report in reports:
        if not isinstance(report, dict) or report.get("error"):
            continue
        if isinstance(report.get("thresholds"), dict) and not thresholds:
            thresholds = report["thresholds"]
        generated_at = report.get("generated_at_unix_seconds") or generated_at
        for root in report.get("roots") or []:
            if not isinstance(root, dict):
                continue
            key = str(root.get("root") or "")
            if not key:
                continue
            existing = roots_by_path.get(key)
            if existing is None or int(root.get("bytes") or 0) >= int(existing.get("bytes") or 0):
                roots_by_path[key] = root
    roots = list(roots_by_path.values())
    roots.sort(key=lambda root: int(root.get("bytes") or 0), reverse=True)
    for root in roots:
        alerts.extend(alert for alert in root.get("alerts") or [] if isinstance(alert, dict))
    summary = {
        "root_count": len(roots),
        "run_count": sum(int(root.get("run_count") or 0) for root in roots),
        "terminal_run_count": sum(int(root.get("terminal_run_count") or 0) for root in roots),
        "partial_count": sum(int(root.get("partial_count") or 0) for root in roots),
        "bytes": sum(int(root.get("bytes") or 0) for root in roots),
        "reclaimable_bytes": sum(int(root.get("reclaimable_bytes") or 0) for root in roots),
        "partial_bytes": sum(int(root.get("partial_bytes") or 0) for root in roots),
        "stale_partial_bytes": sum(int(root.get("stale_partial_bytes") or 0) for root in roots),
        "alert_count": len(alerts),
    }
    return {
        "schema": "synth.optimizer.storage_health.v1",
        "generated_at_unix_seconds": generated_at,
        "thresholds": thresholds,
        "summary": summary,
        "alerts": alerts,
        "roots": roots,
    }


def _run_preferred(candidate: dict, existing: dict) -> bool:
    if candidate.get("ws_url") and not existing.get("ws_url"):
        return True
    if candidate.get("service_url") and not existing.get("service_url"):
        return True
    candidate_state = str(candidate.get("state") or "")
    existing_state = str(existing.get("state") or "")
    if candidate_state == "running" and existing_state != "running":
        return True
    return str(candidate.get("last_activity_at") or "") > str(existing.get("last_activity_at") or "")


def _service_ws_url(service_url: str, run_id: str) -> str:
    return f"{service_url.replace('http://', 'ws://').replace('https://', 'wss://').rstrip('/')}/runs/{quote(run_id, safe='')}/ws"


def _fingerprint(data: dict) -> str:
    runs = [
        (
            run.get("run_id"),
            run.get("state"),
            run.get("source_id"),
            run.get("best_train_reward"),
            run.get("best_heldout_reward"),
            run.get("phase"),
            run.get("stage"),
            run.get("generation"),
            (run.get("usage") or {}).get("total_tokens"),
            run.get("last_activity_at"),
            run.get("ended_at"),
            run.get("worker_id"),
            run.get("scheduler_state"),
            run.get("blocked_reason"),
            run.get("request_status"),
            run.get("heartbeat_state"),
            run.get("stale_reason"),
            run.get("eta"),
            run.get("ws_url"),
            run.get("events_url"),
        )
        for run in data.get("runs", [])
    ]
    summary = data.get("summary") or {}
    sources = data.get("sources") or []
    errors = data.get("source_errors") or []
    return json.dumps(
        {"runs": runs, "summary": summary, "sources": sources, "errors": errors},
        sort_keys=True,
    )


class _Hub:
    """Fans the latest board snapshot out to subscribed SSE clients."""

    def __init__(self, source: BoardSource, *, interval: float) -> None:
        self._source = source
        self._interval = interval
        self._lock = threading.Lock()
        self._subscribers: set[threading.Event] = set()
        try:
            self._latest: dict = source.snapshot()
        except Exception:
            self._latest = _empty_snapshot()
        self._fingerprint = _fingerprint(self._latest)
        self._stop = threading.Event()

    @property
    def latest(self) -> dict:
        with self._lock:
            return self._latest

    def subscribe(self) -> threading.Event:
        event = threading.Event()
        event.set()
        with self._lock:
            self._subscribers.add(event)
        return event

    def unsubscribe(self, event: threading.Event) -> None:
        with self._lock:
            self._subscribers.discard(event)

    def run_forever(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                data = self._source.snapshot()
            except Exception:
                continue
            fingerprint = _fingerprint(data)
            with self._lock:
                if fingerprint == self._fingerprint:
                    continue
                self._latest = data
                self._fingerprint = fingerprint
                wake = list(self._subscribers)
            for event in wake:
                event.set()

    def stop(self) -> None:
        self._stop.set()


# Shared board HTTP helpers. Both the standalone board handler and the console
# handler (docs_server) serve the same board surface; these keep that single
# implementation in one place.


def write_bytes(handler: BaseHTTPRequestHandler, body: bytes, content_type: str) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def write_html(handler: BaseHTTPRequestHandler, text: str) -> None:
    write_bytes(handler, text.encode(), "text/html; charset=utf-8")


def write_json(handler: BaseHTTPRequestHandler, data: dict) -> None:
    write_bytes(handler, json.dumps(data).encode(), "application/json")


def write_json_status(handler: BaseHTTPRequestHandler, status: int, data: dict) -> None:
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict:
    length = int(handler.headers.get("Content-Length") or "0")
    if length <= 0:
        return {}
    body = handler.rfile.read(length)
    if not body:
        return {}
    value = json.loads(body.decode("utf-8"))
    return value if isinstance(value, dict) else {}


def render_board_page(hub: _Hub, source: BoardSource) -> str:
    from .o11y import render_board_html

    return render_board_html(
        hub.latest,
        title=source.title,
        live_endpoint=STREAM_PATH,
        service_url=getattr(source, "service_url", None),
        events_base=EVENTS_BASE,
    )


def stream_board(handler: BaseHTTPRequestHandler, hub: _Hub) -> None:
    """Serve the SSE ``board`` event stream until the client disconnects."""
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.end_headers()
    event = hub.subscribe()
    try:
        while True:
            if event.wait(timeout=15.0):
                event.clear()
                payload = json.dumps(hub.latest)
                handler.wfile.write(f"event: board\ndata: {payload}\n\n".encode())
            else:
                handler.wfile.write(b": ping\n\n")
            handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        hub.unsubscribe(event)


def handle_board_request(
    handler: BaseHTTPRequestHandler,
    raw_path: str,
    query: dict,
    hub: _Hub,
    source: BoardSource,
) -> bool:
    """Serve the shared board API routes; return True if the path was handled."""
    method = handler.command
    if method == "GET" and raw_path == "/api/runs":
        write_json(handler, hub.latest)
    elif method == "GET" and raw_path == "/api/storage":
        write_json(handler, source.workspace_storage())
    elif method == "GET" and raw_path == STREAM_PATH:
        stream_board(handler, hub)
    elif method == "GET" and raw_path.startswith(EVENTS_BASE + "/") and raw_path.endswith("/events"):
        run_id = unquote(raw_path[len(EVENTS_BASE) + 1 : -len("/events")])
        since = int(query.get("since", ["0"])[0])
        write_json(handler, {"run_id": run_id, "events": source.run_events(run_id, since=since)})
    elif method == "GET" and raw_path.startswith(EVENTS_BASE + "/") and raw_path.endswith("/timings"):
        run_id = unquote(raw_path[len(EVENTS_BASE) + 1 : -len("/timings")])
        write_json(handler, source.run_timings(run_id))
    elif method == "GET" and raw_path.startswith(EVENTS_BASE + "/") and raw_path.endswith("/limits"):
        run_id = unquote(raw_path[len(EVENTS_BASE) + 1 : -len("/limits")])
        write_json(handler, source.run_limits(run_id))
    elif method == "GET" and raw_path.startswith(EVENTS_BASE + "/") and raw_path.endswith("/storage"):
        run_id = unquote(raw_path[len(EVENTS_BASE) + 1 : -len("/storage")])
        write_json(handler, source.run_storage(run_id))
    elif method == "POST" and raw_path.startswith(EVENTS_BASE + "/") and raw_path.endswith("/compact"):
        run_id = unquote(raw_path[len(EVENTS_BASE) + 1 : -len("/compact")])
        body = read_json_body(handler)
        profile = str(body.get("profile") or "compact")
        dry_run = bool(body.get("dry_run", True))
        write_json(
            handler,
            source.compact_run_storage(run_id, profile=profile, dry_run=dry_run),
        )
    elif method == "DELETE" and raw_path.startswith(EVENTS_BASE + "/"):
        suffix = raw_path[len(EVENTS_BASE) + 1 :]
        if "/" in suffix:
            return False
        run_id = unquote(suffix)
        write_json(handler, source.delete_run_storage(run_id))
    else:
        return False
    return True


def _handler_factory(hub: _Hub, source: BoardSource):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args) -> None:
            pass

        def _dispatch(self) -> None:
            raw_path = self.path.split("?", 1)[0]
            query = parse_qs(self.path.split("?", 1)[1]) if "?" in self.path else {}
            try:
                if self.command == "GET" and raw_path == "/":
                    write_html(self, render_board_page(hub, source))
                elif handle_board_request(self, raw_path, query, hub, source):
                    pass
                else:
                    self.send_error(404, "not found")
            except HTTPError as exc:
                text = exc.read().decode("utf-8", errors="replace")
                try:
                    payload = json.loads(text) if text else {}
                except json.JSONDecodeError:
                    payload = {"error": text or exc.reason}
                if not isinstance(payload, dict):
                    payload = {"error": str(payload)}
                write_json_status(self, exc.code, payload)
            except (KeyError, ValueError) as exc:
                write_json_status(self, 409, {"error": str(exc)})
            except Exception as exc:
                write_json_status(self, 500, {"error": str(exc)})

        def do_GET(self) -> None:
            self._dispatch()

        def do_POST(self) -> None:
            self._dispatch()

        def do_DELETE(self) -> None:
            self._dispatch()

    return Handler


def service_board_snapshot(service_url: str, *, title: str = "GEPA Run Board") -> dict:
    return ServiceBoardSource(service_url, title=title).snapshot()


def board_snapshot(
    roots: Sequence[str] | None = None,
    *,
    title: str = "GEPA Run Board",
    service_url: str | None = None,
    live_within_seconds: float | None = None,
) -> dict:
    return AggregateSource(
        roots,
        title=title,
        service_url=service_url,
        live_within_seconds=live_within_seconds,
    ).snapshot()


def serve_board(
    _roots: Sequence[str] | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    title: str = "GEPA Run Board",
    interval: float = 2.0,
    service_url: str | None = None,
    live_within_seconds: float | None = None,
) -> None:
    """Run the local live board server until interrupted."""

    source = AggregateSource(
        _roots,
        title=title,
        service_url=service_url,
        live_within_seconds=live_within_seconds,
    )
    hub = _Hub(source, interval=interval)
    poller = threading.Thread(target=hub.run_forever, name="gepa-board-poller", daemon=True)
    poller.start()

    httpd = ThreadingHTTPServer((host, port), _handler_factory(hub, source))
    url = f"http://{host}:{port}/"
    source_note = f"service {service_url}" if service_url else "GEPA_HOME discovery"
    print(f"GEPA run board: {url}  ({source_note}; Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...")
    finally:
        hub.stop()
        httpd.shutdown()


def _project_service_run(run: dict) -> dict:
    config = run.get("config") or {}
    usage = _usage_dict(run.get("usage") or {})
    status = str(run.get("status") or "unknown")
    state = _board_state(status)
    outcome = run.get("outcome") or {}
    best = outcome.get("best") if isinstance(outcome, dict) else None
    best = best if isinstance(best, dict) else {}
    submitted_at = _parse_ts(run.get("submitted_at"))
    started_at = _parse_ts(run.get("started_at")) or submitted_at
    finished_at = _parse_ts(run.get("finished_at"))
    last_activity = finished_at or started_at or submitted_at
    cost_usd = _number(usage.get("cost_usd"))
    return {
        "run_id": str(run.get("run_id") or ""),
        "domain": _infer_domain(config),
        "task": _project_task_summary(config),
        "budgets": _project_budget_summary(config),
        "state": state,
        "run_dir": "",
        "started_at": _iso(started_at),
        "ended_at": _iso(finished_at),
        "last_activity_at": _iso(last_activity),
        "duration_seconds": _duration_seconds(started_at, finished_at, state),
        "cost_usd": cost_usd,
        "best_candidate_id": run.get("best_candidate_id") or best.get("candidate_id"),
        "best_train_reward": _number(run.get("best_train_reward")),
        "best_heldout_reward": _number(
            run.get("best_heldout_reward") or best.get("heldout_score")
        ),
        "acceptance_score": None,
        "phase": run.get("phase") or status,
        "stage": status,
        "generation": run.get("generation") or (run.get("totals") or {}).get("generations"),
        "candidate_count": run.get("candidate_count"),
        "worker_id": run.get("worker_id"),
        "request_status": None,
        "lease_expires_at": _iso(_parse_ts(run.get("lease_expires_at"))),
        "seconds_until_lease_expiry": None,
        "worker_last_progress_at": None,
        "last_run_event_at": None,
        "heartbeat_state": None,
        "stale_reason": None,
        "why_not_running": None,
        "scheduler_state": None,
        "blocked_reason": None,
        "blocked_by_run_id": None,
            "active_evaluation": None,
            "queue_counts": {},
            "checkpoint_sequence": None,
            "eta": None,
            "usage": usage,
            "timing_summary": run.get("timing_summary") if isinstance(run.get("timing_summary"), dict) else {},
            "failure": _failure(outcome),
            "score_chart_path": None,
        }


def _project_task_summary(config: dict) -> dict:
    taskset = config.get("taskset") if isinstance(config.get("taskset"), dict) else {}
    advanced = config.get("advanced") if isinstance(config.get("advanced"), dict) else {}
    pipeline = advanced.get("pipeline") if isinstance(advanced.get("pipeline"), dict) else {}
    train_ids = taskset.get("train_ids") or []
    heldout_ids = taskset.get("heldout_ids") or []
    return {
        "train_rows": len(train_ids) if isinstance(train_ids, list) else None,
        "heldout_rows": len(heldout_ids) if isinstance(heldout_ids, list) else None,
        "minibatch_rows": pipeline.get("minibatch_size"),
        "proposals_per_generation": pipeline.get("proposals_per_generation"),
    }


def _project_budget_summary(config: dict) -> dict:
    advanced = config.get("advanced") if isinstance(config.get("advanced"), dict) else {}
    budgets = advanced.get("budgets") if isinstance(advanced.get("budgets"), dict) else {}
    stop_conditions = config.get("stop_conditions")
    max_total = None
    max_train = budgets.get("max_train_rollouts")
    max_heldout = budgets.get("max_heldout_rollouts")
    if isinstance(stop_conditions, list):
        for condition in stop_conditions:
            if not isinstance(condition, dict) or condition.get("kind") != "max_rollouts":
                continue
            max_total = condition.get("n")
            max_train = condition.get("train", max_train)
            max_heldout = condition.get("heldout", max_heldout)
            break
    if max_total is None and max_train is not None and max_heldout is not None:
        try:
            max_total = int(max_train) + int(max_heldout)
        except (TypeError, ValueError):
            max_total = None
    return {
        "max_total_rollouts": _int_or_none(max_total),
        "max_train_rollouts": _int_or_none(max_train),
        "max_heldout_rollouts": _int_or_none(max_heldout),
    }


def _usage_dict(usage: dict) -> dict:
    prompt = _int_or_none(usage.get("prompt_tokens", usage.get("input_tokens")))
    completion = _int_or_none(usage.get("completion_tokens", usage.get("output_tokens")))
    total = _int_or_none(usage.get("total_tokens"))
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "proposer_calls": _int_or_none(usage.get("proposer_calls")),
        "rollout_calls": _int_or_none(usage.get("rollout_calls")),
        "cost_usd": _number(usage.get("cost_usd")),
    }


def _project_event_lines(text: str, *, since: int = 0) -> list[dict]:
    events: list[dict] = []
    seq = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        seq += 1
        if seq <= since:
            continue
        fields = raw.get("fields")
        payload = dict(fields) if isinstance(fields, dict) else {"value": fields}
        payload["source_event_type"] = raw.get("type", "event")
        if raw.get("message"):
            payload["message"] = raw["message"]
        events.append(
            {
                "seq": seq,
                "ts": raw.get("ts", ""),
                "kind": raw.get("type", "event"),
                "payload": payload,
            }
        )
    return events


def _board_state(status: str) -> str:
    if status == "queued":
        return "queued"
    if status == "running":
        return "running"
    if status == "paused":
        return "paused"
    if status == "succeeded":
        return "succeeded"
    if status in {"failed", "cancelled"}:
        return "failed"
    return "unknown"


def _failure(outcome: dict) -> dict | None:
    if not isinstance(outcome, dict) or outcome.get("result") not in {"failed", "cancelled"}:
        return None
    error = outcome.get("error") if isinstance(outcome.get("error"), dict) else {}
    return {
        "failure_type": outcome.get("result") or "failed",
        "reason_code": error.get("kind") or outcome.get("reason") or "unknown",
        "message": error.get("message") or outcome.get("reason") or "",
        "retryable": False,
    }


def _infer_domain(config: dict) -> str:
    taskset = config.get("taskset") if isinstance(config.get("taskset"), dict) else {}
    container = config.get("container_url") or (config.get("container") or {}).get("url")
    ids = taskset.get("train_ids") or []
    if isinstance(ids, list) and ids:
        first = str(ids[0])
        return first.split(":", 1)[0].split("/", 1)[0] or "unknown"
    if container:
        text = str(container)
        if "banking" in text.lower():
            return "banking77"
    return "unknown"


def _board_sort_key(run: dict) -> tuple[str, str]:
    ts = run.get("started_at") or run.get("last_activity_at") or ""
    return (str(ts), str(run.get("run_id") or ""))


def _duration_seconds(start: datetime | None, end: datetime | None, state: str) -> float | None:
    if start is None:
        return None
    stop = _now() if state in {"queued", "running", "paused"} else end
    if stop is None:
        return None
    return max(0.0, (stop - start).total_seconds())


def _normalize_scheduler_projection(scheduler: dict) -> dict:
    normalized = dict(scheduler)
    workers = scheduler.get("workers")
    if isinstance(workers, list):
        normalized["workers"] = [
            _normalize_scheduler_worker(worker) if isinstance(worker, dict) else worker
            for worker in workers
        ]
    queued = scheduler.get("queued")
    if isinstance(queued, list):
        normalized["queued"] = [
            _normalize_scheduler_item(item) if isinstance(item, dict) else item for item in queued
        ]
    return normalized


def _normalize_run_status_projection(run_status: dict) -> dict:
    normalized = dict(run_status)
    workers = run_status.get("worker_slots")
    if isinstance(workers, list):
        normalized["worker_slots"] = [
            _normalize_scheduler_worker(worker) if isinstance(worker, dict) else worker
            for worker in workers
        ]
    active = run_status.get("active_leases")
    if isinstance(active, list):
        normalized["active_leases"] = [
            _normalize_scheduler_worker(worker) if isinstance(worker, dict) else worker
            for worker in active
        ]
    queued = run_status.get("queued_reasons")
    if isinstance(queued, list):
        normalized["queued_reasons"] = [
            _normalize_scheduler_item(item) if isinstance(item, dict) else item for item in queued
        ]
    return normalized


def _normalize_scheduler_worker(worker: dict) -> dict:
    normalized = dict(worker)
    for key in (
        "last_heartbeat_at",
        "last_progress_at",
        "last_worker_progress_at",
        "last_run_event_at",
        "lease_expires_at",
        "leased_at",
    ):
        if key in normalized:
            normalized[key] = _iso(_parse_ts(normalized.get(key)))
    return normalized


def _normalize_scheduler_item(item: dict) -> dict:
    normalized = dict(item)
    for key in ("submitted_at", "last_progress_at", "last_worker_progress_at", "last_run_event_at"):
        if key in normalized:
            normalized[key] = _iso(_parse_ts(normalized.get(key)))
    return normalized


def _parse_ts(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value)
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat().replace("+00:00", "Z")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sum_present(values) -> float | int | None:
    present = []
    for value in values:
        if value is None or value == "":
            continue
        try:
            present.append(value if isinstance(value, int) and not isinstance(value, bool) else float(value))
        except (TypeError, ValueError):
            continue
    if not present:
        return None
    if all(isinstance(value, int) and not isinstance(value, bool) for value in present):
        return int(sum(present))
    return float(sum(present))


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_dict(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, int] = {}
    for key, raw in value.items():
        try:
            result[str(key)] = int(raw or 0)
        except (TypeError, ValueError):
            continue
    return result


def _int_or(value: object, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _empty_snapshot() -> dict:
    return {
        "schema": "synth.gepa_run_board.v1",
        "source": "service",
        "generated_at": _iso(_now()),
        "summary": {
            "total": 0,
            "queued": 0,
            "running": 0,
            "paused": 0,
            "succeeded": 0,
            "failed": 0,
            "unknown": 0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
        },
        "scheduler": {},
        "run_status": {},
        "runs": [],
    }


def _workers(scheduler: dict) -> list[dict]:
    workers = scheduler.get("workers") if isinstance(scheduler, dict) else None
    return [worker for worker in workers if isinstance(worker, dict)] if isinstance(workers, list) else []


def _worker_slots(run_status: dict) -> list[dict]:
    workers = run_status.get("worker_slots") if isinstance(run_status, dict) else None
    return [worker for worker in workers if isinstance(worker, dict)] if isinstance(workers, list) else []


def _queued_items(scheduler: dict) -> list[dict]:
    queued = scheduler.get("queued") if isinstance(scheduler, dict) else None
    return [item for item in queued if isinstance(item, dict)] if isinstance(queued, list) else []


def _queued_reasons(run_status: dict) -> list[dict]:
    queued = run_status.get("queued_reasons") if isinstance(run_status, dict) else None
    return [item for item in queued if isinstance(item, dict)] if isinstance(queued, list) else []


def _is_stale_worker(worker: dict) -> bool:
    if worker.get("stale") is not None:
        return bool(worker.get("stale"))
    if worker.get("state") != "active":
        return False
    expires_at = _parse_ts(worker.get("lease_expires_at") or worker.get("last_heartbeat_at"))
    return expires_at is not None and expires_at < _now()
