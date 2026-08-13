"""Local observability board for GEPA runs.

Discovery protocol
------------------
Every GEPA run appends to a ``run_registry.jsonl`` at the root of its output
directory (``<output_dir>/run_registry.jsonl``) the moment it starts, and again
when it reaches a terminal state. Each record carries the run id, run directory,
the per-run ``events.jsonl`` / ``events.normalized.jsonl`` paths, the
``result_manifest.json`` path, and rolling usage/cost. The registry is therefore
the authoritative index of which runs exist locally and which are still live.

Three projection sources, with a fixed precedence (never a fallback chain):

- ``run_registry.jsonl``      -- authoritative for *lifecycle* (started / finished
  / failed) and for the paths of every other artifact.
- ``events.jsonl``            -- authoritative for *live* progress of a RUNNING
  run: current phase, generation, best reward so far, rollout progress, and the
  heartbeat timestamp used to detect staleness. The normalized feed is written
  only when a run terminalizes.
- ``result_manifest.json``    -- authoritative for the *final* metrics of a
  terminal run: best candidate rewards, failure detail, end-to-end timing.

Nouns (kept deliberately distinct):

- ``RunStatus``  -- the projected status of one GEPA run.
- ``LiveProgress`` -- the live projection of a running run from its event feed.
- ``RunBoard``   -- the index over many ``RunStatus`` rows, rendered as HTML.

This is a read-only projection. It never mutates run directories and never
executes run-supplied content. The static HTML it emits embeds its own data and
needs no server; the live board (``board_server``) layers an SSE stream on top.
"""

from __future__ import annotations

import html
import json
import logging
import socket
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

MANIFEST_NAME = "result_manifest.json"
REGISTRY_NAME = "run_registry.jsonl"

# A run is only credited as genuinely RUNNING if its event feed advanced within
# this window. A `started` run with no terminal record and no recent heartbeat is
# not "running" — we cannot tell if its process is alive — so it is UNKNOWN.
#
# The window must exceed GEPA's longest expected silent span: a single full-model
# proposer call routinely runs 300s+ with no events, and tau2-style rollouts add
# more. 20 minutes keeps a healthy run "running" through a proposal while still
# flagging genuinely abandoned runs (the stale ones we see are hours/days idle).
LIVE_WITHIN_SECONDS = 1200.0
LOGGER = logging.getLogger(__name__)


class BoardSource(Protocol):
    @property
    def source_id(self) -> str: ...

    @property
    def title(self) -> str: ...

    def snapshot(self) -> dict: ...

    def run_events(self, run_id: str, *, since: int = 0) -> list[dict]: ...

    def run_timings(self, run_id: str) -> dict: ...

    def run_limits(self, run_id: str) -> dict: ...

    def run_storage(self, run_id: str) -> dict: ...

    def compact_run_storage(self, run_id: str, *, profile: str, dry_run: bool) -> dict: ...

    def delete_run_storage(self, run_id: str) -> dict: ...


class RunState(StrEnum):
    """Authoritative lifecycle state of a GEPA run.

    UNKNOWN is distinct from RUNNING: a run that started, never recorded a
    terminal state, and has gone silent is of indeterminate state (likely a
    crashed or killed process), not a healthy in-progress run.
    """

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN = "unknown"

    @classmethod
    def from_registry_status(cls, status: str) -> RunState:
        if status == "finished":
            return cls.SUCCEEDED
        if status == "failed":
            return cls.FAILED
        return cls.RUNNING


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# --------------------------------------------------------------------------
# Registry: the discovery index.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RegistryRecord:
    run_id: str
    status: str
    ts: datetime | None
    run_dir: Path
    manifest_path: Path | None
    event_feed_path: Path | None
    normalized_event_feed_path: Path | None
    best_candidate_id: str | None
    cost_usd: float | None
    usage: dict

    @classmethod
    def from_json(cls, data: dict) -> RegistryRecord:
        return cls(
            run_id=str(data["run_id"]),
            status=str(data.get("status", "started")),
            ts=_parse_ts(data.get("ts")),
            run_dir=Path(data["run_dir"]),
            manifest_path=_opt_path(data.get("manifest_path")),
            event_feed_path=_opt_path(data.get("event_feed_path")),
            normalized_event_feed_path=_opt_path(data.get("normalized_event_feed_path")),
            best_candidate_id=data.get("best_candidate_id"),
            cost_usd=data.get("cost_usd"),
            usage=data.get("usage") or {},
        )


def _opt_path(value: object) -> Path | None:
    return Path(str(value)) if value else None


def latest_registry_records(roots: Iterable[Path]) -> dict[str, RegistryRecord]:
    """Collapse every ``run_registry.jsonl`` under the roots to the latest
    record per run id. The latest terminal/started record wins by timestamp."""

    latest: dict[str, RegistryRecord] = {}
    for registry in _discover_files(roots, REGISTRY_NAME):
        for line in registry.read_text().splitlines():
            for raw in _json_values_from_line(line):
                record = RegistryRecord.from_json(raw)
                current = latest.get(record.run_id)
                if current is None or _ts_key(record.ts) >= _ts_key(current.ts):
                    latest[record.run_id] = record
    return latest


def _json_values_from_line(line: str) -> list[dict]:
    text = line.strip()
    if not text:
        return []
    decoder = json.JSONDecoder()
    values: list[dict] = []
    pos = 0
    while pos < len(text):
        value, pos = decoder.raw_decode(text, pos)
        if isinstance(value, dict):
            values.append(value)
        while pos < len(text) and text[pos].isspace():
            pos += 1
    return values


def _ts_key(ts: datetime | None) -> float:
    return ts.timestamp() if ts else 0.0


def _discover_files(roots: Iterable[Path], name: str) -> list[Path]:
    seen: set[Path] = set()
    found: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        candidates = [root / name] if root.is_dir() else []
        candidates.extend(sorted(root.rglob(name)))
        for path in candidates:
            resolved = path.resolve()
            if path.is_file() and resolved not in seen:
                seen.add(resolved)
                found.append(path)
    return found


# --------------------------------------------------------------------------
# Usage / failure value objects.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    proposer_calls: int = 0
    rollout_calls: int = 0

    @classmethod
    def from_dict(cls, usage: dict) -> RunUsage:
        return cls(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            total_tokens=int(usage.get("total_tokens", 0)),
            proposer_calls=int(usage.get("proposer_calls", 0)),
            rollout_calls=int(usage.get("rollout_calls", 0)),
        )


@dataclass(frozen=True)
class RunFailure:
    failure_type: str
    reason_code: str
    message: str
    retryable: bool

    @classmethod
    def from_manifest(cls, failure: dict) -> RunFailure:
        inner = failure.get("failure", {})
        return cls(
            failure_type=str(inner.get("failure_type", "unknown")),
            reason_code=str(inner.get("reason_code", "unknown")),
            message=str(inner.get("message") or failure.get("message", "")),
            retryable=bool(inner.get("retryable", False)),
        )


# --------------------------------------------------------------------------
# Live projection from a run's raw event feed.
# --------------------------------------------------------------------------


def project_run_events(path: Path | None, *, since: int = 0) -> list[dict]:
    """Project a run's raw event feed into the same ``{seq, ts, kind,
    payload}`` shape the service streams over its per-run WebSocket.

    This is the disk-backed equivalent of the service's projection, used for the
    drill-down of runs that are not managed by a standing service. ``since`` lets
    a caller resume after a sequence number, mirroring the WS ``since`` cursor.
    """

    events: list[dict] = []
    if not path or not path.exists():
        return events
    seq = 0
    for line in path.read_text().splitlines():
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


@dataclass(frozen=True)
class LiveProgress:
    phase: str | None
    stage: str | None
    generation: int | None
    best_train_reward: float | None
    best_heldout_reward: float | None
    best_candidate_id: str | None
    container_url: str | None
    completed_rollouts: int | None
    last_activity_at: datetime | None
    started_at: datetime | None
    usage: RunUsage
    eta: dict | None = None
    terminal_state: RunState | None = None

    @classmethod
    def from_event_feed(cls, path: Path | None) -> LiveProgress:
        phase = stage = None
        generation = completed_rollouts = None
        best = best_heldout = None
        best_candidate_id = None
        container_url = None
        first_ts = last_ts = None
        usage: dict = {}
        eta = None
        terminal_state = None
        if path and path.exists():
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning(
                        "skipping malformed GEPA event line path=%s line=%s", path, lineno
                    )
                    continue
                fields = event.get("fields") or {}
                ts = _parse_ts(event.get("ts"))
                if ts is not None:
                    first_ts = first_ts or ts
                    last_ts = ts
                etype = event.get("type")
                if etype == "optimizer.state.transitioned":
                    phase = fields.get("to", phase)
                    details = fields.get("details") or {}
                    stage = details.get("stage", stage)
                    if "generation" in details:
                        generation = details["generation"]
                elif etype == "gepa.run.started":
                    container_url = fields.get("container_url", container_url)
                elif etype == "frontier.updated":
                    best = fields.get("best_train_reward", best)
                elif etype == "candidate.evaluated" and best is None:
                    best = fields.get("train_reward", best)
                elif etype == "score_chart.written":
                    best_candidate_id = fields.get("best_candidate_id", best_candidate_id)
                    candidates = fields.get("candidates") or []
                    if isinstance(candidates, list):
                        chosen = next(
                            (
                                candidate
                                for candidate in candidates
                                if isinstance(candidate, dict) and candidate.get("is_best")
                            ),
                            None,
                        )
                        if chosen:
                            best = chosen.get("train_reward", best)
                            best_heldout = chosen.get("heldout_reward", best_heldout)
                elif etype == "gepa.run.finished":
                    terminal_state = RunState.SUCCEEDED
                    best_candidate_id = fields.get("best_candidate_id", best_candidate_id)
                    best_heldout = fields.get("heldout_reward", best_heldout)
                    if isinstance(fields.get("usage"), dict):
                        usage = fields["usage"]
                elif etype == "runtime.job.completed":
                    stage = fields.get("active_stage", stage)
                    concurrency = fields.get("adaptive_rollout_concurrency") or {}
                    if "completed_rollouts" in concurrency:
                        completed_rollouts = concurrency["completed_rollouts"]
                elif etype == "optimizer.limit.estimate_updated":
                    eta = project_limit_eta(fields)
                if isinstance(fields.get("usage"), dict):
                    usage = fields["usage"]
        return cls(
            phase=phase,
            stage=stage,
            generation=generation,
            best_train_reward=best,
            best_heldout_reward=best_heldout,
            best_candidate_id=best_candidate_id,
            container_url=container_url,
            completed_rollouts=completed_rollouts,
            last_activity_at=last_ts,
            started_at=first_ts,
            usage=RunUsage.from_dict(usage),
            eta=eta,
            terminal_state=terminal_state,
        )


# --------------------------------------------------------------------------
# RunStatus: the projected status of a single run.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunStatus:
    run_id: str
    domain: str
    state: RunState
    run_dir: Path
    started_at: datetime | None
    ended_at: datetime | None
    last_activity_at: datetime | None
    cost_usd: float
    usage: RunUsage
    best_candidate_id: str | None = None
    best_train_reward: float | None = None
    best_heldout_reward: float | None = None
    acceptance_score: float | None = None
    phase: str | None = None
    stage: str | None = None
    generation: int | None = None
    eta: dict | None = None
    failure: RunFailure | None = None
    score_chart_path: Path | None = None

    @property
    def is_running(self) -> bool:
        return self.state is RunState.RUNNING

    @property
    def duration_seconds(self) -> float | None:
        if self.started_at is None:
            return None
        if self.state is RunState.RUNNING:
            end = self.last_activity_at or _now()
        elif self.ended_at is not None:
            end = self.ended_at
        else:
            end = self.last_activity_at  # UNKNOWN: only as far as we last saw it
        if end is None:
            return None
        return max(0.0, (end - self.started_at).total_seconds())

    @classmethod
    def resolve(
        cls, record: RegistryRecord, *, live_within_seconds: float = LIVE_WITHIN_SECONDS
    ) -> RunStatus:
        state = RunState.from_registry_status(record.status)
        domain = _infer_domain(record.run_dir)
        manifest = _read_json(record.manifest_path)

        if state is RunState.RUNNING:
            # The raw events.jsonl is flushed per event and exists while the run is
            # live; events.normalized.jsonl is only written at terminalization.
            live = LiveProgress.from_event_feed(record.event_feed_path)
            if live.terminal_state is not None:
                return cls(
                    run_id=record.run_id,
                    domain=domain,
                    state=live.terminal_state,
                    run_dir=record.run_dir,
                    started_at=record.ts or live.started_at,
                    ended_at=live.last_activity_at,
                    last_activity_at=live.last_activity_at,
                    cost_usd=float(record.cost_usd or 0.0),
                    usage=live.usage if live.usage.total_tokens else RunUsage.from_dict(record.usage),
                    best_candidate_id=live.best_candidate_id or record.best_candidate_id,
                    best_train_reward=live.best_train_reward,
                    best_heldout_reward=live.best_heldout_reward,
                    phase=live.phase,
                    stage=live.stage,
                    generation=live.generation,
                    eta=live.eta,
                )
            started = record.ts or live.started_at
            heartbeat = live.last_activity_at or started
            if _is_dead_local_url(live.container_url):
                state = RunState.UNKNOWN
            # Demote to UNKNOWN if the run has gone silent: started but no terminal
            # record and no event-feed progress within the live window.
            elif heartbeat is None or (_now() - heartbeat).total_seconds() > live_within_seconds:
                state = RunState.UNKNOWN
            usage = live.usage if live.usage.total_tokens else RunUsage.from_dict(record.usage)
            return cls(
                run_id=record.run_id,
                domain=domain,
                state=state,
                run_dir=record.run_dir,
                started_at=started,
                ended_at=None,
                last_activity_at=live.last_activity_at,
                cost_usd=float(record.cost_usd or 0.0),
                usage=usage,
                best_candidate_id=record.best_candidate_id,
                best_train_reward=live.best_train_reward,
                phase=live.phase,
                stage=live.stage,
                generation=live.generation,
                eta=live.eta,
            )

        # Terminal: the manifest is authoritative for final metrics.
        return cls._from_terminal(record, state, domain, manifest)

    @classmethod
    def _from_terminal(
        cls, record: RegistryRecord, state: RunState, domain: str, manifest: dict | None
    ) -> RunStatus:
        manifest = manifest or {}
        history = manifest.get("state_history") or []
        best = manifest.get("best_candidate") or {}
        failure_block = manifest.get("failure")
        chart = manifest.get("score_chart_path")
        usage = manifest.get("usage") or record.usage
        return cls(
            run_id=record.run_id,
            domain=domain,
            state=state,
            run_dir=record.run_dir,
            started_at=_parse_ts(history[0].get("at")) if history else record.ts,
            ended_at=_parse_ts(history[-1].get("at")) if history else record.ts,
            last_activity_at=_parse_ts(history[-1].get("at")) if history else record.ts,
            cost_usd=float(manifest.get("cost_usd") or record.cost_usd or 0.0),
            usage=RunUsage.from_dict(usage),
            best_candidate_id=best.get("candidate_id") or record.best_candidate_id,
            best_train_reward=best.get("train_reward"),
            best_heldout_reward=best.get("heldout_reward"),
            acceptance_score=best.get("acceptance_score"),
            failure=RunFailure.from_manifest(failure_block) if failure_block else None,
            score_chart_path=Path(chart) if chart else None,
        )

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "domain": self.domain,
            "state": self.state.value,
            "run_dir": str(self.run_dir),
            "started_at": _iso(self.started_at),
            "ended_at": _iso(self.ended_at),
            "last_activity_at": _iso(self.last_activity_at),
            "duration_seconds": self.duration_seconds,
            "cost_usd": self.cost_usd,
            "best_candidate_id": self.best_candidate_id,
            "best_train_reward": self.best_train_reward,
            "best_heldout_reward": self.best_heldout_reward,
            "acceptance_score": self.acceptance_score,
            "phase": self.phase,
            "stage": self.stage,
            "generation": self.generation,
            "eta": self.eta,
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
                "proposer_calls": self.usage.proposer_calls,
                "rollout_calls": self.usage.rollout_calls,
            },
            "failure": (
                {
                    "failure_type": self.failure.failure_type,
                    "reason_code": self.failure.reason_code,
                    "message": self.failure.message,
                    "retryable": self.failure.retryable,
                }
                if self.failure
                else None
            ),
            "score_chart_path": str(self.score_chart_path) if self.score_chart_path else None,
        }


def _iso(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts else None


def _read_json(path: Path | None) -> dict | None:
    if path and path.exists():
        return json.loads(path.read_text())
    return None


def _is_dead_local_url(url: str | None) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((parsed.hostname, port), timeout=0.15):
            return False
    except OSError:
        return True


def _infer_domain(run_dir: Path) -> str:
    parts = run_dir.parts
    if "runs" in parts:
        idx = parts.index("runs")
        if idx > 0:
            return parts[idx - 1]
    return run_dir.parent.name


# Genuinely-running runs float to the very top; everything else (succeeded,
# failed, unknown) is ordered purely by its most recent timestamp, newest first.
def _run_time(run: RunStatus) -> datetime | None:
    return run.ended_at or run.last_activity_at or run.started_at


def _board_sort_key(run: RunStatus) -> tuple:
    return (1 if run.state is RunState.RUNNING else 0, _ts_key(_run_time(run)))


# --------------------------------------------------------------------------
# RunBoard: the index.
# --------------------------------------------------------------------------


@dataclass
class RunBoard:
    runs: list[RunStatus]

    @classmethod
    def from_roots(
        cls, roots: Sequence[Path | str], *, live_within_seconds: float = LIVE_WITHIN_SECONDS
    ) -> RunBoard:
        records = latest_registry_records(Path(r) for r in roots)
        runs = [
            RunStatus.resolve(record, live_within_seconds=live_within_seconds)
            for record in records.values()
        ]
        runs.sort(key=_board_sort_key, reverse=True)
        return cls(runs=runs)

    @property
    def total(self) -> int:
        return len(self.runs)

    def count(self, state: RunState) -> int:
        return sum(1 for r in self.runs if r.state is state)

    @property
    def total_cost_usd(self) -> float:
        return sum(r.cost_usd for r in self.runs)

    @property
    def total_tokens(self) -> int:
        return sum(r.usage.total_tokens for r in self.runs)

    def to_data(self) -> dict:
        return {
            "schema": "synth.gepa_run_board.v1",
            "generated_at": _iso(_now()),
            "summary": {
                "total": self.total,
                "running": self.count(RunState.RUNNING),
                "succeeded": self.count(RunState.SUCCEEDED),
                "failed": self.count(RunState.FAILED),
                "unknown": self.count(RunState.UNKNOWN),
                "total_cost_usd": self.total_cost_usd,
                "total_tokens": self.total_tokens,
            },
            "runs": [r.to_dict() for r in self.runs],
        }

    def render_html(self, *, title: str = "GEPA Run Board", live_endpoint: str | None = None) -> str:
        return render_board_html(self.to_data(), title=title, live_endpoint=live_endpoint)

    def write_html(self, path: Path | str, *, title: str = "GEPA Run Board") -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render_html(title=title))
        return path


# --------------------------------------------------------------------------
# HTML rendering. Server-rendered initial table plus a small vanilla-JS client
# that, when given a live SSE endpoint, patches rows in place as runs advance.
# --------------------------------------------------------------------------


def _fmt_reward(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _fmt_cost(value: float | None) -> str:
    return "—" if value is None else f"${value:.4f}"


def _fmt_tokens(value: int | None) -> str:
    return "—" if value is None else f"{value:,}"


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None


def _limit_kind_label(kind: object) -> str:
    if not kind:
        return "limit"
    if isinstance(kind, str):
        return kind.replace("_", " ")
    if isinstance(kind, dict) and kind.get("custom"):
        return str(kind["custom"]).replace("_", " ")
    return "limit"


def project_limit_eta(snapshot: dict | None) -> dict | None:
    if not isinstance(snapshot, dict):
        return None
    nearest = snapshot.get("nearest") or snapshot.get("nearest_limit")
    if not isinstance(nearest, dict):
        return None
    limits = snapshot.get("limits") if isinstance(snapshot.get("limits"), list) else []
    status = nearest
    limit_id = nearest.get("limit_id")
    if not nearest.get("forecast") and limit_id:
        status = next(
            (
                item
                for item in limits
                if isinstance(item, dict)
                and isinstance(item.get("forecast"), dict)
                and item["forecast"].get("limit_id") == limit_id
            ),
            nearest,
        )
    forecast = status.get("forecast") if isinstance(status.get("forecast"), dict) else nearest
    seconds = _float_or_none(forecast.get("seconds_to_limit"))
    kind = status.get("kind")
    definition = status.get("definition") if isinstance(status.get("definition"), dict) else {}
    if kind is None:
        kind = definition.get("kind")
    if seconds is None:
        return None
    return {
        "kind": _limit_kind_label(kind),
        "seconds_to_limit": seconds,
        "seconds_to_limit_low": _float_or_none(forecast.get("seconds_to_limit_low")),
        "seconds_to_limit_high": _float_or_none(forecast.get("seconds_to_limit_high")),
        "confidence": forecast.get("confidence"),
        "model": forecast.get("model"),
        "updated_at": forecast.get("updated_at") or snapshot.get("generated_at"),
    }


def _eta_label(eta: dict | None) -> str:
    if not isinstance(eta, dict):
        return "—"
    seconds = _float_or_none(eta.get("seconds_to_limit"))
    if seconds is None:
        return "—"
    seconds = max(0.0, seconds)
    value = _fmt_duration(seconds)
    low = _float_or_none(eta.get("seconds_to_limit_low"))
    high = _float_or_none(eta.get("seconds_to_limit_high"))
    if low is not None and high is not None:
        value += f" [{_fmt_duration(low)}-{_fmt_duration(high)}]"
    confidence = eta.get("confidence")
    confidence_suffix = f" · {confidence}" if confidence else ""
    return f"{eta.get('kind') or 'limit'} · {value}{confidence_suffix}"


def _eta_html(eta: dict | None) -> str:
    if not isinstance(eta, dict):
        return "<span class='eta-empty'>—</span>"
    seconds = _float_or_none(eta.get("seconds_to_limit"))
    if seconds is None:
        return "<span class='eta-empty'>—</span>"
    seconds = max(0.0, seconds)
    value = _fmt_duration(seconds)
    low = _float_or_none(eta.get("seconds_to_limit_low"))
    high = _float_or_none(eta.get("seconds_to_limit_high"))
    interval = ""
    if low is not None and high is not None:
        interval = (
            "<div class='eta-range'>"
            f"<span class='eta-range-label'>range</span>"
            f"<span>{_esc(_fmt_duration(low))}-{_esc(_fmt_duration(high))}</span>"
            "</div>"
        )
    kind = _esc(eta.get("kind") or "limit")
    confidence = eta.get("confidence")
    confidence_label = f" · {_esc(confidence)}" if confidence else ""
    return (
        f"<div class='eta-box'><div class='eta-main'><span class='eta-mid'>mid</span>{_esc(value)}</div>{interval}"
        f"<div class='eta-sub'>{kind}{confidence_label}</div></div>"
    )


def _fmt_ts(value: str | None) -> str:
    if not value:
        return "—"
    return value.replace("T", " ").split("+")[0].split(".")[0]


def _fmt_age(value: str | None) -> str:
    if not value:
        return "—"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return "—"
    seconds = max(0, int((_now() - parsed).total_seconds()))
    if seconds < 90:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m ago"
    return f"{minutes // 60}h ago"


def _fmt_updated(run: dict) -> str:
    value = run.get("last_activity_at") or run.get("ended_at") or run.get("started_at")
    absolute = _fmt_ts(value)
    if not value:
        return absolute
    age = _fmt_age(value)
    if age == "—":
        return absolute
    return f"{age}<br><span class='muted'>{absolute}</span>"


def _esc(value: object) -> str:
    return html.escape(str(value))


def _phase_label(run: dict) -> str:
    bits = []
    seen = set()
    active = run.get("active_evaluation") if isinstance(run.get("active_evaluation"), dict) else {}
    active_stage = active.get("stage")
    if active_stage:
        bits.append(f"active {active_stage}")
        seen.add(active_stage)
    row_count = active.get("row_count")
    scored_count = active.get("scored_count")
    if row_count:
        bits.append(f"scored {int(scored_count or 0)}/{int(row_count)}")
    candidate_eval_count = active.get("candidate_evaluation_count")
    if candidate_eval_count:
        bits.append(f"{int(candidate_eval_count)} evals")
    queue_counts = run.get("queue_counts") if isinstance(run.get("queue_counts"), dict) else {}
    active_queues = [f"{name} {count}" for name, count in sorted(queue_counts.items()) if count]
    if active_queues:
        bits.append("queues " + ", ".join(active_queues))
    if run.get("worker_id"):
        bits.append(f"worker {run['worker_id']}")
    if run.get("blocked_reason"):
        blocked = f"blocked {run['blocked_reason']}"
        if run.get("blocked_by_run_id"):
            blocked += f" by {run['blocked_by_run_id']}"
        bits.append(blocked)
    elif run.get("scheduler_state") == "runnable":
        bits.append("runnable")
    for bit in (run.get("phase"), run.get("stage")):
        if bit and bit not in seen:
            bits.append(bit)
            seen.add(bit)
    gen = run.get("generation")
    if gen is not None:
        bits.append(f"gen {gen}")
    return " · ".join(bits)


def _render_rows(runs: list[dict]) -> str:
    rows: list[str] = []
    for run in runs:
        state = run["state"]
        liveish = state in {"queued", "running", "paused"}
        detail = ""
        if run.get("failure"):
            f = run["failure"]
            detail = (
                f"<div class='fail'>{_esc(f['failure_type'])} · {_esc(f['reason_code'])}"
                f"{' · retryable' if f['retryable'] else ''}"
                f"<div class='failmsg'>{_esc(f['message'])}</div></div>"
            )
        elif liveish:
            ph = _phase_label(run)
            if ph:
                detail = f"<div class='phase'>{_esc(ph)}</div>"
        rows.append(
            "<tr data-run='{rid}' data-state='{state}' data-domain='{domain}'>"
            "<td><span class='pill {state}'>{label}</span></td>"
            "<td class='mono'>{rid}{detail}</td>"
            "<td>{domain}</td>"
            "<td class='num train'>{train}</td>"
            "<td class='num heldout'>{heldout}</td>"
            "<td class='eta'>{eta}</td>"
            "<td class='num dur'>{dur}</td>"
            "<td class='storage-cell'>{storage}</td>"
            "<td class='num tokens'>{tokens}</td>"
            "<td class='num cost'>{cost}</td>"
            "<td class='ts updated'>{updated}</td>"
            "</tr>".format(
                rid=_esc(run["run_id"]),
                state=_esc(state),
                domain=_esc(run["domain"]),
                label=_esc(state),
                detail=detail,
                train=_fmt_reward(run["best_train_reward"]),
                heldout=_fmt_reward(run["best_heldout_reward"]),
                eta=_eta_html(run.get("eta")),
                dur=_fmt_duration(run["duration_seconds"]),
                storage=_storage_badge(run),
                tokens=_fmt_tokens((run.get("usage") or {}).get("total_tokens")),
                cost=_fmt_cost(run.get("cost_usd")),
                updated=_fmt_updated(run),
            )
        )
    return "\n".join(rows)


def _storage_badge(run: dict) -> str:
    state = str(run.get("state") or "")
    if state in {"succeeded", "failed", "cancelled"}:
        return "<span class='storage-badge ready'>inspect</span>"
    if state in {"queued", "running", "paused"}:
        return "<span class='storage-badge live'>live</span>"
    return "<span class='storage-badge'>unknown</span>"


def render_board_html(
    data: dict,
    *,
    title: str,
    live_endpoint: str | None = None,
    service_url: str | None = None,
    events_base: str | None = None,
) -> str:
    s = data["summary"]
    queued = int(s.get("queued", 0) or 0)
    running = int(s.get("running", 0) or 0)
    paused = int(s.get("paused", 0) or 0)
    domains = sorted({r["domain"] for r in data["runs"]})
    domain_opts = "".join(f"<option value='{_esc(d)}'>{_esc(d)}</option>" for d in domains)
    rows_html = _render_rows(data["runs"])
    embedded = html.escape(json.dumps(data), quote=False)
    live_json = json.dumps(live_endpoint)
    service_json = json.dumps(service_url)
    events_base_json = json.dumps(events_base)
    service_progress = _fmt_age(s.get("service_last_progress_at"))
    board_poll = _fmt_age(data.get("generated_at"))
    oldest_queued = _fmt_duration(s.get("service_oldest_queued_age_seconds"))
    service_running = int(s.get("service_running_count") or 0)
    service_workers = int(s.get("service_worker_count") or 0)
    active_workers = int(s.get("service_active_workers") or 0)
    queued_runnable = int(s.get("service_queued_runnable") or 0)
    queued_blocked = int(s.get("service_queued_blocked") or 0)
    cost_label = _fmt_cost(s.get("total_cost_usd"))
    tokens_label = (
        "unknown tokens" if s.get("total_tokens") is None else f"{s['total_tokens']:,} tokens"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>
  :root {{
    --bg:#0a0a0a; --panel:#111111; --inset:#0a0a0a; --border:#1f2937; --text:#f5f5f5; --muted:#9ca3af;
    --ok:#22c55e; --fail:#f87171; --run:#FF5C00; --stale:#d97757; --score:#6b7280; --blue:#6a9bcc;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
    font:14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; }}
  header {{ padding:24px 28px 8px; display:flex; align-items:baseline; gap:14px; }}
  h1 {{ margin:0; font-size:20px; }}
  .live {{ font-size:11px; color:var(--muted); display:flex; align-items:center; gap:6px; }}
  .dot {{ width:8px; height:8px; border-radius:50%; background:var(--muted); }}
  .dot.on {{ background:var(--ok); box-shadow:0 0 6px var(--ok); }}
  .sub {{ color:var(--muted); font-size:12px; padding:0 28px; }}
  .cards {{ display:flex; gap:12px; flex-wrap:wrap; padding:16px 28px; }}
  .card {{ background:var(--panel); border:1px solid var(--border); border-radius:8px;
    padding:12px 16px; min-width:104px; }}
  .card .n {{ font-size:22px; font-weight:600; }}
  .card .l {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
  .card.queue .n {{ color:var(--stale); }} .card.run .n {{ color:var(--run); }} .card.ok .n {{ color:var(--ok); }}
  .card.fail .n {{ color:var(--fail); }} .card.unknown .n {{ color:var(--stale); }}
  .status-panel {{ margin:0 28px 14px; padding:14px 16px; background:var(--panel);
    border:1px solid var(--border); border-radius:8px; }}
  .status-head {{ display:flex; justify-content:space-between; gap:16px; align-items:baseline; margin-bottom:10px; }}
  .status-title {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  .status-note {{ color:var(--muted); font-size:11px; text-align:right; }}
  .status-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(118px,1fr)); gap:8px; margin-bottom:12px; }}
  .status-stat {{ background:var(--inset); border:1px solid var(--border); border-radius:6px; padding:8px 10px; min-width:0; }}
  .status-stat .k {{ color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.04em; white-space:nowrap; }}
  .status-stat .v {{ margin-top:2px; font-size:15px; font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }}
  .status-stat .v.run {{ color:var(--run); }} .status-stat .v.ok {{ color:var(--ok); }}
  .status-stat .v.warn {{ color:var(--stale); }} .status-stat .v.fail {{ color:var(--fail); }}
  .status-lanes {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }}
  .status-lane-title {{ color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.04em; margin-bottom:5px; }}
  .chiprow {{ display:flex; flex-wrap:wrap; gap:6px; align-items:center; }}
  .chip {{ display:inline-flex; max-width:100%; gap:5px; align-items:baseline; border:1px solid rgba(139,148,158,.35);
    border-radius:999px; padding:2px 7px; color:var(--muted); font-size:11px; }}
  .chip.run {{ color:var(--run); border-color:rgba(255,92,0,.45); }}
  .chip.warn {{ color:var(--stale); border-color:rgba(217,119,87,.55); }}
  .chip.fail {{ color:var(--fail); border-color:rgba(248,113,113,.55); }}
  .chip strong {{ color:var(--text); font-weight:500; }}
  .queue-list {{ display:flex; flex-direction:column; gap:4px; }}
  .queue-item {{ display:grid; grid-template-columns:minmax(120px, 1fr) minmax(140px, 1.2fr) auto;
    gap:8px; color:var(--muted); font-size:11px; border-bottom:1px solid rgba(31,41,55,.7); padding-bottom:4px; }}
  .queue-item .rid {{ color:var(--text); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  .queue-item .why.run {{ color:var(--run); }} .queue-item .why.warn {{ color:var(--stale); }}
  .controls {{ padding:4px 28px 12px; display:flex; gap:10px; }}
  input, select {{ background:var(--panel); color:var(--text); border:1px solid var(--border);
    border-radius:6px; padding:6px 10px; font:inherit; }}
  table {{ width:calc(100% - 56px); margin:0 28px 40px; border-collapse:collapse; }}
  th, td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--border); vertical-align:top; }}
  th {{ color:var(--muted); font-weight:500; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
  tr:hover td {{ background:#141414; }}
  tr[data-state="queued"] td {{ background:rgba(217,119,87,.07); }}
  tr[data-state="running"] td {{ background:rgba(255,92,0,.07); }}
  tr[data-state="paused"] td {{ background:rgba(217,119,87,.07); }}
  .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  .eta {{ color:var(--run); min-width:190px; max-width:240px; }}
  .eta-box {{ display:flex; flex-direction:column; gap:3px; align-items:flex-start; }}
  .eta-main {{ color:var(--text); font-size:20px; line-height:1.05; font-weight:800; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  .eta-mid {{ color:var(--muted); font-size:10px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; margin-right:6px; vertical-align:middle; }}
  .eta-range {{ display:inline-flex; gap:6px; align-items:center; color:var(--run); background:rgba(255,92,0,.10); border:1px solid rgba(255,92,0,.28); border-radius:6px; padding:2px 6px; font-size:11px; line-height:1.2; white-space:nowrap; font-variant-numeric:tabular-nums; }}
  .eta-range-label {{ color:var(--muted); font-size:10px; font-weight:700; letter-spacing:.04em; text-transform:uppercase; }}
  .eta-sub {{ color:var(--muted); font-size:11px; line-height:1.2; white-space:nowrap; }}
  .eta-empty {{ color:var(--run); }}
  .ts {{ color:var(--muted); white-space:nowrap; }}
  .muted {{ color:var(--muted); }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px; font-weight:600; }}
  .pill.succeeded {{ background:rgba(34,197,94,.13); color:var(--ok); }}
  .pill.failed {{ background:rgba(248,113,113,.13); color:var(--fail); }}
  .pill.queued {{ background:rgba(217,119,87,.13); color:var(--stale); }}
  .pill.running {{ background:rgba(255,92,0,.13); color:var(--run); }}
  .pill.paused {{ background:rgba(217,119,87,.13); color:var(--stale); }}
  .pill.unknown {{ background:rgba(217,119,87,.13); color:var(--stale); }}
  .storage-cell {{ white-space:nowrap; }}
  .storage-badge {{ display:inline-flex; border:1px solid rgba(139,148,158,.35); border-radius:999px;
    padding:2px 7px; color:var(--muted); font-size:11px; }}
  .storage-badge.ready {{ color:var(--ok); border-color:rgba(34,197,94,.45); }}
  .storage-badge.live {{ color:var(--run); border-color:rgba(255,92,0,.45); }}
  .fail {{ color:var(--fail); font-size:11px; margin-top:4px; }}
  .failmsg {{ color:var(--muted); margin-top:2px; max-width:520px; }}
  .phase {{ color:var(--run); font-size:11px; margin-top:4px; }}
  .empty {{ padding:40px 28px; color:var(--muted); }}
  @keyframes flash {{ from {{ background:rgba(255,92,0,.28); }} to {{ background:transparent; }} }}
  tr.bump td {{ animation:flash 1s ease-out; }}
  tbody tr {{ cursor:pointer; }}
  #scrim {{ position:fixed; inset:0; background:rgba(0,0,0,.45); z-index:9; }}
  [hidden] {{ display:none !important; }}
  #drawer {{ position:fixed; top:0; right:0; bottom:0; width:min(620px,92vw); z-index:10;
    background:var(--panel); border-left:1px solid var(--border); display:flex; flex-direction:column;
    box-shadow:-12px 0 30px rgba(0,0,0,.4); }}
  .drawer-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px;
    padding:18px 20px; border-bottom:1px solid var(--border); }}
  #drawer-title {{ font-size:15px; }}
  .drawer-sub {{ color:var(--muted); font-size:12px; margin-top:4px; }}
  .drawer-meta {{ color:var(--run); font-size:11px; margin-top:3px; }}
  .statgrid {{ display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:8px; margin-bottom:10px; }}
  .stat {{ background:var(--inset); border:1px solid var(--border); border-radius:6px; padding:7px 8px; min-width:0; }}
  .stat .k {{ color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.04em; white-space:nowrap; }}
  .stat .v {{ margin-top:2px; font-size:12px; line-height:1.25; font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }}
  .stat .v.ok {{ color:var(--ok); }} .stat .v.run {{ color:var(--run); }} .stat .v.warn {{ color:var(--stale); }}
  .queuegrid {{ display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:10px; }}
  .queuecard {{ background:var(--inset); border:1px solid var(--border); border-radius:6px; padding:10px; min-width:0; }}
  .qtop {{ display:flex; justify-content:space-between; gap:8px; align-items:baseline; margin-bottom:8px; }}
  .qtitle {{ color:var(--text); font-size:12px; }}
  .qstate {{ color:var(--muted); font-size:11px; font-variant-numeric:tabular-nums; }}
  .qstate.run {{ color:var(--run); }} .qstate.ok {{ color:var(--ok); }} .qstate.warn {{ color:var(--stale); }}
  .qbar {{ height:5px; border-radius:999px; background:#141414; border:1px solid var(--border); overflow:hidden; margin-bottom:8px; }}
  .qfill {{ height:100%; background:var(--run); }}
  .qfill.ok {{ background:var(--ok); }} .qfill.warn {{ background:var(--stale); }}
  .qstats {{ display:grid; grid-template-columns:repeat(2, minmax(0,1fr)); gap:5px 10px; }}
  .qstat {{ min-width:0; }}
  .qstat .k {{ color:var(--muted); font-size:10px; text-transform:uppercase; letter-spacing:.04em; white-space:nowrap; }}
  .qstat .v {{ margin-top:1px; font-size:12px; font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }}
  .chart-t {{ color:var(--muted); font-size:11px; margin:6px 0 2px; }}
  .chart {{ background:var(--inset); border:1px solid var(--border); border-radius:6px; padding:4px; }}
  .chart svg text {{ font-family:ui-monospace, Menlo, monospace; }}
  .leg {{ display:flex; gap:12px; font-size:10px; color:var(--muted); margin-top:3px; }}
  .leg i {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:3px; vertical-align:middle; }}
  .waterfall {{ background:var(--inset); border:1px solid var(--border); border-radius:6px; overflow:auto; }}
  .waterfall svg {{ display:block; min-width:560px; }}
  .wf-label {{ fill:var(--muted); font-size:9px; }}
  .wf-id {{ fill:var(--text); font-size:10px; }}
  .wf-axis {{ stroke:var(--border); }}
  .wf-line {{ stroke:var(--border); stroke-width:1.5; }}
  .wf-stage {{ stroke-width:5; stroke-linecap:round; opacity:.9; }}
  .wf-dot {{ stroke:var(--inset); stroke-width:1.5; }}
  .drawer-actions {{ display:flex; gap:6px; flex-shrink:0; }}
  .ctl {{ background:var(--inset); color:var(--text); border:1px solid var(--border); border-radius:6px;
    padding:5px 10px; font:inherit; font-size:12px; cursor:pointer; }}
  .ctl:hover {{ border-color:var(--run); }}
  .ctl[disabled] {{ opacity:.45; cursor:not-allowed; }}
  #btn-stop, #btn-cancel {{ color:var(--fail); }}
  .drawer-conn {{ padding:6px 20px; font-size:11px; color:var(--muted); border-bottom:1px solid var(--border); }}
  .drawer-conn .live {{ color:var(--ok); }}
  .drawer-body {{ flex:1; overflow:auto; }}
  .dsec {{ border-bottom:1px solid var(--border); padding:12px 20px; }}
  .dsec-h {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.04em;
    margin-bottom:8px; cursor:default; }}
  details.dsec > summary.dsec-h {{ cursor:pointer; }}
  .dcount {{ color:var(--muted); text-transform:none; letter-spacing:0; }}
  .dnone {{ color:var(--muted); font-size:12px; padding:4px 0; }}
  .storage-actions {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }}
  .storage-note {{ color:var(--muted); font-size:11px; margin-top:8px; }}
  .storage-warning {{ color:var(--fail); font-size:11px; margin-top:8px; }}
  .storage-list {{ display:flex; flex-direction:column; gap:4px; margin-top:8px; }}
  .storage-row {{ display:grid; grid-template-columns:80px minmax(0,1fr); gap:8px; font-size:12px;
    border-bottom:1px solid rgba(31,41,55,.7); padding:3px 0; }}
  .storage-row .bytes {{ color:var(--muted); text-align:right; font-variant-numeric:tabular-nums; }}
  .dframe {{ display:flex; flex-direction:column; gap:4px; }}
  .frow {{ display:flex; gap:10px; align-items:baseline; font-size:12px; padding:3px 0;
    border-bottom:1px solid rgba(31,41,55,.7); }}
  .frow .g {{ color:var(--muted); min-width:54px; }}
  .frow .best {{ color:var(--ok); font-variant-numeric:tabular-nums; min-width:84px; }}
  .frow .up {{ color:var(--ok); }} .frow .same {{ color:var(--muted); }}
  .frow .cnt {{ color:var(--muted); }}
  #d-cands {{ width:100%; border-collapse:collapse; font-size:12px; }}
  #d-cands th {{ text-align:left; color:var(--muted); font-weight:500; font-size:10px;
    text-transform:uppercase; padding:4px 8px 4px 0; }}
  #d-cands td {{ padding:5px 8px 5px 0; border-bottom:1px solid rgba(31,41,55,.7);
    vertical-align:top; font-variant-numeric:tabular-nums; }}
  #d-cands .num {{ text-align:right; }}
  .res-acc {{ color:var(--ok); }} .res-rej {{ color:var(--fail); }} .res-seed {{ color:var(--run); }}
  .res-pend {{ color:var(--stale); }}
  .creason {{ color:var(--muted); }}
  .drawer-log {{ max-height:300px; overflow:auto; margin-top:6px; }}
  .ev {{ padding:5px 0; border-bottom:1px solid rgba(31,41,55,.7); }}
  .ev .meta {{ color:var(--muted); font-size:11px; display:flex; gap:8px; }}
  .ev .kind {{ color:var(--run); }}
  .ev .msg {{ margin-top:2px; font-size:12px; }}
  .ev .facts {{ margin-top:3px; display:flex; flex-wrap:wrap; gap:6px; font-size:11px; color:var(--muted); }}
  .ev .fact {{ border:1px solid rgba(139,148,158,.35); border-radius:999px; padding:1px 6px; }}
  .ev.terminal .kind {{ color:var(--ok); }}
  .ev.err .kind {{ color:var(--fail); }}
</style>
</head>
<body>
<header>
  <h1>{_esc(title)}</h1>
  <span class="live"><span id="dot" class="dot"></span><span id="livetext">static</span></span>
</header>
<div class="sub" id="subline">
  {s['total']} run(s) · {queued} queued · {running} running · workers {active_workers}/{service_workers} · queue {queued_runnable} runnable/{queued_blocked} blocked · service running {service_running} · service progress {service_progress} · oldest queued {oldest_queued} · board poll {board_poll} · {s['unknown']} unknown · {cost_label} · {tokens_label}
</div>
<div class="cards">
  <div class="card queue"><div class="n" data-k="queued">{queued}</div><div class="l">Queued</div></div>
  <div class="card run"><div class="n" data-k="running">{running}</div><div class="l">Running</div></div>
  <div class="card queue"><div class="n" data-k="paused">{paused}</div><div class="l">Paused</div></div>
  <div class="card ok"><div class="n" data-k="succeeded">{s['succeeded']}</div><div class="l">Succeeded</div></div>
  <div class="card fail"><div class="n" data-k="failed">{s['failed']}</div><div class="l">Failed</div></div>
  <div class="card unknown"><div class="n" data-k="unknown">{s['unknown']}</div><div class="l">Unknown</div></div>
  <div class="card"><div class="n" data-k="total">{s['total']}</div><div class="l">Total</div></div>
</div>
<section id="run-status" class="status-panel"></section>
<div class="controls">
  <input id="q" type="search" placeholder="filter run id…" autocomplete="off">
  <select id="state">
    <option value="">all states</option>
    <option value="queued">queued</option>
    <option value="running">running</option>
    <option value="paused">paused</option>
    <option value="succeeded">succeeded</option>
    <option value="failed">failed</option>
    <option value="unknown">unknown</option>
  </select>
  <select id="domain"><option value="">all domains</option>{domain_opts}</select>
</div>
<table id="board">
  <thead><tr>
    <th>State</th><th>Run</th><th>Domain</th>
    <th class="num">Train</th><th class="num">Heldout</th><th>ETA</th><th class="num">Duration</th>
    <th>Storage</th><th class="num">Tokens</th><th class="num">Cost</th><th class="ts">Last Updated</th>
  </tr></thead>
  <tbody>
{rows_html}
  </tbody>
</table>
<div id="empty" class="empty" hidden>No runs match the current filter.</div>
<aside id="drawer" hidden>
  <div class="drawer-head">
    <div>
      <div id="drawer-title" class="mono"></div>
      <div id="drawer-sub" class="drawer-sub"></div>
      <div id="drawer-meta" class="drawer-meta"></div>
    </div>
    <div class="drawer-actions">
      <button id="btn-stop" class="ctl" hidden>stop</button>
      <button id="btn-cancel" class="ctl" hidden>cancel</button>
      <button id="drawer-close" class="ctl">close</button>
    </div>
  </div>
  <div id="drawer-conn" class="drawer-conn"></div>
  <div class="drawer-body">
    <div class="dsec"><div class="dsec-h">Runtime</div>
      <div id="d-runstate" class="statgrid"><div class="dnone">no runtime stats yet</div></div>
      <div class="chart-t">Per-limit ETA forecast over time</div>
      <div id="chart-limit-eta" class="chart"><div class="dnone">no limit forecast samples yet</div></div>
    </div>
    <div class="dsec"><div class="dsec-h">Storage</div>
      <div id="d-storage"><div class="dnone">loading storage report…</div></div>
    </div>
    <div class="dsec"><div class="dsec-h">Throughput</div>
      <div id="d-throughput" class="statgrid"><div class="dnone">no throughput stats yet</div></div>
      <div class="chart-t">Tokens/min over time</div>
      <div id="chart-throughput" class="chart"><div class="dnone">no throughput samples yet</div></div>
      <div class="chart-t">Proposer round time</div>
      <div id="d-proposer-rounds" class="dframe"><div class="dnone">no proposer rounds yet</div></div>
    </div>
    <div class="dsec"><div class="dsec-h">Queues</div>
      <div id="d-queues" class="queuegrid"><div class="dnone">no queue stats yet</div></div>
    </div>
    <div class="dsec"><div class="dsec-h">Progress</div>
      <div class="chart-t">Best train + minibatch + heldout by generation</div>
      <div id="chart-best" class="chart"><div class="dnone">no data yet</div></div>
      <div class="chart-t">Minibatch reward by candidate</div>
      <div id="chart-mb" class="chart"><div class="dnone">no data yet</div></div>
      <div class="leg"><span><i style="background:#6a9bcc"></i>seed</span>
        <span><i style="background:#22c55e"></i>accepted</span>
        <span><i style="background:#f87171"></i>rejected</span></div>
    </div>
    <div class="dsec"><div class="dsec-h">Candidate state waterfall</div>
      <div class="chart-t">State machine trajectory from candidate creation to final state</div>
      <div id="chart-waterfall" class="waterfall"><div class="dnone">no candidate state transitions yet</div></div>
      <div class="leg"><span><i style="background:#FF5C00"></i>created</span>
        <span><i style="background:#d97757"></i>rollout</span>
        <span><i style="background:#6b7280"></i>scored</span>
        <span><i style="background:#22c55e"></i>accepted/final</span>
        <span><i style="background:#f87171"></i>rejected</span></div>
    </div>
    <div class="dsec"><div class="dsec-h">Frontier / Pareto</div>
      <div class="chart-t">Aggregate train + Pareto coverage by candidate index</div>
      <div id="chart-pareto" class="chart"><div class="dnone">no aggregate stats yet</div></div>
      <div id="d-frontier" class="dframe"><div class="dnone">no frontier updates yet</div></div>
    </div>
    <div class="dsec"><div class="dsec-h">Candidates <span id="d-candcount" class="dcount"></span></div>
      <table id="d-cands"><thead><tr>
        <th>Candidate</th><th class="num">Idx</th><th>Gen</th><th class="num">Minibatch</th><th class="num">Train</th>
        <th class="num">Heldout</th><th>Result</th>
      </tr></thead><tbody><tr><td colspan="7" class="dnone">no candidates yet</td></tr></tbody></table>
    </div>
    <div class="dsec"><div class="dsec-h">Recent events <span id="d-evcount" class="dcount"></span></div>
      <div id="drawer-log" class="drawer-log"><div class="dnone">no events yet</div></div></div>
  </div>
</aside>
<div id="scrim" hidden></div>
<script type="application/json" id="board-data">{embedded}</script>
<script>
{_BOARD_JS}
window.__LIVE_ENDPOINT__ = {live_json};
window.__SERVICE_URL__ = {service_json};
window.__EVENTS_BASE__ = {events_base_json};
GepaBoard.init();
</script>
</body>
</html>
"""


# Vanilla client: server-rendered rows are the source of truth on load; when a
# live SSE endpoint is present, each `board` event replaces the table from the
# pushed snapshot (running runs already sorted to the top server-side).
_BOARD_JS = r"""
var GepaBoard = (function () {
  var q, st, dm, body, empty, boardData=null;
  function fmtReward(v){ return v==null? '—' : Number(v).toFixed(3); }
  function fmtCost(v){ return v==null||v===''? '—' : '$'+Number(v).toFixed(4); }
  function fmtTokens(v){ return v==null||v===''? '—' : Number(v).toLocaleString(); }
  function fmtDur(s){
    if(s==null) return '—';
    if(s<60) return Math.round(s)+'s';
    var m=Math.floor(s/60), x=Math.floor(s%60);
    if(m<60) return m+'m '+String(x).padStart(2,'0')+'s';
    var h=Math.floor(m/60); m=m%60; return h+'h '+String(m).padStart(2,'0')+'m';
  }
  function etaLabel(eta){
    if(!eta || eta.seconds_to_limit==null || isNaN(Number(eta.seconds_to_limit))) return '—';
    var seconds=Math.max(0, Number(eta.seconds_to_limit));
    var value=fmtDur(seconds);
    if(eta.seconds_to_limit_low!=null && eta.seconds_to_limit_high!=null){
      value+=' ['+fmtDur(Number(eta.seconds_to_limit_low))+'-'+fmtDur(Number(eta.seconds_to_limit_high))+']';
    }
    return (eta.kind || 'limit')+' · '+value+(eta.confidence ? ' · '+eta.confidence : '');
  }
  function etaHtml(eta){
    if(!eta || eta.seconds_to_limit==null || isNaN(Number(eta.seconds_to_limit))) return "<span class='eta-empty'>—</span>";
    var seconds=Math.max(0, Number(eta.seconds_to_limit));
    var interval='';
    if(eta.seconds_to_limit_low!=null && eta.seconds_to_limit_high!=null){
      interval="<div class='eta-range'><span class='eta-range-label'>range</span><span>"
        +esc(fmtDur(Number(eta.seconds_to_limit_low)))+"-"+esc(fmtDur(Number(eta.seconds_to_limit_high)))+"</span></div>";
    }
    var sub=esc(eta.kind || 'limit')+(eta.confidence ? ' · '+esc(eta.confidence) : '');
    return "<div class='eta-box'><div class='eta-main'><span class='eta-mid'>mid</span>"+esc(fmtDur(seconds))+"</div>"+interval+"<div class='eta-sub'>"+sub+"</div></div>";
  }
  function fmtCost(v){ return v==null || isNaN(Number(v)) ? '—' : '$'+Number(v).toFixed(4); }
  function parseTs(v){
    if(!v) return NaN;
    var s=String(v).trim();
    if(/^\\d{4}-\\d{2}-\\d{2}[ T]\\d{2}:\\d{2}:\\d{2}(\\.\\d+)?$/.test(s)){
      s=s.replace(' ','T')+'Z';
    }
    return Date.parse(s);
  }
  function fmtTs(v){ return v? String(v).replace('T',' ').split('+')[0].split('.')[0] : '—'; }
  function agoLabel(iso){
    if(!iso) return '';
    var t=parseTs(iso); if(isNaN(t)) return '';
    var s=(Date.now()-t)/1000;
    if(s<90) return Math.round(s)+'s ago';
    var m=Math.floor(s/60); if(m<60) return m+'m ago';
    return Math.floor(m/60)+'h ago';
  }
  function ageOrDash(iso){ return agoLabel(iso) || '—'; }
  function fmtUpdated(r){
    var v=r.last_activity_at || r.ended_at || r.started_at;
    if(!v) return '—';
    return esc(agoLabel(v)||'—')+"<br><span class='muted'>"+esc(fmtTs(v))+"</span>";
  }
  function shortRunId(id){ return String(id||'').replace(/^gepa_/,'').slice(0,12) || '—'; }
  function serviceCounts(data){
    var s=(data&&data.summary)||{};
    var rs=(data&&data.run_status)||{};
    var rsc=rs.counts||{};
    var c=s.service_status_counts||{};
    var workers=rs.worker_slots || ((((data||{}).scheduler||{}).workers)||[]);
    var queued=rs.queued_reasons || ((((data||{}).scheduler||{}).queued)||[]);
    var active=workers.filter(function(w){ return w && w.state==='active'; });
    var leased=active.filter(function(w){ return w.request_status==='leased'; }).length;
    var rawRunning=active.filter(function(w){ return w.request_status==='running'; }).length;
    return {
      queued:rsc.queued!=null ? rsc.queued : (c.queued!=null ? c.queued : (s.queued||0)),
      leased:rsc.leased!=null ? rsc.leased : (s.service_leased_count!=null ? s.service_leased_count : leased),
      rawRunning:rsc.running!=null ? rsc.running : (s.service_raw_running_count!=null ? s.service_raw_running_count : rawRunning),
      terminal:rsc.terminal!=null ? rsc.terminal : (s.service_terminal_count!=null ? s.service_terminal_count : ((c.succeeded||0)+(c.failed||0)+(c.cancelled||0))),
      runnable:rsc.queued_runnable!=null ? rsc.queued_runnable : (s.service_queued_runnable||0),
      blocked:rsc.queued_blocked!=null ? rsc.queued_blocked : (s.service_queued_blocked||0),
      stale:rsc.stale_leases!=null ? rsc.stale_leases : (s.service_stale_lease_count||0),
      workers:workers,
      active:active,
      queuedItems:queued
    };
  }
  function leaseExpiresAt(w){ return w.lease_expires_at || w.last_heartbeat_at || null; }
  function isStaleWorker(w){
    if(w && w.stale!=null) return !!w.stale;
    if(!w || w.state!=='active') return false;
    var expires=parseTs(leaseExpiresAt(w));
    return !isNaN(expires) && expires<Date.now();
  }
  function expiryLabel(w){
    if(w.seconds_until_lease_expiry!=null && !isNaN(Number(w.seconds_until_lease_expiry))){
      var seconds=Number(w.seconds_until_lease_expiry);
      return seconds>=0 ? 'lease +' + fmtDur(seconds) : 'lease expired ' + fmtDur(Math.abs(seconds)) + ' ago';
    }
    var exp=leaseExpiresAt(w);
    return exp ? 'lease exp '+fmtTs(exp) : 'lease —';
  }
  function reasonLabel(reason){
    if(!reason) return 'runnable; waiting for worker tick';
    if(reason==='worker_capacity') return 'waiting for an idle worker';
    if(reason==='container_exclusive_conflict') return 'waiting for container lock';
    if(reason==='cache_namespace_conflict') return 'waiting for cache namespace';
    if(reason==='manual_step') return 'manual step required';
    return reason;
  }
  function statusStat(label, value, cls){
    return "<div class='status-stat'><div class='k'>"+esc(label)+"</div><div class='v "+(cls||'')+"'>"+esc(value)+"</div></div>";
  }
  function workerChip(w){
    var stale=isStaleWorker(w), status=w.request_status || w.state || 'active';
    var heartbeat=w.heartbeat_state || (stale ? 'stale' : 'heartbeating');
    var cls=stale ? 'fail' : (heartbeat==='heartbeating' || status==='leased' || status==='running' ? 'run' : 'warn');
    var title=[
      w.worker_id||'',
      'status '+status,
      'heartbeat '+heartbeat,
      expiryLabel(w),
      'worker progress '+(fmtTs(w.last_worker_progress_at || w.last_progress_at)),
      'run event '+(fmtTs(w.last_run_event_at))
    ].join(' · ');
    var bits=["<span class='chip "+cls+"' title='"+esc(title)+"'>",
      "<strong>"+esc(w.slot!=null ? '#'+w.slot : 'worker')+"</strong>",
      esc(status),
      esc(heartbeat),
      esc(shortRunId(w.run_id))];
    var progress=agoLabel(w.last_worker_progress_at || w.last_progress_at);
    if(progress) bits.push('worker '+esc(progress));
    var runEvent=agoLabel(w.last_run_event_at);
    if(runEvent) bits.push('event '+esc(runEvent));
    bits.push(esc(expiryLabel(w)));
    if(stale) bits.push(esc(w.stale_reason || 'stale lease'));
    bits.push("</span>");
    return bits.join(' ');
  }
  function queuedReasonItem(item){
    var cls=item.reason ? 'warn' : 'run';
    var age=agoLabel(item.submitted_at);
    var why=item.why_not_running || reasonLabel(item.reason);
    if(item.blocked_by_run_id) why += ' by '+shortRunId(item.blocked_by_run_id);
    return "<div class='queue-item'><div class='rid' title='"+esc(item.run_id)+"'>"+esc(shortRunId(item.run_id))+"</div>"
      +"<div class='why "+cls+"'>"+esc(why)+"</div><div>"+esc(age||'—')+"</div></div>";
  }
  function renderRunStatus(data){
    var el=document.getElementById('run-status');
    if(!el || !data) return;
    var s=data.summary||{}, c=serviceCounts(data);
    var workerTotal=s.service_worker_count || c.workers.length || 0;
    var activeWorkers=s.service_active_workers!=null ? s.service_active_workers : c.active.length;
    var idleWorkers=s.service_idle_workers!=null ? s.service_idle_workers : Math.max(0, workerTotal-activeWorkers);
    var workers=c.active.length ? c.active.map(workerChip).join('') : "<span class='chip'><strong>idle</strong> no active workers</span>";
    if(idleWorkers>0) workers += "<span class='chip'><strong>"+esc(idleWorkers)+"</strong> idle</span>";
    var queued=c.queuedItems.length
      ? "<div class='queue-list'>"+c.queuedItems.slice(0,8).map(queuedReasonItem).join('')+"</div>"
      : "<span class='chip'><strong>empty</strong> no queued runs</span>";
    var moreQueued=c.queuedItems.length>8 ? "<span class='chip warn'><strong>+"+esc(c.queuedItems.length-8)+"</strong> more queued</span>" : "";
    var storage=data.storage_health||{}, storageSummary=storage.summary||{}, storageAlerts=storage.alerts||[];
    var warnings=[];
    if(c.stale) warnings.push(c.stale+' stale lease'+(c.stale===1?'':'s'));
    if(c.blocked) warnings.push(c.blocked+' blocked queued');
    if(storageAlerts.length){
      var top=storageAlerts[0]||{};
      var target=top.run_id || top.root || top.path || 'workspace';
      warnings.push(storageAlerts.length+' storage alert'+(storageAlerts.length===1?'':'s')+' · '+(top.kind||'storage')+' '+fmtBytes(top.bytes)+' '+target);
    }
    el.innerHTML="<div class='status-head'><div class='status-title'>Run Status</div>"
      +"<div class='status-note'>last progress "+esc(ageOrDash(s.service_last_progress_at))
      +" · board poll "+esc(ageOrDash(data.generated_at))+"</div></div>"
      +"<div class='status-grid'>"
      +statusStat('Queued', c.queued+' total', c.queued ? 'warn' : '')
      +statusStat('Leased', c.leased, c.leased ? 'run' : '')
      +statusStat('Running', c.rawRunning, c.rawRunning ? 'run' : '')
      +statusStat('Terminal', c.terminal, c.terminal ? 'ok' : '')
      +statusStat('Workers', activeWorkers+'/'+workerTotal, activeWorkers ? 'run' : '')
      +statusStat('Oldest Queued', fmtDur(s.service_oldest_queued_age_seconds), c.queued ? 'warn' : '')
      +statusStat('Storage', fmtBytes(storageSummary.bytes), storageAlerts.length ? 'warn' : '')
      +statusStat('Stale Partials', fmtBytes(storageSummary.stale_partial_bytes), storageSummary.stale_partial_bytes ? 'warn' : '')
      +"</div>"
      +"<div class='status-lanes'><div><div class='status-lane-title'>Worker Occupancy</div>"
      +"<div class='chiprow'>"+workers+"</div></div>"
      +"<div><div class='status-lane-title'>Queued Runs: why not running</div>"+queued+moreQueued+"</div></div>"
      +(warnings.length ? "<div class='chiprow' style='margin-top:10px'><span class='chip fail'><strong>warn</strong> "+esc(warnings.join(' · '))+"</span></div>" : "");
  }
  function phaseLabel(r){
    var b=[], seen={};
    var active=r.active_evaluation||{};
    if(active.stage){ b.push('active '+active.stage); seen[active.stage]=1; }
    if(active.row_count){
      b.push('scored '+Number(active.scored_count||0).toLocaleString()+'/'+Number(active.row_count).toLocaleString());
    }
    if(active.candidate_evaluation_count){
      b.push(Number(active.candidate_evaluation_count).toLocaleString()+' evals');
    }
    var queues=r.queue_counts||{};
    var qb=Object.keys(queues).sort().filter(function(k){ return queues[k]; })
      .map(function(k){ return k+' '+queues[k]; });
    if(qb.length) b.push('queues '+qb.join(', '));
    if(r.worker_id) b.push('worker '+r.worker_id);
    if(r.heartbeat_state) b.push('heartbeat '+r.heartbeat_state);
    var workerAgo=agoLabel(r.worker_last_progress_at);
    if(workerAgo) b.push('worker progress '+workerAgo);
    var eventAgo=agoLabel(r.last_run_event_at);
    if(eventAgo) b.push('run event '+eventAgo);
    if(r.seconds_until_lease_expiry!=null && !isNaN(Number(r.seconds_until_lease_expiry))){
      var lease=Number(r.seconds_until_lease_expiry);
      b.push(lease>=0 ? 'lease +'+fmtDur(lease) : 'lease expired '+fmtDur(Math.abs(lease))+' ago');
    }
    if(r.stale_reason) b.push('stale '+r.stale_reason);
    if(r.blocked_reason){
      var blocked='blocked '+(r.why_not_running || r.blocked_reason);
      if(r.blocked_by_run_id) blocked += ' by '+r.blocked_by_run_id;
      b.push(blocked);
    } else if(r.scheduler_state==='runnable') {
      b.push('runnable');
    }
    [r.phase, r.stage].forEach(function(x){ if(x && !seen[x]){ b.push(x); seen[x]=1; } });
    if(r.generation!=null) b.push('gen '+r.generation);
    var ago=agoLabel(r.last_activity_at); if(ago) b.push(ago);
    return b.join(' · ');
  }
  function rowHtml(r){
    var liveish = r.state==='queued' || r.state==='running' || r.state==='paused';
    var detail='';
    if(r.failure){
      detail = "<div class='fail'>"+esc(r.failure.failure_type)+" · "+esc(r.failure.reason_code)
        +(r.failure.retryable?' · retryable':'')
        +"<div class='failmsg'>"+esc(r.failure.message)+"</div></div>";
    } else if(liveish){
      var ph=phaseLabel(r); if(ph) detail="<div class='phase'>"+esc(ph)+"</div>";
    }
    return "<tr data-run='"+esc(r.run_id)+"' data-state='"+r.state+"' data-domain='"+esc(r.domain)+"'>"
      +"<td><span class='pill "+r.state+"'>"+esc(r.state)+"</span></td>"
      +"<td class='mono'>"+esc(r.run_id)+detail+"</td>"
      +"<td>"+esc(r.domain)+"</td>"
      +"<td class='num'>"+fmtReward(r.best_train_reward)+"</td>"
      +"<td class='num'>"+fmtReward(r.best_heldout_reward)+"</td>"
      +"<td class='eta'>"+etaHtml(r.eta)+"</td>"
      +"<td class='num'>"+fmtDur(r.duration_seconds)+"</td>"
      +"<td class='storage-cell'>"+storageBadge(r)+"</td>"
      +"<td class='num'>"+fmtTokens((r.usage||{}).total_tokens)+"</td>"
      +"<td class='num'>"+fmtCost(r.cost_usd)+"</td>"
      +"<td class='ts'>"+fmtUpdated(r)+"</td></tr>";
  }
  function storageBadge(r){
    if(r.state==='succeeded' || r.state==='failed' || r.state==='cancelled'){
      return "<span class='storage-badge ready'>inspect</span>";
    }
    if(r.state==='queued' || r.state==='running' || r.state==='paused'){
      return "<span class='storage-badge live'>live</span>";
    }
    return "<span class='storage-badge'>unknown</span>";
  }
  function esc(s){ var d=document.createElement('div'); d.textContent=(s==null?'':String(s)); return d.innerHTML; }
  function apply(){
    var term=q.value.trim().toLowerCase(), state=st.value, domain=dm.value, shown=0;
    Array.prototype.forEach.call(body.querySelectorAll('tr'), function(r){
      var ok=(!state||r.dataset.state===state)&&(!domain||r.dataset.domain===domain)
        &&(!term||r.cells[1].textContent.toLowerCase().indexOf(term)>=0);
      r.hidden=!ok; if(ok) shown++;
    });
    empty.hidden = shown!==0;
  }
  function patch(data){
    boardData=data;
    renderRunStatus(data);
    var prev={}; Array.prototype.forEach.call(body.querySelectorAll('tr'), function(r){
      prev[r.dataset.run]=r.dataset.state+'|'+r.cells[3].textContent;
    });
    body.innerHTML = data.runs.map(rowHtml).join('');
    Array.prototype.forEach.call(body.querySelectorAll('tr'), function(r){
      var sig=r.dataset.state+'|'+r.cells[3].textContent;
      if(prev[r.dataset.run]!==undefined && prev[r.dataset.run]!==sig) r.classList.add('bump');
    });
    var s=data.summary;
    document.querySelectorAll('[data-k]').forEach(function(el){ el.textContent=s[el.dataset.k] || 0; });
    document.getElementById('subline').textContent =
      s.total+' run(s) · '+(s.queued||0)+' queued · '+(s.running||0)+' running'
      +' · workers '+(s.service_active_workers||0)+'/'+(s.service_worker_count||0)
      +' · queue '+(s.service_queued_runnable||0)+' runnable/'+(s.service_queued_blocked||0)+' blocked'
      +' · service running '+(s.service_running_count||0)
      +' · service progress '+ageOrDash(s.service_last_progress_at)
      +' · oldest queued '+fmtDur(s.service_oldest_queued_age_seconds)
      +' · board poll '+ageOrDash(data.generated_at)
      +' · '+(s.unknown||0)+' unknown · '+fmtCost(s.total_cost_usd)
      +' · '+(s.total_tokens==null||s.total_tokens===''? 'unknown tokens' : Number(s.total_tokens).toLocaleString()+' tokens');
    // keep domain filter options in sync
    var have={}; Array.prototype.forEach.call(dm.options, function(o){ have[o.value]=1; });
    Array.from(new Set(data.runs.map(function(r){return r.domain;}))).sort().forEach(function(d){
      if(!have[d]){ var o=document.createElement('option'); o.value=o.textContent=d; dm.appendChild(o); }
    });
    apply();
    if(curRun && !ws && filePollState){
      var selected = data.runs.find(function(r){ return r.run_id===curRun; });
      if(selected && selected.state!==filePollState) loadFileEvents(curRun, selected.state, eventsUrlForRun(selected, curRun));
    }
  }
  function connect(url){
    var dot=document.getElementById('dot'), txt=document.getElementById('livetext');
    var es=new EventSource(url);
    es.addEventListener('board', function(ev){ patch(JSON.parse(ev.data)); });
    es.onopen=function(){ dot.classList.add('on'); txt.textContent='live'; };
    es.onerror=function(){ dot.classList.remove('on'); txt.textContent='reconnecting…'; };
  }
  // ---- per-run drill-down -------------------------------------------------
  var drawer, scrim, dlog, dconn, dtitle, dsub, btnStop, btnCancel, ws=null, curRun=null;
  var fileTimer=null, filePollState=null, curRunSummary=null;
  var D=null, candCountEl, evCountEl, frameEl, candBodyEl, dmeta, runStateEl, queueEl, throughputEl, proposerRoundEl, waterfallEl, limitEtaEl, storageEl, curRunStorage=null;
  var RECENT=14;
  function shortId(id){ if(!id) return '—'; return String(id).replace(/^gepa_/,'').slice(0,10); }
  function newDetail(){ return {
    cands:{}, order:[], gen:null, frontier:[], best:null, events:[], task:null, container:null,
    candidateStats:{generated:{}, minibatch:{}, minibatchPassed:{}, frontierImproved:{}},
    budgets:{maxTotalRollouts:null, maxTrainRollouts:null, maxHeldoutRollouts:null},
    state:{from:null,to:null,trigger:null,message:null,stage:null,generation:null,rollouts:null,rows:null,candidates:null,at:null},
	    summary:{bestHeldout:null, durationSeconds:null},
	    limits:{snapshot:null, history:[]},
	    runtime:{rolloutCalls:0, rolloutWall:0, rolloutBatches:0, lastRolloutRate:null, lastRolloutCount:null,
      liveRolloutRate:null, liveScoredRollouts:null, liveRolloutStage:null, liveRolloutSamples:[],
      maxRolloutRate:null, lastRolloutStage:null, policyModel:null, policyCalls:null, policyTokens:null,
      policyJobs:null, proposerModel:null, proposerCalls:null, proposerTokens:null, proposerJobs:null,
      totalTokens:null, totalCost:null, totalCostSeeded:false, policyCost:null, proposerCost:null,
      workers:null, cacheHits:0, cacheMisses:0, dispatchChunks:0},
    throughput:{rollout:[], proposer:[], all:[], rolloutTokens:0, proposerTokens:0,
      rolloutWall:0, proposerWall:0, lastRollout:null, lastProposer:null},
    proposerRounds:{active:{}, order:[], totalWall:0},
    timings:{ids:{}, rows:[], summary:null, throughputBackfilled:{rollout:false, proposer:false}},
    queues:{
      rollout:{status:'idle', stage:null, generation:null, queued:0, running:0, completed:0,
        totalQueued:0, totalCompleted:0, batches:0, lastRate:null, liveRate:null, scored:null,
        candidateCount:null, rowCount:null},
      candidate:{status:'idle', generation:null, requested:0, returned:0, registered:0, minibatch:0,
        accepted:0, rejected:0, totalRequested:0, totalReturned:0, totalRegistered:0, totalMinibatch:0,
        totalAccepted:0, totalRejected:0, warnings:0, model:null, frontierSize:null, parent:null}
    }
  }; }
  function ensureCand(id){
    if(!id) return null;
    if(!D.cands[id]){ D.cands[id]={id:id, idx:D.order.length, gen:D.gen, parent:null, mb:null, train:null, heldout:null, result:null, reason:null, seed:false, timeline:[]}; D.order.push(id); }
    else if(D.cands[id].gen==null && D.gen!=null){ D.cands[id].gen=D.gen; }
    return D.cands[id];
  }
  function candidateMark(id, at, state, label, cls, details){
    var c=ensureCand(id);
    if(!c) return;
    c.timeline.push({t:at, state:state, label:label, cls:cls, details:details||''});
  }
  function markGeneratedCandidate(id){
    if(id) D.candidateStats.generated[id]=1;
  }
  function markMinibatchCandidate(id, passed){
    if(!id) return;
    D.candidateStats.generated[id]=1;
    D.candidateStats.minibatch[id]=1;
    if(passed) D.candidateStats.minibatchPassed[id]=1;
  }
  function markFrontierImprovedCandidate(id){
    if(!id) return;
    D.candidateStats.generated[id]=1;
    D.candidateStats.frontierImproved[id]=1;
  }
  function coverageStats(p){
    var c=p.coverage||{};
    var trainTotal=p.train_row_count||p.train_task_id_count||c.train_row_count||c.train_task_id_count;
    var reached=p.covered_train_example_count||p.covered_train_task_id_count
      ||c.covered_train_example_count||c.covered_train_task_id_count;
    var reachedPct=p.covered_train_example_percent||p.covered_train_task_id_percent
      ||c.covered_train_example_percent||c.covered_train_task_id_percent;
    var best=p.best_candidate_example_count||p.best_candidate_task_id_count
      ||c.best_candidate_example_count||c.best_candidate_task_id_count;
    var bestPct=p.best_candidate_example_coverage_percent||p.best_candidate_task_id_coverage_percent
      ||c.best_candidate_example_coverage_percent||c.best_candidate_task_id_coverage_percent;
    return {total:trainTotal, reached:reached, reachedPct:reachedPct, best:best, bestPct:bestPct};
  }
  function addRuntimeUsage(usage){
    if(!usage) return;
    if(usage.total_tokens!=null) D.runtime.totalTokens = Number(usage.total_tokens);
    if(usage.rollout_calls!=null) D.runtime.rolloutCalls = Number(usage.rollout_calls);
    if(usage.proposer_calls!=null) D.runtime.proposerCalls = Number(usage.proposer_calls);
  }
  function updateWorkers(p){
    var a=p.adaptive_rollout_concurrency||{};
    if(p.configured_rollout_workers!=null || p.static_rollout_workers!=null || a.current_limit!=null){
      D.runtime.workers={configured:p.configured_rollout_workers, static:p.static_rollout_workers,
        limit:a.current_limit, min:a.min_limit, max:a.max_limit};
    }
  }
  function setCandidateMinibatchRows(rollouts, candidates){
    if(!rollouts || !candidates || D.task && D.task.candidateMinibatchRows!=null) return;
    var n=Number(rollouts), c=Number(candidates);
    if(!n || !c) return;
    var per=n/c;
    if(per>0 && Math.abs(per-Math.round(per))<0.001){
      D.task=D.task||{};
      D.task.candidateMinibatchRows=Math.round(per);
    }
  }
  function findRun(runId){
    if(!boardData || !Array.isArray(boardData.runs)) return null;
    return boardData.runs.find(function(r){ return r.run_id===runId; }) || null;
  }
  function seedRunSummary(r){
    if(!r) return;
    D.summary.bestHeldout = r.best_heldout_reward!=null ? r.best_heldout_reward : D.summary.bestHeldout;
    D.summary.durationSeconds = r.duration_seconds!=null ? r.duration_seconds : D.summary.durationSeconds;
    D.state.to = r.state || D.state.to;
    D.state.stage = r.stage || D.state.stage;
    if(r.generation!=null){ D.gen=r.generation; D.state.generation=r.generation; }
    if(r.cost_usd!=null){ D.runtime.totalCost = Number(r.cost_usd); D.runtime.totalCostSeeded=true; }
    if(r.timing_summary && typeof r.timing_summary==='object'){
      D.timings.summary=r.timing_summary;
    }
    if(r.task && typeof r.task==='object'){
      D.task=D.task||{};
      if(r.task.train_rows!=null) D.task.trainRows=Number(r.task.train_rows);
      if(r.task.heldout_rows!=null) D.task.heldoutRows=Number(r.task.heldout_rows);
      if(r.task.minibatch_rows!=null) D.task.candidateMinibatchRows=Number(r.task.minibatch_rows);
      if(r.task.proposals_per_generation!=null) D.task.proposalsPerGeneration=Number(r.task.proposals_per_generation);
    }
    if(r.budgets && typeof r.budgets==='object'){
      if(r.budgets.max_total_rollouts!=null) D.budgets.maxTotalRollouts=Number(r.budgets.max_total_rollouts);
      if(r.budgets.max_train_rollouts!=null) D.budgets.maxTrainRollouts=Number(r.budgets.max_train_rollouts);
      if(r.budgets.max_heldout_rollouts!=null) D.budgets.maxHeldoutRollouts=Number(r.budgets.max_heldout_rollouts);
    }
    if(r.usage){
      D.runtime.totalTokens = r.usage.total_tokens!=null ? Number(r.usage.total_tokens) : D.runtime.totalTokens;
      D.runtime.rolloutCalls = r.usage.rollout_calls!=null ? Number(r.usage.rollout_calls) : D.runtime.rolloutCalls;
      D.runtime.proposerCalls = r.usage.proposer_calls!=null ? Number(r.usage.proposer_calls) : D.runtime.proposerCalls;
    }
  }
  function updateQueuesFromTransition(p, dt){
    var rq=D.queues.rollout, cq=D.queues.candidate;
    if(p.to==='proposing' || p.trigger==='proposer_started'){
      cq.status='proposing';
      cq.generation=dt.generation!=null ? dt.generation : cq.generation;
      cq.requested=Number(dt.proposal_count||0);
      cq.totalRequested += cq.requested;
      cq.model=dt.model||cq.model;
      cq.frontierSize=dt.frontier_size!=null ? dt.frontier_size : cq.frontierSize;
      cq.parent=dt.parent_candidate_id||cq.parent;
    } else if(p.trigger==='proposer_finished'){
      cq.status='returned';
      cq.generation=dt.generation!=null ? dt.generation : cq.generation;
      cq.returned=Number(dt.proposal_count||0);
    }
    if(p.trigger==='rollouts_queued' || p.to==='rollout_queueing'){
      rq.status='queued';
      rq.stage=dt.stage||rq.stage;
      rq.generation=dt.generation!=null ? dt.generation : rq.generation;
      rq.queued=Number(dt.rollout_count||0);
      rq.running=0;
      rq.totalQueued += rq.queued;
      rq.candidateCount=dt.candidate_count!=null ? dt.candidate_count : rq.candidateCount;
      rq.rowCount=dt.row_count!=null ? dt.row_count : rq.rowCount;
    } else if(p.trigger==='rollouts_started' || p.to==='rollout_running'){
      rq.status='running';
      rq.stage=dt.stage||rq.stage;
      rq.generation=dt.generation!=null ? dt.generation : rq.generation;
      rq.running=Number(dt.rollout_count||rq.queued||0);
      rq.candidateCount=dt.candidate_count!=null ? dt.candidate_count : rq.candidateCount;
      rq.rowCount=dt.row_count!=null ? dt.row_count : rq.rowCount;
    } else if(p.trigger==='rollouts_finished'){
      rq.status='evaluating';
      rq.running=0;
      rq.stage=dt.stage||rq.stage;
    } else if(p.to==='completed' || p.to==='succeeded'){
      rq.status='idle';
      rq.running=0;
      cq.status='complete';
    }
  }
  function roundKey(gen){
    return gen==null ? 'unknown' : String(gen);
  }
  function startProposerRound(ev, dt){
    var gen=dt.generation!=null ? dt.generation : D.state.generation;
    var key=roundKey(gen);
    var active=D.proposerRounds.active[key];
    if(!active){
      active={generation:gen, start:eventMs(ev), end:null, duration:null, proposalCount:null,
        model:null, cost:null, warnings:null};
      D.proposerRounds.active[key]=active;
      D.proposerRounds.order.push(active);
    }
    active.start=eventMs(ev);
    active.model=dt.model||active.model;
    active.proposalCount=dt.proposal_count!=null ? Number(dt.proposal_count) : active.proposalCount;
  }
  function finishProposerRound(ev, p){
    var gen=p.generation!=null ? p.generation : D.state.generation;
    var key=roundKey(gen);
    var active=D.proposerRounds.active[key];
    if(!active){
      active={generation:gen, start:null, end:null, duration:null, proposalCount:null,
        model:null, cost:null, warnings:null};
      D.proposerRounds.active[key]=active;
      D.proposerRounds.order.push(active);
    }
    active.end=eventMs(ev);
    if(active.start!=null && active.end>=active.start){
      var prior=active.duration||0;
      active.duration=(active.end-active.start)/1000;
      D.proposerRounds.totalWall += active.duration-prior;
    }
    active.model=p.model||active.model;
    active.proposalCount=p.proposal_count!=null ? Number(p.proposal_count) : active.proposalCount;
    active.cost=p.cost_usd!=null ? Number(p.cost_usd) : active.cost;
    active.warnings=p.warning_count!=null ? Number(p.warning_count) : active.warnings;
  }
  function mergeProposerRuntime(p){
    var gen=p.generation!=null ? p.generation : D.state.generation;
    var key=roundKey(gen);
    var active=D.proposerRounds.active[key];
    if(!active){
      active={generation:gen, start:null, end:null, duration:null, proposalCount:null,
        model:null, cost:null, warnings:null};
      D.proposerRounds.active[key]=active;
      D.proposerRounds.order.push(active);
    }
    active.runtimeWall=p.wall_seconds!=null ? Number(p.wall_seconds) : active.runtimeWall;
    active.tokens=p.total_tokens!=null ? Number(p.total_tokens) : active.tokens;
    active.tpm=p.tokens_per_minute!=null ? Number(p.tokens_per_minute) : active.tpm;
    active.tps=p.tokens_per_second!=null ? Number(p.tokens_per_second) : active.tps;
    active.model=p.model||active.model;
    active.proposalCount=p.proposal_count!=null ? Number(p.proposal_count) : active.proposalCount;
    active.cost=p.cost_usd!=null ? Number(p.cost_usd) : active.cost;
  }
  function proposerRoundSeconds(r){
    return r.runtimeWall!=null && !isNaN(Number(r.runtimeWall)) ? Number(r.runtimeWall) : r.duration;
  }
  function eventMs(ev){
    var raw=ev.ts || (ev.payload && ev.payload.ts) || null;
    if(!raw) return Date.now();
    var t=parseTs(raw);
    return isNaN(t) ? Date.now() : t;
  }
  function recordRolloutProgress(ev, p){
    if(p.scored_rollouts==null) return;
    var scored=Number(p.scored_rollouts);
    if(isNaN(scored)) return;
    var stage=p.stage || p.active_stage || D.queues.rollout.stage || D.runtime.liveRolloutStage || null;
    var rq=D.queues.rollout;
    rq.scored = rq.scored==null ? scored : Math.max(rq.scored, scored);
    rq.stage = stage || rq.stage;
    if(rq.status==='idle' || rq.status==='queued' || rq.status==='running') rq.status='scoring';
    D.runtime.liveScoredRollouts = D.runtime.liveScoredRollouts==null
      ? scored : Math.max(D.runtime.liveScoredRollouts, scored);
    if(D.runtime.liveRolloutStage!==stage){
      D.runtime.liveRolloutStage=stage;
      D.runtime.liveRolloutSamples=[];
    }
    var t=eventMs(ev);
    var samples=D.runtime.liveRolloutSamples;
    var prev=samples.length ? samples[samples.length-1] : null;
    if(!prev || scored>=prev.scored){
      samples.push({t:t, scored:scored});
    } else {
      samples.length=0;
      samples.push({t:t, scored:scored});
    }
    var cutoff=t-60000;
    while(samples.length>2 && samples[0].t<cutoff) samples.shift();
    while(samples.length>30) samples.shift();
    var first=samples[0], last=samples[samples.length-1];
    if(first && last && last.scored>first.scored && last.t>first.t){
      var rate=(last.scored-first.scored)/((last.t-first.t)/1000)*60;
      D.runtime.liveRolloutRate=rate;
      rq.liveRate=rate;
    }
  }
  function recordThroughputSample(ev, p, kind){
    var toks=Number(p.total_tokens||0), wall=Number(p.wall_seconds||0);
    var tpm=p.tokens_per_minute!=null ? Number(p.tokens_per_minute) : null;
    var tps=p.tokens_per_second!=null ? Number(p.tokens_per_second) : null;
    if((tpm==null || isNaN(tpm)) && toks>0 && wall>0) tpm=toks/wall*60;
    if((tps==null || isNaN(tps)) && toks>0 && wall>0) tps=toks/wall;
    if((tpm==null || isNaN(tpm)) && (tps==null || isNaN(tps))) return;
    var sample={kind:kind, t:eventMs(ev), seq:D.events.length+1, tokens:toks, wall:wall,
      tps:tps, tpm:tpm, model:p.model||null, stage:p.stage||p.active_stage||null,
      generation:p.generation!=null ? p.generation : null};
    D.throughput[kind].push(sample);
    D.throughput.all.push(sample);
    if(kind==='rollout'){
      D.throughput.rolloutTokens += toks;
      D.throughput.rolloutWall += wall;
      D.throughput.lastRollout=sample;
    } else {
      D.throughput.proposerTokens += toks;
      D.throughput.proposerWall += wall;
      D.throughput.lastProposer=sample;
    }
  }
  function timingMs(record){
    var raw=record.finished_at || record.started_at || record.recorded_at;
    var t=raw ? parseTs(raw) : Date.now();
    return isNaN(t) ? Date.now() : t;
  }
  function applyTimingSummary(summary){
    if(!summary) return;
    if(summary.rollout_count!=null) D.runtime.rolloutCalls=Math.max(D.runtime.rolloutCalls||0, Number(summary.rollout_count));
    if(summary.rollout_batch_count!=null) D.runtime.rolloutBatches=Math.max(D.runtime.rolloutBatches||0, Number(summary.rollout_batch_count));
    if(summary.rollout_total_seconds!=null) D.runtime.rolloutWall=Math.max(D.runtime.rolloutWall||0, Number(summary.rollout_total_seconds));
    if(summary.proposer_round_count!=null) D.runtime.proposerJobs=Math.max(D.runtime.proposerJobs||0, Number(summary.proposer_round_count));
  }
  function mergeTimingThroughput(record){
    var kind=record.lane;
    if(kind!=='rollout' && kind!=='proposer') return;
    if(D.throughput[kind].length && !D.timings.throughputBackfilled[kind]) return;
    var toks=Number(record.total_tokens||0), wall=Number(record.wall_seconds||0);
    if(!toks || !wall) return;
    var sample={kind:kind, t:timingMs(record), seq:D.throughput.all.length+1, tokens:toks, wall:wall,
      tps:toks/wall, tpm:toks/wall*60, model:record.metadata&&record.metadata.model||null,
      stage:record.stage||null, generation:record.generation!=null ? record.generation : null};
    D.throughput[kind].push(sample);
    D.throughput.all.push(sample);
    if(kind==='rollout'){
      D.timings.throughputBackfilled.rollout=true;
      D.throughput.rolloutTokens += toks;
      D.throughput.rolloutWall += wall;
      D.throughput.lastRollout=sample;
    } else {
      D.timings.throughputBackfilled.proposer=true;
      D.throughput.proposerTokens += toks;
      D.throughput.proposerWall += wall;
      D.throughput.lastProposer=sample;
    }
  }
  function mergeTimingProposer(record){
    if(record.lane!=='proposer') return;
    var gen=record.generation!=null ? record.generation : D.state.generation;
    var key=roundKey(gen);
    var active=D.proposerRounds.active[key];
    if(!active){
      active={generation:gen, start:null, end:null, duration:null, proposalCount:null,
        model:null, cost:null, warnings:null};
      D.proposerRounds.active[key]=active;
      D.proposerRounds.order.push(active);
    }
    active.start=record.started_at ? parseTs(record.started_at) : active.start;
    active.end=record.finished_at ? parseTs(record.finished_at) : active.end;
    active.runtimeWall=record.wall_seconds!=null ? Number(record.wall_seconds) : active.runtimeWall;
    active.tokens=record.total_tokens!=null ? Number(record.total_tokens) : active.tokens;
    active.cost=record.cost_usd!=null ? Number(record.cost_usd) : active.cost;
    active.proposalCount=record.item_count!=null ? Number(record.item_count) : active.proposalCount;
    if(record.metadata && record.metadata.model) active.model=record.metadata.model;
  }
  function applyTimingRows(payload){
    if(!payload || !Array.isArray(payload.timings)) return;
    D.timings.summary=payload.summary||D.timings.summary;
    applyTimingSummary(D.timings.summary);
    payload.timings.forEach(function(record){
      var id=record.timing_id || [record.lane,record.kind,record.started_at,record.finished_at].join(':');
      if(D.timings.ids[id]) return;
      D.timings.ids[id]=true;
      D.timings.rows.push(record);
      if(record.lane==='proposer') mergeTimingProposer(record);
      mergeTimingThroughput(record);
    });
  }
  function recordRuntimeJob(ev, p){
    var lane=p.lane || p.runtime_kind || p.effect_kind || '';
    var isRollout=lane==='rollout' || lane==='rollout_batch' || lane==='container_rollout'
      || p.rollout_count!=null || p.active_stage==='rollout';
    var isProposer=lane==='proposer' || lane==='candidate_proposal' || p.proposal_count!=null;
    if(p.stage || p.active_stage) D.state.stage=p.stage||p.active_stage;
    if(p.generation!=null){ D.gen=p.generation; D.state.generation=p.generation; }
    if(p.model && isRollout) D.runtime.policyModel=p.model;
    if(p.model && isProposer) D.runtime.proposerModel=p.model;
    if(p.total_tokens!=null){
      var toks=Number(p.total_tokens);
      if(D.runtime.totalTokens==null) D.runtime.totalTokens=0;
      D.runtime.totalTokens += toks;
      if(isRollout){ if(D.runtime.policyTokens==null) D.runtime.policyTokens=0; D.runtime.policyTokens += toks; }
      if(isProposer){ if(D.runtime.proposerTokens==null) D.runtime.proposerTokens=0; D.runtime.proposerTokens += toks; }
    }
    if(p.cost_usd!=null){
      var cost=Number(p.cost_usd);
      if(!isNaN(cost)){
        if(D.runtime.totalCost==null) D.runtime.totalCost=0;
        if(!D.runtime.totalCostSeeded) D.runtime.totalCost += cost;
        if(isRollout){ if(D.runtime.policyCost==null) D.runtime.policyCost=0; D.runtime.policyCost += cost; }
        if(isProposer){ if(D.runtime.proposerCost==null) D.runtime.proposerCost=0; D.runtime.proposerCost += cost; }
      }
    }
    D.runtime.cacheHits += Number(p.cache_hits||0);
    D.runtime.cacheMisses += Number(p.cache_misses||0);
    D.runtime.dispatchChunks += Number(p.dispatch_chunk_count||0);
    updateWorkers(p);
    if(isRollout){
      var n=Number(p.rollout_count||0), wall=Number(p.wall_seconds||0);
      if(n){ D.runtime.rolloutCalls += n; D.runtime.lastRolloutCount=n; }
      if(wall){ D.runtime.rolloutWall += wall; D.runtime.rolloutBatches += 1; }
      if(n && wall){
        var rate=n/wall*60;
        D.runtime.lastRolloutRate=rate;
        D.runtime.maxRolloutRate=D.runtime.maxRolloutRate==null ? rate : Math.max(D.runtime.maxRolloutRate, rate);
        D.queues.rollout.lastRate=rate;
      }
      D.runtime.lastRolloutStage=p.stage||p.active_stage||D.runtime.lastRolloutStage;
      D.queues.rollout.completed += n;
      D.queues.rollout.totalCompleted += n;
      D.queues.rollout.batches += 1;
      D.queues.rollout.stage=p.stage||p.active_stage||D.queues.rollout.stage;
      D.queues.rollout.generation=p.generation!=null ? p.generation : D.queues.rollout.generation;
      D.queues.rollout.candidateCount=p.candidate_count!=null ? p.candidate_count : D.queues.rollout.candidateCount;
      recordThroughputSample(ev, p, 'rollout');
    } else if(isProposer){
      if(D.runtime.proposerJobs==null) D.runtime.proposerJobs=0;
      D.runtime.proposerJobs += 1;
      if(p.proposal_count!=null){
        if(D.runtime.proposerCalls==null) D.runtime.proposerCalls=0;
        D.runtime.proposerCalls += Number(p.proposal_count);
      }
      mergeProposerRuntime(p);
      recordThroughputSample(ev, p, 'proposer');
    }
  }
  function applyRuntimeSummary(p){
    var rs=p.runtime_summary||{}, policy=rs.policy||{}, proposer=rs.proposer||{};
    if(policy.model) D.runtime.policyModel=policy.model;
    if(policy.calls!=null) D.runtime.policyCalls=Number(policy.calls);
    if(policy.total_tokens!=null) D.runtime.policyTokens=Number(policy.total_tokens);
    if(policy.jobs!=null) D.runtime.policyJobs=Number(policy.jobs);
    if(policy.cost_usd!=null) D.runtime.policyCost=Number(policy.cost_usd);
    if(policy.wall_seconds!=null) D.runtime.rolloutWall=Number(policy.wall_seconds);
    if(proposer.model) D.runtime.proposerModel=proposer.model;
    if(proposer.calls!=null) D.runtime.proposerCalls=Number(proposer.calls);
    if(proposer.total_tokens!=null) D.runtime.proposerTokens=Number(proposer.total_tokens);
    if(proposer.jobs!=null) D.runtime.proposerJobs=Number(proposer.jobs);
    if(proposer.cost_usd!=null) D.runtime.proposerCost=Number(proposer.cost_usd);
    if(p.cost_usd!=null) D.runtime.totalCost=Number(p.cost_usd);
    if(p.heldout_reward!=null) D.summary.bestHeldout=p.heldout_reward;
    addRuntimeUsage(p.usage);
    if(p.rollout_count!=null){
      D.runtime.rolloutCalls=Number(p.rollout_count);
      D.queues.rollout.totalCompleted=Number(p.rollout_count);
    }
  }
  function ingest(ev){
    var t=ev.kind, p=ev.payload||{};
    if(t==='optimizer.state.transitioned'){ var dt=p.details||{};
      D.state.from=p.from||D.state.from; D.state.to=p.to||D.state.to; D.state.trigger=p.trigger||D.state.trigger;
      D.state.message=p.message||D.state.message; D.state.at=ev.ts||p.at||D.state.at;
      if(dt.generation!=null){ D.gen=dt.generation; D.state.generation=dt.generation; }
      if(dt.stage) D.state.stage=dt.stage;
      if(dt.rollout_count!=null) D.state.rollouts=dt.rollout_count;
      if(dt.row_count!=null) D.state.rows=dt.row_count;
      if(dt.candidate_count!=null) D.state.candidates=dt.candidate_count;
      if(dt.stage==='candidate_minibatch'){
        setCandidateMinibatchRows(dt.rollout_count, dt.candidate_count);
      }
      if(dt.heldout_rows!=null || dt.train_rows!=null || dt.minibatch_rows!=null){
        D.task=D.task||{};
        if(dt.train_rows!=null) D.task.trainRows=dt.train_rows;
        if(dt.minibatch_rows!=null) D.task.minibatchRows=dt.minibatch_rows;
        if(dt.heldout_rows!=null) D.task.heldoutRows=dt.heldout_rows;
      }
      updateQueuesFromTransition(p, dt);
      if(dt.candidate_id){
        var stage=dt.stage||p.to||'state';
        var cls=stage==='heldout' ? 'final' : (p.to==='evaluating' ? 'score' : 'rollout');
        candidateMark(dt.candidate_id, eventMs(ev), stage, p.to||stage, cls, p.message||'');
      }
      if(p.trigger==='proposer_started' || p.to==='proposing') startProposerRound(ev, dt);
    }
    else if(t==='proposer.completed'){ if(p.generation!=null) D.gen=p.generation;
      D.queues.candidate.status='returned';
      D.queues.candidate.generation=p.generation!=null ? p.generation : D.queues.candidate.generation;
      D.queues.candidate.returned=Number(p.proposal_count||0);
      D.queues.candidate.totalReturned += D.queues.candidate.returned;
      D.queues.candidate.warnings += Number(p.warning_count||0);
      finishProposerRound(ev, p);
    }
    else if(t==='container.task_info.loaded'){ var ds=p.dataset||{}, ev2=p.evaluation||{};
      var task=p.task||{}, ts=p.taskset||{}, sampling=ts.sampling||{}, os=p.output_space||{};
      D.task={dataset:ds.dataset_id||task.name||task.task_id, taskId:task.task_id, metric:ev2.primary_metric,
        split:ds.default_split||ts.default_split, trainRows:sampling.train_sample, heldoutRows:sampling.test_sample,
        minibatchRows:D.task&&D.task.minibatchRows!=null ? D.task.minibatchRows : sampling.minibatch_sample,
        totalRows:ts.row_count, labels:os.label_count}; }
    else if(t==='taskset.tasks.loaded'){
      D.task=D.task||{};
      if(p.train_rows!=null) D.task.trainRows=p.train_rows;
      if(p.minibatch_rows!=null) D.task.minibatchRows=p.minibatch_rows;
      if(p.heldout_rows!=null) D.task.heldoutRows=p.heldout_rows;
    }
    else if(t==='gepa.run.started'){ if(p.container_url) D.container=p.container_url; }
    else if(t==='runtime.job.completed'){ recordRuntimeJob(ev, p); }
    else if(t==='rollout.failure_rate.updated'){ recordRolloutProgress(ev, p); }
    else if(t==='gepa.run.finished'){ D.state.to='completed'; applyRuntimeSummary(p); }
    else if(t==='candidate.registered'){ var c=ensureCand(p.candidate_id);
      if(p.generation!=null) c.gen=p.generation;
      if(p.source==='seed'){ c.result='seed'; c.seed=true; c.gen=0; }
      else { D.queues.candidate.registered += 1; D.queues.candidate.totalRegistered += 1; D.queues.candidate.status='registered'; }
      candidateMark(p.candidate_id, eventMs(ev), 'created', p.source==='seed' ? 'seed' : 'created', 'created', p.message||''); }
    else if(t==='candidate.evaluated'){ var c2=ensureCand(p.candidate_id); if(p.train_reward!=null) c2.train=p.train_reward;
      candidateMark(p.candidate_id, eventMs(ev), 'train_scored', 'train '+fmtReward(p.train_reward), 'score', p.message||''); }
    else if(t==='candidate.minibatch_evaluated'){ var c3=ensureCand(p.candidate_id);
      if(p.minibatch_reward!=null) c3.mb=p.minibatch_reward; if(p.parent_id) c3.parent=p.parent_id;
      markMinibatchCandidate(p.candidate_id, p.accepted_minibatch===true);
      candidateMark(p.candidate_id, eventMs(ev), 'minibatch_scored', 'mb '+fmtReward(p.minibatch_reward), 'score', p.message||'');
      D.queues.candidate.minibatch += 1; D.queues.candidate.totalMinibatch += 1; D.queues.candidate.status='scoring'; }
    else if(t==='candidate.accepted' || t==='candidate.rejected'){ var c4=ensureCand(p.candidate_id);
      var score=p.score||{};
      var stage=score.evaluation_stage||p.evaluation_stage||null;
      c4.result=(t==='candidate.accepted')?'accepted':(stage==='candidate_full_train'?'rejected_full_train':(stage==='candidate_minibatch'?'rejected_minibatch':'rejected'));
      c4.reason=p.reason||null;
      if(p.candidate_minibatch_reward!=null) c4.mb=p.candidate_minibatch_reward;
      if(p.candidate_train_reward!=null) c4.train=p.candidate_train_reward; if(p.parent_id) c4.parent=p.parent_id;
      if(p.accepted_minibatch===true || score.accepted_minibatch===true || stage==='candidate_minibatch' && t==='candidate.accepted'){
        markMinibatchCandidate(p.candidate_id, true);
      } else if(stage==='candidate_minibatch' || p.candidate_minibatch_reward!=null){
        markMinibatchCandidate(p.candidate_id, false);
      } else {
        markGeneratedCandidate(p.candidate_id);
      }
      if(t==='candidate.accepted' && (p.accepted_full_train===true || score.accepted_full_train===true || stage==='candidate_full_train' || p.candidate_train_reward!=null)){
        markFrontierImprovedCandidate(p.candidate_id);
      }
      candidateMark(p.candidate_id, eventMs(ev), c4.result, resultLabel(c4.result), c4.result==='accepted' ? 'accepted' : 'rejected', p.reason||p.message||'');
      if(t==='candidate.accepted'){ D.queues.candidate.accepted += 1; D.queues.candidate.totalAccepted += 1; }
      else { D.queues.candidate.rejected += 1; D.queues.candidate.totalRejected += 1; }
    }
    else if(t==='heldout.completed'){ var c5=ensureCand(p.candidate_id);
      if(p.heldout_reward!=null){ c5.heldout=p.heldout_reward;
        D.summary.bestHeldout = D.summary.bestHeldout==null ? p.heldout_reward : Math.max(D.summary.bestHeldout, p.heldout_reward); }
      if(p.train_reward!=null) c5.train=p.train_reward;
      candidateMark(p.candidate_id, eventMs(ev), 'heldout', 'heldout '+fmtReward(p.heldout_reward), 'final', p.message||''); }
    else if(t==='score_chart.written' && Array.isArray(p.candidates)){
      p.candidates.forEach(function(sc){ var cc=ensureCand(sc.candidate_id);
        if(sc.index!=null) cc.idx=sc.index;
        if(sc.source==='seed'){ cc.result='seed'; cc.seed=true; }
        else markGeneratedCandidate(sc.candidate_id);
        if(sc.status==='accepted'){ cc.result='accepted'; markFrontierImprovedCandidate(sc.candidate_id); }
        else if(sc.status && String(sc.status).indexOf('rejected')>=0) cc.result=sc.status;
        if(sc.status==='accepted' || sc.status==='rejected_full_train') markMinibatchCandidate(sc.candidate_id, true);
        if(sc.train_reward!=null) cc.train=sc.train_reward;
        if(sc.heldout_reward!=null) cc.heldout=sc.heldout_reward;
        if(sc.is_best) cc.best=true; }); }
    else if(t==='frontier.updated' || t==='frontier.snapshot'){
      var prev=D.best; D.best=p.best_train_reward;
      var cov=coverageStats(p);
      D.frontier.push({gen:D.gen, best:p.best_train_reward, count:p.candidate_count,
        best_id:p.best_candidate_id, changed:p.changed_candidate_id,
        idx:p.candidate_count!=null ? Number(p.candidate_count)-1 : null,
        frontierSize:p.frontier_size, reached:cov.reached, reachedPct:cov.reachedPct,
        bestCount:cov.best, bestPct:cov.bestPct, total:cov.total,
        improved:(prev!=null && p.best_train_reward!=null && p.best_train_reward>prev)});
    }
  }
  function aMin(a){ return a.reduce(function(m,v){return v<m?v:m;}, a[0]); }
  function aMax(a){ return a.reduce(function(m,v){return v>m?v:m;}, a[0]); }
  function axNum(v){ return (Math.round(v*1000)/1000).toString(); }
  function resColor(r){ return r==='accepted'?'#22c55e':r==='seed'?'#6a9bcc':(r==='rejected' || String(r||'').indexOf('rejected')===0)?'#f87171':'#d97757'; }
  function resultLabel(r){
    if(r==='rejected_minibatch') return 'rejected minibatch';
    if(r==='rejected_full_train') return 'rejected full train';
    return r || 'pending';
  }
  function fact(label, value){
    if(value==null || value==='') return '';
    return "<span class='fact'>"+esc(label)+" "+esc(value)+"</span>";
  }
  function pct(v){ return v==null ? null : (Math.round(Number(v)*10)/10).toFixed(1)+'%'; }
  function eventFacts(p){
    var cov=coverageStats(p), u=p.usage||{};
    var bits=[
      fact('seeds reached', cov.reached!=null ? cov.reached+(cov.total!=null?'/'+cov.total:'') : null),
      fact('reached', pct(cov.reachedPct)),
      fact('best per-seed reward', cov.best!=null ? cov.best+(cov.total!=null?'/'+cov.total:'') : null),
      fact('best reward', pct(cov.bestPct)),
      fact('tokens', u.total_tokens!=null ? Number(u.total_tokens).toLocaleString() : null),
      fact('rollouts', u.rollout_calls),
      fact('proposer calls', u.proposer_calls)
    ].filter(Boolean);
    return bits.length ? "<div class='facts'>"+bits.join('')+"</div>" : "";
  }
  var CW=560, CH=120, CL=42, CR=12, CT=10, CB=22;
  function chartShell(inner){
    return "<svg viewBox='0 0 "+CW+" "+CH+"' style='display:block;width:100%;height:auto'>"
      +"<line x1='"+CL+"' y1='"+CT+"' x2='"+CL+"' y2='"+(CH-CB)+"' stroke='#1f2937'/>"
      +"<line x1='"+CL+"' y1='"+(CH-CB)+"' x2='"+(CW-CR)+"' y2='"+(CH-CB)+"' stroke='#1f2937'/>"+inner+"</svg>";
  }
  function yLabels(ymin,ymax,Y){
    return "<text x='4' y='"+(Y(ymax)+3).toFixed(1)+"' fill='#9ca3af' font-size='9'>"+axNum(ymax)+"</text>"
      +"<text x='4' y='"+(Y(ymin)+3).toFixed(1)+"' fill='#9ca3af' font-size='9'>"+axNum(ymin)+"</text>";
  }
  function heldoutByGeneration(){
    var best={};
    D.order.forEach(function(id){
      var c=D.cands[id];
      if(c.heldout==null) return;
      var g=c.gen==null?0:Number(c.gen);
      if(best[g]==null || c.heldout>best[g]) best[g]=c.heldout;
    });
    return Object.keys(best).map(function(g){ return {gen:Number(g), heldout:best[g]}; })
      .sort(function(a,b){ return a.gen-b.gen; });
  }
  function minibatchByGeneration(){
    var best={};
    D.order.forEach(function(id){
      var c=D.cands[id];
      if(c.mb==null) return;
      var g=c.gen==null?0:Number(c.gen);
      if(best[g]==null || c.mb>best[g]) best[g]=c.mb;
    });
    return Object.keys(best).map(function(g){ return {gen:Number(g), minibatch:best[g]}; })
      .sort(function(a,b){ return a.gen-b.gen; });
  }
  function bestTrainSvg(){
    var fs=D.frontier.filter(function(f){return f.best!=null;});
    var hs=heldoutByGeneration();
    var ms=minibatchByGeneration();
    if(!fs.length && !hs.length && !ms.length) return "<div class='dnone'>no data yet</div>";
    var trainYs=fs.map(function(f){return f.best;});
    var minibatchYs=ms.map(function(m){return m.minibatch;});
    var heldoutYs=hs.map(function(h){return h.heldout;});
    var ys=trainYs.concat(minibatchYs).concat(heldoutYs);
    var ymin=aMin(ys.concat([0])), ymax=aMax(ys.concat([1])); if(ymax===ymin) ymax=ymin+1;
    var gens=fs.map(function(f){return f.gen==null?0:Number(f.gen);})
      .concat(ms.map(function(m){return m.gen;}))
      .concat(hs.map(function(h){return h.gen;}));
    var gmin=aMin(gens), gmax=aMax(gens);
    function XGen(g){ return CL+(CW-CL-CR)*(gmax<=gmin?0.5:(g-gmin)/(gmax-gmin)); }
    function Y(v){ return CT+(CH-CT-CB)*(1-(v-ymin)/(ymax-ymin)); }
    var trainPath=fs.map(function(f,i){ var x=XGen(f.gen==null?0:Number(f.gen));
      return (i?'L':'M')+x.toFixed(1)+' '+Y(f.best).toFixed(1); }).join(' ');
    var trainLine=trainPath?"<path d='"+trainPath+"' fill='none' stroke='#22c55e' stroke-width='1.5'/>":"";
    var trainDots=fs.map(function(f){ var x=XGen(f.gen==null?0:Number(f.gen));
      return "<circle cx='"+x.toFixed(1)+"' cy='"+Y(f.best).toFixed(1)+"' r='2.6' fill='#22c55e'><title>train best "+fmtReward(f.best)+"</title></circle>"; }).join('');
    var miniPath=ms.map(function(m,i){ return (i?'L':'M')+XGen(m.gen).toFixed(1)+' '+Y(m.minibatch).toFixed(1); }).join(' ');
    var miniLine=miniPath?"<path d='"+miniPath+"' fill='none' stroke='#FF5C00' stroke-width='1.4' stroke-dasharray='2 3'/>":"";
    var miniDots=ms.map(function(m){ return "<circle cx='"+XGen(m.gen).toFixed(1)+"' cy='"+Y(m.minibatch).toFixed(1)+"' r='2.8' fill='#FF5C00'><title>minibatch best "+fmtReward(m.minibatch)+"</title></circle>"; }).join('');
    var heldPath=hs.map(function(h,i){ return (i?'L':'M')+XGen(h.gen).toFixed(1)+' '+Y(h.heldout).toFixed(1); }).join(' ');
    var heldLine=heldPath?"<path d='"+heldPath+"' fill='none' stroke='#d97757' stroke-width='1.4' stroke-dasharray='4 3'/>":"";
    var heldDots=hs.map(function(h){ return "<circle cx='"+XGen(h.gen).toFixed(1)+"' cy='"+Y(h.heldout).toFixed(1)+"' r='2.8' fill='#d97757'><title>heldout best "+fmtReward(h.heldout)+"</title></circle>"; }).join('');
    var seen={}; var xlab=gens.sort(function(a,b){return a-b;}).map(function(g){ if(seen[g])return ''; seen[g]=1;
      return "<text x='"+XGen(g).toFixed(1)+"' y='"+(CH-6)+"' fill='#9ca3af' font-size='9' text-anchor='middle'>"+(g===0?'seed':g)+"</text>"; }).join('');
    var legend="<g transform='translate("+(CW-226)+" 12)'><circle cx='0' cy='0' r='3' fill='#22c55e'/>"
      +"<text x='7' y='3' fill='#9ca3af' font-size='9'>train</text>"
      +"<circle cx='56' cy='0' r='3' fill='#FF5C00'/>"
      +"<text x='63' y='3' fill='#9ca3af' font-size='9'>minibatch</text>"
      +"<circle cx='134' cy='0' r='3' fill='#d97757'/>"
      +"<text x='141' y='3' fill='#9ca3af' font-size='9'>heldout</text></g>";
    return chartShell(trainLine+miniLine+heldLine+trainDots+miniDots+heldDots+xlab+yLabels(ymin,ymax,Y)+legend);
  }
  function minibatchSvg(){
    var cs=D.order.map(function(id){return D.cands[id];}).filter(function(c){return c.mb!=null;});
    if(!cs.length) return "<div class='dnone'>no minibatch scores yet</div>";
    var gens=cs.map(function(c){return c.gen==null?-1:c.gen;});
    var gmin=aMin(gens), gmax=aMax(gens);
    var ys=cs.map(function(c){return c.mb;});
    var ymin=aMin(ys.concat([0])), ymax=aMax(ys.concat([1])); if(ymax===ymin) ymax=ymin+1;
    function X(g){ return CL+(CW-CL-CR)*(gmax<=gmin?0.5:(g-gmin)/(gmax-gmin)); }
    function Y(v){ return CT+(CH-CT-CB)*(1-(v-ymin)/(ymax-ymin)); }
    var counts={};
    var dots=cs.map(function(c){ var g=c.gen==null?-1:c.gen; var j=(counts[g]=(counts[g]||0)+1)-1;
      var jit=(j%2?1:-1)*Math.ceil(j/2)*4;
      return "<circle cx='"+(X(g)+jit).toFixed(1)+"' cy='"+Y(c.mb).toFixed(1)+"' r='3' fill='"+resColor(c.result)+"' opacity='0.85'/>"; }).join('');
    var seen={}; var xlab=gens.map(function(g){ if(seen[g])return ''; seen[g]=1;
      return "<text x='"+X(g).toFixed(1)+"' y='"+(CH-6)+"' fill='#9ca3af' font-size='9' text-anchor='middle'>"+(g<0?'seed':g)+"</text>"; }).join('');
    return chartShell(dots+xlab+yLabels(ymin,ymax,Y));
  }
  function paretoSvg(){
    var fs=D.frontier.filter(function(f){return f.idx!=null && (f.best!=null || f.reachedPct!=null || f.bestPct!=null);});
    if(!fs.length) return "<div class='dnone'>no aggregate stats yet</div>";
    var xs=fs.map(function(f){return Number(f.idx);});
    var ys=[];
    fs.forEach(function(f){
      if(f.best!=null) ys.push(f.best);
      if(f.reachedPct!=null) ys.push(Number(f.reachedPct)/100);
      if(f.bestPct!=null) ys.push(Number(f.bestPct)/100);
    });
    var xmin=aMin(xs), xmax=aMax(xs);
    var ymin=aMin(ys.concat([0])), ymax=aMax(ys.concat([1])); if(ymax===ymin) ymax=ymin+1;
    function X(i){ return CL+(CW-CL-CR)*(xmax<=xmin?0.5:(Number(i)-xmin)/(xmax-xmin)); }
    function Y(v){ return CT+(CH-CT-CB)*(1-(v-ymin)/(ymax-ymin)); }
    function line(key, color, dash, title){
      var pts=fs.filter(function(f){return f[key]!=null;}).map(function(f){
        var y=key==='best' ? f[key] : Number(f[key])/100;
        return {idx:f.idx, y:y, raw:f[key]};
      });
      var path=pts.map(function(p,i){ return (i?'L':'M')+X(p.idx).toFixed(1)+' '+Y(p.y).toFixed(1); }).join(' ');
      var stroke=path?"<path d='"+path+"' fill='none' stroke='"+color+"' stroke-width='1.4'"+(dash?" stroke-dasharray='"+dash+"'":"")+"/>":"";
      var dots=pts.map(function(p){ return "<circle cx='"+X(p.idx).toFixed(1)+"' cy='"+Y(p.y).toFixed(1)+"' r='2.6' fill='"+color+"'><title>"+title+" @ idx "+p.idx+" "+(key==='best'?fmtReward(p.raw):pct(p.raw))+"</title></circle>"; }).join('');
      return stroke+dots;
    }
    var seen={}; var xlab=xs.map(function(i){ if(seen[i])return ''; seen[i]=1;
      return "<text x='"+X(i).toFixed(1)+"' y='"+(CH-6)+"' fill='#9ca3af' font-size='9' text-anchor='middle'>"+i+"</text>"; }).join('');
    var legend="<g transform='translate("+(CW-262)+" 12)'><circle cx='0' cy='0' r='3' fill='#22c55e'/>"
      +"<text x='7' y='3' fill='#9ca3af' font-size='9'>best train</text>"
      +"<circle cx='78' cy='0' r='3' fill='#FF5C00'/>"
      +"<text x='85' y='3' fill='#9ca3af' font-size='9'>pareto reached</text>"
      +"<circle cx='180' cy='0' r='3' fill='#d97757'/>"
      +"<text x='187' y='3' fill='#9ca3af' font-size='9'>best seed</text></g>";
    return chartShell(
      line('best','#22c55e',null,'best train')
      +line('reachedPct','#FF5C00','2 3','pareto reached')
      +line('bestPct','#d97757','4 3','best per-seed reward')
      +xlab+yLabels(ymin,ymax,Y)+legend
    );
  }
  function countText(n,total){ return n!=null ? n+(total!=null?'/'+total:'') : '—'; }
  function fmtInt(v){ return v==null || isNaN(Number(v)) ? '—' : Number(v).toLocaleString(); }
  function fmtRate(v){ return v==null || isNaN(Number(v)) ? '—' : (Math.round(Number(v)*10)/10).toLocaleString()+'/min'; }
  function fmtTok(v){ return v==null || isNaN(Number(v)) ? '—' : Number(v).toLocaleString()+' tok'; }
  function objCount(o){ return Object.keys(o||{}).length; }
  function fmtPctCount(n,d){ return d ? fmtInt(n)+'/'+fmtInt(d)+' ('+(Number(n)/Number(d)*100).toFixed(1)+'%)' : '—'; }
  function trajectoryCount(){
    return D.order.reduce(function(n,id){
      var c=D.cands[id];
      return n+(c && c.timeline && c.timeline.length ? 1 : 0);
    }, 0);
  }
  function compactUrl(u){
    if(!u) return '—';
    try{ var x=new URL(u); return x.host; }catch(e){ return String(u); }
  }
  function stat(label, value, cls){
    return "<div class='stat'><div class='k'>"+esc(label)+"</div><div class='v "+(cls||'')+"' title='"+esc(value)+"'>"+esc(value)+"</div></div>";
  }
  function fmtTps(v){
    return v==null || isNaN(Number(v)) ? '—' : (Math.round(Number(v)*10)/10).toLocaleString()+'/s';
  }
  function avgTpm(tokens, wall){
    return tokens && wall ? tokens/wall*60 : null;
  }
  function avgTps(tokens, wall){
    return tokens && wall ? tokens/wall : null;
  }
  function throughputSvg(){
    var pts=D.throughput.all.filter(function(p){return p.tpm!=null && !isNaN(Number(p.tpm));});
    if(!pts.length) return "<div class='dnone'>no throughput samples yet</div>";
    var xs=pts.map(function(p){return p.seq;});
    var ys=pts.map(function(p){return Number(p.tpm);});
    var xmin=aMin(xs), xmax=aMax(xs);
    var ymin=0, ymax=aMax(ys); if(ymax<=0) ymax=1;
    function X(i){ return CL+(CW-CL-CR)*(xmax<=xmin?0.5:(Number(i)-xmin)/(xmax-xmin)); }
    function Y(v){ return CT+(CH-CT-CB)*(1-(Number(v)-ymin)/(ymax-ymin)); }
    function series(kind,color,title){
      var ss=pts.filter(function(p){return p.kind===kind;});
      var path=ss.map(function(p,i){ return (i?'L':'M')+X(p.seq).toFixed(1)+' '+Y(p.tpm).toFixed(1); }).join(' ');
      var line=path?"<path d='"+path+"' fill='none' stroke='"+color+"' stroke-width='1.5'/>":"";
      var dots=ss.map(function(p){ return "<circle cx='"+X(p.seq).toFixed(1)+"' cy='"+Y(p.tpm).toFixed(1)+"' r='2.8' fill='"+color+"'><title>"+title+" "+fmtRate(p.tpm)+" · "+fmtTps(p.tps)+" · "+fmtTok(p.tokens)+" · "+fmtDur(p.wall)+"</title></circle>"; }).join('');
      return line+dots;
    }
    var xlab=pts.filter(function(_,i){return i===0 || i===pts.length-1;}).map(function(p){
      return "<text x='"+X(p.seq).toFixed(1)+"' y='"+(CH-6)+"' fill='#9ca3af' font-size='9' text-anchor='middle'>#"+p.seq+"</text>";
    }).join('');
    var legend="<g transform='translate("+(CW-150)+" 12)'><circle cx='0' cy='0' r='3' fill='#FF5C00'/>"
      +"<text x='7' y='3' fill='#9ca3af' font-size='9'>rollout</text>"
      +"<circle cx='70' cy='0' r='3' fill='#d97757'/>"
      +"<text x='77' y='3' fill='#9ca3af' font-size='9'>proposer</text></g>";
    return chartShell(series('rollout','#FF5C00','rollout')+series('proposer','#d97757','proposer')
      +xlab+yLabels(ymin,ymax,Y)+legend);
  }
  function limitEtaSvg(){
    var hist=(D.limits.history||[]).filter(function(sample){ return sample && sample.limits && sample.limits.length; });
    if(!hist.length) return "<div class='dnone'>no limit forecast samples yet</div>";
    var pts=[];
    hist.forEach(function(sample){
      sample.limits.forEach(function(item){
        var seconds=item.seconds_to_limit;
        if(seconds==null || isNaN(Number(seconds))) return;
        pts.push({
          t:sample.t,
          label:item.label,
          seconds:Number(seconds),
          low:item.seconds_to_limit_low==null ? null : Number(item.seconds_to_limit_low),
          high:item.seconds_to_limit_high==null ? null : Number(item.seconds_to_limit_high),
          confidence:item.confidence||'',
          utilization:item.utilization,
          remaining:item.remaining
        });
      });
    });
    if(!pts.length) return "<div class='dnone'>limits exist, forecast ETA not available yet</div>";
    var times=pts.map(function(p){return p.t;}), ys=pts.map(function(p){return p.seconds;});
    pts.forEach(function(p){ if(p.low!=null && !isNaN(p.low)) ys.push(p.low); if(p.high!=null && !isNaN(p.high)) ys.push(p.high); });
    var tmin=aMin(times), tmax=aMax(times); if(tmax<=tmin) tmax=tmin+1000;
    var ymin=0, ymax=aMax(ys); if(ymax<=0) ymax=1;
    function X(t){ return CL+(CW-CL-CR)*((Number(t)-tmin)/(tmax-tmin)); }
    function Y(v){ return CT+(CH-CT-CB)*(1-(Number(v)-ymin)/(ymax-ymin)); }
    var labels=[];
    pts.forEach(function(p){ if(labels.indexOf(p.label)<0) labels.push(p.label); });
    var colors=['#FF5C00','#22c55e','#d97757','#6a9bcc','#f87171','#a78bfa'];
    function color(label){ var i=labels.indexOf(label); return colors[(i<0?0:i)%colors.length]; }
    function series(label){
      var ss=pts.filter(function(p){return p.label===label;}).sort(function(a,b){return a.t-b.t;});
      var path=ss.map(function(p,i){ return (i?'L':'M')+X(p.t).toFixed(1)+' '+Y(p.seconds).toFixed(1); }).join(' ');
      var c=color(label);
      var band=ss.map(function(p){
        if(p.low==null || p.high==null || isNaN(p.low) || isNaN(p.high)) return '';
        return "<line x1='"+X(p.t).toFixed(1)+"' y1='"+Y(p.low).toFixed(1)+"' x2='"+X(p.t).toFixed(1)+"' y2='"+Y(p.high).toFixed(1)+"' stroke='"+c+"' stroke-opacity='0.25'/>";
      }).join('');
      var line=path?"<path d='"+path+"' fill='none' stroke='"+c+"' stroke-width='1.5'/>":"";
      var dots=ss.map(function(p){
        var title=label+" ETA "+fmtDur(p.seconds)
          +(p.low!=null && p.high!=null ? " ["+fmtDur(p.low)+"–"+fmtDur(p.high)+"]" : "")
          +(p.confidence ? " · "+p.confidence : "")
          +(p.utilization!=null ? " · "+pct(Number(p.utilization)*100)+" used" : "");
        return "<circle cx='"+X(p.t).toFixed(1)+"' cy='"+Y(p.seconds).toFixed(1)+"' r='2.7' fill='"+c+"'><title>"+esc(title)+"</title></circle>";
      }).join('');
      return band+line+dots;
    }
    var xlab=hist.filter(function(_,i){return i===0 || i===hist.length-1;}).map(function(s){
      return "<text x='"+X(s.t).toFixed(1)+"' y='"+(CH-6)+"' fill='#9ca3af' font-size='9' text-anchor='middle'>"+esc(fmtTs(s.at))+"</text>";
    }).join('');
    var legend="<g transform='translate("+(CW-190)+" 12)'>"+labels.slice(0,3).map(function(label,i){
      return "<circle cx='0' cy='"+(i*12)+"' r='3' fill='"+color(label)+"'/>"
        +"<text x='7' y='"+(i*12+3)+"' fill='#9ca3af' font-size='9'>"+esc(label)+"</text>";
    }).join('')+"</g>";
    return chartShell(labels.map(series).join('')+xlab+yLabels(ymin,ymax,Y)+legend);
  }
  function renderLimitEtaChart(){
    if(!limitEtaEl) return;
    limitEtaEl.innerHTML=limitEtaSvg();
  }
  function renderProposerRounds(){
    if(!proposerRoundEl) return;
    var rounds=D.proposerRounds.order.filter(function(r){
      return r && (r.start!=null || r.end!=null || r.runtimeWall!=null || r.duration!=null);
    });
    if(!rounds.length){
      proposerRoundEl.innerHTML="<div class='dnone'>no proposer rounds yet</div>";
      return;
    }
    var total=rounds.reduce(function(s,r){ return s+(proposerRoundSeconds(r)||0); },0);
    var avg=rounds.length ? total/rounds.length : null;
    var head="<div class='frow'><span class='same'>•</span><span class='g'>total</span>"
      +"<span class='best'>"+esc(fmtDur(total))+"</span>"
      +"<span class='cnt'>avg "+esc(fmtDur(avg))+"</span>"
      +"<span class='cnt'>"+esc(fmtInt(rounds.length))+" rounds</span></div>";
    var rows=rounds.map(function(r){
      var secs=proposerRoundSeconds(r);
      var perf = r.tpm!=null ? fmtRate(r.tpm) : (r.tps!=null ? fmtTps(r.tps) : '');
      var cost = r.cost!=null ? fmtCost(r.cost) : '';
      return "<div class='frow'><span class='up'>▲</span>"
        +"<span class='g'>gen "+esc(r.generation!=null ? r.generation : '—')+"</span>"
        +"<span class='best'>"+esc(fmtDur(secs))+"</span>"
        +"<span class='cnt'>"+esc(r.proposalCount!=null ? fmtInt(r.proposalCount)+' proposals' : 'proposals —')+"</span>"
        +"<span class='cnt'>"+esc(perf)+"</span>"
        +"<span class='cnt'>"+esc(cost)+"</span>"
        +"<span class='creason'>"+esc(r.model||'')+"</span></div>";
    }).join('');
    proposerRoundEl.innerHTML=head+rows;
  }
  function renderThroughput(){
    var lr=D.throughput.lastRollout, lp=D.throughput.lastProposer;
    var rows=[
      stat('Rollout TPM', fmtRate(avgTpm(D.throughput.rolloutTokens,D.throughput.rolloutWall)),
        D.throughput.rolloutWall ? 'ok' : ''),
      stat('Rollout TPS', fmtTps(avgTps(D.throughput.rolloutTokens,D.throughput.rolloutWall)),
        D.throughput.rolloutWall ? 'ok' : ''),
      stat('Rollout last', lr ? fmtRate(lr.tpm)+' · '+fmtTps(lr.tps) : '—'),
      stat('Proposer TPM', fmtRate(avgTpm(D.throughput.proposerTokens,D.throughput.proposerWall)),
        D.throughput.proposerWall ? 'warn' : ''),
      stat('Proposer TPS', fmtTps(avgTps(D.throughput.proposerTokens,D.throughput.proposerWall)),
        D.throughput.proposerWall ? 'warn' : ''),
      stat('Proposer last', lp ? fmtRate(lp.tpm)+' · '+fmtTps(lp.tps) : '—')
    ];
    throughputEl.innerHTML=rows.join('');
    document.getElementById('chart-throughput').innerHTML=throughputSvg();
    renderProposerRounds();
  }
  function stateClass(s){
    if(!s) return 'warn';
    if(s==='completed' || s==='succeeded') return 'ok';
    if(s.indexOf('rollout')>=0 || s==='proposing' || s==='evaluating' || s==='ready' || s==='initializing') return 'run';
    return 'warn';
  }
  function queueLabel(){
    var s=D.state.to||'unknown';
    if(s==='completed' || s==='succeeded') return 'idle';
    if(s==='rollout_queueing') return 'queued '+fmtInt(D.state.rollouts);
    if(s==='rollout_running') return 'running '+fmtInt(D.state.rollouts||D.runtime.lastRolloutCount);
    if(D.runtime.lastRolloutStage) return D.runtime.lastRolloutStage+' done';
    return '—';
  }
  function workerLabel(){
    var w=D.runtime.workers; if(!w) return '—';
    var bits=[];
    if(w.configured!=null) bits.push('cfg '+w.configured);
    if(w.static!=null) bits.push('static '+w.static);
    if(w.limit!=null) bits.push('limit '+w.limit);
    return bits.length ? bits.join(' · ') : '—';
  }
  function taskLabel(){
    if(!D.task) return '—';
    var bits=[];
    if(D.task.dataset) bits.push(D.task.dataset);
    if(D.task.trainRows!=null) bits.push('train '+D.task.trainRows);
    if(D.task.candidateMinibatchRows!=null) bits.push('minibatch '+D.task.candidateMinibatchRows+'/cand');
    if(D.task.minibatchRows!=null) bits.push('pool '+D.task.minibatchRows);
    if(D.task.heldoutRows!=null) bits.push('heldout '+D.task.heldoutRows);
    if(D.task.labels!=null) bits.push(D.task.labels+' labels');
    return bits.length ? bits.join(' · ') : '—';
  }
  function dataLabel(){
    if(!D.task) return '—';
    var bits=[];
    if(D.task.trainRows!=null) bits.push('train '+fmtInt(D.task.trainRows));
    if(D.task.candidateMinibatchRows!=null) bits.push('minibatch '+fmtInt(D.task.candidateMinibatchRows)+'/cand');
    if(D.task.minibatchRows!=null) bits.push('pool '+fmtInt(D.task.minibatchRows));
    if(D.task.heldoutRows!=null) bits.push('heldout '+fmtInt(D.task.heldoutRows));
    if(D.task.labels!=null) bits.push(fmtInt(D.task.labels)+' labels');
    return bits.length ? bits.join(' · ') : '—';
  }
  function budgetLabel(){
    var b=D.budgets; if(!b) return '—';
    var bits=[];
    if(b.maxTrainRollouts!=null) bits.push('train '+fmtInt(b.maxTrainRollouts));
    if(b.maxHeldoutRollouts!=null) bits.push('heldout '+fmtInt(b.maxHeldoutRollouts));
    if(b.maxTotalRollouts!=null) bits.push('total '+fmtInt(b.maxTotalRollouts));
    return bits.length ? bits.join(' · ') : '—';
  }
  function pctNum(done,total){
    if(!total) return 0;
    return Math.max(0, Math.min(100, done/total*100));
  }
  function qstat(label,value){
    return "<div class='qstat'><div class='k'>"+esc(label)+"</div><div class='v'>"+esc(value)+"</div></div>";
  }
  function qstatIf(label,value){
    return value==null ? '' : qstat(label,value);
  }
  function qclass(status){
    if(status==='idle' || status==='complete') return 'ok';
    if(status==='queued' || status==='running' || status==='proposing' || status==='scoring') return 'run';
    if(status==='evaluating' || status==='registered' || status==='returned') return 'warn';
    return '';
  }
  function qcard(title, status, done, total, stats){
    var cls=qclass(status);
    return "<div class='queuecard'><div class='qtop'><div class='qtitle'>"+esc(title)+"</div>"
      +"<div class='qstate "+cls+"'>"+esc(status||'unknown')+"</div></div>"
      +"<div class='qbar'><div class='qfill "+cls+"' style='width:"+pctNum(done,total).toFixed(1)+"%'></div></div>"
      +"<div class='qstats'>"+stats.join('')+"</div></div>";
  }
  function renderQueues(){
    var rq=D.queues.rollout, cq=D.queues.candidate;
    var rolloutTotal=Math.max(rq.totalQueued||0, rq.totalCompleted||0, rq.queued||0, rq.running||0);
    var rolloutDone=rq.totalCompleted || rq.completed;
    var registeredFromList=D.order.map(function(id){return D.cands[id];}).filter(function(c){return !c.seed;}).length;
    var registered=Math.max(cq.totalRegistered||0, registeredFromList);
    var candTotal=Math.max(cq.totalRequested||0, registered, cq.requested||0);
    var candDone=(cq.totalAccepted||0)+(cq.totalRejected||0);
    var pending=Math.max(0, registered-candDone);
    var active=(rq.status==='queued' || rq.status==='running') ? (rq.running||rq.queued) : null;
    var rqStats=[
      qstatIf('Active batch', active ? fmtInt(active)+' rollouts' : null),
      qstatIf('Completed', rolloutTotal ? fmtInt(rolloutDone)+'/'+fmtInt(rolloutTotal) : null),
      qstatIf('Scored total', rq.scored!=null ? fmtInt(rq.scored) : null),
      qstatIf('Stage', rq.stage||null),
      qstatIf('Generation', rq.generation!=null ? rq.generation : null),
      qstatIf('Candidates', rq.candidateCount!=null ? fmtInt(rq.candidateCount) : null),
      qstatIf('Rows', rq.rowCount!=null ? fmtInt(rq.rowCount) : null),
      qstatIf('Batches', rq.batches ? fmtInt(rq.batches) : null),
      qstatIf('Live rate', rq.liveRate!=null ? fmtRate(rq.liveRate) : null),
      qstatIf('Last rate', rq.lastRate!=null ? fmtRate(rq.lastRate) : null)
    ].filter(Boolean);
    if(!rqStats.length) rqStats=[qstat('State','waiting for rollout events')];
    var proposalOpen=(cq.status==='proposing');
    var hasCandidateFlow=registered || cq.totalMinibatch || candDone || cq.totalReturned;
    var cqStats=[
      qstatIf('Requested', (cq.totalRequested||cq.requested) ? fmtInt(cq.totalRequested||cq.requested) : null),
      qstatIf('Returned', proposalOpen ? 'waiting' : (cq.totalReturned ? fmtInt(cq.totalReturned) : null)),
      qstatIf('Registered', hasCandidateFlow ? fmtInt(registered) : null),
      qstatIf('Minibatch scored', hasCandidateFlow ? fmtInt(cq.totalMinibatch) : null),
      qstatIf('Accepted', hasCandidateFlow ? fmtInt(cq.totalAccepted) : null),
      qstatIf('Rejected', hasCandidateFlow ? fmtInt(cq.totalRejected) : null),
      qstatIf('Pending', hasCandidateFlow ? fmtInt(pending) : null),
      qstatIf('Model', cq.model||D.runtime.proposerModel||null),
      qstatIf('Generation', cq.generation!=null ? cq.generation : null),
      qstatIf('Frontier', cq.frontierSize!=null ? fmtInt(cq.frontierSize) : null)
    ].filter(Boolean);
    if(!cqStats.length) cqStats=[qstat('State','waiting for candidate events')];
    queueEl.innerHTML=qcard('Rollout queue', rq.status, rolloutDone, rolloutTotal, rqStats)
      +qcard('Candidate queue', cq.status, candDone, candTotal, cqStats);
  }
	  function renderRunState(){
	    var s=D.state.to||'unknown';
	    var aggRate=(D.runtime.rolloutCalls && D.runtime.rolloutWall) ? D.runtime.rolloutCalls/D.runtime.rolloutWall*60 : null;
    var policyCalls=D.runtime.policyCalls!=null ? D.runtime.policyCalls : D.runtime.rolloutCalls;
    var policy=D.runtime.policyModel ? D.runtime.policyModel+' · '+fmtInt(policyCalls)+' calls' : fmtInt(policyCalls)+' calls';
    var proposer=D.runtime.proposerModel ? D.runtime.proposerModel+' · '+fmtInt(D.runtime.proposerCalls)+' calls'
      : (D.runtime.proposerCalls!=null ? fmtInt(D.runtime.proposerCalls)+' calls' : '—');
    var cache=(D.runtime.cacheHits || D.runtime.cacheMisses) ? fmtInt(D.runtime.cacheHits)+' hit · '+fmtInt(D.runtime.cacheMisses)+' miss' : '—';
    var generatedCandidates=objCount(D.candidateStats.generated);
    var minibatchPassed=objCount(D.candidateStats.minibatchPassed);
    var frontierImproved=objCount(D.candidateStats.frontierImproved);
    var trackedCandidates=D.order.length;
    var trajectories=trajectoryCount();
    var rows=[
      stat('State', s, stateClass(s)),
      stat('Best final', fmtReward(D.summary.bestHeldout), D.summary.bestHeldout!=null ? 'ok' : ''),
      stat('Time taken', fmtDur(D.summary.durationSeconds)),
      stat('Candidates', generatedCandidates ? fmtInt(generatedCandidates) : '—'),
      stat('Minibatch pass', fmtPctCount(minibatchPassed, generatedCandidates), minibatchPassed ? 'ok' : ''),
      stat('Frontier improved', fmtPctCount(frontierImproved, generatedCandidates), frontierImproved ? 'ok' : ''),
      stat('Trajectories', trackedCandidates ? fmtPctCount(trajectories, trackedCandidates) : '—', trackedCandidates && trajectories===trackedCandidates ? 'ok' : (trackedCandidates ? 'warn' : '')),
      stat('Policy cost', fmtCost(D.runtime.policyCost), D.runtime.policyCost!=null ? 'ok' : ''),
      stat('Proposer cost', fmtCost(D.runtime.proposerCost), D.runtime.proposerCost!=null ? 'ok' : ''),
	      stat('Total cost', fmtCost(D.runtime.totalCost), D.runtime.totalCost!=null ? 'ok' : ''),
	      stat('Data', dataLabel()),
	      stat('Budget', budgetLabel()),
	      stat('Nearest ETA', limitEtaLabel(), D.limits.snapshot && D.limits.snapshot.nearest_limit ? 'run' : ''),
	      stat('Stage', D.state.stage||D.runtime.lastRolloutStage||'—'),
      stat('Generation', D.state.generation!=null ? D.state.generation : (D.gen!=null ? D.gen : '—')),
      stat('Live rollouts/min', fmtRate(D.runtime.liveRolloutRate), D.runtime.liveRolloutRate!=null ? 'run' : ''),
      stat('Rollouts/min', fmtRate(aggRate), aggRate!=null ? 'ok' : ''),
      stat('Last batch/min', fmtRate(D.runtime.lastRolloutRate)),
      stat('Policy', policy),
      stat('Proposer', proposer),
      stat('Tokens', fmtTok(D.runtime.totalTokens)),
      stat('Workers', workerLabel()),
      stat('Cache', cache),
      stat('Container', compactUrl(D.container)),
      stat('Task', taskLabel()),
      stat('Rollouts', fmtInt(D.runtime.rolloutCalls)),
      stat('Dispatch chunks', fmtInt(D.runtime.dispatchChunks))
    ];
    runStateEl.innerHTML=rows.join('');
  }
  function renderMeta(){
    var b=[]; if(D.task){ if(D.task.dataset) b.push('task '+D.task.dataset); if(D.task.metric) b.push('metric '+D.task.metric); }
    if(D.container) b.push(D.container);
    dmeta.textContent=b.join(' · ');
  }
  function waterfallColor(cls){
    if(cls==='accepted' || cls==='final') return '#22c55e';
    if(cls==='rejected') return '#f87171';
    if(cls==='rollout') return '#d97757';
    if(cls==='score') return '#6b7280';
    return '#FF5C00';
  }
  function dedupeMarks(marks){
    var seen={};
    return marks.filter(function(m){
      var key=[m.t,m.state,m.label,m.cls].join('|');
      if(seen[key]) return false;
      seen[key]=1;
      return true;
    }).sort(function(a,b){ return a.t-b.t; });
  }
  function candidateWaterfallSvg(){
    var rows=D.order.map(function(id){ return D.cands[id]; })
      .filter(function(c){ return c && c.timeline && c.timeline.length; })
      .map(function(c){ c.timeline=dedupeMarks(c.timeline); return c; });
    if(!rows.length) return "<div class='dnone'>no candidate state transitions yet</div>";
    rows.sort(function(a,b){
      var at=a.timeline[0] ? a.timeline[0].t : 0, bt=b.timeline[0] ? b.timeline[0].t : 0;
      if(at!==bt) return at-bt;
      return (a.idx||0)-(b.idx||0);
    });
    var times=[];
    rows.forEach(function(c){ c.timeline.forEach(function(m){ if(m.t!=null && !isNaN(m.t)) times.push(m.t); }); });
    var tmin=aMin(times), tmax=aMax(times); if(tmax<=tmin) tmax=tmin+1000;
    var left=104, right=16, top=24, rowH=24, bottom=28;
    var w=620, h=top+rows.length*rowH+bottom;
    function X(t){ return left+(w-left-right)*((Number(t)-tmin)/(tmax-tmin)); }
    function Y(i){ return top+i*rowH+8; }
    function age(t){ return fmtDur((Number(t)-tmin)/1000); }
    var out=[];
    out.push("<svg viewBox='0 0 "+w+" "+h+"' width='100%' height='"+h+"' role='img' aria-label='Candidate state waterfall'>");
    out.push("<line class='wf-axis' x1='"+left+"' y1='12' x2='"+(w-right)+"' y2='12'/>");
    out.push("<text class='wf-label' x='"+left+"' y='9' text-anchor='middle'>0s</text>");
    out.push("<text class='wf-label' x='"+(w-right)+"' y='9' text-anchor='end'>"+esc(age(tmax))+"</text>");
    rows.forEach(function(c,i){
      var y=Y(i), marks=c.timeline;
      var first=marks[0], last=marks[marks.length-1];
      var result=c.result||last.state||'pending';
      var label=shortId(c.id)+(c.seed?' seed':'');
      out.push("<text class='wf-id' x='8' y='"+(y+3)+"'>"+esc(label)+"</text>");
      out.push("<line class='wf-line' x1='"+X(first.t).toFixed(1)+"' y1='"+y+"' x2='"+X(last.t).toFixed(1)+"' y2='"+y+"'/>");
      for(var j=1;j<marks.length;j++){
        var a=marks[j-1], b=marks[j];
        out.push("<line class='wf-stage' x1='"+X(a.t).toFixed(1)+"' y1='"+y+"' x2='"+X(b.t).toFixed(1)+"' y2='"+y+"' stroke='"+waterfallColor(b.cls)+"'/>");
      }
      marks.forEach(function(m){
        var title=shortId(c.id)+" · "+m.label+" · +"+age(m.t)+(m.details?" · "+m.details:"");
        out.push("<circle class='wf-dot' cx='"+X(m.t).toFixed(1)+"' cy='"+y+"' r='4' fill='"+waterfallColor(m.cls)+"'><title>"+esc(title)+"</title></circle>");
      });
      out.push("<text class='wf-label' x='"+(w-right)+"' y='"+(y+3)+"' text-anchor='end'>"+esc(result)+"</text>");
    });
    out.push("</svg>");
    return out.join('');
  }
  function renderDetail(){
    renderMeta();
    renderRunState();
    renderLimitEtaChart();
    renderThroughput();
    renderQueues();
    document.getElementById('chart-best').innerHTML = bestTrainSvg();
    document.getElementById('chart-mb').innerHTML = minibatchSvg();
    document.getElementById('chart-pareto').innerHTML = paretoSvg();
    waterfallEl.innerHTML = candidateWaterfallSvg();
    // frontier timeline, newest first
    if(D.frontier.length){
      frameEl.innerHTML = D.frontier.slice().reverse().map(function(f){
        var mark = f.improved? "<span class='up'>▲</span>" : "<span class='same'>•</span>";
        return "<div class='frow'>"+mark
          +"<span class='g'>idx "+(f.idx==null?'—':f.idx)+"</span>"
          +"<span class='g'>gen "+(f.gen==null?'—':f.gen)+"</span>"
          +"<span class='best'>best "+fmtReward(f.best)+"</span>"
          +"<span class='cnt'>pareto "+countText(f.reached,f.total)+"</span>"
          +"<span class='cnt'>best-seed "+countText(f.bestCount,f.total)+"</span>"
          +"<span class='cnt'>frontier "+(f.frontierSize!=null?f.frontierSize:'—')+"</span>"
          +"<span class='cnt'>"+(f.count!=null?f.count+' cand':'')+"</span>"
          +"<span class='creason'>"+(f.changed?('Δ '+shortId(f.changed)):'')+"</span></div>";
      }).join('');
    }
    // candidates, newest first
    if(D.order.length){
      candCountEl.textContent = '('+D.order.length+')';
      candBodyEl.innerHTML = D.order.slice().reverse().map(function(id){
        var c=D.cands[id];
        var res = c.result==='accepted'? "<span class='res-acc'>accepted</span>"
          : (c.result==='rejected' || String(c.result||'').indexOf('rejected')===0)? "<span class='res-rej'>"+esc(resultLabel(c.result))+"</span>"
          : c.result==='seed'? "<span class='res-seed'>seed</span>"
          : "<span class='res-pend'>pending</span>";
        var reason = c.reason? " <span class='creason'>"+esc(c.reason)+"</span>" : "";
        var star = c.best? " <span class='res-acc'>★</span>" : "";
        return "<tr><td title='"+esc(c.id)+"'>"+esc(shortId(c.id))+star
          +(c.parent?" <span class='creason'>← "+esc(shortId(c.parent))+"</span>":"")+"</td>"
          +"<td class='num'>"+(c.idx==null?'—':c.idx)+"</td>"
          +"<td>"+(c.seed?'seed':(c.gen==null?'—':c.gen))+"</td>"
          +"<td class='num'>"+fmtReward(c.mb)+"</td>"
          +"<td class='num'>"+fmtReward(c.train)+"</td>"
          +"<td class='num'>"+fmtReward(c.heldout)+"</td>"
          +"<td>"+res+reason+"</td></tr>";
      }).join('');
    }
    // recent events, newest first, capped
    if(D.events.length){
      evCountEl.textContent='('+D.events.length+', showing '+Math.min(RECENT,D.events.length)+')';
      dlog.innerHTML = D.events.slice(-RECENT).reverse().map(function(ev){
        var p=ev.payload||{}; var msg=p.message||'';
        return "<div class='ev'><div class='meta'><span>#"+(ev.seq!=null?ev.seq:'')+"</span>"
          +"<span class='kind'>"+esc(ev.kind)+"</span><span>"+fmtTs(ev.ts)+"</span></div>"
          +(msg?("<div class='msg'>"+esc(msg)+"</div>"):"")
          +eventFacts(p)+"</div>";
      }).join('');
    }
  }
  function wsUrlForRun(r){
    if(r && r.ws_url) return r.ws_url;
    var u=window.__SERVICE_URL__;
    if(!u || !r || r.state!=='running') return null;
    return u.replace(/^http/,'ws').replace(/\/$/,'')+'/runs/'+encodeURIComponent(r.run_id)+'/ws';
  }
  function eventsUrlForRun(r, runId){
    if(r && r.events_url) return r.events_url;
    var base=window.__EVENTS_BASE__;
    return base ? base+'/'+encodeURIComponent(runId)+'/events' : null;
  }
	  function timingsUrlForRun(r, runId){
	    if(r && r.timings_url) return r.timings_url;
	    var base=window.__EVENTS_BASE__;
	    return base ? base+'/'+encodeURIComponent(runId)+'/timings' : null;
	  }
  function limitsUrlForRun(r, runId){
    if(r && r.limits_url) return r.limits_url;
    var base=window.__EVENTS_BASE__;
    return base ? base+'/'+encodeURIComponent(runId)+'/limits' : null;
  }
  function storageUrlForRun(r, runId){
    if(r && r.storage_url && window.__EVENTS_BASE__) return r.storage_url;
    var base=window.__EVENTS_BASE__;
    return base ? base+'/'+encodeURIComponent(runId)+'/storage' : null;
  }
  function runApiUrl(runId, suffix){
    var base=window.__EVENTS_BASE__;
    return base ? base+'/'+encodeURIComponent(runId)+(suffix||'') : null;
  }
  function fetchJson(url, opts){
    return fetch(url, opts).then(function(r){
      return r.text().then(function(text){
        var data={};
        if(text){ try{ data=JSON.parse(text); }catch(e){ data={error:text}; } }
        if(!r.ok){
          var msg=data.error || data.message || (data.error_code ? data.error_code : r.statusText);
          throw new Error(msg || ('HTTP '+r.status));
        }
        return data;
      });
    });
  }
  function fmtBytes(v){
    var n=Number(v||0), units=['B','KB','MB','GB','TB'];
    for(var i=0;i<units.length;i++){
      if(n<1024 || i===units.length-1) return units[i]==='B' ? Math.round(n)+'B' : n.toFixed(1)+units[i];
      n=n/1024;
    }
    return n.toFixed(1)+'TB';
  }
  function storageList(items, labelKey){
    if(!items || !items.length) return "<div class='dnone'>none</div>";
    return "<div class='storage-list'>"+items.slice(0,10).map(function(item){
      var label=item[labelKey] || item.relative_path || item.name || item.path || 'artifact';
      return "<div class='storage-row'><div class='bytes'>"+esc(fmtBytes(item.bytes))+"</div>"
        +"<div title='"+esc(item.path||label)+"'>"+esc(label)+"</div></div>";
    }).join('')+"</div>";
  }
  function renderStorageReport(report, preview){
    if(!storageEl) return;
    if(!report || report.error){
      storageEl.innerHTML="<div class='storage-warning'>"+esc(report&&report.error||'storage report failed')+"</div>";
      return;
    }
    var rec=report.recommendation||{}, terminal=!!report.terminal;
    var profile=rec.profile || 'compact';
    var safe=Array.isArray(report.safe_actions) ? report.safe_actions : [];
    var canCompact=terminal && safe.indexOf('compact')>=0;
    var canDelete=terminal && safe.indexOf('delete')>=0;
    var manifest=report.compaction_manifest||{};
    var checkpoint=report.checkpoint_compaction||{};
    var body="<div class='statgrid'>"
      +stat('Size', fmtBytes(report.bytes))
      +stat('Reclaimable', fmtBytes(report.reclaimable_bytes), report.reclaimable_bytes ? 'ok' : '')
      +stat('Status', (report.terminal_status||'unknown')+(terminal?' terminal':' live'), terminal?'ok':'run')
      +stat('Recommended', (rec.action||'none')+(rec.profile?':'+rec.profile:''), rec.action==='compact'?'ok':'')
      +stat('Runtime homes', fmtBytes(report.generated_runtime&&report.generated_runtime.bytes))
      +stat('Checkpoint reclaim', fmtBytes(checkpoint.estimated_reclaim_bytes))
      +"</div>"
      +"<div class='storage-note'>"+esc(rec.reason||'no recommendation')+"</div>";
    if(manifest.exists){
      body+="<div class='storage-note'>last compaction: "+esc(manifest.profile||'profile unknown')
        +" · reclaimed "+esc(fmtBytes(manifest.actual_reclaim_bytes))+"</div>";
    }
    if(preview){
      body+="<div class='storage-note'>dry run: "+esc(preview.profile||profile)
        +" would reclaim "+esc(fmtBytes(preview.estimated_reclaim_bytes))+"</div>"
        +storageList(preview.removed_paths||[], 'path');
    }
    body+="<div class='storage-actions'>"
      +"<button class='ctl' data-storage-action='dry-compact' data-profile='debug' "+(canCompact?'':'disabled')+">dry debug</button>"
      +"<button class='ctl' data-storage-action='dry-compact' data-profile='compact' "+(canCompact?'':'disabled')+">dry compact</button>"
      +"<button class='ctl' data-storage-action='dry-compact' data-profile='minimal' "+(canCompact?'':'disabled')+">dry minimal</button>"
      +"<button class='ctl' data-storage-action='apply-compact' data-profile='"+esc(preview&&preview.profile||profile)+"' "+(preview&&canCompact?'':'disabled')+">apply compact</button>"
      +"<button class='ctl' data-storage-action='delete' "+(canDelete?'':'disabled')+">delete</button>"
      +"</div>";
    if(!terminal) body+="<div class='storage-warning'>cleanup disabled until the run is terminal</div>";
    body+="<div class='chart-t'>Artifacts</div>"+storageList(report.artifact_summary||[], 'name')
      +"<div class='chart-t'>Top files</div>"+storageList(report.top_files||[], 'relative_path');
    var sqlite=report.sqlite||[];
    if(sqlite.length){
      body+="<div class='chart-t'>SQLite</div>"+sqlite.map(function(db){
        var rows=(db.objects||[]).slice(0,8).map(function(obj){
          return "<div class='storage-row'><div class='bytes'>"+esc(fmtBytes(obj.bytes))+"</div><div>"+esc(obj.name)+"</div></div>";
        }).join('');
        return "<div class='storage-note'>"+esc(db.path)+" · "+esc(fmtBytes(db.bytes))+"</div>"
          +(rows?"<div class='storage-list'>"+rows+"</div>":"<div class='dnone'>dbstat unavailable</div>");
      }).join('');
    }
    storageEl.innerHTML=body;
  }
  function loadStorageReport(runId){
    if(!storageEl) return;
    var url=storageUrlForRun(curRunSummary, runId);
    storageEl.innerHTML="<div class='dnone'>loading storage report...</div>";
    if(!url){ storageEl.innerHTML="<div class='dnone'>storage actions require the live board server</div>"; return; }
    fetchJson(url).then(function(report){
      if(curRun!==runId) return;
      curRunStorage=report;
      renderStorageReport(report);
    }).catch(function(error){
      if(curRun===runId) renderStorageReport({error:error.message});
    });
  }
  function refreshBoardSnapshot(){
    fetchJson('/api/runs').then(patch).catch(function(error){
      if(dconn) setConn('board refresh failed: '+error.message);
    });
  }
  function setStorageBusy(busy){
    if(!storageEl) return;
    storageEl.querySelectorAll('button').forEach(function(button){ button.disabled=busy || button.disabled; });
  }
  function compactStorage(profile, dryRun){
    if(!curRun) return;
    var url=runApiUrl(curRun, '/compact');
    if(!url) return;
    if(!dryRun && !confirm('Apply '+profile+' compaction to '+curRun+'?')) return;
    setStorageBusy(true);
    fetchJson(url, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({profile:profile, dry_run:dryRun})
    }).then(function(report){
      if(curRunStorage && dryRun){
        renderStorageReport(curRunStorage, report);
      } else {
        loadStorageReport(curRun);
        refreshBoardSnapshot();
      }
    }).catch(function(error){
      renderStorageReport({error:error.message});
    });
  }
  function deleteStorageRun(){
    if(!curRun) return;
    var typed=prompt('Type '+curRun+' to delete this terminal run and its local artifacts.');
    if(typed!==curRun) return;
    var url=runApiUrl(curRun, '');
    if(!url) return;
    setStorageBusy(true);
    fetchJson(url, {method:'DELETE'}).then(function(){
      closeDrawer();
      refreshBoardSnapshot();
    }).catch(function(error){
      renderStorageReport({error:error.message});
    });
  }
	  function applyLimitSnapshot(snapshot){
	    if(!snapshot || typeof snapshot !== 'object') return;
	    D.limits.snapshot=snapshot;
	    var rows=Array.isArray(snapshot.limits) ? snapshot.limits : [];
	    var at=snapshot.generated_at || (snapshot.nearest_limit && snapshot.nearest_limit.updated_at) || new Date().toISOString();
	    var t=Date.parse(at); if(isNaN(t)) t=Date.now();
	    var sample={at:at, t:t, limits:rows.map(function(status){
	      var definition=status.definition||{}, forecast=status.forecast||{};
	      return {
	        limit_id:definition.limit_id || forecast.limit_id || '',
	        label:limitKindLabel(definition.kind || forecast.kind),
	        seconds_to_limit:forecast.seconds_to_limit,
	        seconds_to_limit_low:forecast.seconds_to_limit_low,
	        seconds_to_limit_high:forecast.seconds_to_limit_high,
	        confidence:forecast.confidence,
	        utilization:status.utilization,
	        remaining:status.remaining
	      };
	    }).filter(function(item){ return item.limit_id || item.label; })};
	    if(sample.limits.length){
	      var history=D.limits.history||[];
	      var last=history.length ? history[history.length-1] : null;
	      if(!last || last.at!==sample.at || JSON.stringify(last.limits)!==JSON.stringify(sample.limits)){
	        history.push(sample);
	        if(history.length>80) history=history.slice(history.length-80);
	        D.limits.history=history;
	      }
	    }
	  }
	  function limitKindLabel(kind){
	    if(!kind) return 'limit';
	    if(typeof kind === 'string') return kind.replace(/_/g,' ');
	    if(kind.custom) return String(kind.custom).replace(/_/g,' ');
	    return 'limit';
	  }
	  function limitEtaLabel(){
	    var snap=D.limits.snapshot, nearest=snap && snap.nearest_limit;
	    if(!nearest) return '—';
	    var status=(snap.limits||[]).find(function(item){ return item.forecast && item.forecast.limit_id===nearest.limit_id; });
	    var kind=status && status.definition ? limitKindLabel(status.definition.kind) : 'limit';
	    var eta='unknown';
	    if(nearest.seconds_to_limit!=null){
	      eta=fmtDur(Number(nearest.seconds_to_limit));
	      if(nearest.seconds_to_limit_low!=null && nearest.seconds_to_limit_high!=null){
	        eta+=' ['+fmtDur(Number(nearest.seconds_to_limit_low))+'–'+fmtDur(Number(nearest.seconds_to_limit_high))+']';
	      }
	    }
	    var conf=nearest.confidence ? ' · '+nearest.confidence : '';
	    return kind+' · '+eta+conf;
	  }
  function closeDrawer(){
    if(ws){ try{ws.close();}catch(e){} ws=null; }
    if(fileTimer){ clearTimeout(fileTimer); fileTimer=null; }
    filePollState=null;
    curRun=null; curRunStorage=null; drawer.hidden=true; scrim.hidden=true;
    btnStop.hidden=true; btnCancel.hidden=true;
  }
  function onEvent(ev){ ingest(ev); D.events.push(ev); }
  function setConn(text, live){ dconn.innerHTML = live? "<span class='live'>● "+esc(text)+"</span>" : esc(text); }
  function openDrawer(runId, state){
    closeDrawer(); curRun=runId; drawer.hidden=false; scrim.hidden=false;
    dtitle.textContent=runId; dsub.textContent='';
    D=newDetail(); curRunSummary=findRun(runId); seedRunSummary(curRunSummary);
    evCountEl.textContent=''; dmeta.textContent='';
    dlog.innerHTML="<div class='dnone'>loading event feed…</div>";
    if(storageEl) storageEl.innerHTML="<div class='dnone'>loading storage report...</div>";
    renderRunState();
    throughputEl.innerHTML="<div class='dnone'>no throughput stats yet</div>";
    document.getElementById('chart-throughput').innerHTML="<div class='dnone'>no throughput samples yet</div>";
    document.getElementById('chart-limit-eta').innerHTML="<div class='dnone'>no limit forecast samples yet</div>";
    proposerRoundEl.innerHTML="<div class='dnone'>no proposer rounds yet</div>";
    queueEl.innerHTML="<div class='dnone'>loading queue events…</div>";
    frameEl.innerHTML="<div class='dnone'>loading frontier updates…</div>";
    document.getElementById('chart-best').innerHTML="<div class='dnone'>loading progress data…</div>";
    document.getElementById('chart-mb').innerHTML="<div class='dnone'>loading minibatch scores…</div>";
    document.getElementById('chart-pareto').innerHTML="<div class='dnone'>loading aggregate stats…</div>";
    waterfallEl.innerHTML="<div class='dnone'>loading candidate trajectories…</div>";
    candBodyEl.innerHTML="<tr><td colspan='7' class='dnone'>loading candidates…</td></tr>";
    candCountEl.textContent='';
    loadStorageReport(runId);
    var wsUrl=wsUrlForRun(curRunSummary);
    if(wsUrl && state==='running'){ streamWs(wsUrl, runId); }
    else { loadFileEvents(runId, state, eventsUrlForRun(curRunSummary, runId)); }
  }
  function streamWs(wsUrl, runId){
    setConn('connecting…');
    ws=new WebSocket(wsUrl);
    ws.onopen=function(){ ws.send(JSON.stringify({type:'subscribe', kinds:[], since:0})); };
    ws.onmessage=function(m){
      var f; try{ f=JSON.parse(m.data); }catch(e){ return; }
      if(f.type==='subscribed'){ setConn('live (websocket)', true);
        btnStop.hidden=false; btnCancel.hidden=false; }
	      else if(f.type==='event'){ onEvent(f); renderDetail(); }
	      else if(f.type==='status'){ if(f.run){ curRunSummary=f.run; seedRunSummary(f.run); applyLimitSnapshot(f.run.limits); dsub.textContent=runSub(f.run); renderRunState(); renderLimitEtaChart(); } }
	      else if(f.type==='terminal'){ if(f.run){ curRunSummary=f.run; seedRunSummary(f.run); dsub.textContent=runSub(f.run); renderRunState(); }
	        loadTimingRows(runId); loadLimitSnapshot(runId);
	        setConn('closed'); btnStop.hidden=true; btnCancel.hidden=true; }
      else if(f.type==='error'){ setConn('error'); }
    };
    ws.onerror=function(){ setConn('websocket connection error'); };
    ws.onclose=function(){ if(curRun===runId && D && !D.events.length) setConn('closed'); };
  }
  function runSub(r){
    var b=[]; if(r.phase) b.push(r.phase); if(r.stage) b.push(r.stage);
    if(r.generation!=null) b.push('gen '+r.generation);
    if(r.best_train_reward!=null) b.push('best '+Number(r.best_train_reward).toFixed(3));
    return b.join(' · ');
  }
  function loadFileEvents(runId, state, eventsUrl){
    if(!eventsUrl){ setConn('no live source (static export)'); return; }
    setConn(state==='running'?'reading event feed (no service)':'event feed');
    if(fileTimer){ clearTimeout(fileTimer); fileTimer=null; }
	    filePollState=state;
	    var timingsUrl=timingsUrlForRun(curRunSummary, runId);
	    var limitsUrl=limitsUrlForRun(curRunSummary, runId);
	    Promise.all([
	      fetch(eventsUrl).then(function(r){return r.json();}),
	      (timingsUrl ? fetch(timingsUrl).then(function(r){return r.json();}) : Promise.resolve(null))
	        .catch(function(){ return null; }),
	      (limitsUrl ? fetch(limitsUrl).then(function(r){return r.json();}) : Promise.resolve(null))
	        .catch(function(){ return null; })
	    ]).then(function(results){
	        if(curRun!==runId) return;
	        var d=results[0]||{}, timings=results[1], limits=results[2];
	        var limitHistory=D && D.limits ? D.limits.history : [];
	        D=newDetail(); D.limits.history=limitHistory; seedRunSummary(curRunSummary); (d.events||[]).forEach(onEvent); applyTimingRows(timings); applyLimitSnapshot(limits); renderDetail();
        if(!d.events||!d.events.length) setConn('no events recorded');
        if(filePollState==='running') fileTimer=setTimeout(function(){ loadFileEvents(runId, filePollState, eventsUrl); }, 2000);
      })
      .catch(function(){ setConn('failed to load events'); });
  }
	  function loadTimingRows(runId){
	    var timingsUrl=timingsUrlForRun(curRunSummary, runId);
	    if(!timingsUrl) return;
	    fetch(timingsUrl).then(function(r){return r.json();})
	      .then(function(d){ if(curRun===runId && D){ applyTimingRows(d); renderDetail(); } })
	      .catch(function(){});
	  }
	  function loadLimitSnapshot(runId){
	    var limitsUrl=limitsUrlForRun(curRunSummary, runId);
	    if(!limitsUrl) return;
	    fetch(limitsUrl).then(function(r){return r.json();})
	      .then(function(d){ if(curRun===runId && D){ applyLimitSnapshot(d); renderDetail(); } })
	      .catch(function(){});
	  }
  function sendControl(kind){ if(ws && ws.readyState===1) ws.send(JSON.stringify({type:kind})); }

  return { init:function(){
    try{ boardData=JSON.parse(document.getElementById('board-data').textContent); }catch(e){ boardData=null; }
    q=document.getElementById('q'); st=document.getElementById('state');
    dm=document.getElementById('domain'); body=document.querySelector('#board tbody');
    empty=document.getElementById('empty');
    drawer=document.getElementById('drawer'); scrim=document.getElementById('scrim');
    dlog=document.getElementById('drawer-log'); dconn=document.getElementById('drawer-conn');
    dtitle=document.getElementById('drawer-title'); dsub=document.getElementById('drawer-sub');
    btnStop=document.getElementById('btn-stop'); btnCancel=document.getElementById('btn-cancel');
    frameEl=document.getElementById('d-frontier'); candBodyEl=document.querySelector('#d-cands tbody');
    candCountEl=document.getElementById('d-candcount'); evCountEl=document.getElementById('d-evcount');
    dmeta=document.getElementById('drawer-meta'); runStateEl=document.getElementById('d-runstate');
    queueEl=document.getElementById('d-queues'); throughputEl=document.getElementById('d-throughput');
    proposerRoundEl=document.getElementById('d-proposer-rounds');
    waterfallEl=document.getElementById('chart-waterfall');
    limitEtaEl=document.getElementById('chart-limit-eta');
    storageEl=document.getElementById('d-storage');
    [q,st,dm].forEach(function(el){ el.addEventListener('input', apply); });
    document.getElementById('drawer-close').addEventListener('click', closeDrawer);
    scrim.addEventListener('click', closeDrawer);
    document.addEventListener('keydown', function(e){ if(e.key==='Escape') closeDrawer(); });
    btnStop.addEventListener('click', function(){ sendControl('stop'); });
    btnCancel.addEventListener('click', function(){ sendControl('cancel'); });
    storageEl.addEventListener('click', function(e){
      var button=e.target.closest('button[data-storage-action]'); if(!button) return;
      var action=button.dataset.storageAction, profile=button.dataset.profile || 'compact';
      if(action==='dry-compact') compactStorage(profile, true);
      else if(action==='apply-compact') compactStorage(profile, false);
      else if(action==='delete') deleteStorageRun();
    });
    body.addEventListener('click', function(e){
      var tr=e.target.closest('tr'); if(!tr) return;
      openDrawer(tr.dataset.run, tr.dataset.state);
    });
    apply();
    renderRunStatus(boardData);
    if(window.__LIVE_ENDPOINT__) connect(window.__LIVE_ENDPOINT__);
  }};
})();
"""
