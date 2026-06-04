from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable


@dataclass(frozen=True)
class Transition:
    seq: int
    ts_unix_ms: int
    entity_type: str
    entity_id: str
    from_state: str | None
    to_state: str
    trigger: str
    generation: int | None
    parent_id: str | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RunStats:
    run_dir: Path
    run_id: str
    task: str
    model: str
    rollout_p50_seconds: float | None
    rollout_p95_seconds: float | None
    proposer_p50_seconds: float | None
    proposer_p95_seconds: float | None
    minibatch_passed: int
    minibatch_rejected: int
    candidate_count: int
    candidates_per_proposer_minute: float
    passing_candidates_per_proposer_minute: float
    pareto_delta_per_proposer_minute: float
    max_concurrent_rollouts: int
    upstream_429_rate: float
    wall_seconds: float
    rollout_count: int
    proposer_round_count: int
    total_tokens: int
    cost_usd: float

    @property
    def minibatch_pass_rate(self) -> float | None:
        denominator = self.minibatch_passed + self.minibatch_rejected
        if denominator <= 0:
            return None
        return self.minibatch_passed / denominator

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir),
            "run_id": self.run_id,
            "task": self.task,
            "model": self.model,
            "rollout_p50_seconds": self.rollout_p50_seconds,
            "rollout_p95_seconds": self.rollout_p95_seconds,
            "proposer_p50_seconds": self.proposer_p50_seconds,
            "proposer_p95_seconds": self.proposer_p95_seconds,
            "minibatch_passed": self.minibatch_passed,
            "minibatch_rejected": self.minibatch_rejected,
            "minibatch_pass_rate": self.minibatch_pass_rate,
            "candidate_count": self.candidate_count,
            "candidates_per_proposer_minute": self.candidates_per_proposer_minute,
            "passing_candidates_per_proposer_minute": self.passing_candidates_per_proposer_minute,
            "pareto_delta_per_proposer_minute": self.pareto_delta_per_proposer_minute,
            "max_concurrent_rollouts": self.max_concurrent_rollouts,
            "upstream_429_rate": self.upstream_429_rate,
            "wall_seconds": self.wall_seconds,
            "rollout_count": self.rollout_count,
            "proposer_round_count": self.proposer_round_count,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }


def eval_stats_for_roots(roots: Iterable[str | Path], *, write_json: bool = True) -> list[RunStats]:
    run_dirs = discover_transition_run_dirs([Path(root) for root in roots])
    stats = [eval_stats_for_run(run_dir) for run_dir in run_dirs]
    if write_json:
        for stat in stats:
            (stat.run_dir / "stats.json").write_text(
                json.dumps(stat.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
    return stats


def discover_transition_run_dirs(roots: list[Path]) -> list[Path]:
    run_dirs: set[Path] = set()
    for root in roots:
        if root.is_file() and root.name == "transitions.sqlite":
            run_dirs.add(root.parent)
            continue
        if (root / "transitions.sqlite").exists():
            run_dirs.add(root)
            continue
        for path in root.rglob("transitions.sqlite"):
            run_dirs.add(path.parent)
    return sorted(run_dirs)


def eval_stats_for_run(run_dir: Path) -> RunStats:
    transitions = read_transitions(run_dir / "transitions.sqlite")
    if not transitions:
        raise ValueError(f"{run_dir} has no transitions")

    run_id = run_id_from_transitions(run_dir, transitions)
    task = task_from_run_id(run_id)
    model = model_from_transitions(transitions)
    wall_seconds = (max(row.ts_unix_ms for row in transitions) - min(row.ts_unix_ms for row in transitions)) / 1000.0

    rollout_spans = state_spans(transitions, "rollout", "running", {"completed", "failed", "cached", "cancelled"})
    proposer_spans = state_spans(transitions, "proposer_round", "generating", {"returned"})
    proposer_minutes = max(sum(proposer_spans) / 60.0, 1e-9)

    minibatch_passed = distinct_entity_count(transitions, "candidate", "accepted_minibatch")
    minibatch_rejected = distinct_entity_count(transitions, "candidate", "rejected_minibatch")
    candidate_count = distinct_generated_candidate_count(transitions)
    accepted_full_train = distinct_generated_entity_count(transitions, "candidate", "accepted")
    rollout_terminal_rows = [
        row
        for row in transitions
        if row.entity_type == "rollout"
        and row.to_state in {"completed", "failed", "cached", "cancelled"}
    ]
    upstream_429_count = sum(1 for row in rollout_terminal_rows if transition_has_429(row))

    total_tokens = 0
    cost_usd = 0.0
    for row in transitions:
        if row.entity_type == "proposer_round" and row.to_state != "closed":
            continue
        if row.entity_type == "rollout" and row.to_state not in {"completed", "failed", "cached", "cancelled"}:
            continue
        if row.entity_type not in {"proposer_round", "rollout"}:
            continue
        cost_usd += float(row.metadata.get("cost_usd") or 0.0)
        usage = row.metadata.get("usage")
        if isinstance(usage, dict):
            total_tokens += int(usage.get("total_tokens") or usage.get("totalTokens") or 0)

    return RunStats(
        run_dir=run_dir,
        run_id=run_id,
        task=task,
        model=model,
        rollout_p50_seconds=percentile(rollout_spans, 0.50),
        rollout_p95_seconds=percentile(rollout_spans, 0.95),
        proposer_p50_seconds=percentile(proposer_spans, 0.50),
        proposer_p95_seconds=percentile(proposer_spans, 0.95),
        minibatch_passed=minibatch_passed,
        minibatch_rejected=minibatch_rejected,
        candidate_count=candidate_count,
        candidates_per_proposer_minute=candidate_count / proposer_minutes,
        passing_candidates_per_proposer_minute=minibatch_passed / proposer_minutes,
        pareto_delta_per_proposer_minute=accepted_full_train / proposer_minutes,
        max_concurrent_rollouts=max_concurrent_rollouts(transitions),
        upstream_429_rate=(upstream_429_count / len(rollout_terminal_rows)) if rollout_terminal_rows else 0.0,
        wall_seconds=wall_seconds,
        rollout_count=len(rollout_terminal_rows),
        proposer_round_count=len({row.entity_id for row in transitions if row.entity_type == "proposer_round"}),
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )


def read_transitions(path: Path) -> list[Transition]:
    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT seq, ts_unix_ms, entity_type, entity_id, from_state, to_state,
                   trigger, generation, parent_id, metadata
            FROM transitions
            ORDER BY seq
            """
        ).fetchall()
    return [
        Transition(
            seq=int(row["seq"]),
            ts_unix_ms=int(row["ts_unix_ms"]),
            entity_type=str(row["entity_type"]),
            entity_id=str(row["entity_id"]),
            from_state=row["from_state"],
            to_state=str(row["to_state"]),
            trigger=str(row["trigger"]),
            generation=row["generation"],
            parent_id=row["parent_id"],
            metadata=parse_metadata(row["metadata"]),
        )
        for row in rows
    ]


def parse_metadata(raw: str | bytes | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def run_id_from_transitions(run_dir: Path, transitions: list[Transition]) -> str:
    for row in transitions:
        if row.entity_type == "run":
            run_id = row.metadata.get("run_id") or row.entity_id
            if isinstance(run_id, str) and run_id:
                return run_id
    return run_dir.name


def task_from_run_id(run_id: str) -> str:
    if "_synth_gepa" in run_id:
        return run_id.split("_synth_gepa", 1)[0]
    if "_gepa" in run_id:
        return run_id.split("_gepa", 1)[0]
    parts = run_id.split("_")
    return "_".join(parts[:2]) if len(parts) > 2 else run_id


def model_from_transitions(transitions: list[Transition]) -> str:
    for row in transitions:
        model = row.metadata.get("proposer_model")
        if isinstance(model, str) and model:
            return short_model(model)
    for row in transitions:
        if row.entity_type == "proposer_round":
            model = row.metadata.get("model")
            if isinstance(model, str) and model:
                return short_model(model)
    return "unknown"


def short_model(model: str) -> str:
    return model.rsplit("/", 1)[-1].replace("gpt-5.4-", "")


def state_spans(
    transitions: list[Transition],
    entity_type: str,
    start_state: str,
    terminal_states: set[str],
) -> list[float]:
    spans: list[float] = []
    by_entity: dict[str, list[Transition]] = {}
    for row in transitions:
        if row.entity_type == entity_type:
            by_entity.setdefault(row.entity_id, []).append(row)
    for rows in by_entity.values():
        start_ms: int | None = None
        for row in rows:
            if row.to_state == start_state:
                start_ms = row.ts_unix_ms
            elif start_ms is not None and row.to_state in terminal_states:
                spans.append(max(row.ts_unix_ms - start_ms, 0) / 1000.0)
                start_ms = None
    return spans


def distinct_entity_count(transitions: list[Transition], entity_type: str, to_state: str) -> int:
    return len(
        {
            row.entity_id
            for row in transitions
            if row.entity_type == entity_type and row.to_state == to_state
        }
    )


def distinct_trigger_count(transitions: list[Transition], entity_type: str, trigger: str) -> int:
    return len(
        {
            row.entity_id
            for row in transitions
            if row.entity_type == entity_type and row.trigger == trigger
        }
    )


def distinct_generated_candidate_count(transitions: list[Transition]) -> int:
    return len(
        {
            row.entity_id
            for row in transitions
            if row.entity_type == "candidate"
            and row.trigger == "registered"
            and row.parent_id is not None
        }
    )


def distinct_generated_entity_count(
    transitions: list[Transition], entity_type: str, to_state: str
) -> int:
    return len(
        {
            row.entity_id
            for row in transitions
            if row.entity_type == entity_type
            and row.to_state == to_state
            and row.parent_id is not None
        }
    )


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if quantile == 0.50:
        return float(median(ordered))
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return float(ordered[index])


def max_concurrent_rollouts(transitions: list[Transition]) -> int:
    events: list[tuple[int, int]] = []
    for row in transitions:
        if row.entity_type != "rollout":
            continue
        if row.to_state == "running":
            events.append((row.ts_unix_ms, 1))
        elif row.to_state in {"completed", "failed", "cancelled"}:
            events.append((row.ts_unix_ms, -1))
    active = 0
    max_active = 0
    for _, delta in sorted(events, key=lambda item: (item[0], -item[1])):
        active += delta
        max_active = max(max_active, active)
    return max_active


def transition_has_429(row: Transition) -> bool:
    status = row.metadata.get("status_code") or row.metadata.get("http_status")
    if status == 429 or status == "429":
        return True
    failure = row.metadata.get("failure")
    if isinstance(failure, dict):
        status = failure.get("status_code") or failure.get("http_status")
        return status == 429 or status == "429"
    return False


def render_eval_stats_table(stats: list[RunStats]) -> str:
    lines = []
    lines.append(f"GEPA profile: {len(stats)} run(s)")
    lines.append("")
    header = (
        f"{'task':<16} {'model':<12} {'rollout p50/p95':<18} {'prop LLM p50/p95':<19} "
        f"{'mb pass%':>8} {'mb pass':>8} {'cand/min':>9} {'pass/min':>9} "
        f"{'pDelta/min':>10} {'maxRoll':>7} {'429%':>6}"
    )
    lines.append(header)
    lines.append(
        f"{'-' * 16} {'-' * 12} {'-' * 18} {'-' * 19} "
        f"{'-' * 8} {'-' * 8} {'-' * 9} {'-' * 9} {'-' * 10} {'-' * 7} {'-' * 6}"
    )
    for row in stats:
        lines.append(
            f"{row.task:<16.16} {row.model:<12.12} "
            f"{format_pair(row.rollout_p50_seconds, row.rollout_p95_seconds):<18} "
            f"{format_pair(row.proposer_p50_seconds, row.proposer_p95_seconds):<19} "
            f"{format_percent(row.minibatch_pass_rate):>8} "
            f"{row.minibatch_passed}/{row.minibatch_passed + row.minibatch_rejected:<6} "
            f"{row.candidates_per_proposer_minute:>9.2f} "
            f"{row.passing_candidates_per_proposer_minute:>9.2f} "
            f"{row.pareto_delta_per_proposer_minute:>10.3f} "
            f"{row.max_concurrent_rollouts:>7} "
            f"{row.upstream_429_rate * 100:>5.1f}"
        )
    lines.append("")
    for row in stats:
        lines.append(
            f"{row.run_id}: wall={row.wall_seconds:.1f}s rollouts={row.rollout_count} "
            f"candidates={row.candidate_count} mb_pass={row.minibatch_passed} "
            f"tokens={row.total_tokens} cost=${row.cost_usd:.4f} "
            f"generations={generation_count_hint(row.run_dir)}"
        )
    return "\n".join(lines)


def format_pair(left: float | None, right: float | None) -> str:
    if left is None or right is None:
        return "n/a"
    return f"{left:.1f}s / {right:.1f}s"


def format_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def generation_count_hint(run_dir: Path) -> str:
    try:
        transitions = read_transitions(run_dir / "transitions.sqlite")
    except (OSError, sqlite3.Error):
        return "n/a"
    generations = {
        row.generation
        for row in transitions
        if row.generation is not None and row.entity_type in {"candidate", "proposer_round"}
    }
    return str(len(generations)) if generations else "n/a"
