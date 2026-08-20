from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .pricing import episode_cost_usd

DEFAULT_EXPLORATION_REDUCE = "mean"


def _flag_weight(raw: Mapping[str, Any], include_key: str, weight_key: str) -> float:
    if weight_key in raw and raw[weight_key] is not None:
        return float(raw[weight_key])
    return 1.0 if raw.get(include_key) else 0.0


def _combine_settings(combine: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(combine or {})
    reduce = str(raw.get("exploration_reduce") or DEFAULT_EXPLORATION_REDUCE).strip().lower()
    if reduce not in {"mean", "sum"}:
        raise ValueError("combine.exploration_reduce must be 'mean' or 'sum'")
    missing = str(raw.get("missing") or "zero").strip().lower()
    if missing not in {"zero", "fail"}:
        raise ValueError("combine.missing must be 'zero' or 'fail'")
    return {
        "exploration_reduce": reduce,
        "exploration_weight": float(raw.get("exploration_weight", 1.0)),
        "exploitation_weight": float(raw.get("exploitation_weight", 1.0)),
        "eval_uplift_weight": float(raw.get("eval_uplift_weight", 1.0)),
        "missing": missing,
        "include_confidence": bool(raw.get("include_confidence", False)),
        "include_time": bool(raw.get("include_time", False)),
        "include_cost": bool(raw.get("include_cost", False)),
        "include_milestones": bool(raw.get("include_milestones", False)),
        "include_rubrics": bool(raw.get("include_rubrics", False)),
        "confidence_weight": _flag_weight(raw, "include_confidence", "confidence_weight"),
        "time_weight": _flag_weight(raw, "include_time", "time_weight"),
        "cost_weight": _flag_weight(raw, "include_cost", "cost_weight"),
        "milestones_weight": _flag_weight(raw, "include_milestones", "milestones_weight"),
        "rubrics_weight": _flag_weight(raw, "include_rubrics", "rubrics_weight"),
    }


def _candidate_id(candidate: dict[str, Any]) -> str:
    return str(candidate.get("candidate_id") or "")


def _by_id(candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        cid: candidate
        for candidate in candidates
        if (cid := _candidate_id(candidate))
    }


def _lookup(candidates: list[dict[str, Any]], ids: list[str]) -> list[dict[str, Any]]:
    index = _by_id(candidates)
    return [index[cid] for cid in ids if cid in index]


def _train_scores(candidate: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    seed_rewards = candidate.get("seed_rewards") or {}
    sources: list[Any] = []
    if isinstance(seed_rewards, dict):
        sources.extend(seed_rewards.get("train") or [])
        sources.extend(seed_rewards.get("minibatch") or [])
    for key in ("train_scores", "minibatch_scores"):
        sources.extend(candidate.get(key) or [])
    for row in sources:
        if not isinstance(row, dict):
            continue
        example_id = str(row.get("example_id") or row.get("task_id") or "")
        if example_id and example_id not in out:
            out[example_id] = float(row.get("reward") or 0.0)
    return out


def _heldout_scores(candidate: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    seed_rewards = candidate.get("seed_rewards") or {}
    if isinstance(seed_rewards, dict):
        for row in seed_rewards.get("heldout") or []:
            if not isinstance(row, dict):
                continue
            example_id = str(row.get("example_id") or row.get("task_id") or "")
            if example_id and example_id not in out:
                out[example_id] = float(row.get("reward") or 0.0)
    for row in candidate.get("heldout_scores") or []:
        if not isinstance(row, dict):
            continue
        example_id = str(row.get("example_id") or row.get("task_id") or "")
        if example_id and example_id not in out:
            out[example_id] = float(row.get("reward") or 0.0)
    return out


def _heldout_mean(candidate: dict[str, Any]) -> float | None:
    per_seed = _heldout_scores(candidate)
    if per_seed:
        return sum(per_seed.values()) / len(per_seed)
    for key in ("heldout_reward", "heldout_score"):
        raw = candidate.get(key)
        if raw is not None:
            return float(raw)
    return None


def _train_mean(candidate: dict[str, Any]) -> float:
    scores = _train_scores(candidate)
    if scores:
        return sum(scores.values()) / len(scores)
    for key in ("train_reward", "train_score"):
        raw = candidate.get(key)
        if raw is not None:
            return float(raw)
    return float("-inf")


def _best_id(candidates: list[dict[str, Any]], preferred: str | None = None) -> str | None:
    if preferred:
        wanted = str(preferred)
        if any(_candidate_id(candidate) == wanted for candidate in candidates):
            return wanted
    scored = [candidate for candidate in candidates if _train_mean(candidate) != float("-inf")]
    pool = scored or candidates
    if not pool:
        return None
    return _candidate_id(max(pool, key=_train_mean)) or None


def archive_max_per_seed(candidates: list[dict[str, Any]]) -> dict[str, tuple[str, float]]:
    best: dict[str, tuple[str, float]] = {}
    for candidate in candidates:
        cid = _candidate_id(candidate)
        for example_id, reward in _train_scores(candidate).items():
            current = best.get(example_id)
            if current is None or reward > current[1]:
                best[example_id] = (cid, reward)
    return best


def _mean_on(candidates: list[dict[str, Any]], example_ids: set[str]) -> float:
    values: list[float] = []
    for candidate in candidates:
        scores = _train_scores(candidate)
        row = [scores[eid] for eid in example_ids if eid in scores]
        if row:
            values.append(sum(row) / len(row))
    if not values:
        return 0.0
    return sum(values) / len(values)


def _mean_heldout(candidates: list[dict[str, Any]]) -> float | None:
    values = [
        value
        for candidate in candidates
        if (value := _heldout_mean(candidate)) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def _parse_ts(raw: Any) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _cursor_generation(cursor: Mapping[str, Any] | None) -> float:
    if not isinstance(cursor, dict):
        return 0.0
    raw = cursor.get("generation")
    if raw is None:
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def _mean_confidence(candidates: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for candidate in candidates:
        raw = candidate.get("acceptance_score")
        if raw is None and isinstance(candidate.get("acceptance_metadata"), dict):
            raw = candidate["acceptance_metadata"].get("confidence")
        if raw is None:
            continue
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def _mean_rubric(candidates: list[dict[str, Any]]) -> float | None:
    values: list[float] = []
    for candidate in candidates:
        details = candidate.get("reward_details")
        if not isinstance(details, dict):
            details = {}
        metadata = candidate.get("acceptance_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        for blob in (details, metadata, candidate):
            for key in ("rubric", "rubrics", "rubric_reward", "process_reward"):
                raw = blob.get(key) if isinstance(blob, dict) else None
                if raw is None:
                    continue
                try:
                    values.append(float(raw))
                    break
                except (TypeError, ValueError, AttributeError):
                    continue
    if not values:
        return None
    return sum(values) / len(values)


_SEVERITY_SCORE = {
    "none": 1.0,
    "low": 0.75,
    "medium": 0.5,
    "high": 0.25,
    "critical": 0.0,
}


def _jesterky_annotation_rows(context: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    root = (context or {}).get("output_dir")
    if not root:
        return []
    states = sorted(
        Path(str(root)).glob("**/proposer_workspaces/generation_*/state/jesterky_trace_annotations.jsonl")
    )
    if not states:
        return []
    rows: list[dict[str, Any]] = []
    try:
        text = states[-1].read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _jesterky_confidence(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    bad = 0
    for row in rows:
        severity = str(row.get("severity") or "").strip().lower()
        if row.get("blocker") or severity in {"high", "critical"}:
            bad += 1
    return 1.0 - (bad / len(rows))


def _jesterky_rubric(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    scores: list[float] = []
    for row in rows:
        raw = row.get("reward")
        try:
            reward = None if raw is None else float(raw)
        except (TypeError, ValueError):
            reward = None
        if reward is not None and reward != 0.0:
            scores.append(reward)
            continue
        severity = str(row.get("severity") or "none").strip().lower()
        scores.append(_SEVERITY_SCORE.get(severity, 0.5))
    return sum(scores) / len(scores)
    if not values:
        return None
    return sum(values) / len(values)


def _optional_terms(
    settings: Mapping[str, Any],
    *,
    episode_candidates: list[dict[str, Any]],
    context: Mapping[str, Any] | None,
) -> tuple[dict[str, float], dict[str, Any] | None]:
    ctx = dict(context or {})
    episode = ctx.get("episode") if isinstance(ctx.get("episode"), dict) else {}
    pre = ctx.get("pre_cursor") if isinstance(ctx.get("pre_cursor"), dict) else {}
    post = ctx.get("post_cursor") if isinstance(ctx.get("post_cursor"), dict) else {}
    extras: dict[str, float] = {}
    cost_report: dict[str, Any] | None = None
    missing_fail = settings["missing"] == "fail"

    if settings["include_confidence"]:
        value = _mean_confidence(episode_candidates)
        if value is None:
            value = _jesterky_confidence(_jesterky_annotation_rows(ctx))
        if value is None and missing_fail:
            raise ValueError("confidence evidence missing and combine.missing='fail'")
        extras["confidence"] = 0.0 if value is None else value

    if settings["include_time"]:
        start = _parse_ts(ctx.get("created_at"))
        end = _parse_ts(ctx.get("completed_at"))
        elapsed = (end - start).total_seconds() if start and end else None
        budget = episode.get("max_wall_seconds")
        try:
            denom = float(budget) if budget is not None else 1800.0
        except (TypeError, ValueError):
            denom = 1800.0
        if elapsed is None and missing_fail:
            raise ValueError("time evidence missing and combine.missing='fail'")
        extras["time"] = 0.0 if elapsed is None else -min(1.0, max(0.0, elapsed) / max(1.0, denom))

    if settings["include_cost"]:
        cost_report = episode_cost_usd(ctx, missing=settings["missing"])
        delta = float(cost_report["episode_cost_usd"])
        budget = episode.get("max_spend_usd")
        try:
            denom = float(budget) if budget is not None else 15.0
        except (TypeError, ValueError):
            denom = 15.0
        extras["cost"] = -min(1.0, delta / max(1e-9, denom))

    if settings["include_milestones"]:
        delta = _cursor_generation(post) - _cursor_generation(pre)
        rounds = episode.get("proposer_rounds") or 1
        try:
            denom = float(rounds)
        except (TypeError, ValueError):
            denom = 1.0
        extras["milestones"] = min(1.0, max(0.0, delta / max(1.0, denom)))

    if settings["include_rubrics"]:
        value = _mean_rubric(episode_candidates)
        if value is None:
            value = _jesterky_rubric(_jesterky_annotation_rows(ctx))
        if value is None and missing_fail:
            raise ValueError("rubric evidence missing and combine.missing='fail'")
        extras["rubrics"] = 0.0 if value is None else value
    return extras, cost_report


def score_episode(
    *,
    pre_fork: list[dict[str, Any]],
    episode_candidates: list[dict[str, Any]],
    post_candidates: list[dict[str, Any]] | None = None,
    best_candidate_id: str | None = None,
    combine: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train exploration + train exploitation from train/minibatch scores.

    Eval uplift is heldout(post-episode best) − heldout(pre-fork best), with
    both scores read from the terminal cursor after GEPA's heldout pass.

    `train_exploration` is mean per-seed displacement by default so an integer
    count of binary minibatch flips cannot dominate the other two terms.
    Pass combine.exploration_reduce="sum" to restore the unnormalized total.
    """
    episode_ids = {eid for candidate in episode_candidates for eid in _train_scores(candidate)}
    prior = {
        eid: holder
        for eid, holder in archive_max_per_seed(pre_fork).items()
        if eid in episode_ids
    }
    new_max = {
        eid: holder
        for eid, holder in archive_max_per_seed(pre_fork + episode_candidates).items()
        if eid in episode_ids
    }
    settings = _combine_settings(combine)
    exploration_rows = []
    train_exploration_sum = 0.0
    for example_id, (cid, reward) in sorted(new_max.items()):
        old_cid, old_reward = prior.get(example_id, ("", float("-inf")))
        delta = reward - old_reward if old_reward != float("-inf") else reward
        if cid != old_cid and delta > 0:
            train_exploration_sum += delta
            exploration_rows.append(
                {
                    "example_id": example_id,
                    "new_candidate_id": cid,
                    "prior_candidate_id": old_cid or None,
                    "delta": delta,
                }
            )
    denom = max(1, len(episode_ids))
    train_exploration = (
        train_exploration_sum
        if settings["exploration_reduce"] == "sum"
        else train_exploration_sum / denom
    )
    pre_fork_mean = _mean_on(pre_fork, episode_ids)
    episode_mean = _mean_on(episode_candidates, episode_ids)
    train_exploitation = episode_mean - pre_fork_mean

    post = list(post_candidates) if post_candidates is not None else list(pre_fork) + list(episode_candidates)
    pre_ids = [_candidate_id(candidate) for candidate in pre_fork if _candidate_id(candidate)]
    episode_ids_list = [
        _candidate_id(candidate) for candidate in episode_candidates if _candidate_id(candidate)
    ]
    pre_from_post = _lookup(post, pre_ids) or list(pre_fork)
    episode_from_post = _lookup(post, episode_ids_list) or list(episode_candidates)

    pre_best_id = _best_id(pre_from_post)
    post_best_id = _best_id(post, preferred=str(best_candidate_id) if best_candidate_id else None)
    post_index = _by_id(post)
    pre_best = post_index.get(pre_best_id or "") or (pre_from_post[0] if pre_from_post else None)
    post_best = post_index.get(post_best_id or "") or (episode_from_post[-1] if episode_from_post else None)
    pre_best_heldout = _heldout_mean(pre_best) if pre_best else None
    post_best_heldout = _heldout_mean(post_best) if post_best else None
    pre_fork_heldout = _mean_heldout(pre_from_post)
    episode_heldout = _mean_heldout(episode_from_post)
    if episode_heldout is not None and pre_fork_heldout is not None:
        eval_uplift = episode_heldout - pre_fork_heldout
        heldout_evaluated = True
    elif post_best_heldout is not None and pre_fork_heldout is not None:
        eval_uplift = post_best_heldout - pre_fork_heldout
        heldout_evaluated = True
    else:
        eval_uplift = 0.0
        heldout_evaluated = False
        if settings["missing"] == "fail":
            raise ValueError("heldout evidence missing and combine.missing='fail'")
    extras, cost_report = _optional_terms(
        settings,
        episode_candidates=episode_candidates,
        context=context,
    )
    reward = (
        settings["exploration_weight"] * train_exploration
        + settings["exploitation_weight"] * train_exploitation
    )
    if heldout_evaluated:
        reward += settings["eval_uplift_weight"] * eval_uplift
    for name, value in extras.items():
        reward += settings.get(f"{name}_weight", 0.0) * value
    return {
        "train_exploration": train_exploration,
        "train_exploitation": train_exploitation,
        "eval_uplift": eval_uplift,
        "train_exploration_sum": train_exploration_sum,
        "exploration_reduce": settings["exploration_reduce"],
        "exploration": train_exploration,
        "exploitation": train_exploitation,
        "reward": reward,
        "reward_weights": {
            "train_exploration": settings["exploration_weight"],
            "train_exploitation": settings["exploitation_weight"],
            "eval_uplift": settings["eval_uplift_weight"],
        },
        "optional_terms": extras,
        "episode_cost_usd": None if cost_report is None else cost_report["episode_cost_usd"],
        "cost_pricing": cost_report,
        "exploration_rows": exploration_rows,
        "pre_fork_mean": pre_fork_mean,
        "episode_mean": episode_mean,
        "pre_fork_heldout_mean": pre_fork_heldout,
        "episode_heldout_mean": episode_heldout,
        "pre_best_candidate_id": pre_best_id,
        "post_best_candidate_id": post_best_id,
        "pre_best_heldout": pre_best_heldout,
        "post_best_heldout": post_best_heldout,
        "heldout_evaluated": heldout_evaluated,
        "scored_example_ids": len(episode_ids),
        "episode_candidate_ids": episode_ids_list,
        "episode_heldout_rewards": [
            {
                "candidate_id": _candidate_id(candidate),
                "heldout_reward": _heldout_mean(candidate),
            }
            for candidate in episode_from_post
        ],
    }
