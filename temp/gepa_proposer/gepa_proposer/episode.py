from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

from .fixtures import _usage_totals

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "completed"}
DEFAULT_PROPOSER_IO_TIMEOUT_SECONDS = 300
DEFAULT_PROPOSER_ROUNDS = 1
DEFAULT_TRAIN_ROLLOUT_HEADROOM = 8000
DEFAULT_HELDOUT_ROLLOUT_HEADROOM = 8000

BANKING77_SEED = (
    "Classify the customer banking query into exactly one Banking77 intent. "
    "Return only the label."
)
HEALTHBENCH_SEED = (
    "You are a careful health assistant. Give accurate, relevant, and safe guidance. "
    "State uncertainty, ask useful follow-up questions, and recommend appropriate "
    "professional or emergency care when warranted."
)
OFFICEQA_SEED = (
    "You answer questions about U.S. Treasury documents. Use only the supplied "
    "source text. Be exact on figures, years, and units. Return only the final "
    "answer string, with no explanation."
)
CRAFTER_SEED = (
    "You are a Crafter agent. Respond with <tool_call>"
    '{"name":"crafter_interact","arguments":{"actions_list":["<action>"]}}'
    "</tool_call>. Pick reasonable actions."
)
TAU2_SEED = (
    "You are a retail customer-service agent. Authenticate the user by email or "
    "name + zip code, then follow the store policy for cancel, modify, return, "
    "and exchange. Confirm before any database update. One tool call per turn."
)


def parse_episode(payload: dict[str, Any] | None) -> dict[str, Any]:
    raw = payload or {}
    episode = raw.get("episode") if isinstance(raw.get("episode"), dict) else {}
    proposer_rounds = episode.get("proposer_rounds", DEFAULT_PROPOSER_ROUNDS)
    try:
        proposer_rounds = int(proposer_rounds)
    except (TypeError, ValueError) as exc:
        raise ValueError("episode.proposer_rounds must be a positive integer") from exc
    if proposer_rounds < 1:
        raise ValueError("episode.proposer_rounds must be a positive integer")
    parsed: dict[str, Any] = {
        "proposer_rounds": proposer_rounds,
        "skip_heldout": bool(episode["skip_heldout"])
        if "skip_heldout" in episode
        else False,
    }
    for key, caster in (
        ("max_rollouts", int),
        ("max_wall_seconds", int),
        ("max_spend_usd", float),
    ):
        if episode.get(key) is None:
            continue
        try:
            value = caster(episode[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"episode.{key} must be a positive number") from exc
        if value <= 0:
            raise ValueError(f"episode.{key} must be a positive number")
        parsed[key] = value
    operator = raw.get("operator") if isinstance(raw.get("operator"), dict) else episode.get("operator")
    if isinstance(operator, dict):
        parsed["operator"] = operator
    for key in (
        "pipeline_mode",
        "proposals_per_generation",
        "rollout_workers",
        "cache_namespace",
        "schema_repair_rounds",
        "jesterky_bulk",
    ):
        if episode.get(key) is not None:
            parsed[key] = episode[key]
    return parsed


def row_ids(rows: Any) -> list[str]:
    ids: list[str] = []
    if not isinstance(rows, list):
        return ids
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_id = str(row.get("task_id") or row.get("example_id") or "").strip()
        if task_id:
            ids.append(task_id)
    return ids


def pools_from_cursor(cursor: dict[str, Any]) -> dict[str, list[str]]:
    train = row_ids(cursor.get("train_rows"))
    minibatch = row_ids(cursor.get("minibatch_rows")) or train
    reflection = row_ids(cursor.get("reflection_rows")) or minibatch
    heldout = row_ids(cursor.get("heldout_rows"))
    if not train:
        raise ValueError("fixture cursor is missing train_rows")
    if not heldout:
        raise ValueError("fixture cursor is missing heldout_rows")
    return {
        "pareto": train,
        "minibatch": minibatch,
        "reflection": reflection,
        "heldout": heldout,
    }


def stop_condition(episode: dict[str, Any]) -> dict[str, Any]:
    condition = {
        "kind": "episode",
        "proposer_rounds": episode["proposer_rounds"],
        "skip_heldout": episode["skip_heldout"],
    }
    for key in ("max_rollouts", "max_wall_seconds", "max_spend_usd"):
        if key in episode:
            condition[key] = episode[key]
    return condition


def inner_policy_spec(downstream: dict[str, Any] | None) -> dict[str, Any]:
    policy = dict((downstream or {}).get("policy") or {})
    spec: dict[str, Any] = {
        "provider": policy.get("provider")
        or os.environ.get("BANKING77_POLICY_PROVIDER")
        or "openai",
        "model": policy.get("model") or os.environ.get("BANKING77_POLICY_MODEL") or "gpt-4.1-nano",
        "api_family": policy.get("api_family") or "chat_completions",
        "credentials": {
            "resolver": "env",
            "env_var": policy.get("env_var")
            or os.environ.get("BANKING77_POLICY_CREDENTIAL_ENV")
            or "OPENAI_API_KEY",
        },
    }
    if policy.get("base_url"):
        spec["base_url"] = policy["base_url"]
    if policy.get("max_tokens") is not None:
        spec["max_tokens"] = int(policy["max_tokens"])
    return spec


def proposer_spec(arm: dict[str, Any]) -> dict[str, Any]:
    provider = arm.get("provider") or "openai"
    if provider == "openrouter":
        return _openrouter_proposer_spec(arm)
    # Per-arm override first: parallel arms that share one codex_home race on
    # ~/.codex/models_cache.json (observed 2026-08-20 as "failed to load models
    # cache" then "failed to refresh available models: timeout waiting for child
    # process to exit"). Give each arm its own copy when running them together.
    auth_mode = arm.get("auth_mode") or os.environ.get("GEPA_PROPOSER_AUTH_MODE") or "chatgpt"
    spec: dict[str, Any] = {
        "provider": provider,
        "model": arm.get("model") or "gpt-5.6-luna",
        "api_family": arm.get("api_family")
        or os.environ.get("GEPA_PROPOSER_API_FAMILY")
        or "chat_completions",
        "auth_mode": auth_mode,
        "reasoning_effort": arm.get("reasoning_effort") or "low",
        "credentials": {
            "resolver": "env",
            "env_var": os.environ.get("GEPA_PROPOSER_CREDENTIAL_ENV") or "OPENAI_API_KEY",
        },
    }
    if auth_mode in {"chatgpt", "host"}:
        spec["copy_host_auth"] = True
        spec["codex_home"] = (
            arm.get("codex_home")
            or os.environ.get("GEPA_PROPOSER_CODEX_HOME")
            or str(Path.home() / ".codex")
        )
    return spec


def _openrouter_proposer_spec(arm: dict[str, Any]) -> dict[str, Any]:
    """OpenRouter proposer arm (e.g. nvidia/nemotron-3.5-lightning).

    The engine validates this shape in config.rs: backend must stay
    codex_app_server, auth_mode must be api_key/auto, api_key_env must be set and
    must not be OPENAI_API_KEY, and any model outside the verified allowlist
    (currently just x-ai/grok-4.3) needs allow_unverified_model.
    """
    model = arm.get("model")
    if not model:
        raise ValueError("openrouter proposer arm requires a model slug")
    api_key_env = arm.get("api_key_env") or "OPENROUTER_API_KEY"
    if api_key_env == "OPENAI_API_KEY":
        raise ValueError("openrouter proposer must not use OPENAI_API_KEY")
    if not os.environ.get(api_key_env):
        raise ValueError(f"openrouter proposer arm needs {api_key_env} in the environment")
    # The gepa-service-v1 wire spec is deny_unknown_fields and narrower than the
    # internal config: `backend` is not on the wire (it already defaults to
    # codex_app_server, which is what OpenRouter requires) and `api_key_env` is
    # carried by credentials.env_var, which the service maps onto
    # config.proposer.api_key_env for auth_mode=api_key.
    spec: dict[str, Any] = {
        "provider": "openrouter",
        "model": model,
        "api_family": arm.get("api_family") or "chat_completions",
        "auth_mode": "api_key",
        "allow_unverified_model": bool(arm.get("allow_unverified_model", True)),
        "credentials": {"resolver": "env", "env_var": api_key_env},
    }
    effort = arm.get("reasoning_effort")
    if effort:
        spec["reasoning_effort"] = effort
    # Codex sizes an unrecognised slug's turn from its context window, not from a
    # per-call output cap, so an OpenRouter model needs these or a large reflection
    # turn gets compacted/truncated mid-flight.
    for field in ("model_context_window", "model_auto_compact_token_limit"):
        if arm.get(field) is not None:
            spec[field] = int(arm[field])
    return spec


def _pipeline_mode(raw: Any) -> str:
    mode = str(raw or "sync_serial").strip().lower()
    if mode in {"combee", "flash", "flashevolve", "flash_evolve"}:
        return "flash_evolve"
    if mode in {"async", "pipelined", "async_pipelined"}:
        return "async_pipelined"
    return "sync_serial"


def cache_spec(arm: dict[str, Any], episode: dict[str, Any], run_id: str) -> dict[str, Any]:
    namespace = (
        episode.get("cache_namespace")
        or f"gepa-proposer:{arm.get('model')}:{arm.get('reasoning_effort')}:{run_id}"
    )
    path = episode.get("cache_path") or os.environ.get("GEPA_PROPOSER_CACHE_DIR")
    spec: dict[str, Any] = {"mode": "readwrite", "namespace": namespace}
    if path:
        spec["path"] = str(path)
    return spec


def proposer_io_spec(arm: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    """Per-arm proposer I/O budget.

    The service default is 300s. A large-context proposer reflecting over a
    ~480k-token prompt can still be streaming past that: nvidia/nemotron-3.5-lightning
    died on `codex app-server timed out waiting for response` with a received_tail
    showing the turn mid-flight (item/agentMessage/delta -> item/completed ->
    turn/plan/updated), i.e. the model was answering and the wait gave up. Slow
    arms therefore need a longer budget than fast ones, so this is arm-scoped.
    """
    timeout = (
        arm.get("proposer_timeout_seconds")
        or episode.get("proposer_timeout_seconds")
        or os.environ.get("GEPA_PROPOSER_IO_TIMEOUT_SECONDS")
        or DEFAULT_PROPOSER_IO_TIMEOUT_SECONDS
    )
    spec: dict[str, Any] = {"timeout_seconds": int(timeout)}
    # Parallel arms sharing one codex_home race on the models cache; proposer_io
    # carries a per-run override the wire already accepts.
    codex_home = arm.get("codex_home") or episode.get("codex_home")
    if codex_home:
        spec["codex_home"] = str(codex_home)
    repair = arm.get("schema_repair_rounds")
    if repair is None:
        repair = episode.get("schema_repair_rounds")
    if repair is not None:
        spec["schema_repair_rounds"] = int(repair)
    return spec


def build_run_request(
    *,
    spec: dict[str, Any],
    cursor: dict[str, Any],
    arm: dict[str, Any],
    episode: dict[str, Any],
    container_url: str,
    run_id: str | None = None,
    output_dir: str | None = None,
) -> dict[str, Any]:
    pools = pools_from_cursor(cursor)
    body: dict[str, Any] = {
        "container_url": container_url,
        "policy": inner_policy_spec(spec.get("downstream") if isinstance(spec.get("downstream"), dict) else None),
        "proposer": proposer_spec(arm),
        "taskset": {
            "train_ids": pools["pareto"],
            "heldout_ids": pools["heldout"],
        },
        "task_pools": pools,
        "stop_conditions": [stop_condition(episode)],
        "advanced": {
            "pipeline": {
                "mode": _pipeline_mode(episode.get("pipeline_mode") or "sync_serial"),
                "proposals_per_generation": int(
                    episode.get("proposals_per_generation")
                    or os.environ.get("GEPA_PROPOSER_PROPOSALS")
                    or 6
                ),
                "rollout_workers": int(
                    episode.get("rollout_workers") or os.environ.get("GEPA_PROPOSER_ROLLOUT_WORKERS") or 30
                ),
            },
            "budgets": {
                "max_train_rollouts": DEFAULT_TRAIN_ROLLOUT_HEADROOM,
                "max_heldout_rollouts": max(
                    DEFAULT_HELDOUT_ROLLOUT_HEADROOM, len(pools["heldout"])
                ),
            },
            "proposer_io": proposer_io_spec(arm, episode),
        },
        "fixture": {
            "schema": spec.get("schema") or "gepa_cursor_fixture.v1",
            "fixture_id": spec.get("fixture_id"),
            "source_run_id": spec.get("source_run_id"),
            "source_checkpoint_id": spec.get("source_checkpoint_id"),
            "generation": spec.get("generation"),
            "snapshot_sha256": spec.get("snapshot_sha256"),
            "checkpoint": _checkpoint_with_usage(spec.get("checkpoint")),
        },
    }
    if output_dir:
        body["output_dir"] = output_dir
    operator = episode.get("operator")
    if isinstance(operator, dict) and operator:
        body["advanced"]["operator"] = operator
    if episode.get("jesterky_bulk") is not None:
        body["advanced"]["jesterky_workflow"] = {"bulk": bool(episode["jesterky_bulk"])}
    return body


def _checkpoint_with_usage(checkpoint: Any) -> Any:
    if not isinstance(checkpoint, dict):
        return checkpoint
    patched = copy.deepcopy(checkpoint)
    patched["usage"] = _usage_totals(patched.get("usage"))
    snapshot = patched.get("snapshot")
    if isinstance(snapshot, dict):
        snapshot["usage"] = _usage_totals(snapshot.get("usage"))
        patched["snapshot"] = snapshot
    return patched


def program_for_task(spec: dict[str, Any]) -> dict[str, Any]:
    downstream = spec.get("downstream") if isinstance(spec.get("downstream"), dict) else {}
    field = str(downstream.get("candidate_field") or "stage2_system")
    candidates = (spec.get("cursor") or {}).get("candidates") or []
    seed = ""
    if candidates and isinstance(candidates[0], dict):
        payload = candidates[0].get("payload") or {}
        if isinstance(payload, dict):
            seed = str(payload.get(field) or "")
    if not seed:
        if str(downstream.get("id") or "") == "officeqa":
            seed = OFFICEQA_SEED
        elif field == "domain_policy" or str(downstream.get("id") or "") == "tau2":
            seed = TAU2_SEED
        elif field == "react_system_prompt" or str(downstream.get("id") or "") == "crafter":
            seed = CRAFTER_SEED
        elif field == "system_prompt":
            seed = HEALTHBENCH_SEED
        else:
            seed = BANKING77_SEED
    return {
        "version": "prompt_program.v1",
        "program_id": f"gepa_proposer_{spec.get('task_id')}",
        "modules": [
            {
                "module_id": field,
                "role": "system",
                "content": seed,
                "mutable": True,
                "candidate_field": field,
                "template_variables": [],
            }
        ],
        "target_modules": [
            {
                "module_id": field,
                "candidate_field": field,
                "objective": "outcome_reward",
            }
        ],
        "seed_candidate": {field: seed},
        "rollout_overlay_schema": {"candidate_fields": [field]},
    }
