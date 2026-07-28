from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_ROOT = REPO_ROOT / "dev_examples" / "better_gepa"
BANKING77_ROOT = REPO_ROOT / "dev_examples" / "banking77"
PROFILES_ROOT = DEV_ROOT / "profiles"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from synth_optimizers.gepa import _toml_dumps  # noqa: E402
from dev_examples.banking77.banking77_synth_gepa_dev import (  # noqa: E402
    DEFAULT_PORT,
    DEV_ROOT as BANKING77_DEV_ROOT,
    GepaDevCompute,
    _write_gepa_toml,
)


PROFILE_FILES = {
    "openai_baseline": "banking77_openai_baseline.toml",
    "openai_baseline_docker": "banking77_openai_baseline_docker.toml",
    "openrouter_grok43": "banking77_openrouter_grok43.toml",
    "openrouter_grok43_docker": "banking77_openrouter_grok43_docker.toml",
    "openrouter_nemotron_ultra": "banking77_openrouter_nemotron_ultra.toml",
    "nvidia_nemotron_ultra": "banking77_nvidia_nemotron_ultra.toml",
    "deepseek_v4_flash": "banking77_deepseek_v4_flash.toml",
    "chatgpt_mini": "banking77_chatgpt_mini_proposer.toml",
}

PROFILE_REQUIRED_ENV = {
    "openai_baseline": ("OPENAI_API_KEY",),
    "openai_baseline_docker": ("OPENAI_API_KEY",),
    "openrouter_grok43": ("OPENAI_API_KEY", "OPENROUTER_API_KEY"),
    "openrouter_grok43_docker": ("OPENAI_API_KEY", "OPENROUTER_API_KEY"),
    "openrouter_nemotron_ultra": ("OPENAI_API_KEY", "OPENROUTER_API_KEY"),
    "nvidia_nemotron_ultra": ("OPENAI_API_KEY", "NVIDIA_API_KEY"),
    "deepseek_v4_flash": ("OPENAI_API_KEY", "DEEPSEEK_API_KEY"),
    "chatgpt_mini": ("OPENAI_API_KEY",),
}


@dataclass(frozen=True)
class AcceptanceReport:
    profile: str
    mode: str
    run_id: str
    config_path: str
    run_dir: str
    log_path: str
    event_path: str
    manifest_path: str
    compare_json_path: str
    proposer_tokens: int
    policy_tokens: int
    total_tokens: int
    usage_line_count: int
    final_state: str
    runtime_substrate: str
    proposal_manifest_path: str
    proposal_count: int
    docker_staging_cleaned: bool | None


def main() -> int:
    parser = argparse.ArgumentParser(description="Better GEPA usage/auth acceptance harness")
    parser.add_argument("--profile", choices=sorted(PROFILE_FILES), required=True)
    parser.add_argument("--mode", choices=["cost_stop", "token_stop"], default="cost_stop")
    parser.add_argument("--substrate", choices=["local", "docker"], default=None)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    load_env_keys(PROFILE_REQUIRED_ENV[args.profile])
    requested_substrate = substrate_for_profile(args.profile, args.substrate)
    require_profile_ready(args.profile, requested_substrate)

    run_id = (
        f"acceptance_{args.profile}_{args.mode}_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    )
    output_dir = BANKING77_DEV_ROOT / "runs" / run_id
    config_path = build_acceptance_config(
        profile=args.profile,
        mode=args.mode,
        substrate=args.substrate,
        port=args.port,
        run_id=run_id,
        output_dir=output_dir,
    )
    log_path = output_dir / "acceptance_stdout.log"
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "from synth_optimizers._synth_optimizers import GepaRun; "
            "GepaRun.from_toml(sys.argv[1]).execute()"
        ),
        str(config_path),
    ]
    env = os.environ.copy()
    env["SYNTH_OPTIMIZERS_TERMINAL"] = "1"
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        raise SystemExit(f"acceptance run failed with exit {completed.returncode}; see {log_path}")

    report = inspect_acceptance_run(
        profile=args.profile,
        mode=args.mode,
        run_id=run_id,
        config_path=config_path,
        output_dir=output_dir,
        log_path=log_path,
        acceptance_command=acceptance_command(args),
    )
    report_path = output_dir / "acceptance_report.json"
    report_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


def build_acceptance_config(
    *,
    profile: str,
    mode: str,
    substrate: str | None,
    port: int,
    run_id: str,
    output_dir: Path,
) -> Path:
    compute = GepaDevCompute(
        max_generations=1,
        proposals_per_generation=1,
        minibatch_size=2,
        max_total_rollouts=12,
        max_cost_usd=0.15 if mode == "cost_stop" else 50.0,
        proposer_model="gpt-5.4-mini",
    )
    base_config_path = _write_gepa_toml(
        port=port,
        run_id=run_id,
        output_dir=output_dir,
        compute=compute,
        policy_concurrency=8,
    )
    payload = tomllib.loads(base_config_path.read_text())
    deep_merge(payload, load_toml(PROFILES_ROOT / "_termination_limits.toml"))
    deep_merge(payload, load_toml(PROFILES_ROOT / PROFILE_FILES[profile]))
    apply_substrate_override(payload, substrate)
    apply_docker_image_override(payload)
    apply_acceptance_taskset(payload)
    apply_mode(payload, mode)
    remove_planned_only_fields(payload)
    normalize_profile_paths(payload)
    config_path = output_dir / f"{run_id}.acceptance.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(_toml_dumps(payload))
    return config_path


def inspect_acceptance_run(
    *,
    profile: str,
    mode: str,
    run_id: str,
    config_path: Path,
    output_dir: Path,
    log_path: Path,
    acceptance_command: list[str],
) -> AcceptanceReport:
    run_dir = output_dir / run_id
    event_path = run_dir / "events.jsonl"
    manifest_path = run_dir / "result_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    config = tomllib.loads(config_path.read_text())
    log_text = log_path.read_text()
    events = [json.loads(line) for line in event_path.read_text().splitlines() if line.strip()]
    runtime_jobs = [event for event in events if event.get("type") == "runtime.job.completed"]
    proposer_jobs = [
        event for event in runtime_jobs if event.get("fields", {}).get("runtime_kind") == "proposer"
    ]
    if not proposer_jobs:
        raise SystemExit(f"{profile}/{mode} did not produce a proposer runtime job")
    if "usage total=" not in log_text:
        raise SystemExit(f"{profile}/{mode} terminal log missing running usage lines")
    runtime_summary = final_runtime_summary(events)
    proposer_event = final_proposer_event(events)
    runtime_substrate = str(
        proposer_event.get("fields", {}).get("runtime_substrate")
        or tomllib.loads(config_path.read_text())
        .get("proposer", {})
        .get("runtime_substrate", "local")
    )
    proposal_manifest_path = proposer_manifest_path(run_dir, proposer_event)
    proposal_manifest = json.loads(proposal_manifest_path.read_text())
    proposals = proposal_manifest.get("proposals")
    if not isinstance(proposals, list) or not proposals:
        raise SystemExit(
            f"{profile}/{mode} proposal manifest has no proposals: {proposal_manifest_path}"
        )
    proposer_tokens = int(runtime_summary["proposer"]["total_tokens"])
    policy_tokens = int(runtime_summary["policy"]["total_tokens"])
    total_tokens = proposer_tokens + policy_tokens
    if proposer_tokens <= 0:
        raise SystemExit(f"{profile}/{mode} proposer tokens were not recorded")
    if total_tokens <= 0:
        raise SystemExit(f"{profile}/{mode} total usage tokens were not recorded")
    docker_staging_cleaned = inspect_docker_acceptance(
        runtime_substrate=runtime_substrate,
        profile=profile,
        mode=mode,
        run_dir=run_dir,
        log_text=log_text,
    )
    compare_json_path = write_compare_artifact(
        profile=profile,
        mode=mode,
        run_id=run_id,
        config_path=config_path,
        run_dir=run_dir,
        log_path=log_path,
        event_path=event_path,
        manifest_path=manifest_path,
        proposal_manifest_path=proposal_manifest_path,
        runtime_summary=runtime_summary,
        manifest=manifest,
        config=config,
        events=events,
        acceptance_command=acceptance_command,
    )
    return AcceptanceReport(
        profile=profile,
        mode=mode,
        run_id=run_id,
        config_path=str(config_path),
        run_dir=str(run_dir),
        log_path=str(log_path),
        event_path=str(event_path),
        manifest_path=str(manifest_path),
        compare_json_path=str(compare_json_path),
        proposer_tokens=proposer_tokens,
        policy_tokens=policy_tokens,
        total_tokens=total_tokens,
        usage_line_count=log_text.count("usage total="),
        final_state=final_state(events),
        runtime_substrate=runtime_substrate,
        proposal_manifest_path=str(proposal_manifest_path),
        proposal_count=len(proposals),
        docker_staging_cleaned=docker_staging_cleaned,
    )


def write_compare_artifact(
    *,
    profile: str,
    mode: str,
    run_id: str,
    config_path: Path,
    run_dir: Path,
    log_path: Path,
    event_path: Path,
    manifest_path: Path,
    proposal_manifest_path: Path,
    runtime_summary: dict[str, Any],
    manifest: dict[str, Any],
    config: dict[str, Any],
    events: list[dict[str, Any]],
    acceptance_command: list[str],
) -> Path:
    candidate_registry = load_candidate_registry(manifest)
    best_candidate = manifest.get("best_candidate")
    if not isinstance(best_candidate, dict):
        raise SystemExit("result_manifest.json missing best_candidate object")
    candidate_id = str(best_candidate.get("candidate_id") or "")
    if not candidate_id:
        raise SystemExit("best_candidate missing candidate_id")
    candidate = candidate_by_id(candidate_registry, candidate_id) or best_candidate
    baseline = seed_candidate(candidate_registry)
    if baseline is None:
        parent_id = candidate.get("parent_id")
        baseline = candidate_by_id(candidate_registry, str(parent_id)) if parent_id else None
    if baseline is None:
        raise SystemExit("could not identify baseline candidate for compare artifact")

    baseline_heldout = metric_float(baseline, "heldout_reward")
    candidate_heldout = metric_float(candidate, "heldout_reward")
    baseline_visible = metric_float(baseline, "train_reward")
    candidate_visible = metric_float(candidate, "train_reward")
    score_basis = "heldout_reward"
    if baseline_heldout is None or candidate_heldout is None:
        score_basis = "train_reward; heldout_reward unavailable"
    compare = {
        "schema_version": "synth.gepa_compare.v1",
        "profile": profile,
        "mode": mode,
        "status": "works" if final_state(events) == "finished" else "broken",
        "tier": "smoke",
        "repo": "optimizers",
        "repo_sha": git_text("rev-parse", "HEAD"),
        "repo_dirty": bool(git_text("status", "--short", allow_empty=True)),
        "command": shell_join(acceptance_command),
        "cwd": str(REPO_ROOT),
        "run_id": run_id,
        "artifact_root": str(run_dir),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "event_path": str(event_path),
        "log_path": str(log_path),
        "proposal_manifest_path": str(proposal_manifest_path),
        "workspace_db_path": manifest.get("workspace_db_path"),
        "baseline_candidate_id": baseline.get("candidate_id"),
        "candidate_id": candidate_id,
        "baseline_score": baseline_heldout,
        "candidate_score": candidate_heldout,
        "heldout_score": candidate_heldout,
        "uplift": delta(candidate_heldout, baseline_heldout),
        "baseline_visible_score": baseline_visible,
        "candidate_visible_score": candidate_visible,
        "visible_uplift": delta(candidate_visible, baseline_visible),
        "score_basis": score_basis,
        "cost_usd": metric_float(manifest, "cost_usd") or 0.0,
        "wall_time_seconds": wall_time_seconds(events),
        "model_alias": model_alias(config, runtime_summary, "proposer"),
        "policy_model_alias": model_alias(config, runtime_summary, "policy"),
        "runtime_substrate": config.get("proposer", {}).get("runtime_substrate", "local"),
        "usage": manifest.get("usage", {}),
        "taskset": {
            "visible_split": config.get("taskset", {}).get("train_split"),
            "heldout_split": config.get("taskset", {}).get("heldout_split"),
            "visible_task_ids": config.get("taskset", {}).get("train_ids", []),
            "heldout_task_ids": config.get("taskset", {}).get("heldout_ids", []),
            "task_pools": config.get("gepa", {}).get("task_pools", {}),
        },
    }
    compare_path = run_dir / "compare.json"
    compare_path.write_text(json.dumps(compare, indent=2, sort_keys=True) + "\n")
    return compare_path


def acceptance_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "dev_examples/better_gepa/run_acceptance.py",
        "--profile",
        str(args.profile),
        "--mode",
        str(args.mode),
    ]
    if args.substrate is not None:
        command.extend(["--substrate", str(args.substrate)])
    if args.port != DEFAULT_PORT:
        command.extend(["--port", str(args.port)])
    return command


def load_candidate_registry(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    registry_path_value = manifest.get("candidate_registry_path")
    if not isinstance(registry_path_value, str) or not registry_path_value:
        raise SystemExit("result_manifest.json missing candidate_registry_path")
    registry_path = Path(registry_path_value)
    if not registry_path.is_file():
        raise SystemExit(f"candidate registry missing: {registry_path}")
    registry = json.loads(registry_path.read_text())
    if not isinstance(registry, list):
        raise SystemExit(f"candidate registry is not a list: {registry_path}")
    if not all(isinstance(candidate, dict) for candidate in registry):
        raise SystemExit(f"candidate registry has non-object entries: {registry_path}")
    return registry


def seed_candidate(registry: list[dict[str, Any]]) -> dict[str, Any] | None:
    for candidate in registry:
        if candidate.get("source") == "seed":
            return candidate
    for candidate in registry:
        if candidate.get("parent_id") is None:
            return candidate
    return None


def candidate_by_id(registry: list[dict[str, Any]], candidate_id: str) -> dict[str, Any] | None:
    for candidate in registry:
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    return None


def metric_float(source: dict[str, Any], key: str) -> float | None:
    value = source.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise SystemExit(f"{key} must be numeric when present; got {value!r}")


def delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def model_alias(
    config: dict[str, Any],
    runtime_summary: dict[str, Any],
    runtime_kind: str,
) -> str:
    runtime_bucket = runtime_summary.get(runtime_kind)
    if isinstance(runtime_bucket, dict) and runtime_bucket.get("model"):
        return str(runtime_bucket["model"])
    config_bucket = config.get(runtime_kind)
    if isinstance(config_bucket, dict) and config_bucket.get("model"):
        return str(config_bucket["model"])
    return "unknown"


def wall_time_seconds(events: list[dict[str, Any]]) -> float | None:
    timestamps = [parse_event_time(event.get("ts")) for event in events if event.get("ts")]
    parsed = [timestamp for timestamp in timestamps if timestamp is not None]
    if len(parsed) < 2:
        return None
    return (max(parsed) - min(parsed)).total_seconds()


def parse_event_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def git_text(*args: str, allow_empty: bool = False) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {completed.stderr.strip()}")
    output = completed.stdout.strip()
    if not output and not allow_empty:
        raise SystemExit(f"git {' '.join(args)} returned no output")
    return output


def shell_join(args: list[str]) -> str:
    return " ".join(shell_quote(arg) for arg in args)


def shell_quote(value: str) -> str:
    if value and all(char.isalnum() or char in "@%_+=:,./-" for char in value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def final_runtime_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    finished = [event for event in events if event.get("type") == "gepa.run.finished"]
    if not finished:
        raise SystemExit("run did not emit gepa.run.finished")
    summary = finished[-1].get("fields", {}).get("runtime_summary")
    if not isinstance(summary, dict):
        raise SystemExit("gepa.run.finished missing runtime_summary")
    return summary


def final_state(events: list[dict[str, Any]]) -> str:
    for event in reversed(events):
        event_type = event.get("type")
        if event_type == "gepa.run.finished":
            return "finished"
        if event_type == "gepa.run.failed":
            return "failed"
    return "unknown"


def final_proposer_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        if event.get("type") == "proposer.completed":
            return event
    raise SystemExit("run did not emit proposer.completed")


def proposer_manifest_path(run_dir: Path, proposer_event: dict[str, Any]) -> Path:
    workspace = proposer_event.get("fields", {}).get("workspace")
    if isinstance(workspace, str) and workspace:
        path = Path(workspace)
    else:
        path = run_dir / "proposer_workspaces" / "generation_000"
    manifest_path = path / "proposal" / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"proposal manifest missing: {manifest_path}")
    return manifest_path


def inspect_docker_acceptance(
    *,
    runtime_substrate: str,
    profile: str,
    mode: str,
    run_dir: Path,
    log_text: str,
) -> bool | None:
    if runtime_substrate != "docker":
        return None
    if ".codex_app_server_entrypoint.sh" in log_text and "No such file" in log_text:
        raise SystemExit(f"{profile}/{mode} docker proposer logged missing entrypoint")
    response_path = (
        run_dir
        / "proposer_workspaces"
        / "generation_000"
        / ".agent_artifacts"
        / "opencode_response.json"
    )
    if not response_path.is_file():
        raise SystemExit(f"{profile}/{mode} docker response artifact missing: {response_path}")
    response = json.loads(response_path.read_text())
    receipt = response.get("supervisor_receipt")
    if not isinstance(receipt, dict):
        raise SystemExit(f"{profile}/{mode} docker response missing supervisor_receipt")
    staging_dir = receipt.get("staging_dir")
    if not isinstance(staging_dir, str) or not staging_dir:
        raise SystemExit(f"{profile}/{mode} docker response missing staging_dir receipt")
    if Path(staging_dir).exists():
        raise SystemExit(f"{profile}/{mode} docker staging dir was not cleaned: {staging_dir}")
    return True


def load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text())


def deep_merge(left: dict[str, Any], right: dict[str, Any]) -> None:
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(left.get(key), dict):
            deep_merge(left[key], value)
        else:
            left[key] = value


def apply_mode(payload: dict[str, Any], mode: str) -> None:
    gepa = payload["gepa"]
    if mode == "cost_stop":
        gepa["max_cost_usd"] = 0.15
        gepa["max_prompt_tokens"] = 2_000_000
        gepa["max_completion_tokens"] = 500_000
        gepa["max_total_tokens"] = 2_500_000
    else:
        gepa["max_cost_usd"] = 50.0
        gepa["max_prompt_tokens"] = 60_000
        gepa["max_completion_tokens"] = 20_000
        gepa["max_total_tokens"] = 80_000


def apply_substrate_override(payload: dict[str, Any], substrate: str | None) -> None:
    if substrate is None:
        return
    proposer = payload.setdefault("proposer", {})
    proposer["runtime_substrate"] = substrate
    if substrate == "local":
        proposer.pop("docker", None)
        return
    proposer.setdefault("docker", {})
    proposer["docker"].setdefault(
        "image",
        "ghcr.io/synth-laboratories/codex-gepa-proposer:2026-05-31",
    )
    proposer["docker"].setdefault("workspace_mount_path", "/workspace")
    proposer["docker"].setdefault("network", "bridge")
    proposer["docker"].setdefault("extra_env", {})


def apply_docker_image_override(payload: dict[str, Any]) -> None:
    image = os.environ.get("SYNTH_GEPA_DOCKER_PROPOSER_IMAGE")
    if not image:
        return
    proposer = payload.setdefault("proposer", {})
    if proposer.get("runtime_substrate") != "docker":
        return
    proposer.setdefault("docker", {})
    proposer["docker"]["image"] = image


def apply_acceptance_taskset(payload: dict[str, Any]) -> None:
    payload["taskset"]["train_ids"] = [f"train:{task_id}" for task_id in range(4)]
    payload["taskset"]["heldout_ids"] = [f"test:{task_id}" for task_id in range(2)]
    # Task pools live under [gepa] for a standalone `GepaRun.from_toml` config,
    # matching GepaConfig.to_toml_dict (gepa["task_pools"]) and config.gepa.task_pools.
    # (The service request uses a sibling `task_pools` field — same canonical home,
    # different transport.)
    payload.setdefault("gepa", {})["task_pools"] = {
        "pareto": [f"train:{task_id}" for task_id in range(2)],
        "minibatch": [f"train:{task_id}" for task_id in range(2)],
        "reflection": [f"train:{task_id}" for task_id in range(4)],
        "heldout": [f"test:{task_id}" for task_id in range(2)],
    }
    command = payload["container"]["command"]
    command.insert(4, "BANKING77_POLICY_TIMEOUT_SECONDS=60")


def remove_planned_only_fields(payload: dict[str, Any]) -> None:
    payload.get("run", {}).pop("non_western_provider", None)
    payload.get("gepa", {}).pop("stop_policy", None)


def normalize_profile_paths(payload: dict[str, Any]) -> None:
    proposer = payload.get("proposer", {})
    codex_home = proposer.get("codex_home")
    if isinstance(codex_home, str) and codex_home.startswith("~/"):
        proposer["codex_home"] = str(Path(codex_home).expanduser())
    if proposer.get("auth_mode") == "chatgpt":
        proposer.pop("api_key_env", None)


def load_env_keys(keys: tuple[str, ...]) -> None:
    env_files = [
        BANKING77_ROOT / ".env",
        REPO_ROOT.parent / "synth-ai" / ".env",
        REPO_ROOT.parent / "synth-dev" / ".env.shared",
        REPO_ROOT.parent / "backend" / ".env.local",
    ]
    for key in keys:
        if os.environ.get(key):
            continue
        value = env_value(env_files, key)
        if value:
            os.environ[key] = value
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        raise SystemExit(f"missing required env for acceptance: {', '.join(missing)}")


def env_value(env_files: list[Path], key: str) -> str:
    prefix = f"{key}="
    for env_file in env_files:
        if not env_file.is_file():
            continue
        for line in env_file.read_text().splitlines():
            if line.startswith(prefix):
                return line[len(prefix) :].strip().strip('"')
    return ""


def substrate_for_profile(profile: str, override: str | None) -> str:
    if override is not None:
        return override
    if profile.endswith("_docker"):
        return "docker"
    return "local"


def require_profile_ready(profile: str, substrate: str) -> None:
    if profile == "chatgpt_mini":
        auth_path = Path("~/.codex/auth.json").expanduser()
        if not auth_path.is_file():
            raise SystemExit("chatgpt_mini requires ~/.codex/auth.json from `codex auth login`")
        if substrate == "docker":
            raise SystemExit("chatgpt_mini docker acceptance is deferred in v1")
    if substrate == "docker" and not docker_available():
        print("SKIP docker acceptance: `docker info` failed; Docker/OrbStack is not available")
        raise SystemExit(0)


def docker_available() -> bool:
    completed = subprocess.run(
        ["docker", "info"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


if __name__ == "__main__":
    raise SystemExit(main())
