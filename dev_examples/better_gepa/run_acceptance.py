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

from synth_optimizers.gepa import _toml_dumps


REPO_ROOT = Path(__file__).resolve().parents[2]
DEV_ROOT = REPO_ROOT / "dev_examples" / "better_gepa"
BANKING77_ROOT = REPO_ROOT / "dev_examples" / "banking77"
PROFILES_ROOT = DEV_ROOT / "profiles"

sys.path.insert(0, str(REPO_ROOT))

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
    "deepseek_v4_flash": "banking77_deepseek_v4_flash.toml",
    "chatgpt_mini": "banking77_chatgpt_mini_proposer.toml",
}

PROFILE_REQUIRED_ENV = {
    "openai_baseline": ("OPENAI_API_KEY",),
    "openai_baseline_docker": ("OPENAI_API_KEY",),
    "openrouter_grok43": ("OPENAI_API_KEY", "OPENROUTER_API_KEY"),
    "openrouter_grok43_docker": ("OPENAI_API_KEY", "OPENROUTER_API_KEY"),
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
        proposer_model="gpt-5.4-nano",
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
) -> AcceptanceReport:
    run_dir = output_dir / run_id
    event_path = run_dir / "events.jsonl"
    manifest_path = run_dir / "result_manifest.json"
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
    return AcceptanceReport(
        profile=profile,
        mode=mode,
        run_id=run_id,
        config_path=str(config_path),
        run_dir=str(run_dir),
        log_path=str(log_path),
        event_path=str(event_path),
        manifest_path=str(manifest_path),
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
