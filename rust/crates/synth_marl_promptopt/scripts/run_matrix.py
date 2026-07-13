#!/usr/bin/env python3
"""Run one frozen GEPA/MARL prompt-optimizer matrix over GameBench Rust services."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path


ALGORITHMS = ("gepa", "coma", "ic3net", "imac", "rode")
ENVIRONMENTS = ("craftax", "dungeongrid", "overcooked")
SERVICE_ENVIRONMENTS = {
    "craftax": "craftax-multiplayer",
    "dungeongrid": "dungeongrid-multiplayer",
    "overcooked": "overcooked-v2-multiplayer",
}
PROGRAM_FIELDS = ("shared_instruction", "communication_policy", "role_prompts")
SEED_CANDIDATE = {
    "shared_instruction": "PRIORITY=SAFETY",
    "communication_policy": (
        "SPEAK=ALWAYS; MAX_CHARS=120; REQUEST=ACTION_ONLY; "
        "HANDOFF=DIRECT; FOLLOWER_REPLY=ACK"
    ),
    "role_prompts": "ROLE_ASSIGNMENT=FLEXIBLE",
}


def craftax_catalog() -> tuple[list[str], list[str], list[str]]:
    probes = (
        "iron_handoff",
        "food_rescue",
        "miner_craft_pipeline",
        "expiring_request_repair",
    )

    def rows(split: str, seeds: range) -> list[str]:
        return [
            f"craftax_coordination_v1:{split}:{seed}:{probe}"
            for seed in seeds
            for probe in probes
        ]

    return rows("train", range(1001, 1013)), rows("selection", range(2001, 2007)), rows(
        "heldout", range(3001, 3007)
    )


def dungeongrid_catalog() -> tuple[list[str], list[str], list[str]]:
    transforms = ("identity", "mirror_x", "mirror_y", "rotate_180")
    role_orders = ("original", "swapped")
    probes = ("pre_breach", "carrier_assignment", "extraction_handoff")

    def rows(split: str, scenarios: tuple[str, ...]) -> list[str]:
        return [
            (
                f"dungeongrid_coordination_v1:{split}:{scenario}:"
                f"{transform}:{role_order}:{probe}"
            )
            for scenario in scenarios
            for transform in transforms
            for role_order in role_orders
            for probe in probes
        ]

    return (
        rows("train", ("blackwater_bell_breach", "cinder_mage_threshold")),
        rows("selection", ("chokepoint_passage",)),
        rows("heldout", ("frost_mirror_crossfire",)),
    )


def overcooked_catalog() -> tuple[list[str], list[str], list[str]]:
    templates = {
        "train": (
            ("train_hidden_1101", "hidden_recipe_reveal", 16),
            ("train_roles_1102", "ingredient_cook_role_assignment", 16),
            ("train_handoff_1103", "delivery_handoff", 16),
        ),
        "selection": (
            ("selection_hidden_2101", "hidden_recipe_reveal", 8),
            ("selection_roles_2102", "ingredient_cook_role_assignment", 8),
            ("selection_handoff_2103", "delivery_handoff", 8),
        ),
        "heldout": (
            ("heldout_hidden_3101", "hidden_recipe_reveal", 8),
            ("heldout_roles_3102", "ingredient_cook_role_assignment", 8),
            ("heldout_handoff_3103", "delivery_handoff", 8),
        ),
    }

    def rows(split: str) -> list[str]:
        return [
            f"overcooked_v2_coordination_v1:{split}:{row_id}:r{replica:02}:{probe}"
            for row_id, probe, count in templates[split]
            for replica in range(1, count + 1)
        ]

    return rows("train"), rows("selection"), rows("heldout")


CATALOGS = {
    "craftax": craftax_catalog,
    "dungeongrid": dungeongrid_catalog,
    "overcooked": overcooked_catalog,
}


def toml_string(value: str | Path) -> str:
    return json.dumps(str(value))


def toml_array(values: list[str] | tuple[str, ...]) -> str:
    body = "".join(f"  {toml_string(value)},\n" for value in values)
    return f"[\n{body}]"


def profile_text(
    *,
    environment: str,
    service_url: str,
    output_dir: Path,
    train_ids: list[str],
    selection_ids: list[str],
    heldout_ids: list[str],
    proposer_model: str,
    reasoning_effort: str,
    max_generations: int,
    train_rollouts: int,
    heldout_rollouts: int,
    seed: int,
) -> str:
    public_train_ids = train_ids + selection_ids
    return f'''[run]
run_id = {toml_string(f"marl_matrix_gepa_{environment}")}
output_dir = {toml_string(output_dir)}
seed = {seed}

[container]
url = {toml_string(service_url)}

[taskset]
train_split = "train"
heldout_split = "heldout"
train_ids = {toml_array(public_train_ids)}
heldout_ids = {toml_array(heldout_ids)}

[candidate]
target_modules = {toml_array(PROGRAM_FIELDS)}

[seed_candidate]
shared_instruction = {toml_string(SEED_CANDIDATE["shared_instruction"])}
communication_policy = {toml_string(SEED_CANDIDATE["communication_policy"])}
role_prompts = {toml_string(SEED_CANDIDATE["role_prompts"])}

[policy]
enabled = false
provider = "openai"
model = "gpt-4.1-nano"
proxy_mode = "proxy_only"

[proposer]
backend = "codex_app_server"
runtime_substrate = "local"
execution_mode = "local_process"
provider = "openai"
auth_mode = "chatgpt"
codex_home = {toml_string(Path.home() / ".codex")}
copy_host_auth = true
model = {toml_string(proposer_model)}
reasoning_effort = {toml_string(reasoning_effort)}
timeout_seconds = 900
message_stall_timeout_seconds = 180
sandbox_mode = "workspace-write"
approval_policy = "never"

[gepa]
max_generations = {max_generations}
proposals_per_generation = 2
minibatch_size = 12
max_total_rollouts = {train_rollouts + heldout_rollouts}
max_train_rollouts = {train_rollouts}
max_heldout_rollouts = {heldout_rollouts}
rollout_submission_mode = "sync"
rollout_failure_rate_tolerance = 0.0
frontier_type = "per_example"

[gepa.task_pools]
pareto = {toml_array(selection_ids)}
minibatch = {toml_array(train_ids)}
reflection = {toml_array(train_ids)}
heldout = {toml_array(heldout_ids)}
'''


def variant_text(
    *, profile: Path, variant: str, environment: str, output_dir: Path, seed: int
) -> str:
    return f'''gepa_profile = {toml_string(profile)}
variant = {toml_string(variant)}

[run]
run_id = {toml_string(f"marl_matrix_{variant}_{environment}")}
output_dir = {toml_string(output_dir)}
seed = {seed}

[experiment]
selection_candidates_per_generation = 1
minimum_rows_per_candidate = 72
require_disjoint_splits = true
require_exact_rollout_budget = true
compare_seed_on_heldout = true
'''


def parse_csv(value: str, allowed: tuple[str, ...]) -> list[str]:
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected).difference(allowed))
    if unknown:
        raise SystemExit(f"unsupported values {unknown}; allowed={list(allowed)}")
    return selected


def require_service(url: str, expected_environment: str) -> None:
    with urllib.request.urlopen(f"{url.rstrip('/')}/health", timeout=10) as response:
        payload = json.load(response)
    if payload.get("status") != "ok" or payload.get("environment") != expected_environment:
        raise RuntimeError(
            f"unexpected service at {url}: expected={expected_environment!r} payload={payload!r}"
        )


def result_summary(algorithm: str, payload: dict[str, object]) -> dict[str, object]:
    if algorithm == "gepa":
        best = payload.get("best_candidate")
        return {
            "manifest_path": payload.get("manifest_path"),
            "cost_usd": payload.get("cost_usd"),
            "best_candidate": best,
        }
    return {
        key: payload.get(key)
        for key in (
            "manifest_path",
            "candidate_count",
            "rollout_count",
            "champion_candidate_id",
            "frontier_candidate_ids",
            "heldout_seed_score",
            "heldout_champion_score",
            "heldout_uplift",
            "budget",
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    parser.add_argument("--environments", default=",".join(ENVIRONMENTS))
    parser.add_argument("--craftax-url", default="http://127.0.0.1:18788")
    parser.add_argument("--dungeongrid-url", default="http://127.0.0.1:18789")
    parser.add_argument("--overcooked-url", default="http://127.0.0.1:18790")
    parser.add_argument("--proposer-model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-generations", type=int, default=2)
    parser.add_argument("--train-rollouts", type=int, default=768)
    parser.add_argument("--heldout-rollouts", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--run-timeout-seconds", type=int, default=1800)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    optimizer_root = Path(__file__).resolve().parents[4]
    output_root = (args.output_root or optimizer_root / ".out/marl_promptopt_rust_matrix").resolve()
    config_dir = output_root / "configs"
    log_dir = output_root / "logs"
    run_dir = output_root / "runs"
    for path in (config_dir, log_dir, run_dir):
        path.mkdir(parents=True, exist_ok=True)

    algorithms = parse_csv(args.algorithms, ALGORITHMS)
    environments = parse_csv(args.environments, ENVIRONMENTS)
    urls = {
        "craftax": args.craftax_url,
        "dungeongrid": args.dungeongrid_url,
        "overcooked": args.overcooked_url,
    }
    profiles: dict[str, Path] = {}
    wrappers: dict[tuple[str, str], Path] = {}
    for environment in environments:
        train_ids, selection_ids, heldout_ids = CATALOGS[environment]()
        if (len(train_ids), len(selection_ids), len(heldout_ids)) != (48, 24, 24):
            raise RuntimeError(f"unexpected {environment} catalog counts")
        profile = config_dir / f"{environment}_gepa.toml"
        profile.write_text(
            profile_text(
                environment=environment,
                service_url=urls[environment],
                output_dir=run_dir / f"gepa_{environment}",
                train_ids=train_ids,
                selection_ids=selection_ids,
                heldout_ids=heldout_ids,
                proposer_model=args.proposer_model,
                reasoning_effort=args.reasoning_effort,
                max_generations=args.max_generations,
                train_rollouts=args.train_rollouts,
                heldout_rollouts=args.heldout_rollouts,
                seed=args.seed,
            ),
            encoding="utf-8",
        )
        profiles[environment] = profile
        for algorithm in algorithms:
            if algorithm == "gepa":
                continue
            wrapper = config_dir / f"{environment}_{algorithm}.toml"
            wrapper.write_text(
                variant_text(
                    profile=profile,
                    variant=algorithm,
                    environment=environment,
                    output_dir=run_dir / f"{algorithm}_{environment}",
                    seed=args.seed,
                ),
                encoding="utf-8",
            )
            wrappers[(environment, algorithm)] = wrapper

    if args.generate_only:
        print(json.dumps({"config_dir": str(config_dir), "generated": True}, indent=2))
        return 0

    for environment in environments:
        require_service(urls[environment], SERVICE_ENVIRONMENTS[environment])

    receipt: dict[str, object] = {
        "schema_version": "marl_promptopt_rust_matrix.v1",
        "optimizer_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=optimizer_root, text=True
        ).strip(),
        "algorithms": algorithms,
        "environments": environments,
        "budgets": {
            "train_rollouts": args.train_rollouts,
            "heldout_rollouts": args.heldout_rollouts,
            "max_generations": args.max_generations,
            "proposals_per_generation": 2,
            "minibatch_size": 12,
        },
        "runs": [],
    }
    process_env = os.environ.copy()
    process_env["SYNTH_OPTIMIZERS_DISABLE_USAGE_REGISTRATION"] = "1"
    process_env["RUST_BACKTRACE"] = "1"

    for environment in environments:
        for algorithm in algorithms:
            config = profiles[environment] if algorithm == "gepa" else wrappers[(environment, algorithm)]
            binary = "gepa_baseline" if algorithm == "gepa" else "marl_promptopt"
            command = [
                "cargo",
                "run",
                "-p",
                "synth_marl_promptopt",
                "--bin",
                binary,
                "--",
                "--config",
                str(config),
            ]
            run_name = f"{environment}_{algorithm}"
            stdout_path = log_dir / f"{run_name}.stdout.json"
            stderr_path = log_dir / f"{run_name}.stderr.log"
            print(f"START {run_name}", flush=True)
            started = time.time()
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                try:
                    completed = subprocess.run(
                        command,
                        cwd=optimizer_root,
                        env=process_env,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        timeout=args.run_timeout_seconds,
                        check=False,
                    )
                    returncode = completed.returncode
                    failure = None
                except subprocess.TimeoutExpired:
                    returncode = 124
                    failure = f"timeout after {args.run_timeout_seconds}s"
            record: dict[str, object] = {
                "environment": environment,
                "algorithm": algorithm,
                "config": str(config),
                "returncode": returncode,
                "duration_seconds": round(time.time() - started, 3),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            }
            if failure is not None:
                record["failure"] = failure
            elif returncode == 0:
                try:
                    payload = json.loads(stdout_path.read_text(encoding="utf-8"))
                    record["summary"] = result_summary(algorithm, payload)
                except (OSError, json.JSONDecodeError) as error:
                    record["failure"] = f"result parse failed: {error}"
            else:
                record["failure"] = f"process exited {returncode}"
            receipt["runs"].append(record)
            (output_root / "matrix_receipt.json").write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            print(
                f"END {run_name} returncode={returncode} duration={record['duration_seconds']}",
                flush=True,
            )
            if returncode != 0 and args.fail_fast:
                return returncode

    return 0 if all(run["returncode"] == 0 for run in receipt["runs"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
