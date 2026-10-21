"""Installed HealthBench Eval example: freeze, build, run, watch and read.

Launch uses Containers' OCI primitive; scientific selection and result adoption
remain with EvalRunner. Nothing in read/watch launches or retries compute.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from datetime import UTC
from pathlib import Path
from types import SimpleNamespace

from .models import EvalContractError

RECIPE = "eval.healthbench.native-smoke.v1"


def digest(value):
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def freeze(args):
    """Materialize a portable startup/input bundle before any model execution."""
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = args.evals_root.resolve() / "containers/images/healthbench2"
    shutil.copytree(
        source / "healthbench_chat",
        root / "healthbench_chat",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    for name in ("eval_adapter.py", "eval_native.py", "build_eval.py"):
        shutil.copy2(source / name, root / name)
    (root / "build_eval.py").chmod(0o755)
    wheelhouse = root / "wheels"
    shutil.copytree(args.wheels, wheelhouse)
    # Refuse unfrozen base identities and foreign platform/native dependency drift.
    if "@sha256:" not in args.base_image:
        raise ValueError("base_image_digest_required")
    all_rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    selected = all_rows[:3]
    if len(selected) != 3:
        raise ValueError("three_distinct_source_rows_required")
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in selected)
    (root / "dataset.jsonl").write_text(data)
    prompt = {
        "system_prompt": "You are a careful medical assistant. Give useful, calibrated advice and identify when professional care is needed. Be concise: use at most 400 words."
    }
    (root / "prompt.json").write_text(json.dumps(prompt, indent=2) + "\n")
    runtime_hash = (
        "sha256:" + hashlib.sha256((root / "healthbench_chat/runtime.py").read_bytes()).hexdigest()
    )
    protocol = {
        "schema_version": "healthbench.eval-protocol.v2",
        "execution_profile": "native-prepare-start.v1",
        "purpose": "execution_support_only",
        "reference_protocol_equivalent": False,
        "grader_protocol": "healthbench.local_rubric.v1",
        "grader_source_digest": runtime_hash,
        "dataset_digest": "sha256:" + hashlib.sha256(data.encode()).hexdigest(),
        "source_dataset_digest": "sha256:" + hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "prompt_digest": digest(prompt),
        "trial_max_usd": 0.8,
        "proposer": {"enabled": False, "calls": 0},
        "truncation": "refuse_length_finish",
        "panels": {
            name: [
                {
                    "seed": seed,
                    "prompt_id": selected[seed].get("prompt_id"),
                    "row_digest": digest(
                        {"prompt": selected[seed]["prompt"], "rubrics": selected[seed]["rubrics"]}
                    ),
                }
            ]
            for seed, name in enumerate(("baseline", "selection", "final"))
        },
    }
    models = json.loads(args.prices.read_text())
    for role, model, limit in (
        ("answer", "openai/gpt-4.1-mini", 1536),
        ("judge", "openai/gpt-4.1", 512),
    ):
        item = next(m for m in models["data"] if m["id"] == model)
        protocol[role] = {
            "provider": "openrouter",
            "model": "gpt-4.1-2025-04-14" if role == "judge" else model,
            "wire_model": model,
            "base_url": "https://openrouter.ai/api/v1",
            "max_tokens": limit,
            "input_usd_per_token": float(item["pricing"]["prompt"]),
            "output_usd_per_token": float(item["pricing"]["completion"]),
        }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    shutil.copy2(args.prices, root / "provider-models.json")
    wheels = sorted(wheelhouse.glob("*.whl"))
    if not wheels or len([w for w in wheels if w.name.startswith("synth_containers-")]) != 1:
        raise ValueError("one_containers_wheel_required")
    # Fully offline installation: all transitive wheels are immutable build inputs.
    (root / "Dockerfile").write_text(
        f'FROM {args.base_image}\nWORKDIR /app\nCOPY wheels /wheels\nRUN pip install --no-index --find-links=/wheels /wheels/*.whl && pip check\nCOPY . /app/release\nENV PYTHONPATH=/app/release PYTHONUNBUFFERED=1\nCMD ["python", "/app/release/eval_adapter.py"]\n'
    )
    wheel = next(w for w in wheels if w.name.startswith("synth_containers-"))
    manifest = f'''schema_version = "synth.container-package.v1"
package_id = "healthbench2-eval-1.1.0"
[build]
entrypoint = "build_eval.py"
inputs = ["Dockerfile", "eval_adapter.py", "eval_native.py", "healthbench_chat", "wheels", "dataset.jsonl", "prompt.json", "protocol.json", "provider-models.json"]
platform = "{args.platform}"
[runtime]
profile = "command"
command = ["python", "/app/release/eval_adapter.py"]
working_directory = "/app"
[runtime.readiness]
kind = "adapter"
adapter = "healthbench-http-eval-v1"
[[contracts]]
name = "eval"
version = "eval.container-result.v1"
[[contracts]]
name = "gepa"
version = "synth_optimizers.gepa.v2"
[[contracts]]
name = "containers"
version = "container_contract.v1"
[[credentials]]
name = "OPENROUTER_API_KEY"
reference = "secretref://eval/healthbench/openrouter"
[[dependencies]]
name = "synth-containers"
reference = "wheels/{wheel.name}"
digest = "sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}"
'''
    (root / "container-package.toml").write_text(manifest)
    print(root / "container-package.toml")


class SharedOciExecutor:
    def __init__(self):
        from synth_containers.oci_trial import OciTrialExecutor

        self.executor = OciTrialExecutor()

    def resolve_reference(self, image, pin):
        # A local build returns an OCI config digest, independently of mutable
        # recipe tags. Resolve that exact identity through the shared primitive.
        # Tagged containerd images also expose repository digests. Passing a
        # raw ID as a repository name to resolve_reference would produce
        # sha256:ID@sha256:ID; this local profile verifies and runs the ID itself.
        actual, _ = self.executor.image_digests(pin)
        if actual != pin:
            raise EvalContractError("pinned_local_image_missing_or_mismatched")
        return pin

    def run(self, request, **kwargs):
        from synth_containers.oci_trial import ContainerRuntimeError, TrialRunRequest

        from .executor import ContainerRuntimeError as EvalRuntimeError

        if request.provider_admission_dir is not None:
            raise EvalContractError("HealthBench adapter does not accept Reflection admission")
        try:
            return self.executor.run(
                TrialRunRequest(
                    trial_id=request.trial_id,
                    image_reference=request.image_reference,
                    input_dir=request.input_dir,
                    policy_dir=request.policy_dir,
                    output_dir=request.output_dir,
                    limits=request.limits,
                    network=request.network,
                    secrets=request.secrets,
                    extra_hosts=request.extra_hosts,
                ),
                **kwargs,
            )
        except ContainerRuntimeError as error:
            raise EvalRuntimeError(str(error)) from error


def run(args):
    args.home.mkdir(parents=True, exist_ok=True)
    with (args.home / "healthbench-controller.lock").open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lease.seek(0)
            owner = json.load(lease)
            if owner.get("run_id") != args.run_id:
                raise ValueError("eval_home_owned_by_different_run")
            watch(SimpleNamespace(home=args.home, run_id=args.run_id, once=False))
            return result_exit_code(args)
        lease.seek(0)
        lease.truncate()
        json.dump({"run_id": args.run_id}, lease)
        lease.flush()
        os.fsync(lease.fileno())
        if (args.home / "runs" / args.run_id / "result_manifest.json").exists():
            read(args)
            return result_exit_code(args)
        return _run(args)


def _run(args):
    from synth_containers.package_build import plan_container_package
    from synth_containers.package_models import ContainerArtifactDescriptor

    from .home import EvalHome
    from .runner import run_worker
    from .staging import CandidateSource, stage_candidate_set

    descriptor = ContainerArtifactDescriptor.model_validate_json(args.artifact.read_text())
    if descriptor.plan.manifest.package_id != "healthbench2-eval-1.1.0":
        raise ValueError("wrong_healthbench_release")
    if (
        plan_container_package(args.bundle / "container-package.toml").input_digest
        != descriptor.plan.input_digest
    ):
        raise ValueError("release_inputs_drift")
    home = EvalHome.open(args.home.resolve())
    for operation in home.run_dir(args.run_id).glob("trials/*/output/operation.json"):
        if not (operation.parent.parent / "job_result.json").exists():
            raise ValueError("unsettled_operation_requires_original_effect_reconciliation")
    pin = descriptor.output.artifact.digest
    retained = home.root / "healthbench-artifact.json"
    if retained.exists() and json.loads(retained.read_text()) != json.loads(
        args.artifact.read_text()
    ):
        raise ValueError("existing_home_has_different_release")
    if not retained.exists():
        retained.write_text(args.artifact.read_text())
    # Home pins only this trusted catalog recipe, never a candidate-provided image.
    (home.root / "pins.toml").write_text(f'[pins."{RECIPE}"]\nimage_digest = "{pin}"\n')
    manifest = home.root / "healthbench-worker.json"
    if not manifest.exists():
        candidates = stage_candidate_set(
            home,
            [
                CandidateSource(
                    "baseline",
                    args.bundle / "prompt.json",
                    "prompt.json",
                    "healthbench-prompt.v1",
                    True,
                )
            ],
        )
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": "eval.worker-manifest.v1",
                    "run_id": args.run_id,
                    "recipe_id": RECIPE,
                    "home": str(home.root),
                    "candidate_set_path": str(candidates.root / "candidate_set.json"),
                },
                indent=2,
            )
        )
    else:
        previous = json.loads(manifest.read_text())
        if previous["run_id"] != args.run_id:
            raise ValueError("existing_home_has_different_run")
    code = run_worker(manifest, executor=SharedOciExecutor(), stream=sys.stdout)
    return code or result_exit_code(args)


def result_exit_code(args):
    value = json.loads((args.home / "runs" / args.run_id / "result_manifest.json").read_text())
    return 0 if value.get("trials") and all(t["valid"] for t in value["trials"]) else 1


def read(args):
    root = args.home.resolve() / "runs" / args.run_id
    manifest = root / "result_manifest.json"
    if not manifest.is_file():
        raise ValueError("result_not_committed")
    value = json.loads(manifest.read_text())
    for artifact in value["artifacts"]:
        path = Path(artifact["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError("artifact_missing_or_outside_run")
        if "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() != artifact["digest"]:
            raise ValueError("artifact_digest_mismatch")
    print(json.dumps(value, indent=2))
    return 0


def watch(args):
    """At most ten rollout rows and a fixed status line; read-only reconnect."""
    root = args.home.resolve() / "runs" / args.run_id
    path = root / "events.jsonl"
    while True:
        rows = {}
        status = "waiting"
        for line in path.read_text().splitlines() if path.exists() else []:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # a writer may still be committing the last line
            status = event["event"]
            if event.get("trial_id"):
                rows[event["trial_id"]] = event
        display = []
        for key, event in rows.items():
            native = event.get("container_event", {})
            operation = root / "trials" / key / "output/operation.json"
            state = json.loads(operation.read_text()).get("state") if operation.exists() else None
            job = root / "trials" / key / "job_result.json"
            if state and job.exists() and json.loads(job.read_text())["status"] == "cancelled":
                state = "cancelled; native last=" + state
            display.append(
                f"{key}  {state + '  ' if state else ''}{native.get('kind', native.get('type', native.get('event', event['event'])))}"
            )
        if sys.stdout.isatty():
            columns, height = shutil.get_terminal_size()
            visible = min(10, max(1, height - 2))
            left_width = max(12, columns - 34)
            right = [
                "Status",
                status,
                f"Trials: {len(rows)}",
                f"Visible: {min(visible, len(rows))}",
                f"Hidden: {max(0, len(rows) - visible)}",
            ]
            print("\033[H", end="")
            for i in range(visible):
                left = (display[-visible:] + [""] * visible)[i]
                label = right[i] if i < len(right) else ""
                print(f"{left[:left_width]:<{left_width}}  {label[:30]:<30}\033[K")
        else:
            for line in display[-10:]:
                print(line)
            for _ in range(10 - min(10, len(rows))):
                print()
            print(f"Status: {status}")
        if args.once or (root / "result_manifest.json").exists():
            return 0
        time.sleep(0.5)


def reconcile(args):
    """Read provider billing metadata for the original generations, without compute.

    OpenRouter: https://openrouter.ai/docs/api/api-reference/generations/get-generation
    """
    import concurrent.futures
    from datetime import datetime

    import httpx

    root = args.home.resolve() / "runs" / args.run_id
    calls = []
    for path in sorted(root.glob("trials/*/output/provider.jsonl")):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        reserved = [r for r in records if r["state"] == "reserved"]
        completed = [r for r in records if r["state"] == "completed"]
        if len(reserved) != len(completed):
            raise ValueError("unsettled_provider_effect")
        for record in completed:
            receipt = record.get("provider_receipt") or {}
            if not receipt.get("id") or receipt.get("provider") != "openrouter":
                raise ValueError("provider_generation_receipt_missing")
            calls.append(
                {
                    "trial_id": path.parents[1].name,
                    "call_id": record["call_id"],
                    "role": record["role"],
                    "generation_id": receipt["id"],
                    "completion_usage": receipt.get("usage"),
                }
            )
    if not calls or len({c["generation_id"] for c in calls}) != len(calls):
        raise ValueError("missing_or_duplicate_generation_ids")
    with httpx.Client(
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]}, timeout=30
    ) as client:

        def fetch(call):
            response = client.get(
                "https://openrouter.ai/api/v1/generation", params={"id": call["generation_id"]}
            )
            if response.status_code == 404:
                # Generation metadata can lag or be unavailable. The completed
                # response's provider-reported cost is a separate authority;
                # retain the exact source instead of inventing metadata.
                cost = (call.get("completion_usage") or {}).get("cost")
                if isinstance(cost, (int, float)) and cost >= 0:
                    return {
                        **call,
                        "provider": {"id": call["generation_id"], "total_cost": cost},
                        "charge_source": "completion_usage",
                        "metadata_lookup_status": 404,
                    }
            response.raise_for_status()
            data = response.json()["data"]
            cost = data.get("total_cost")
            if (
                data["id"] != call["generation_id"]
                or not isinstance(cost, (int, float))
                or cost < 0
            ):
                raise ValueError("provider_charge_not_final")
            return {**call, "provider": data, "charge_source": "generation_metadata"}

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            charges = list(pool.map(fetch, calls))
    receipt = {
        "run_id": args.run_id,
        "charge_final": True,
        "source": "openrouter_provider_reported_charges",
        "observed_at": datetime.now(UTC).isoformat(),
        "generations": charges,
        "total_cost_usd": sum(c["provider"]["total_cost"] for c in charges),
    }
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "generations"}))
    return 0


def cancel(args):
    from .runner import request_cancel

    found = request_cancel(args.home, args.run_id)
    print(
        json.dumps(
            {"run_id": args.run_id, "cancel_requested": found, "authority": "owned_oci_supervisor"}
        )
    )
    return 0 if found else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m synth_optimizers.eval.healthbench")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("freeze")
    for name in ("evals-root", "wheels", "dataset", "prices", "output"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--base-image", required=True)
    p.add_argument("--platform", default="linux/arm64", choices=("linux/arm64", "linux/amd64"))
    p = commands.add_parser("run")
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--home", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    for name in ("watch", "read", "reconcile", "cancel"):
        p = commands.add_parser(name)
        p.add_argument("--home", type=Path, required=True)
        p.add_argument("--run-id", required=True)
        if name == "watch":
            p.add_argument("--once", action="store_true")
        elif name == "reconcile":
            p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    return {
        "freeze": freeze,
        "run": run,
        "watch": watch,
        "read": read,
        "reconcile": reconcile,
        "cancel": cancel,
    }[args.command](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
