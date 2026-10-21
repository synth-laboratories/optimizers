"""Installed native Banking77 Eval example; no proposer or paid rubric judge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from . import healthbench as shared

RECIPE = "eval.banking77.native-smoke.v1"
PACKAGE = "banking77-eval-1.0.0"
PROFILE = {"package_id": PACKAGE, "recipe": RECIPE, "policy_kind": "banking77-prompt.v1"}


def freeze(args):
    """Freeze upstream bytes, private gold, split roles and exact executable inputs."""
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = args.evals_root.resolve() / "containers/images"
    shutil.copytree(
        source / "banking77/banking77_classify",
        root / "banking77_classify",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy2(source / "banking77/eval_adapter.py", root / "eval_adapter.py")
    # One native consumer and provider reservation implementation, shared with
    # the qualified HealthBench profile; all bytes become declared build inputs.
    for original, destination in (
        ("eval_native.py", "eval_native.py"),
        ("eval_adapter.py", "eval_budget.py"),
        ("build_eval.py", "build_eval.py"),
    ):
        shutil.copy2(source / "healthbench2" / original, root / destination)
    (root / "build_eval.py").chmod(0o755)
    if "@sha256:" not in args.base_image:
        raise ValueError("base_image_digest_required")
    shutil.copytree(args.wheels, root / "wheels")
    shutil.copytree(args.dataset, root / "dataset")
    provenance = json.loads((root / "dataset/source-provenance.json").read_text())
    commit = provenance["commit"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("upstream_commit_required")
    rows, hashes = {}, {}
    for split in ("train", "heldout"):
        path = root / "dataset" / f"banking77-{split}.csv"
        hashes[split] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        if provenance["files"][split]["digest"] != hashes[split]:
            raise ValueError("source_provenance_digest_mismatch")
        with path.open(newline="") as handle:
            rows[split] = list(csv.DictReader(handle))
    # These are source split indices, preserved end-to-end without reindexing.
    seeds = {"baseline": 0, "selection": 40, "final": 80}
    if len(rows["heldout"]) <= max(seeds.values()):
        raise ValueError("source_panel_rows_missing")
    world = root / "banking77_classify/world.py"
    # Import only the frozen dependency-free rendering/scoring module.
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("banking77_freeze_world", world)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    prompt = {"system_prompt": module.CLASSIFY_SYSTEM}
    (root / "prompt.json").write_text(json.dumps(prompt, indent=2) + "\n")
    prices = json.loads(args.prices.read_text())
    model = "openai/gpt-4.1-mini"
    price = next(m for m in prices["data"] if m["id"] == model)["pricing"]
    panels = {
        name: [
            {
                "seed": seed,
                "split": "heldout",
                "row_digest": shared.digest(
                    {
                        "text": rows["heldout"][seed]["text"],
                        "label": rows["heldout"][seed]["category"],
                    }
                ),
            }
        ]
        for name, seed in seeds.items()
    }
    if (
        len({r[0]["row_digest"] for r in panels.values()}) != 3
        or len({rows["heldout"][s]["text"] for s in seeds.values()}) != 3
    ):
        raise ValueError("row_identity_overlap_or_mismatch")
    protocol = {
        "schema_version": "banking77.eval-protocol.v1",
        "execution_profile": "native-prepare-start.v1",
        "purpose": "execution_support_only",
        "source_provenance": provenance,
        "scoring_protocol": "banking77.normalized-label-accuracy.v1",
        "scorer_sources": {
            name: "sha256:"
            + hashlib.sha256((root / "banking77_classify" / name).read_bytes()).hexdigest()
            for name in ("runtime.py", "world.py")
        },
        "allowed_labels_digest": shared.digest(module.BANKING77_LABELS),
        "dataset_digests": hashes,
        "panels": panels,
        "prompt_digest": shared.digest(prompt),
        "trial_max_usd": 0.1,
        "truncation": "refuse_length_finish",
        "answer": {
            "provider": "openrouter",
            "wire_model": model,
            "base_url": "https://openrouter.ai/api/v1",
            "max_tokens": 128,
            "input_usd_per_token": float(price["prompt"]),
            "output_usd_per_token": float(price["completion"]),
        },
        "judge": {"kind": "deterministic", "calls": 0, "cost_usd": 0},
        "proposer": {"enabled": False, "calls": 0},
    }
    (root / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    shutil.copy2(args.prices, root / "provider-models.json")
    wheels = list((root / "wheels").glob("synth_containers-*.whl"))
    if len(wheels) != 1:
        raise ValueError("one_containers_wheel_required")
    wheel = wheels[0]
    (root / "Dockerfile").write_text(
        f'FROM {args.base_image}\nWORKDIR /app\nCOPY wheels /wheels\nRUN pip install --no-index --find-links=/wheels /wheels/*.whl && pip check\nCOPY . /app/release\nENV PYTHONPATH=/app/release PYTHONUNBUFFERED=1\nCMD ["python", "/app/release/eval_adapter.py"]\n'
    )
    (root / "container-package.toml").write_text(f'''schema_version = "synth.container-package.v1"
package_id = "{PACKAGE}"
[build]
entrypoint = "build_eval.py"
inputs = ["Dockerfile", "eval_adapter.py", "eval_native.py", "eval_budget.py", "banking77_classify", "wheels", "dataset", "prompt.json", "protocol.json", "provider-models.json"]
platform = "{args.platform}"
[runtime]
profile = "command"
command = ["python", "/app/release/eval_adapter.py"]
working_directory = "/app"
[runtime.readiness]
kind = "adapter"
adapter = "banking77-http-eval-v1"
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
reference = "secretref://eval/banking77/openrouter"
[[dependencies]]
name = "synth-containers"
reference = "wheels/{wheel.name}"
digest = "sha256:{hashlib.sha256(wheel.read_bytes()).hexdigest()}"
''')
    print(root / "container-package.toml")


def run(args):
    return shared.run(args, profile=PROFILE)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m synth_optimizers.eval.banking77")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("freeze")
    for name in ("evals-root", "wheels", "dataset", "prices", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--base-image", required=True)
    p.add_argument("--platform", default="linux/arm64", choices=("linux/arm64", "linux/amd64"))
    p = commands.add_parser("run")
    for name in ("artifact", "bundle", "home"):
        p.add_argument("--" + name, type=Path, required=True)
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
    commands = {
        "freeze": freeze,
        "run": run,
        "watch": shared.watch,
        "read": shared.read,
        "reconcile": shared.reconcile,
        "cancel": shared.cancel,
    }
    return commands[args.command](args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
