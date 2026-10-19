"""Stage committed inputs and build Craftax through the maintained host gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from uuid import uuid4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gamebench", type=Path)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--mbx-linux", type=Path, required=True)
    parser.add_argument("--synth-dev-root", type=Path, required=True)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--evals-root", type=Path, required=True)
    parser.add_argument("--evals-revision", default="HEAD")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    stage = (args.stage or here / ".build" / uuid4().hex).resolve()
    if not stage.is_relative_to(here.parents[1]):
        parser.error("stage must remain in this optimizer worktree")
    stage.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(
        ["git", "-C", str(args.gamebench), "rev-parse", args.revision + "^{commit}"], text=True
    ).strip()
    archive = subprocess.Popen(
        [
            "git",
            "-C",
            str(args.gamebench),
            "archive",
            commit,
            "tasks/craftax-singleplayer",
            "tasks/shared",
        ],
        stdout=subprocess.PIPE,
    )
    gamebench = stage / "gamebench"
    try:
        with tarfile.open(fileobj=archive.stdout, mode="r|") as source:
            for member in source:
                parts = Path(member.name).parts
                if any(
                    p.startswith("target")
                    or p in {"experiments", "reports", "__pycache__", ".pytest_cache"}
                    for p in parts
                ):
                    continue
                destination = gamebench / member.name
                if not destination.resolve().is_relative_to(gamebench.resolve()):
                    raise ValueError("archive path escapes build context")
                if member.isfile():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(source.extractfile(member).read())
                    destination.chmod(member.mode)
                elif member.issym() or member.islnk():
                    raise ValueError(f"source archive link needs explicit review: {member.name}")
        if archive.wait():
            raise ValueError("GameBench archive failed")
    finally:
        if archive.poll() is None:
            archive.terminate()
            archive.wait()
    for name in ("Dockerfile", "target.py"):
        shutil.copy2(here / name, stage / name)
    shutil.copytree(
        here.parent / "shared", stage / "shared", ignore=shutil.ignore_patterns("__pycache__")
    )
    (stage / "toolchain").mkdir()
    shutil.copy2(args.mbx_linux, stage / "toolchain/mbx")
    (stage / "toolchain/mbx").chmod(0o755)
    policy = stage / "build-policy"
    policy.mkdir()
    for name in (
        "storage_target_guard.py",
        "core_dev_build.py",
        "scratch_pin.py",
        "limits.py",
        "heavy_admission.py",
    ):
        shutil.copy2(args.synth_dev_root / "local_dev/scripts" / name, policy / name)
    evals_commit = subprocess.check_output(
        ["git", "-C", str(args.evals_root), "rev-parse", args.evals_revision + "^{commit}"], text=True
    ).strip()
    canonical = stage / "evals"
    model_cards = subprocess.check_output(["git", "-C", str(args.evals_root), "ls-tree", "-r", "--name-only", evals_commit, "core/model_cards"], text=True).splitlines()
    files = ["core/models.py", "core/_env.py", "core/provider_fallback.py",
             "core/party_mcp/native_comm.py", "core/model_cards/openrouter.json",
             "suites/external/nanobench/harbor/environment/reference.py"]
    files += ["suites/nonproduct/craftax/" + name for name in (
        "harness.py", "budget.py", "code_script.py", "failures.py", "failures.toml",
        "achievements.py", "achievements.toml", "tiles.py", "source.py", "source.toml")]
    files = sorted(set(files + [name for name in model_cards if name.endswith(".json")]))
    for name in files:
        path = canonical / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(["git", "-C", str(args.evals_root), "show", f"{evals_commit}:{name}"]))
        # Namespace-only import entrypoints avoid importing unrelated HTTP runners.
        for parent in path.parents:
            if parent == canonical:
                break
            (parent / "__init__.py").touch(exist_ok=True)
    (canonical / "core/party_mcp/__init__.py").write_text("from .native_comm import strip_native_comm_actions\n")
    (canonical / "suites/nonproduct/craftax/__init__.py").write_text("from .source import load_pin\n")
    pin = __import__("tomllib").loads((canonical / "suites/nonproduct/craftax/source.toml").read_text())
    if pin["source"]["commit"] != commit:
        raise ValueError("GameBench must match the canonical frontend harness source pin")
    canonical_manifest = {"evals_commit": evals_commit, "gamebench_commit": commit,
                          "files": {name: hashlib.sha256((canonical / name).read_bytes()).hexdigest() for name in files}}
    (canonical / "canonical-source.json").write_text(json.dumps(canonical_manifest, indent=2) + "\n")
    manifest = {
        "schema_version": "eval.craftax-build-inputs.v1",
        "evals_commit": evals_commit,
        "gamebench_commit": commit,
        "files": {
            str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        },
    }
    (stage / "build-inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    command = [
        "python3",
        str(args.synth_dev_root / "local_dev/scripts/core_dev_build.py"),
        "--label",
        "craftax-eval-target",
        "--wait-seconds",
        "60",
        "--",
        "docker",
        "build",
        "--progress=plain",
        "--build-arg",
        f"GAMEBENCH_SOURCE_COMMIT={commit}",
        "-t",
        "craftax-eval-target",
        str(stage),
    ]
    completed = subprocess.run(command, cwd=here.parents[1], check=False)
    if completed.returncode:
        return completed.returncode
    image = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", "craftax-eval-target"], text=True
    ).strip()
    (stage / "image.json").write_text(
        json.dumps({"image_id": image, "source": manifest}, indent=2) + "\n"
    )
    print(image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
