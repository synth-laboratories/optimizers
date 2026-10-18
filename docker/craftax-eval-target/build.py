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
    manifest = {
        "schema_version": "eval.craftax-build-inputs.v1",
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
