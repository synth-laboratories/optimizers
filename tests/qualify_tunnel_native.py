"""Build the public native DTOs through maintained mbx and explicit target guards."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    records = root / "records/tunnel-v2-20261009"
    records.mkdir(parents=True, exist_ok=True)
    os.environ.update(PYO3_PYTHON=sys.executable, PYO3_USE_ABI3_FORWARD_COMPATIBILITY="1")
    with (records / "native-build.log").open("w") as output:
        result = subprocess.run([sys.executable, str(args.launcher), "--focused", "--", "cargo", "rustc", "--manifest-path", str(root / "Cargo.toml"), "--target-dir", str(root / "target"), "--locked", "--lib", "-p", "synth_optimizers_py", "--", "-C", "link-arg=-undefined", "-C", "link-arg=dynamic_lookup"], cwd=root, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        return result.returncode
    library = root / "target/debug/lib_synth_optimizers.dylib"
    (records / "native-build.json").write_text(json.dumps({"source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(), "sha256": hashlib.sha256(library.read_bytes()).hexdigest(), "path": str(library.relative_to(root)), "scope": "candidate built, not installed or runtime qualified"}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
