#!/usr/bin/env bash
# Build the native-Rust Craftax eval target from a pinned GameBench checkout
# and print the local immutable image id to pin. No registry is required:
#
#   ./build.sh [gamebench-checkout]
#   synth-optimizers eval pin --home <eval home> \
#       --recipe eval.craftax.code-policy.smoke.v1 --digest sha256:...
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gamebench="${1:-$HOME/Documents/GitHub/gamebench}"
task="$gamebench/tasks/craftax-singleplayer"
[ -d "$task" ] || { echo "no craftax task at $task" >&2; exit 1; }
# The tracked linux-aarch64 REPL fixture is cryptographically bound to this
# GameBench source closure. Building the image from a moving checkout can bake
# a fresh binary beside a stale fixture manifest, which the trusted verifier
# correctly rejects before a rollout. Archive the same immutable source ref the
# fixture declares; callers may override only to publish a new fixture/source
# pair deliberately.
# The runnable harness/fixture landed later than the exact Rust source closure
# recorded by that fixture. Stage the harness commit, then overlay gold_rust
# from the manifest-bound source commit so verification and the built binary
# describe the same bytes.
task_ref="${GAMEBENCH_CRAFTAX_TASK_REF:-6403e18388f525321cc3a748953c914553a59531}"
source_ref="${GAMEBENCH_CRAFTAX_SOURCE_REF:-945898b7894803ca148adf58bb4e75601e8115e2}"
task_commit="$(git -C "$gamebench" rev-parse "$task_ref^{commit}")"
source_commit="$(git -C "$gamebench" rev-parse "$source_ref^{commit}")"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cp "$here/Dockerfile" "$here/target.py" "$stage/"
rsync -a --exclude '__pycache__' "$here/../shared/" "$stage/shared/"
# Archive the task at the fixture-bound commit. `tasks/shared` was extracted
# later and does not exist at that historical ref; it is policy-harness code,
# not part of the Rust fixture source closure, so stage it from the resolved
# checkout commit and let the resulting image digest bind that combination.
mkdir -p "$stage/gamebench"
git -C "$gamebench" archive "$task_commit" tasks/craftax-singleplayer \
    | tar -x -C "$stage/gamebench"
git -C "$gamebench" archive "$source_commit" tasks/craftax-singleplayer/gold_rust \
    | tar -x -C "$stage/gamebench"
shared_commit="$(git -C "$gamebench" rev-parse 'HEAD^{commit}')"
git -C "$gamebench" archive "$shared_commit" tasks/shared \
    | tar -x -C "$stage/gamebench"
# Replace the archived task's pre-shared bubblewrap-only implementation with
# the current thin adapter. The shared authority recognizes an outer Linux
# trial container as the security boundary and avoids requiring privileged
# nested user namespaces.
git -C "$gamebench" archive "$shared_commit" \
    tasks/craftax-singleplayer/containers/codepolicy/policy_subprocess.py \
    | tar -x -C "$stage/gamebench"
cp "$here/local_mlx_policy_environment.patch" "$stage/"
git -C "$stage/gamebench" apply "$stage/local_mlx_policy_environment.patch"
# Preserve trusted evaluator tracebacks in the target's captured stderr. The
# upstream sweep intentionally emits only a stable exit-43 receipt; without the
# traceback Workshop cannot distinguish or repair image/source-closure faults.
cp "$here/run_policy_sweep_observability.patch" "$stage/"
git -C "$stage/gamebench" apply "$stage/run_policy_sweep_observability.patch"

docker build \
    --build-arg "GAMEBENCH_SOURCE_COMMIT=$source_commit" \
    -t craftax-eval-target "$stage"
docker image inspect --format '{{.Id}}' craftax-eval-target
