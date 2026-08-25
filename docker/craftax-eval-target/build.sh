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
source_ref="${GAMEBENCH_CRAFTAX_SOURCE_REF:-80c630db6ab35e7c9ae2b79eda51ac2bfc16ad6b}"
commit="$(git -C "$gamebench" rev-parse "$source_ref^{commit}")"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cp "$here/Dockerfile" "$here/target.py" "$stage/"
rsync -a --exclude '__pycache__' "$here/../shared/" "$stage/shared/"
# Archive the task and its `tasks/shared` sibling at the same commit. This also
# excludes caches and build outputs without relying on the checkout's state.
mkdir -p "$stage/gamebench"
git -C "$gamebench" archive "$commit" \
    tasks/craftax-singleplayer tasks/shared | tar -x -C "$stage/gamebench"

docker build \
    --build-arg "GAMEBENCH_SOURCE_COMMIT=$commit" \
    -t craftax-eval-target "$stage"
docker image inspect --format '{{.Id}}' craftax-eval-target
