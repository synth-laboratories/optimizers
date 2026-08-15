#!/usr/bin/env bash
# Build the Craftax eval target from a pinned gamebench checkout and print the
# digest to pin:
#
#   ./build.sh [gamebench-checkout]
#   synth-optimizers eval pin --home <eval home> \
#       --recipe eval.craftax.code-policy.smoke.v1 --digest sha256:...
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
gamebench="${1:-$HOME/Documents/GitHub/gamebench}"
task="$gamebench/tasks/craftax-singleplayer"
[ -d "$task" ] || { echo "no craftax task at $task" >&2; exit 1; }
commit="$(git -C "$gamebench" rev-parse HEAD 2>/dev/null || echo unknown)"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cp "$here/Dockerfile" "$here/target.py" "$stage/"
rsync -a --exclude '__pycache__' "$here/../shared/" "$stage/shared/"
# The task tree only; caches and build outputs would change the digest for no
# behavioural reason.
# The task tree and its `tasks/shared` sibling: `containers/codepolicy` resolves
# the shared policy sandbox relative to the task's grandparent, so the sibling
# has to keep its real position in the tree.
rsync -a --exclude '__pycache__' --exclude '.pytest_cache' --exclude 'target/' \
    "$task/" "$stage/gamebench/tasks/craftax-singleplayer/"
rsync -a --exclude '__pycache__' \
    "$gamebench/tasks/shared/" "$stage/gamebench/tasks/shared/"

docker build \
    --build-arg "GAMEBENCH_SOURCE_COMMIT=$commit" \
    -t craftax-eval-target "$stage"
docker image inspect --format '{{.Id}}' craftax-eval-target
