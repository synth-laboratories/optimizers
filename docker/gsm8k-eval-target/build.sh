#!/usr/bin/env bash
# Build the GSM8K eval target locally from a containers checkout and print the
# image id to pin:
#
#   ./build.sh [containers-checkout]
#   synth-optimizers eval pin --home <eval home> \
#       --recipe eval.mlx.local-policy.smoke.v1 --digest sha256:...
#
# The published image is built by .github/workflows/publish-gsm8k-eval-target.yml
# from the same stage.py, so the only difference is the registry digest.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
containers="${1:-$HOME/Documents/GitHub/containers}"
[ -f "$containers/src/synth_containers/platform/gsm8k_world.py" ] || {
    echo "no gsm8k_world.py under $containers/src" >&2; exit 1; }
commit="$(git -C "$containers" rev-parse HEAD 2>/dev/null || echo unknown)"
python="${PYTHON:-python3}"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
"$python" "$here/stage.py" --containers-src "$containers/src" --out "$stage/context"
revision="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["revision"])' "$stage/context/stage-receipt.json")"

docker build \
    --build-arg "GSM8K_REVISION=$revision" \
    --build-arg "CONTAINERS_SOURCE_COMMIT=$commit" \
    -t gsm8k-eval-target "$stage/context"
docker image inspect --format '{{.Id}}' gsm8k-eval-target
