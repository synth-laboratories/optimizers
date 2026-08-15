#!/usr/bin/env bash
# Build the fixture target and print the digest to pin.
#
#   ./build.sh
#   synth-optimizers eval pin --home <eval home> \
#       --recipe eval.fixture.policy-smoke.v1 --digest sha256:...
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runtime="${1:-docker}"
"$runtime" build -t synth-eval-fixture-target "$here"
"$runtime" image inspect --format '{{.Id}}' synth-eval-fixture-target
