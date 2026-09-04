#!/bin/zsh
# Restart one image's container fresh, then drive one run against it.
set -u
E2E=/Users/joshuapurtell/GitHub/optimizers/docs/e2e
WORK=/tmp/synth-container-first-e2e
NAME=$1        # e.g. banking77
PORT=$2
CFG=$3
pkill -f "serve_${NAME}.py" 2>/dev/null
sleep 1
mkdir -p "$WORK"
rm -f "$WORK/server_${NAME}.log"
rm -rf "$WORK/$NAME" "$WORK/receipts_${NAME}"
if [[ "$NAME" == "dungeongrid" ]]; then
  export SYNTH_DUNGEONGRID_SCENARIOS=/Users/joshuapurtell/GitHub/gamebench/tasks/dungeongrid-multiplayer/defaults/scenarios
fi
cd /Users/joshuapurtell/GitHub/evals
nohup uv run --with uvicorn python "$E2E/serve_${NAME}.py" "$PORT" > "$WORK/server_${NAME}.log" 2>&1 &
for i in $(seq 1 40); do
  if curl -s -m 2 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then break; fi
  sleep 1
done
cd /Users/joshuapurtell/GitHub/optimizers
PYTHONPATH="$E2E" uv run synth-optimizers rl run \
  --config "$CFG" \
  --receipts "$WORK/receipts_${NAME}" \
  --plane e2e_plane:unpaid
STATUS=$?
echo "--- exit $STATUS ---"
exit "$STATUS"
