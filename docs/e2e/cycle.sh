#!/bin/zsh
# Restart the container fresh, then drive one run. The container keeps admitted
# attempts in memory and the executor's idempotency keys are stable across runs
# of the same run_id, so a second run against a live container is a replay of a
# terminal attempt rather than a new one.
set -u
E2E=/Users/joshuapurtell/GitHub/optimizers/docs/e2e
WORK=/tmp/synth-container-first-e2e
WT=/Users/joshuapurtell/GitHub/wt-containers-cispo-conformance
pkill -f serve_container.py 2>/dev/null
pkill -f dump_wire.py 2>/dev/null
sleep 1
mkdir -p "$WORK"
rm -f "$WORK/server.log"
cd "$WT"
nohup uv run --with pytest --with uvicorn python "$E2E/serve_container.py" 8199 > "$WORK/server.log" 2>&1 &
for i in $(seq 1 30); do
  if curl -s -m 2 http://127.0.0.1:8199/health > /dev/null 2>&1; then break; fi
  sleep 1
done
cd /Users/joshuapurtell/GitHub/optimizers
rm -rf "$WORK/receipts" "$WORK/reference"
PYTHONPATH="$E2E" uv run synth-optimizers rl run \
  --config "$E2E/configs/run.toml" \
  --receipts "$WORK/receipts" \
  --plane e2e_plane:unpaid
STATUS=$?
echo "--- exit $STATUS ---"
exit "$STATUS"
