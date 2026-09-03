#!/bin/zsh
# Restart the container fresh, then drive one run. The container keeps admitted
# attempts in memory and the executor's idempotency keys are stable across runs
# of the same run_id, so a second run against a live container is a replay of a
# terminal attempt rather than a new one.
set -u
E2E=/private/tmp/claude-501/-Users-joshuapurtell-GitHub/fae9821e-2959-4ca8-9924-c68492de7d19/scratchpad/e2e
WT=/Users/joshuapurtell/GitHub/wt-containers-cispo-conformance
pkill -f serve_container.py 2>/dev/null
pkill -f dump_wire.py 2>/dev/null
sleep 1
rm -f "$E2E/server.log"
cd "$WT"
nohup uv run --with pytest --with uvicorn python "$E2E/serve_container.py" 8199 > "$E2E/server.log" 2>&1 &
for i in $(seq 1 30); do
  if curl -s -m 2 http://127.0.0.1:8199/health > /dev/null 2>&1; then break; fi
  sleep 1
done
cd /Users/joshuapurtell/GitHub/optimizers
rm -rf "$E2E/work/receipts" "$E2E/work/checkpoints.sqlite3" "$E2E/work/runs"
PYTHONPATH="$E2E" uv run synth-optimizers rl run \
  --config "$E2E/work/run.toml" \
  --receipts "$E2E/work/receipts" \
  --plane e2e_plane:unpaid
echo "--- exit $? ---"
