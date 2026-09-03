#!/bin/zsh
# Restart one image's container fresh, then drive one run against it.
set -u
E2E=/private/tmp/claude-501/-Users-joshuapurtell-GitHub/fae9821e-2959-4ca8-9924-c68492de7d19/scratchpad/e2e
NAME=$1        # e.g. banking77
PORT=$2
CFG=$3
pkill -f "serve_${NAME}.py" 2>/dev/null
sleep 1
rm -f "$E2E/server_${NAME}.log"
cd /Users/joshuapurtell/GitHub/evals
nohup uv run --with uvicorn python "$E2E/serve_${NAME}.py" "$PORT" > "$E2E/server_${NAME}.log" 2>&1 &
for i in $(seq 1 40); do
  if curl -s -m 2 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then break; fi
  sleep 1
done
cd /Users/joshuapurtell/GitHub/optimizers
rm -rf "$E2E/work/receipts_${NAME}" "$E2E/work/runs_${NAME}"
PYTHONPATH="$E2E" uv run synth-optimizers rl run \
  --config "$CFG" \
  --receipts "$E2E/work/receipts_${NAME}" \
  --plane e2e_plane:unpaid
echo "--- exit $? ---"
