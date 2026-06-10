# Hosted Optimizers

Hosted optimizer jobs run through the Synth API with `SYNTH_API_KEY`. The public
client and CLI submit to `/api/v1/optimizers/runs`; the backend owns auth,
billing, durable run status, events, and artifacts. No public workflow requires
`OPTIMIZERS_BETA_SERVICE_TOKEN`.

## GEPA

GEPA supports both local execution and hosted jobs.

```bash
export SYNTH_API_KEY="..."

synth-optimizers gepa submit \
  --config gepa.toml \
  --tunnel-url http://127.0.0.1:8765 \
  --follow

synth-optimizers gepa watch gepa_... --events
```

Use `--container-pool POOL_ID` instead of `--tunnel-url` when the task container
is already hosted in a pool. The options are mutually exclusive.

Python:

```python
from synth_optimizers.hosted import HostedOptimizerClient

client = HostedOptimizerClient()
response = client.submit_gepa_toml(open("gepa.toml", encoding="utf-8").read())
record = client.wait_for_run(response.run_id, timeout_seconds=3600)
events = list(client.event_backfill(response.run_id, limit=3))
```

## GELO

GELO is hosted-only in the public package. There is no public `gelo run`; local
GELO execution remains an internal `optimizers-beta` workflow.

```bash
export SYNTH_API_KEY="..."

synth-optimizers gelo startup

synth-optimizers gelo materialize \
  --preset crafter_smoke \
  --container-url http://127.0.0.1:8943 \
  -o .out/crafter_goex.json

synth-optimizers gelo submit \
  --config .out/crafter_goex.json \
  --tunnel-url http://127.0.0.1:8943 \
  --follow

synth-optimizers gelo watch goex_... --slice board --goex-events
```

Python:

```python
from synth_optimizers.gelo import GeloPreset
from synth_optimizers.hosted import HostedOptimizerClient

client = HostedOptimizerClient()

with client.open_synth_tunnel("http://127.0.0.1:8943") as tunnel:
    config = GeloPreset.crafter_smoke().materialize(container_tunnel=tunnel)
    response = client.submit_gelo(config)
    record = client.wait_for_run(response.run_id, timeout_seconds=3600)

board = client.get_state_slice(response.run_id, "board")
events = list(client.event_backfill(response.run_id, limit=3))
```

## Launch Evidence

Before promoting a hosted optimizer release, capture evidence for the exact
target API and commits being promoted:

- `synth-optimizers gelo startup` shows `go-ex` available and billing feature ids configured.
- GEPA active cancel reaches terminal cancellation through the backend-hosted path.
- GEPA and GELO public run reads survive optimizer service restart/reconcile.
- Public lifecycle event evidence uses bounded `event_backfill()` reads; streaming
  `events()` tails are for monitoring, not finite launch proof.
- GELO `gelo submit --preset crafter_smoke --tunnel-url ... --follow` reaches terminal success.
- `gelo watch RUN_ID --slice board` returns public board rows without private paths.
- Artifacts needed by the public client, such as `checkpoint_frontier`, are readable.
- Billing evidence records optimizer spend for GEPA and GELO, or a launch waiver is recorded.

The hosted launch checklist under `specifications/daily/<date>/` is authoritative
for final promotion evidence.
