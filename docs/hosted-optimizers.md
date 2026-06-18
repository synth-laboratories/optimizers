# Hosted Optimizers

Hosted optimizer jobs run through the Synth API with `SYNTH_API_KEY`. The public
client and CLI submit to `/api/v1/optimizers/runs`; the backend owns auth,
billing, durable run status, events, and artifacts. No public workflow requires
a separate optimizer service token.

By default, `synth-optimizers` also sends a best-effort usage-registration event
when an optimizer run is kicked off. The event is intentionally content-free:
it sends only the optimizer algorithm, client surface, package name, and package
version. It does not send prompts, config, code, artifacts, repo paths, run ids,
or user content; the backend derives source IP from the request context. Disable
it with `SYNTH_OPTIMIZERS_DISABLE_USAGE_REGISTRATION=1`,
`HostedOptimizerClient(register_usage=False)`, `[usage_registration] enabled =
false` for local GEPA TOML, or the CLI `--disable-usage-registration` flag.

## GEPA

GEPA supports both local execution and hosted jobs.

```bash
export SYNTH_API_KEY="..."

synth-optimizers gepa submit \
  --config gepa.toml \
  --tunnel-url http://127.0.0.1:8765 \
  --tunnel-provider synth_tunnel \
  --follow

synth-optimizers gepa watch gepa_... --events
```

Use `--tunnel-provider synth_tunnel`, `cloudflared`, or `ngrok` to choose the
tunnel provider for a local task container. `--tunnel-url` and
`--container-pool POOL_ID` are mutually exclusive.
RunPod can host the task container for proofs; start the selected tunnel
connector in the RunPod pod so `127.0.0.1` resolves to that task service.

Python:

```python
from synth_optimizers import GepaConfig
from synth_optimizers.hosted import HostedOptimizerClient

client = HostedOptimizerClient()
response = client.submit(GepaConfig.from_toml("gepa.toml"))
record = client.wait_for_run(response.run_id, timeout_seconds=3600)
events = list(client.event_backfill(response.run_id, limit=3))
candidates = client.get_state_slice(response.run_id, "candidates")
frontier_events = list(client.algorithm_events(response.run_id, limit=10))
```

## GELO

GELO is hosted-only in the public package. There is no public `gelo run`; local
GELO execution is not part of the public product surface.

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
  --tunnel-provider synth_tunnel \
  --follow

synth-optimizers gelo watch goex_... --slice board --goex-events
```

Python:

```python
from synth_optimizers.gelo import GeloPreset
from synth_optimizers.hosted import HostedOptimizerClient

client = HostedOptimizerClient()

with client.open_tunnel("http://127.0.0.1:8943", provider="synth_tunnel") as tunnel:
    config = GeloPreset.crafter_smoke().to_config(container_tunnel=tunnel)
    response = client.submit(config)
    record = client.wait_for_run(response.run_id, timeout_seconds=3600)

board = client.get_state_slice(response.run_id, "board")
events = list(client.event_backfill(response.run_id, limit=3))
```

GELO submit defaults to launch-promo billing. Use `billing_mode="paid"` in
`client.submit(...)` / `client.submit_gelo(...)`, or `--billing-mode paid` in the
CLI, for normal paid hosted GELO without launch-promo model and concurrency gates.

## Streaming and state

Hosted GEPA and GELO share the generic optimizer observability routes:

- `events()` / `event_backfill()` for lifecycle events.
- `algorithm_event_stream()` / `algorithm_events()` for normalized algorithm events.
- `get_state()` for the latest cursor.
- `get_state_slice("candidates")` and `get_state_slice("frontier")` for live search state.

GELO also exposes compatibility aliases `goex_events()` and `goex_event_stream()`.
Prefer the generic `algorithm_*` methods for new SDK and CLI integrations.

### Extension fields

The public hosted GELO path accepts the documented base Go-Explore prompt-space
configuration. Experimental extension fields are not a launch surface; clients
should omit them unless a Synth operator provides a dated compatibility note for
the target hosted API.

## Launch Evidence

Before promoting a hosted optimizer release, capture evidence for the exact
target API and commits being promoted:

- `synth-optimizers gelo startup` shows `go-ex` available and billing feature ids configured.
- GEPA active cancel reaches terminal cancellation through the backend-hosted path.
- GEPA and GELO public run reads survive optimizer service restart/reconcile.
- Public lifecycle event evidence uses bounded `event_backfill()` reads; streaming
  `events()` tails are for monitoring, not finite launch proof.
- GELO `gelo submit --preset crafter_smoke --tunnel-url ... --follow` reaches terminal success.
- `gelo watch RUN_ID --slice board` returns public board rows without internal paths.
- Artifacts needed by the public client, such as `checkpoint_frontier`, are readable.
- Billing evidence records optimizer spend for GEPA and GELO, or a launch waiver is recorded.

The hosted launch checklist under `specifications/daily/<date>/` is authoritative
for final promotion evidence.
