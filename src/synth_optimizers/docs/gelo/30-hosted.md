# Hosted jobs

GELO runs as a **hosted optimizer job** on Synth (`api.usesynth.ai`), same substrate as GEPA
but with `algorithm: "go-ex"` and a checkpoint-heavy config shape.

## Request flow

```
Client (SYNTH_API_KEY)
  → POST /api/v1/optimizers/runs  { algorithm: "go-ex", config_json: {...} }
  → backend (auth, billing, projection)
  → optimizers-beta /v1/runs
  → HTTP → your task container (rollouts, checkpoints, resume)
```

Durable status, events, and artifacts land in backend PG/Redis/S3 (see storage section in
[container contract](#/containers/contract)). **Env checkpoint blobs for resume stay in your
container** — hosted storage does not replace them.

## Public API routes

| Route | GELO use |
|-------|----------|
| `GET /api/v1/optimizers/startup` | `go-ex` available + submit_supported |
| `POST /api/v1/optimizers/runs` | Submit job |
| `GET .../runs/{id}` | Status, `finalize_state`, result |
| `GET .../events` | Lifecycle SSE |
| `GET .../artifacts/{name}` | Terminal artifacts |
| `GET .../state` | Cursor |
| `GET .../state/{slice}` | `board`, `themes`, … |
| `GET .../goex-events` | NDJSON tail |
| `POST /api/v1/synthtunnel/leases` | Local dev tunnel |

Auth: **`SYNTH_API_KEY` only** on public routes.

## Tunnel {#tunnel}

Hosted workers on Railway cannot reach `127.0.0.1` on your laptop. For local development:

```python
with client.open_tunnel("http://127.0.0.1:8943", provider="synth_tunnel") as tunnel:
    config = materialize_with_tunnel(tunnel)
    client.submit_gelo(config)
```

CLI: `--tunnel-url http://127.0.0.1:8943 --tunnel-provider synth_tunnel` on `gelo submit`.
Supported providers are `synth_tunnel`, `cloudflared`, and `ngrok`. Only SynthTunnel emits
`container.auth_refresh`; cloudflared and ngrok submit as public container URLs.

Container must be up and pass `GET /health` before submit.

## Staging vs prod claims

| Claim | Minimum gate |
|-------|----------------|
| "Submit hosted GELO jobs" | api-dev ReleasePhase C; `startup()` shows `go-ex` |
| "Durable across restarts" | Storage S1+ (not S0 alone) |
| "Reliable uplift optimizer" | Algorithm Tier B — separate from hosting |

Local E2E proof runbook: `optimizers-beta/HOSTED_LOCAL_E2E.md` (uses public client only).

## Billing

Terminal jobs should record `optimizer_algorithm=go-ex` and `optimizer_go_ex_llm_spend`
(Autumn feature). Policy LLM spend runs in-container; proposer spend on hosted worker.
