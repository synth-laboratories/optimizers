# Hosted jobs

GELO runs as a **hosted optimizer job** on Synth (`api.usesynth.ai`), same substrate as GEPA
but with `algorithm: "go-ex"` and a checkpoint-heavy config shape.

## Request flow

```
Client (SYNTH_API_KEY)
  → POST /api/v1/optimizers/runs  { algorithm: "go-ex", config_json: {...} }
  → backend (auth, billing, projection)
  → hosted optimizer worker
  → HTTP → your task container (rollouts, checkpoints, resume)
```

Durable status, events, and artifacts land in backend PG/Redis/S3 (see storage section in
[container contract](#/containers/contract)). **Env checkpoint blobs for resume stay in your
container** — hosted storage does not replace them.

## Launch promo

The GELO launch promo gives the first 20 organizations a `$500` hosted Go-Ex proposer
spend grant, valid for 14 days after claim. Hosted `go-ex` submits auto-claim the grant
when slots remain, attach the GELO launch promo grant, and preflight at least
`$1` of `optimizer_go_ex_llm_spend` headroom before the run is queued. In-container policy
LLM calls are still owned by the task container.

The promo grant is reused for repeat submits from the same organization. Each organization
may have one hosted GELO run in `queued` or `running` status at once while using the promo.
Pass `billing_mode="paid"` in the SDK, or `--billing-mode paid` in the CLI, to submit against
normal hosted optimizer billing without promo slot, grant, concurrency, or GPT-proposer gates.

Promo submits require GPT models for the five paid proposer roles:
`core_proposer`, `aux_hill_climb_proposer`, `aux_data_miner_proposer`,
`aux_consolidate_proposer`, and `aux_consolidate_hill_climb_proposer`.
`theme_verifier_agent` and `terminator_agent` are intentionally exempt.

## Public API routes

| Route | GELO use |
|-------|----------|
| `GET /api/v1/optimizers/gelo-launch-promo/status` | Slots, grant status, expiry, and headroom snapshot |
| `POST /api/v1/optimizers/gelo-launch-promo/claim` | Explicit promo claim; first submit also auto-claims |
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

Local E2E proof should use the public client against the target hosted API.

## Billing

Terminal jobs should record `optimizer_algorithm=go-ex` and `optimizer_go_ex_llm_spend`
Policy LLM spend runs in-container; proposer spend runs on the hosted worker.
