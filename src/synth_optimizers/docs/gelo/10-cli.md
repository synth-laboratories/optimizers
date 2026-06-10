# CLI reference

GELO commands live under `synth-optimizers gelo`. **Hosted submit is the only customer
execution path** — there is no `gelo run`.

```bash
synth-optimizers gelo startup
synth-optimizers gelo materialize --preset crafter_smoke --container-url http://127.0.0.1:8943 -o gelo.json
synth-optimizers gelo submit --config crafter_goex.json --follow
synth-optimizers gelo watch goex_... --slice board
synth-optimizers gelo console --port 8767
```

## `gelo startup`

Read the hosted optimizer startup catalog through the Synth API.

```bash
export SYNTH_API_KEY="..."
synth-optimizers gelo startup --json
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--base-url` | `SYNTH_BACKEND_URL` or `https://api.usesynth.ai` | Synth API base. |
| `--api-key-env` | `SYNTH_API_KEY` | Env var holding the API key. |
| `--timeout-seconds` | `120` | HTTP client timeout. |
| `--json` | off | Print the full startup catalog JSON. |

Uses `HostedOptimizerClient.startup()`.

## `gelo materialize`

Materialize public GELO config JSON from a preset or structured TOML/JSON input.
This writes config only; it does not submit or execute a run.

```bash
synth-optimizers gelo materialize \
  --preset crafter_smoke \
  --container-url http://127.0.0.1:8943 \
  -o .out/crafter_goex.json
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--preset` | source option | Built-in preset. Public presets currently ship `crafter_smoke` and `crafter`. |
| `--toml` | source option | Structured public GELO TOML or JSON input. |
| `--overlay` | — | Structured overlay for `--toml`. |
| `--container-url` | — | Direct container URL. Mutually exclusive with `--container-pool`. |
| `--container-pool` | — | Hosted pool id. Mutually exclusive with `--container-url`. |
| `--container-task-id` | — | Optional task id for `--container-pool`. |
| `--run-id` | auto | Override `run.run_id`. |
| `--proposer-rounds` | preset default | Override preset proposer rounds. |
| `--train-seed-count` | preset default | Override preset train seed count. |
| `--heldout-seed-count` | preset default | Override preset heldout seed count. |
| `--max-rollouts` | preset default | Override preset rollout cap. |
| `--policy-model` | preset default | Override preset rollout policy model. |
| `-o`, `--out` | required | Output JSON path. |
| `--json` | off | Also print the materialized config. |

Uses `GeloPreset` or `GeloMaterializer`.

## `gelo submit`

Submit a hosted GELO job through the Synth API (`SYNTH_API_KEY`).

```bash
export SYNTH_API_KEY="..."
synth-optimizers gelo submit \
  --config .out/crafter_goex_rust.json \
  --tunnel-url http://127.0.0.1:8943 \
  --follow
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--config` | source option | Materialized hosted config JSON (`GeloHostedConfig` wire shape). |
| `--preset` | source option | Built-in preset; materialized before submit. |
| `--toml` | source option | Structured public GELO TOML or JSON input. |
| `--overlay` | — | Structured overlay for `--toml`. |
| `--base-url` | `SYNTH_BACKEND_URL` or `https://api.usesynth.ai` | Synth API base. |
| `--api-key-env` | `SYNTH_API_KEY` | Env var holding the API key. |
| `--run-id` | auto | Optional client run id hint. |
| `--idempotency-key` | — | Idempotent resubmit. |
| `--project-id` | — | Optional project scope. |
| `--container-url` | config value | Override direct container URL. Mutually exclusive with pool and tunnel. |
| `--tunnel-url` | — | Open a SynthTunnel lease to a local container before submit. Mutually exclusive with direct URL and pool. |
| `--container-pool` | — | Hosted pool id. Mutually exclusive with direct URL and tunnel. |
| `--container-task-id` | — | Optional task id for `--container-pool`. |
| `--proposer-rounds` | preset default | Override preset proposer rounds. |
| `--train-seed-count` | preset default | Override preset train seed count. |
| `--heldout-seed-count` | preset default | Override preset heldout seed count. |
| `--max-rollouts` | preset default | Override preset rollout cap. |
| `--policy-model` | preset default | Override preset rollout policy model. |
| `--timeout-seconds` | `120` | HTTP client timeout. |
| `--follow` | off | Stream `/events` until terminal status. |
| `--json` | off | Print submit/terminal JSON. |

Uses `HostedOptimizerClient.submit_gelo()`.

## `gelo watch`

Watch a hosted GELO run through public run/status, state, slice, and Go-Ex event routes.

```bash
synth-optimizers gelo watch goex_... --slice board
synth-optimizers gelo watch goex_... --goex-events --after-seq 0
```

| Flag | Default | Purpose |
|------|---------|---------|
| `RUN_ID` | (required) | Hosted GELO run id. |
| `--base-url` | `SYNTH_BACKEND_URL` or `https://api.usesynth.ai` | Synth API base. |
| `--api-key-env` | `SYNTH_API_KEY` | Env var holding the API key. |
| `--timeout-seconds` | `120` | HTTP client timeout. |
| `--slice` | — | Fetch `agents`, `board`, `candidates`, `data-engine`, `frontier`, or `themes`. |
| `--goex-events` | off | Tail `/goex-events/stream` after the first state snapshot. |
| `--after-seq` | `0` | Public Go-Ex event sequence cursor. |
| `--limit` | `500` | Backfill event limit. |
| `--poll-seconds` | `2` | Poll interval when not tailing Go-Ex SSE. |
| `--once` | off | Print one snapshot and exit. |
| `--json` | off | Print newline-delimited JSON records. |

Uses `HostedOptimizerClient.get_run()`, `get_state()`, `get_state_slice()`, and
`goex_event_stream()`.

## `gelo console`

Serve the **bundled GELO docs** as navigable HTML (markdown rendered client-side via
`/api/docs/page`). Same docs engine as `gepa console`; GELO defaults to the `gelo` docs set.

```bash
synth-optimizers gelo console --host 127.0.0.1 --port 8767
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8767` | Port (GEPA console defaults to 8766). |
| `--docs` | bundled `docs/gelo/` | Override docs root. |
| `--docs-set` | `gelo` | Bundled set name under `synth_optimizers/docs/`. |

Open `http://127.0.0.1:8767/docs/` for the HTML docs viewer.

There is no local GELO run board in the public CLI (hosted observability uses
`get_state` / `goex-events` on the API). The console Dashboard tab may be empty.

## Explicit non-commands

| Not in public CLI | Where it lives |
|-------------------|----------------|
| `gelo run` | **Does not exist** — GELO is hosted-only. |
| `gelo service` | `optimizers-beta` internal serve. |
| Local theme board / TUI | `optimizers-beta` (`goex_tui`, internal scripts). |
