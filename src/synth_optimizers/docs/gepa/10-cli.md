# CLI reference

The CLI is installed as `synth-optimizers` (entry point `synth_optimizers.cli:main`).
Every command is a subcommand of `synth-optimizers`; GEPA lives under `gepa`.

```bash
synth-optimizers gepa run --config gepa.toml
synth-optimizers gepa service --db service.sqlite
synth-optimizers gepa board ~/.gepa/runs --serve
synth-optimizers events compare --left a.jsonl --right b.jsonl
```

## `gepa run`

Run a single GEPA optimization from a config file.

```bash
synth-optimizers gepa run --config gepa.toml
```

| Flag | Values | Purpose |
|------|--------|---------|
| `--config` | path (required) | The run config TOML. |
| `--proposer-execution-mode` | `local_process` `stdio` `websocket` `ws` | Override `[proposer].execution_mode` for Codex app-server runs. `local_process`/`stdio` use stdin/stdout JSON-RPC; `websocket`/`ws` use Codex's experimental local WebSocket listener. |
| `--proposer-model` | model id | Override `[proposer].model` for this run. |
| `--proposer-reasoning-effort` | `none` `low` `medium` `high` | Override `[proposer].reasoning_effort`. |
| `--proposer-service-tier` | `default` `fast` | Override the Codex service tier. `fast` uses Codex Fast mode and requires ChatGPT auth. |
| `--proposer-auth-mode` | `auto` `api_key` `chatgpt` `host` | Override `[proposer].auth_mode`. |
| `--proposer-codex-home` | path | Override `[proposer].codex_home` for ChatGPT-authenticated Codex runs. |
| `--json` | flag | Print the full result JSON instead of the terminal progress view. |

Set `SYNTH_OPTIMIZERS_TERMINAL=1` for a live token/cost split in the terminal while the
run progresses.

## `gepa service`

Run the standing HTTP service — the public worker/workspace surface. Queueing, claiming,
and lifecycle control happen over the `/runs` and `/workspace` routes.

```bash
synth-optimizers gepa service --db service.sqlite --bind 127.0.0.1:8879 --workers 10
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--db` | (required) | SQLite control-plane database. |
| `--bind` | `127.0.0.1:8879` | Host:port to bind. |
| `--worker-id` | auto | Stable worker identity. |
| `--lease-seconds` | `3600` | Lease duration for claimed work. |
| `--workers` | `10` | Worker pool size. |

## `gepa board`

A local, read-only HTML projection of run evidence — `GEPA_HOME` discovery, explicit
registry roots, and any live services it finds.

```bash
# Static HTML file:
synth-optimizers gepa board ~/.gepa/runs --out board.html --open
# Live server with SSE:
synth-optimizers gepa board --serve --port 8765
```

| Flag | Default | Purpose |
|------|---------|---------|
| `roots` (positional) | — | Additional registry roots to include alongside `GEPA_HOME`. |
| `--root` | — | Additional registry root; repeatable. |
| `--out` | `gepa_board.html` | Static HTML output path. |
| `--title` | `GEPA Run Board` | Board title. |
| `--open` | off | Open the board after start. |
| `--serve` | off | Serve a live board over SSE instead of writing a static file. |
| `--host` / `--port` | `127.0.0.1` / `8765` | Live-server bind. |
| `--service-url` | discover | Pin the board to one running `gepa service` (e.g. `http://127.0.0.1:8899`). |
| `--interval` | `2.0` | Live re-projection cadence (seconds). |
| `--json` | off | Print the normalized board JSON instead of writing HTML. |

## `gepa console`

Serve the run **board** and the bundled GEPA docs behind one local port as two tabs
(`Board` | `Docs`), flippable by click or keyboard (`1`/`2`/`T`). The docs ship with the
package, so this works from a plain `pip install` with no repo checkout.

```bash
# Board defaults to GEPA_HOME discovery; docs to the bundled set:
synth-optimizers gepa console --port 8766
# Point the board at explicit run roots:
synth-optimizers gepa console ~/.gepa/runs --root ./runs/final
```

| Flag | Default | Purpose |
|------|---------|---------|
| `roots` (positional) | — | Registry roots for the board, alongside `GEPA_HOME`. |
| `--root` | — | Additional registry root for the board; repeatable. |
| `--title` | `GEPA` | Console title (header + board). |
| `--host` / `--port` | `127.0.0.1` / `8766` | Bind address. |
| `--service-url` | discover | Pin the board to one running `gepa service`. |
| `--interval` | `2.0` | Live re-projection cadence (seconds). |
| `--docs` | bundled | Override the docs directory (defaults to the bundled GEPA docs). |
| `--docs-set` | `gepa` | Which bundled docs set to serve. |

## `gepa eval-stats`

Summarize evaluation stats across one or more runs.

```bash
synth-optimizers gepa eval-stats --runs run_a/ run_b/
```

| Flag | Purpose |
|------|---------|
| `--runs` | One or more run directories or roots containing `transitions.sqlite` files (required). |
| `--no-write-json` | Do not write per-run `stats.json` next to `transitions.sqlite`. |
| `--json` | Print stats JSON instead of the table. |

## `gepa runs compact` / `gepa runs delete`

Maintenance for run storage. Both accept run directories or run IDs, or scan a `--root`.

```bash
synth-optimizers gepa runs compact --root ~/.gepa/runs --all-terminal --older-than 7d --yes
synth-optimizers gepa runs delete  --root ~/.gepa/runs --status failed --yes
```

| Flag | Applies to | Purpose |
|------|-----------|---------|
| `runs` (positional) | both | Run directories or run IDs. |
| `--root` | both | Runs root for run IDs and bulk scans; repeatable. |
| `--all-terminal` | both | Include all terminal-looking runs under `--root`. |
| `--older-than` | both | Only bulk runs older than e.g. `7d`, `12h`, `30m`. |
| `--status` | both | Terminal status to include for bulk scans; repeatable. |
| `--profile` | compact | `debug` `compact` `minimal` (default `compact`). |
| `--yes` | both | Apply the operation (otherwise dry-run). |
| `--json` | both | Machine-readable output. |

> Both default to a dry run. Nothing is compacted or deleted until you pass `--yes`.

## `events replay` / `events compare`

Inspect and diff event feeds.

```bash
synth-optimizers events replay  --events run/events.jsonl
synth-optimizers events compare --left a.jsonl --right b.jsonl
```

| Command | Flags |
|---------|-------|
| `events replay` | `--events` (required) — an `events.jsonl` feed to replay. |
| `events compare` | `--left`, `--right` (required) — two feeds to diff. |
