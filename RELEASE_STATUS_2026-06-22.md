# Optimizers release status — 2026-06-22

## Shipped
- **PyPI `synth-optimizers 0.2.6.dev20260622`** (daily dev build; `pip install --pre`):
  https://pypi.org/project/synth-optimizers/0.2.6.dev20260622/ — mac-arm64 abi3 wheel + sdist.
- `main` fast-forwarded `54c8750 → 62f3a2c`, tag `v0.2.6.dev20260622`.

## Merged into main (clearly-ready)
- `codex/optimizers-public-doc-cleanup-20260618` — public docs/entrypoint cleanup (clean merge).
- `feature/gepa-proposer-cost-fallback` — `fix(gepa)`: price deepseek proposer runtime outcomes.
- **OSS usage telemetry** (`hosted.py`): anonymous `install_id`, `run_complete` event with
  terminal status + uplift number, and `internal` flag (`SYNTH_OPTIMIZERS_INTERNAL`). Opt-out
  respected; forward-compatible (older backends ignore the new fields).

## Deferred — preserved on origin, NOT merged (what to do)
| Branch | What it is | Why deferred | Next action |
|---|---|---|---|
| `feat/gelo-skill-promo-terms` | Large new feature: GEPA run board, o11y schema, hosted config (+4.9k lines) | New-feature WIP, not release-ready | Land via its own reviewed PR when ready |
| `goex/agent-stall-detection` | rust `app_server` no-progress stall window | `app_server.rs` actively in flux locally | Reconcile with local WIP, then PR |
| `codex/optimizer-billing-019e/optimizers` | "Record hosted optimizer startup by default" (+5) | **Superseded** — usage-registration already in main + backend | Verify nothing unique remains, then delete |
| `release/gelo-goex-2026-06-10` | old gepa/tunnels release branch (+5) | **Superseded** by current main | Delete after confirming |
| `wip/optimizers-local-snapshot-2026-06-22` | Snapshot of uncommitted local WIP (gelo.py, tests, app_server, docs) | Untriaged local edits, preserved so they weren't lost | Triage per-file; telemetry already on main |

## Backend dependency (separate repo)
The telemetry client now SENDS `status`/`uplift`/`internal`/`install_id` + `run_complete`, but the
backend ignores them until `optimizer_usage_registrations` gains those columns (+ allows
`run_complete`). That change goes dev → staging → prod separately; until it lands, the growth
dashboard de-dupes OSS DAU/WAU by source IP as a fallback.
