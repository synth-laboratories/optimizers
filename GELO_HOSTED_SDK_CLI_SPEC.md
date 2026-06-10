# GELO public SDK / CLI / hosted API spec

**Date:** 2026-06-09
**Repo:** `optimizers` (public `synth-optimizers` package)
**Companion:** `optimizers-beta/goex_release.txt` (wire types, storage, Railway)
**Audience:** SDK authors, backend deploy, blog CTA copy, staging acceptance

---

## 1. Architecture (locked)

### GEPA vs GELO — local vs hosted (product decision)

| Optimizer | Public **local** execute | Public **hosted** jobs |
|-----------|-------------------------|------------------------|
| **GEPA** | ✅ Yes — `GepaRun`, `gepa run`, `OptimizerRun(GepaConfig).execute()` in `synth-optimizers` | ✅ Yes — `gepa submit`, `submit_gepa` |
| **GELO** | ❌ **No** — not offered in public `synth-optimizers` | ✅ Yes — **only** customer path: `gelo submit`, `submit_gelo` |

**GELO is hosted-only** for customers. That is intentional:

- GELO execution stays on Synth infrastructure (optimizers-beta on Railway), not on the
  user's laptop as a public `gelo run`.
- The public package still provides **config authoring**: `GeloHostedConfig`, `GeloPreset`,
  `materialize` — so users build a job config, then submit it hosted.
- **Internal** engineers use `optimizers-beta` (`goex_local.sh`, `goex_serve.sh`, TUI,
  algo eval) for local iteration; that is not documented as a customer path.

**GEPA stays dual-mode:** run locally against your container for dev/smoke, or submit a
hosted job for production-scale runs. Docs and blog copy must preserve this distinction.

```text
GEPA customer paths:
  local:   synth-optimizers gepa run --config foo.toml
  hosted:  synth-optimizers gepa submit --config foo.toml  (+ SYNTH_API_KEY)

GELO customer path (hosted only):
  hosted:  synth-optimizers gelo submit --preset crafter --tunnel-url ...  (+ SYNTH_API_KEY)
  (no public gelo run)
```

### Request flow (GELO hosted)

```text
Author (SYNTH_API_KEY)
  → backend  api.usesynth.ai  /api/v1/optimizers/*
  → optimizers-beta (private Railway)  /v1/runs  algorithm=go_ex
  → user env container (Crafter, NLE, …)  checkpoint routes

Internal only (not customer contract)
  → optimizers-beta scripts (goex_local, goex_serve, materialize, algo_eval, goex_tui)
```

| Surface | Repo | Customer-facing? |
|---------|------|------------------|
| `HostedOptimizerClient`, `GeloHostedConfig`, `GeloPreset` | `optimizers` | Yes (hosted + authoring) |
| `synth-optimizers gelo *` | `optimizers` | Yes (hosted submit/watch only) |
| `/api/v1/optimizers/*` | `backend` | Yes |
| `GepaRun`, `gepa run` | `optimizers` | Yes (GEPA local — not GELO) |
| `optimizers-beta serve`, `goex_tui`, scoreboards | `optimizers-beta` | No (internal GELO dev) |

**Naming:** public product **GELO**; wire slug **`go-ex`**; run id prefix **`goex_`**.

### Bundled HTML docs (same mechanism as GEPA)

GEPA does **not** ship standalone `.html` files in the repo. Product docs are markdown under
`src/synth_optimizers/docs/<set>/`, bundled as package data (`pyproject.toml` →
`include = ["src/synth_optimizers/docs/**/*"]`), and rendered to HTML at runtime by
`docs_server.py` when you run the console.

| Set | Markdown root | Console command | Default port |
|-----|---------------|-----------------|--------------|
| GEPA | `docs/gepa/` | `synth-optimizers gepa console` | 8766 |
| GELO | `docs/gelo/` | `synth-optimizers gelo console` | 8767 |

Open `http://127.0.0.1:<port>/docs/` for the navigable HTML viewer. Agent runbook:
`skills/gelo/SKILL.md` (checkpoints/storage deep dive). Author reference: bundled
`docs/gelo/` pages (index, CLI, SDK, hosted, containers/contract).

---

## 2. Is GEPA the quality bar?

**Short answer:** GEPA is the right *shape* for hosted + local, but neither algorithm is
fully at “launch quality” on the **hosted CLI** path. GELO should **match GEPA’s hosted
SDK client** and **fix the CLI gaps both share**.

### GEPA today (`synth-optimizers`)

| Capability | Local | Hosted (public) |
|------------|-------|-----------------|
| Typed config + `execute()` | ✅ `GepaConfig`, `GepaRun`, `OptimizerRun` | N/A (hosted is async job) |
| `from_toml()` | ✅ | submit sends `config_toml` |
| Container integration | ✅ `synth-containers` | tunnel / pool via client |
| CLI `run` | ✅ `gepa run` | — |
| CLI `submit` | ✅ `gepa submit` | ✅ uses `HostedOptimizerClient.submit_gepa_toml` |
| CLI presets | ❌ | ❌ |
| CLI `--tunnel-url` | — | ✅ `gepa submit --tunnel-url` |
| CLI `watch` | — | ✅ `gepa watch` lifecycle status/events |
| `HostedOptimizerClient` | ✅ `submit_gepa`, `submit_gepa_toml`, artifacts, events | shared by CLI |
| Board / console / eval-stats | ✅ local GEPA_HOME | ❌ hosted |
| Bundled HTML docs (`docs/<set>/` + console) | ✅ `gepa console` | ✅ `gelo console` |
| Public hosted docs depth | moderate | ✅ `docs/gelo/` + skill |

### GELO today (`synth-optimizers`)

| Capability | Local (public pkg) | Hosted (public) |
|------------|-------------------|-----------------|
| Typed config | ✅ `GeloHostedConfig` sections | ✅ `to_config_json()` |
| `GeloRun.execute()` / local run | ❌ by design | — |
| `GeloPreset` + materialize | ✅ `crafter_smoke`, `crafter` | ✅ config authoring only |
| CLI `submit` | — | ✅ uses `HostedOptimizerClient` |
| CLI `startup` / `watch` | — | ✅ uses `HostedOptimizerClient` |
| CLI `materialize` | — | ✅ preset or structured TOML/JSON |
| State / goex-events on client | ✅ `get_state*`, `goex_events*` | ✅ exposed by `gelo watch` |
| SynthTunnel on client | ✅ `open_synth_tunnel` | ✅ `gelo submit --tunnel-url` |

### Target parity principle

- **Hosted path:** GELO and GEPA share one client (`HostedOptimizerClient`) and one
  backend route family. CLI for both should use the client, support tunnel + pool,
  presets, and `watch`.
- **Local execute:** **GEPA only.** `GepaRun` / `gepa run` remain the public local
  path. **GELO has no public local execute** — not deferred, **excluded** from the
  product: execution is hosted-only; local iteration is an internal optimizers-beta
  concern.
- **Local authoring for GELO:** public package owns **config types, presets,
  materialize** so users never hand-edit JSON copied from internal overlays, then
  **always** `submit_gelo` for execution.

GEPA local SDK quality is **the bar for ergonomics** on config/container setup; GELO
should match that on **authoring + submit → wait → observe → artifacts**, without a
`gelo run` equivalent.

---

## 3. Public SDK surface (normative)

### 3.1 Types

**Existing (ship as-is):**

- `GeloHostedConfig`, sections (`GeloRunSection`, `GeloContainerSection`, …)
- `GeloProposerRole`, `GeloRewardMode`, `GeloCheckpointSemantics`
- `GeloPresetName`, `GeloPreset`, `GeloMaterializer`
- `HostedOptimizerClient`, `SynthTunnelLease`, `OptimizerAlgorithmSlug.GELO`

**Current preset API:**

```python
class GeloPresetName(StrEnum):
    CRAFTER = "crafter"
    CRAFTER_SMOKE = "crafter_smoke"
    NETHACK_SMOKE = "nethack_smoke"
    DUNGEONGRID_PLUS_PICO = "dungeongrid_plus_pico"
    # nethack/dungeongrid fail loudly until their hosted configs are release-ready

@dataclass(frozen=True)
class GeloPreset:
    name: GeloPresetName
    proposer_rounds: int = 3
    train_seed_count: int = 8
    heldout_seed_count: int = 8
    max_rollouts: int = 32
    policy_model: str = "gemini-3.1-flash-lite"

    def materialize(
        self,
        *,
        container_url: str | None = None,
        container_pool: ContainerPoolTarget | Mapping[str, Any] | None = None,
        container_tunnel: SynthTunnelLease | None = None,
    ) -> dict[str, Any]: ...

class GeloMaterializer:
    """toml + overlay → config_json (public facade over optimizers-beta recipes)."""

    @staticmethod
    def from_paths(toml: Path, overlay: Path | None = None) -> GeloMaterializer: ...

    def materialize(
        self,
        *,
        container_url: str | None = None,
        container_pool: ContainerPoolTarget | Mapping[str, Any] | None = None,
        container_tunnel: SynthTunnelLease | None = None,
    ) -> dict[str, Any]: ...
```

Implementation note: `GeloMaterializer` accepts structured public GELO TOML/JSON and
converts launcher-style TOML plus overlays as an import bridge. The **public contract** is
stable JSON out, not `optimizers-beta` paths in user docs. `container_tunnel`
materialization writes `container.auth_refresh` with the SynthTunnel lease id, so long GELO
runs can refresh worker auth through the hosted backend.

### 3.2 `HostedOptimizerClient` (GELO methods)

| Method | Route | Purpose |
|--------|-------|---------|
| `startup()` | `GET /api/v1/optimizers/startup` | Catalog; `go-ex` in `submit_supported` |
| `submit_gelo(config, **kwargs)` | `POST /api/v1/optimizers/runs` | `algorithm: go-ex`, `config_json` |
| `get_run` / `wait_for_run` | `GET .../runs/{id}` | Terminal status + `finalize_state` |
| `get_artifact` | `GET .../artifacts/{name}` | `checkpoint_frontier`, `manifest`, … |
| `events` | SSE `.../events` | Lifecycle |
| `get_state` | `GET .../state` | `GeloRunCursor` |
| `get_state_slice` | `GET .../state/{slice}` | `board`, `themes`, `candidates`, … |
| `goex_events` / `goex_event_stream` | NDJSON / SSE | Incremental GELO slices |
| `open_synth_tunnel` | `POST /api/v1/synthtunnel/leases` | Local container → public URL |

**Submit kwargs (shared with GEPA):** `run_id`, `idempotency_key`, `project_id`,
`container_pool`, `container_tunnel`.

### 3.3 Canonical hosted quickstart (≤30 lines)

```python
from synth_optimizers.hosted import HostedOptimizerClient
from synth_optimizers.gelo import GeloPreset

client = HostedOptimizerClient()  # SYNTH_API_KEY, optional SYNTH_BACKEND_URL

catalog = client.startup()
assert "go-ex" in [a.value for a in catalog.submit_supported]

with client.open_synth_tunnel("http://127.0.0.1:8943") as tunnel:
    config = GeloPreset.crafter_smoke(proposer_rounds=3).materialize(
        container_tunnel=tunnel,
    )
    resp = client.submit_gelo(config)
    record = client.wait_for_run(resp.run_id, timeout_seconds=3600)

assert record.status.value == "succeeded"
board = client.get_state_slice(resp.run_id, "board")
frontier = client.get_artifact(resp.run_id, "checkpoint_frontier")
```

---

## 4. Public CLI surface (normative)

```text
synth-optimizers gelo startup [--base-url] [--json]

synth-optimizers gelo materialize \
  (--preset NAME | --toml PATH) [--overlay PATH] \
  [--container-url URL] \
  [--container-pool POOL_ID] \
  -o PATH

synth-optimizers gelo submit \
  (--preset NAME | --toml PATH | --config PATH) \
  [--base-url] [--api-key-env SYNTH_API_KEY] \
  [--tunnel-url URL] [--container-pool POOL_ID] \
  [--proposer-rounds N] [--run-id] [--idempotency-key] [--project-id] \
  [--follow] [--json]

synth-optimizers gelo watch RUN_ID \
  [--base-url] [--slice board|themes|candidates|frontier] \
  [--goex-events] [--poll-seconds 2] [--json]
```

**Behavior:**

- `submit --preset` calls `GeloPreset.materialize()` then `HostedOptimizerClient.submit_gelo`.
- `submit --tunnel-url` opens tunnel, injects container URL, closes lease on exit.
- `watch` polls `get_state` + optional slice + tails `goex_events` (no service token).
- `materialize` writes JSON only; does not execute.

**Explicitly not in public CLI (GELO hosted-only product rule):**

- `gelo run` — **will not ship**; contrast with `gepa run`. Local GELO execute is
  internal (`optimizers-beta`) only.
- `gelo service` / `gelo board` (→ internal TUI / theme board)

---

## 5. Backend public routes (must be open)

Already implemented locally; **deploy + document** for staging/prod:

| Route | GELO-specific |
|-------|----------------|
| `GET /api/v1/optimizers/startup` | lists `go-ex` available |
| `POST /api/v1/optimizers/runs` | `algorithm: "go-ex"` |
| `GET /api/v1/optimizers/runs/{id}` | `finalize_state` |
| `GET .../events` | shared |
| `GET .../artifacts/{name}` | GELO handle names |
| `GET .../state` | cursor |
| `GET .../state/{slice}` | board, themes, … |
| `GET .../goex-events` | NDJSON tail |
| `GET .../goex-events/stream` | SSE |
| `POST /api/v1/synthtunnel/leases` | local dev + staging |

Auth: **`SYNTH_API_KEY` only** on public routes. Service token is backend ↔
optimizers-beta internal.

---

## 6. GEPA vs GELO — what to unify in one PR series

| Change | Benefit |
|--------|---------|
| Refactor `gepa submit` to use `HostedOptimizerClient` | ✅ done: TOML CLI path calls `submit_gepa_toml` |
| Add `gelo startup` | ✅ done: strict `HostedOptimizerClient.startup()` catalog read |
| Add `gelo watch` | ✅ done: public run/state/slice reads plus optional Go-Ex SSE tail |
| Add `gelo submit --tunnel-url` | ✅ done: opens SynthTunnel and submits through client |
| Add `gepa submit --tunnel-url` | ✅ done: TOML submit path injects SynthTunnel as JSON |
| Add `GeloPreset` + shared preset machinery | ✅ done for `crafter_smoke` and `crafter` |
| Add optional `gepa watch` for lifecycle only | ✅ done: public run/status + lifecycle SSE |
| Add `docs/hosted-optimizers.md` in `optimizers` | ✅ done: public hosted GEPA/GELO guide |

---

## 7. Acceptance criteria (staging)

From `goex_release.txt` §18, plus SDK-specific:

- [ ] `pip install synth-optimizers` version on PyPI includes `submit_gelo`, `GeloPreset`
- [ ] `synth-optimizers gelo startup` shows `go-ex` submit_supported on api-dev
- [ ] `gelo submit --preset crafter_smoke --tunnel-url ... --follow` → `succeeded`
- [ ] `gelo watch <run_id> --slice board` prints theme rows without private paths
- [ ] `get_artifact(..., "checkpoint_frontier")` returns non-empty bytes
- [ ] Billing / usage row records `optimizer_algorithm=go-ex`
- [ ] Quickstart runs with **only** `SYNTH_API_KEY` (no `OPTIMIZERS_BETA_SERVICE_TOKEN`)

---

## 8. Implementation backlog (ordered)

| P | Item | Repo |
|---|------|------|
| Done | `GeloPreset` + `GeloMaterializer` for Crafter presets | `optimizers/gelo.py` |
| Done | CLI: `gelo materialize`, `gelo submit --preset`, `--tunnel-url` | `optimizers/cli.py` |
| Done | CLI: `gelo watch`, `gelo startup` | `optimizers/cli.py` |
| P0 | Staging proof: `gelo submit --preset crafter_smoke --tunnel-url ... --follow` | api-dev + local container |
| Done | `docs/hosted-optimizers.md` (GEPA + GELO) | `optimizers` |
| P1 | Deploy backend go-ex routes to api-dev; ReleasePhase C proof | `backend`, Railway |
| Done | Refactor `gepa submit` → `HostedOptimizerClient` | `optimizers/cli.py` |
| Done | Dev version bump + changelog prep for hosted GEPA/GELO SDK/CLI | `optimizers` |
| P2 | PyPI publish after launch evidence packet clears | `optimizers` |
| — | Public `gelo run` | **Out of scope** — GELO hosted-only; not deferred |

---

## 9. Agent skill (checkpoints + storage)

Public agent guidance for container authors and hosted integrators:

- `skills/gelo/SKILL.md` — per-LLM-call checkpoint contract, `scheduled_checkpoints`
  schema, resume semantics, three-layer storage model, overlay alignment, debug checklist.

Load this skill when implementing or reviewing GELO-compatible containers.

## 10. References

- `optimizers-beta/goex_release.txt` — types, artifacts, storage R1–R11, ReleasePhase A–D
- `optimizers-beta/HOSTED_LOCAL_E2E.md` — local proof via public client
- `optimizers/src/synth_optimizers/hosted.py` — client (implemented)
- `optimizers/src/synth_optimizers/gelo.py` — config types (implemented)
- `optimizers/src/synth_optimizers/cli.py` — `gelo submit`, `gelo startup`, `gelo watch`
- `Jstack/.jstack/daily_notes/2026-06-09/gelo_blog_application_lanes_and_release_quality_plan.md` — checklist §E
