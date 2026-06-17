# SDK reference

Public GELO surface is **hosted**: `HostedOptimizerClient` + typed config sections in
`synth_optimizers.gelo`. There is no `GeloRun.execute()` in the public package.

```python
from synth_optimizers.hosted import HostedOptimizerClient
from synth_optimizers.gelo import (  # noqa
    GeloHostedConfig,
    GeloPluginKind,
    GeloPluginSection,
    GeloPluginsSection,
    GeloPreset,
    GeloProposerRole,
)
```

## Quickstart (hosted)

```python
from synth_optimizers.hosted import HostedOptimizerClient

client = HostedOptimizerClient()  # SYNTH_API_KEY, optional SYNTH_BACKEND_URL

catalog = client.startup()
assert "go-ex" in [a.value for a in catalog.submit_supported]

with client.open_tunnel(
    "http://127.0.0.1:8943",
    provider="synth_tunnel",
) as tunnel:
    config = GeloPreset.crafter_smoke().to_config(container_tunnel=tunnel)
    resp = client.submit(config)
    record = client.wait_for_run(resp.run_id, timeout_seconds=3600)

board = client.get_state_slice(resp.run_id, "board")
frontier = client.get_artifact(resp.run_id, "checkpoint_frontier")
```

## `HostedOptimizerClient` (GELO methods)

| Method | Purpose |
|--------|---------|
| `startup()` | Catalog; `go-ex` in `submit_supported` |
| `submit(config, **kwargs)` | Shared hosted submit; reads `config.algorithm` |
| `submit_gelo(config, **kwargs)` | `POST /api/v1/optimizers/runs` with `algorithm: go-ex` |
| `get_run` / `wait_for_run` | Status + `finalize_state` |
| `get_artifact(name)` | `checkpoint_frontier`, `manifest`, … |
| `events()` | Lifecycle SSE |
| `get_state()` | Run cursor |
| `get_state_slice(slice)` | `board`, `themes`, `candidates`, `frontier`, … |
| `goex_events` / `goex_event_stream` | Incremental GELO NDJSON/SSE |
| `open_tunnel(local_url, provider=...)` | Expose local container with `synth_tunnel`, `cloudflared`, or `ngrok` |
| `open_synth_tunnel(local_url)` | Compatibility wrapper for `open_tunnel(..., provider="synth_tunnel")` |

Submit kwargs: `run_id`, `idempotency_key`, `project_id`, `container_pool`,
`container_tunnel`, `billing_mode`.

Usage registration is on by default for hosted submits. It sends only coarse
package/run-start metadata (`algorithm`, `client_surface`, package name/version);
it does not send prompts, config, code, artifacts, repo paths, run ids, or user
content. The backend derives source IP. Disable it with:

```python
client = HostedOptimizerClient(register_usage=False)
```

## `GeloHostedConfig`

Wire shape for `config_json` on submit. Sections:

- `run` — `run_id`, optional `output_dir`, `seed`
- `container` — `url`, `pool`, headers, `auth_bearer_env`, `auth_refresh`
- `taskset` — `train_seeds`, `heldout_seeds`, `profile`, `target_achievement`, `reward_mode`
- `policy` — in-container rollout LLM (`model`, `provider`, `api_key_env`, `inference_url`)
- `go_ex` — engine: `max_rollouts`, `proposer_rounds`, checkpoint cadence/budget, concurrency
- `seed_candidate` — baseline `react_system_prompt`
- `proposers` — map of `GeloProposerRole` → proposer config
- `plugins` — optional plugin lanes; SFT beta is accepted, RLVR/OPSD fail closed
- `cache`, `disk_budget` — optional

`.to_config_json()` serializes for submit. `GeloHostedConfig.algorithm` is `go-ex`, so it can
be passed directly to `HostedOptimizerClient.submit(config)`.

## Plugin lanes

GELO exposes typed plugin-lane config so public SDK code can name the extension
point without implying every backend exists. SFT is the only accepted beta lane
in this release; RLVR, OPSD, and unknown plugin kinds are rejected by the SDK
materializer and backend submit validation.

```python
from synth_optimizers.gelo import GeloPluginKind, GeloPluginSection, GeloPluginsSection

plugins = GeloPluginsSection(
    lanes=(GeloPluginSection(kind=GeloPluginKind.SFT),),
)
```

## `GeloPreset` and `GeloMaterializer`

```python
from synth_optimizers.gelo import GeloMaterializer, GeloPreset

config = GeloPreset.crafter_smoke(proposer_rounds=3).to_config(
    container_url="http://127.0.0.1:8943",
)

config_from_toml = GeloMaterializer.from_paths(
    "crafter_goex.toml",
    "crafter_goex_overlay.json",
).materialize(container_url="http://127.0.0.1:8943")
```

Public presets currently ship `crafter_smoke` and `crafter`. Other preset names are reserved
and fail loudly until their hosted configs are release-ready. `materialize()` accepts exactly
one container authority: `container_url`, `container_pool`, or `container_tunnel`. SynthTunnel
materialization includes `container.auth_refresh` so the hosted worker can refresh auth during
long GELO runs. Cloudflared and ngrok materialize as public URLs without
`container.auth_refresh`.

Use `.materialize(...)` when you need the raw config JSON dictionary for compatibility.
Materialize from TOML + overlay without hand-editing JSON copied from internal overlays.

GELO submit defaults to launch-promo billing. Pass `billing_mode="paid"` to use normal
paid hosted GELO without launch-promo model and concurrency gates.

## Config authoring vs execution

| In `optimizers` (public) | In `optimizers-beta` (internal) |
|--------------------------|-----------------------------------|
| Types, presets, materialize | `synth_go_ex` execute |
| `submit_gelo`, observability | `goex_local.sh`, serve, TUI |

Customers never need `OPTIMIZERS_BETA_SERVICE_TOKEN`.

## O11y

After submit, poll `get_state` for phase/tick/themes; use `get_state_slice("board")` for
human-readable theme rows; tail `goex_events` for incremental updates. Full normative artifact
names: `goex_release.txt` §8 in optimizers-beta.
