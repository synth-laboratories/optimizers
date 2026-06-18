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
- `plugins` — reserved extension field; omit it for the public hosted GELO path
- `cache`, `disk_budget` — optional

`.to_config_json()` serializes for submit. `GeloHostedConfig.algorithm` is `go-ex`, so it can
be passed directly to `HostedOptimizerClient.submit(config)`.

## Extension fields

GELO exposes a reserved extension section for future compatibility, but the
public hosted path accepts the documented base Go-Explore prompt-space
configuration. Omit extension lanes unless a Synth operator provides a dated
compatibility note for the target hosted API.

```python
from synth_optimizers.gelo import GeloPluginsSection

plugins = GeloPluginsSection(
    lanes=(),
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
Materialize from TOML + overlay without hand-editing hosted worker JSON.

GELO submit defaults to launch-promo billing. Pass `billing_mode="paid"` to use normal
paid hosted GELO without launch-promo model and concurrency gates.

## Config authoring vs execution

| In `optimizers` (public) | Hosted implementation |
|--------------------------|-----------------------|
| Types, presets, materialize | Executes on Synth hosted optimizer workers |
| `submit_gelo`, observability | Exposed through the public Synth API |

Customers only need `SYNTH_API_KEY` for hosted GELO submission and reads.

## O11y

After submit, poll `get_state` for phase/tick/themes; use `get_state_slice("board")` for
human-readable theme rows; tail `goex_events` for incremental updates. Full normative artifact
names: use the public hosted API response and event schema for compatibility.
