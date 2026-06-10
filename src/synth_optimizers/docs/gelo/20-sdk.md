# SDK reference

Public GELO surface is **hosted**: `HostedOptimizerClient` + typed config sections in
`synth_optimizers.gelo`. There is no `GeloRun.execute()` in the public package.

```python
from synth_optimizers.hosted import HostedOptimizerClient
from synth_optimizers.gelo import GeloHostedConfig, GeloPreset, GeloProposerRole  # noqa
```

## Quickstart (hosted)

```python
from synth_optimizers.hosted import HostedOptimizerClient

client = HostedOptimizerClient()  # SYNTH_API_KEY, optional SYNTH_BACKEND_URL

catalog = client.startup()
assert "go-ex" in [a.value for a in catalog.submit_supported]

with client.open_synth_tunnel("http://127.0.0.1:8943") as tunnel:
    config = GeloPreset.crafter_smoke().materialize(container_tunnel=tunnel)
    resp = client.submit_gelo(config)
    record = client.wait_for_run(resp.run_id, timeout_seconds=3600)

board = client.get_state_slice(resp.run_id, "board")
frontier = client.get_artifact(resp.run_id, "checkpoint_frontier")
```

## `HostedOptimizerClient` (GELO methods)

| Method | Purpose |
|--------|---------|
| `startup()` | Catalog; `go-ex` in `submit_supported` |
| `submit_gelo(config, **kwargs)` | `POST /api/v1/optimizers/runs` with `algorithm: go-ex` |
| `get_run` / `wait_for_run` | Status + `finalize_state` |
| `get_artifact(name)` | `checkpoint_frontier`, `manifest`, … |
| `events()` | Lifecycle SSE |
| `get_state()` | Run cursor |
| `get_state_slice(slice)` | `board`, `themes`, `candidates`, `frontier`, … |
| `goex_events` / `goex_event_stream` | Incremental GELO NDJSON/SSE |
| `open_synth_tunnel(local_url)` | Expose local container to hosted worker |

Submit kwargs: `run_id`, `idempotency_key`, `project_id`, `container_pool`, `container_tunnel`.

## `GeloHostedConfig`

Wire shape for `config_json` on submit. Sections:

- `run` — `run_id`, optional `output_dir`, `seed`
- `container` — `url`, `pool`, headers, `auth_bearer_env`, `auth_refresh`
- `taskset` — `train_seeds`, `heldout_seeds`, `profile`, `target_achievement`, `reward_mode`
- `policy` — in-container rollout LLM (`model`, `provider`, `api_key_env`, `inference_url`)
- `go_ex` — engine: `max_rollouts`, `proposer_rounds`, checkpoint cadence/budget, concurrency
- `seed_candidate` — baseline `react_system_prompt`
- `proposers` — map of `GeloProposerRole` → proposer config
- `cache`, `disk_budget` — optional

`.to_config_json()` serializes for submit.

## `GeloPreset` and `GeloMaterializer`

```python
from synth_optimizers.gelo import GeloMaterializer, GeloPreset

config = GeloPreset.crafter_smoke(proposer_rounds=3).materialize(
    container_url="http://127.0.0.1:8943",
)

config_from_toml = GeloMaterializer.from_paths(
    "crafter_goex.toml",
    "crafter_goex_overlay.json",
).materialize(container_url="http://127.0.0.1:8943")
```

Public presets currently ship `crafter_smoke` and `crafter`. Other preset names are reserved
and fail loudly until their hosted configs are release-ready. `materialize()` accepts exactly
one container authority: `container_url`, `container_pool`, or `container_tunnel`. Tunnel
materialization includes `container.auth_refresh` so the hosted worker can refresh SynthTunnel
auth during long GELO runs.

Materialize from TOML + overlay without hand-editing JSON copied from internal overlays.

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
