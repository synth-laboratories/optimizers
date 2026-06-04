# GEPA

GEPA is reflective prompt evolution: it proposes prompt changes, rolls them out against
your task **container**, scores them, keeps a Pareto frontier of candidates, and emits
inspectable run evidence. It ships as `synth-optimizers` — a Rust optimizer core with a
Python CLI and SDK.

There are two ways to drive a run, and they share one config schema (TOML or `GepaConfig`):

- **[CLI](#/cli)** — `synth-optimizers gepa run --config gepa.toml`. Best for one-off runs,
  scripts, and CI. Also exposes the standing service, the run board, eval stats, run
  storage maintenance, and event replay/compare.
- **[SDK](#/sdk)** — `OptimizerRun(GepaConfig(...)).execute()` from Python. Best when you
  build the config programmatically or embed GEPA in a larger pipeline.

And two deeper references:

- **[Algorithms](#/algorithms/overview)** — how GEPA searches, the pipeline modes
  (`sync_serial` / `async_pipelined` / `flash_evolve`), and the proposer + policy layer.
- **[Containers](#/containers/overview)** — the task environments GEPA optimizes: the
  [catalog](#/containers/catalog) of shipped tasks and the [contract](#/containers/contract)
  for authoring your own.

## Install

```bash
pip install synth-optimizers
# or
uv add synth-optimizers
```

A run needs a task **container** (the scored environment) and a proposer (Codex on the
host or in Docker). Policy models run inside your container; the proposer runs on the host.
Rollout requests never carry proposer keys.

## The shape of a run

```
config (TOML / GepaConfig)
  ├─ [container]      which scored environment to talk to
  ├─ [candidate]      which prompt modules to optimize
  ├─ [seed_candidate] the starting prompts
  ├─ [dataset]        train / heldout seeds
  ├─ [policy]         the model that runs rollouts (inside the container)
  └─ [proposer]       the reflective model that proposes prompt edits (Codex)
        │
        ▼
  gepa run  ──►  Pareto search  ──►  run dir (events.jsonl, registry, manifest)
        │
        ▼
  gepa board  ──►  live HTML board of every run
```

## Next

- **[CLI reference](#/cli)** — every `gepa` subcommand and flag.
- **[SDK reference](#/sdk)** — `GepaConfig`, `GepaRun`, `OptimizerRun`, and the config schema.
