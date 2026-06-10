# GELO

GELO (Go-Explore in prompt space) is Synth's **hosted-only** checkpoint-native optimizer
for long-horizon agents. It mines mid-trajectory decision points, labels themes, hill-climbs
partial rollouts, and consolidates prompts — then scores them on heldout seeds.

Unlike GEPA (bundled docs via `gepa console`), GELO does **not** offer public local
execute (`gelo run`).
Customers submit **hosted jobs** via `HostedOptimizerClient` / `synth-optimizers gelo submit`.
Config authoring (presets, materialize, typed `GeloHostedConfig`) lives in this package;
execution runs on Synth infrastructure (`api.usesynth.ai` → optimizers-beta).

## Install

```bash
pip install synth-optimizers
export SYNTH_API_KEY="..."
```

You also need a **GELO-compatible task container** (Tier C for the full flywheel) reachable
from the hosted worker — usually via [SynthTunnel](#/hosted#tunnel) during local development.

## Doc map

- **[CLI](#/cli)** — `gelo startup`, `gelo materialize`, `gelo submit`, `gelo watch`, `gelo console`.
- **[SDK](#/sdk)** — `HostedOptimizerClient`, `GeloHostedConfig`, `GeloPreset`, `GeloMaterializer`.
- **[Hosted jobs](#/hosted)** — backend API, tunnel, observability (`get_state`, `goex-events`).
- **[Containers](#/containers/overview)** — tiers, routes, rollout record, checkpoints, resume,
  storage, metadata, acceptance checklist.

## Product split (GEPA vs GELO)

| | GEPA | GELO |
|--|------|------|
| Public local execute | `gepa run` | **No** |
| Public hosted jobs | `gepa submit` | `gelo submit` (**primary path**) |
| Optimizes | Prompt modules in container program | `react_system_prompt` (v1) |
| Requires checkpoints | No | **Yes** (Tier B+) for full algorithm |

## The shape of a hosted run

```
GeloPreset / materialized config_json
  ├─ container.url or SynthTunnel lease
  ├─ taskset train/heldout seeds
  ├─ policy (in-container rollout LLM)
  ├─ go_ex engine knobs (rounds, budgets, checkpoint cadence)
  ├─ proposers (core, aux miner, verifier, …)
  └─ seed_candidate react_system_prompt
        │
        ▼
  gelo submit  ──►  api.usesynth.ai  ──►  optimizers-beta  ──►  your container
        │
        ▼
  get_state / goex-events / get_artifact(checkpoint_frontier)
```

## Agent skill

Portable runbook for agents and container authors:
[`skills/gelo/SKILL.md`](../../../../skills/gelo/SKILL.md) in the repo (checkpoints + storage deep dive).

Normative hosted API types: [`GELO_HOSTED_SDK_CLI_SPEC.md`](../../../../GELO_HOSTED_SDK_CLI_SPEC.md).

## View these docs as HTML

Bundled markdown is rendered to HTML by the local console (same mechanism as GEPA):

```bash
synth-optimizers gelo console --port 8767
# open http://127.0.0.1:8767/docs/
```

## Next

- **[CLI](#/cli)**
- **[SDK](#/sdk)**
- **[Hosted jobs](#/hosted)**
- **[Container contract](#/containers/contract)** — required reading for env authors.
