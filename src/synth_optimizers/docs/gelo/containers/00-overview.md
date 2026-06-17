# Containers overview

GELO needs a **task container** that can run long-horizon agent rollouts and expose
**checkpoint-native** HTTP routes. The optimizer (hosted) calls your container; it does not
embed your environment.

## Tier ladder

| Tier | What you get | GELO algorithm use |
|------|--------------|-------------------|
| **A** | `/health`, `POST /rollout`, `GET /rollouts/{id}`, `GET /rollouts/{id}/state` | Fresh scoring and smoke measurement; **not** full flywheel |
| **B** | + `checkpoint_schedule`, terminal `scheduled_checkpoints`, `POST /rollouts/{id}/checkpoints`, `POST /rollouts/{parent}/resume`, `POST /rollouts/{id}/terminate` | Checkpoint manufacture and resume branches |
| **C** | + `/metadata`, `/task_info`, `/program`, `/compatibility`, task catalog | **Full** hosted GELO algorithm path |

Blog and hosted product claims for "Go-Explore in prompt space" assume **Tier C** on at least
one reference env (e.g. Crafter).

Public sharing note: this docs bundle intentionally does not include private Crafter source.
Use a public Tier C container implementation or a documented SynthTunnel target before
publishing a runnable Crafter walkthrough.

## Reference implementations

| Env | Notes |
|-----|--------|
| **GameBench Sokoban** | **Recommended public reference.** Self-contained react container (in-process gold), Tier B+ checkpoint/resume, `true_environment_snapshot`. Preset: `sokoban_smoke`. Source: `gamebench/tasks/sokoban-singleplayer/containers/react` |
| GameBench Crafter | In progress (Python + Rust gold); subgoal GELO wiring tracked separately |
| Crafter (legacy) | Original hosted E2E reference; private synth-containers source — confirm publish path before sharing |
| NetHack | Validity cohort; `acceptance.rs` scoring bug is P0 before trust |
| VendingBench | Application lane; separate publish bar |
| Harvey-LAB | Application lane |

For new container authors, start from the GameBench react containers
(`gamebench/tasks/*/containers/react`): they implement the HTTP contract below
(rollout, checkpoint, resume) with an embedded gold engine, so no separate gold
service is required.

Internal requirements doc (optimizers-beta):
`evals/vending_bench/GO_EX_CONTAINER_REQUIREMENTS.md`.

## Doc map

- **[Contract](#/containers/contract)** — routes, rollout record, checkpoints, resume, storage,
  metadata, dispatch, acceptance checklist (normative for env authors).

## SynthTunnel

For local dev, expose the container with SynthTunnel so the hosted worker can reach it.
See [Hosted jobs § Tunnel](#/hosted#tunnel).
