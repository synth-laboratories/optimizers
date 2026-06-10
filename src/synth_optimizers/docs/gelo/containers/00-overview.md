# Containers overview

GELO needs a **task container** that can run long-horizon agent rollouts and expose
**checkpoint-native** HTTP routes. The optimizer (hosted) calls your container; it does not
embed your environment.

## Tier ladder

| Tier | What you get | GELO algorithm use |
|------|--------------|-------------------|
| **A** | `POST /rollout` only | Smoke / measurement; **not** full flywheel |
| **B** | + `POST /rollout/checkpoint` (async) | Intermittent checkpoint mining |
| **C** | + `GET /rollout/{id}`, resume, `scheduled_checkpoints` | **Full** Tier B/C algorithm path |

Blog and hosted product claims for "Go-Explore in prompt space" assume **Tier C** on at least
one reference env (e.g. Crafter).

## Reference implementations

| Env | Notes |
|-----|--------|
| Crafter | Primary hosted E2E reference; Rust chart path for Tier C |
| NetHack | Validity cohort; `acceptance.rs` scoring bug is P0 before trust |
| VendingBench | Application lane; separate publish bar |
| Harvey-LAB | Application lane |

Internal requirements doc (optimizers-beta):
`evals/vending_bench/GO_EX_CONTAINER_REQUIREMENTS.md`.

## Doc map

- **[Contract](#/containers/contract)** — routes, rollout record, checkpoints, resume, storage,
  metadata, dispatch, acceptance checklist (normative for env authors).

## SynthTunnel

For local dev, expose the container with SynthTunnel so the hosted worker can reach it.
See [Hosted jobs § Tunnel](#/hosted#tunnel).
