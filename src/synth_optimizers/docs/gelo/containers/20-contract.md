# Container contract

Normative requirements for GELO-compatible task containers. Consolidates the agent skill,
`GO_EX_CONTAINER_REQUIREMENTS.md`, and algorithm validity notes.

## HTTP routes

### Tier A (minimum)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness before submit |
| `POST` | `/rollout` | Synchronous rollout; returns terminal `RolloutRecord` |

### Tier B+

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/rollout/checkpoint` | Start async rollout with checkpoint schedule |
| `GET` | `/rollout/{rollout_id}` | Poll status; terminal body is full `RolloutRecord` |

Resume and checkpoint bytes are env-specific — often `POST /rollout/resume` or embedded in
checkpoint payloads. Document your env's resume entrypoint in container metadata.

## `scheduled_checkpoints` (per-LLM-call)

Tier C containers should emit checkpoints on a **per-LLM-call** cadence (or equivalent
decision-point schedule), not only episode end.

Wire shape (in rollout request / config):

```json
{
  "scheduled_checkpoints": {
    "mode": "per_llm_call",
    "max_checkpoints": 64,
    "include_prompt_snapshot": true
  }
}
```

**Lifecycle:**

1. `POST /rollout/checkpoint` returns `rollout_id` immediately (`status: running`).
2. Worker polls `GET /rollout/{id}` until `status` is terminal (`completed` | `failed` | `cancelled`).
3. Terminal JSON must parse as `RolloutRecord` with `checkpoints[]` populated when mining is on.

Each checkpoint entry should carry enough state to **resume** from that decision point
(observation, action history, RNG seed, env-specific snapshot bytes).

## Rollout record

Terminal response (sync `/rollout` or async poll) must include:

| Field | Required | Notes |
|-------|----------|-------|
| `status` | yes | `completed` / `failed` / … |
| `reward` / `score` | yes | Env-native scalar for hill-climb |
| `achievement` | yes | Ladder step for taskset targets |
| `trajectory` or `events` | Tier B+ | For theme labeling |
| `checkpoints` | Tier B+ | Ordered; stable ids |
| `metadata` | recommended | `env`, `seed`, `policy_model`, compatibility tags |

**Parse rules:** Hosted worker must not treat HTTP 200 with malformed JSON as success. Failed
rollouts must not be scored as successes (NetHack `acceptance.rs` bug is a counterexample).

## Resume / true snapshot

- Resume from checkpoint `k` must reproduce the same observable state as if the rollout had
  continued without interruption (within env tolerance).
- "True snapshot" means bytes the env can reload — not just a prompt string.
- Optimizer may request partial rollouts from checkpoint mid-trajectory; env must support
  truncated horizons and consistent scoring.

## Three-layer storage

| Layer | Owner | Contents |
|-------|-------|----------|
| **1 — Env resume** | Your container | Checkpoint bytes, env DB/files; required for resume |
| **2 — Optimizer evidence** | Hosted worker / optimizers-beta | Frontier metadata, theme labels, candidate prompts |
| **3 — Hosted durable** | Backend PG/Redis/S3 | Run status, events, terminal artifacts, billing |

Layer 1 is **not** optional for Tier C. Layer 3 does not substitute for env-local resume data.

Retention: cap checkpoint count per rollout (`max_checkpoints`); prune old runs per
`disk_budget` in config.

## Achievement ladder

Tasksets reference `target_achievement` steps. Container must report `achievement` consistently
across train and heldout seeds. Heldout seeds are **measurement only** — not used for search.

## Dispatch kinds

Rollout requests may specify `dispatch_kind` (e.g. full episode, partial from checkpoint,
verification-only). Env must honor or reject explicitly — no silent downgrade.

## Prompt overlay (v1)

Hosted GELO optimizes **`react_system_prompt` only** in v1. Container must apply the candidate
prompt from the request without mixing in optimizer-internal system text.

## Metadata & compatibility

Expose in `/health` or rollout metadata:

- `gepa_geolo_compat` / tier self-report
- env name, version, max horizon
- supported `scheduled_checkpoints.mode` values

Hosted worker uses this for capability gating.

## Async lifecycle

- Do not block on long rollouts on the POST thread unless Tier A sync-only.
- Poll interval and timeouts are worker-side; container should return stable `rollout_id` and
  monotonic status transitions.
- Idempotent re-poll of terminal rollout returns the same record.

## Acceptance checklist (env author)

- [ ] `GET /health` 200 before job submit
- [ ] Tier declared (A/B/C) matches implemented routes
- [ ] `scheduled_checkpoints` honored for Tier C
- [ ] Terminal `RolloutRecord` schema validated
- [ ] Failed rollouts score as failures
- [ ] Resume from checkpoint reproduces state (smoke test)
- [ ] `react_system_prompt` overlay applied correctly
- [ ] Achievement ladder consistent on train + heldout
- [ ] SynthTunnel E2E with hosted submit (local dev)
- [ ] Evidence artifact: sample `checkpoint_frontier` row with real checkpoint ids

## Further reading

- Agent skill: `skills/gelo/SKILL.md` (checkpoint mining + debug checklist)
- Algorithm validity: `optimizers-beta/GO_EX_ALGO_VALIDITY.md`
- Checkpoint mining design: `GO_EX_CHECKPOINT_MINING_DESIGN.md` (optimizers-beta)
