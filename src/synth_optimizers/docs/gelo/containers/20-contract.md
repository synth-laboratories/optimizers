# Container contract

Normative requirements for GELO-compatible task containers. Consolidates the agent skill,
`GO_EX_CONTAINER_REQUIREMENTS.md`, and algorithm validity notes.

## HTTP routes

### Tier A (minimum)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness before submit |
| `POST` | `/rollout` | Start a rollout; sync callers may receive a terminal record, async callers receive a stable `rollout_id` |
| `GET` | `/rollouts/{rollout_id}/state` | Lightweight status poll while the rollout is running |
| `GET` | `/rollouts/{rollout_id}` | Full rollout record; terminal body must be a parseable `RolloutRecord` |

### Tier B+

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/rollouts/{rollout_id}/checkpoints` | Create an explicit branch checkpoint after a rollout |
| `POST` | `/rollouts/{parent_rollout_id}/resume` | Start a child rollout from a parent checkpoint |
| `POST` | `/rollouts/{rollout_id}/terminate` | Request cancellation for a running rollout |

Tier C containers also expose capability metadata used by hosted gating and docs:
`GET /metadata`, `GET /task_info`, `GET /program`, `GET /compatibility?target=go_ex`,
and a task catalog or taskset endpoint.

## `checkpoint_schedule` request and `scheduled_checkpoints` terminal data

Tier C containers should emit checkpoints on a **per-LLM-call** cadence (or equivalent
decision-point schedule), not only episode end.

Wire shape (in rollout request / config):

```json
{
  "submission_mode": "async",
  "checkpoint_schedule": {
    "mode": "per_llm_call",
    "max_checkpoints": 64,
    "include_prompt_snapshot": true
  }
}
```

**Lifecycle:**

1. `POST /rollout` returns `rollout_id` immediately (`status: running`) when
   `submission_mode` is `async`.
2. The container runs the policy loop in the background and snapshots true env state after
   policy LLM calls.
3. Worker polls `GET /rollouts/{id}/state` until `status` is terminal
   (`completed` | `failed` | `cancelled`).
4. Worker fetches `GET /rollouts/{id}` for the full terminal record. Hosted GELO must not
   score a degraded `/state` payload as a successful terminal rollout.
5. Terminal JSON must parse as `RolloutRecord` with `scheduled_checkpoints[]` populated
   when `checkpoint_schedule.mode = "per_llm_call"`.

Each checkpoint entry should carry enough state to **resume** from that decision point
(observation, action history, RNG seed, env-specific snapshot bytes).

## Rollout record

Terminal response (sync `/rollout` or async terminal fetch) must include:

| Field | Required | Notes |
|-------|----------|-------|
| `status` | yes | `completed` / `failed` / … |
| `rollout_id` | yes | Stable id used by `/state`, terminal fetch, checkpoints, and resume |
| `outcome_reward` or nested `reward_info.outcome_reward` / `summary.outcome_reward` | yes | Numeric terminal score for hill-climb |
| `summary`, `usage`, `turns`, `events` | yes | Use `{}` or `[]`; do not emit JSON `null` |
| `scheduled_checkpoints` | Tier B+ when requested | Non-null array for `checkpoint_schedule.mode = "per_llm_call"` |
| `final_achievements` | Tier B+ when requested | Terminal milestone labels |
| `metadata` | yes | Use `{}` not JSON `null`; include `env`, `seed`, `policy_model`, compatibility tags when available |

**Parse rules:** Hosted worker must not treat HTTP 200 with malformed JSON as success. Failed
rollouts must not be scored as successes (NetHack `acceptance.rs` bug is a counterexample).

Per checkpoint entry, include:

| Field | Required | Notes |
|-------|----------|-------|
| `checkpoint_id` | yes | Stable id resolvable by `POST /rollouts/{parent}/resume` |
| `policy_llm_call_index` | yes | Consistent policy-turn index |
| `reward` | yes | Numeric progress signal at that decision point |
| `achievements` | yes | Array of milestone ids unlocked so far; may be empty |

## Resume / true snapshot

- Resume from checkpoint `k` must reproduce the same observable state as if the rollout had
  continued without interruption (within env tolerance).
- "True snapshot" means bytes the env can reload — not just a prompt string.
- Optimizer may request partial rollouts from checkpoint mid-trajectory; env must support
  truncated horizons and consistent scoring.
- Resume route is `POST /rollouts/{parent_rollout_id}/resume`; the request carries the
  `checkpoint_id` and candidate policy. It returns a new child `rollout_id`.
- Resume children should not re-emit per-LLM-call schedules unless explicitly requested for a
  new full-lane child.

## Three-layer storage

| Layer | Owner | Contents |
|-------|-------|----------|
| **1 — Env resume** | Your container | Checkpoint bytes, env DB/files; required for resume |
| **2 — Optimizer evidence** | Hosted optimizer worker | Frontier metadata, theme labels, candidate prompts |
| **3 — Hosted durable** | Backend PG/Redis/S3 | Run status, events, terminal artifacts, billing |

Layer 1 is **not** optional for Tier C. Layer 3 does not substitute for env-local resume data.

Retention: cap checkpoint count per rollout (`max_checkpoints`); prune old runs per
`disk_budget` in config.

## Achievement ladder

Tasksets reference `target_achievement` steps. Container must report
`scheduled_checkpoints[].achievements` and `final_achievements` consistently across train and
heldout seeds. Heldout seeds are **measurement only** — not used for search.

## Dispatch kinds

Rollout requests may specify `dispatch_kind` (e.g. full episode, partial from checkpoint,
verification-only). Env must honor or reject explicitly — no silent downgrade.

## Prompt overlay (v1)

Hosted GELO optimizes **`react_system_prompt` only** in v1. Container must apply the candidate
prompt from the request without mixing in optimizer-internal system text.

## Metadata & compatibility

Expose in `/health` or rollout metadata:

- `gelo_compat` / tier self-report
- env name, version, max horizon
- supported `checkpoint_schedule.mode` values
- `GET /compatibility?target=go_ex` with `supported: true` for Tier B/C containers

Hosted worker uses this for capability gating.

## Async lifecycle

- Do not block on long rollouts on the POST thread unless Tier A sync-only.
- Poll interval and timeouts are worker-side; container should return stable `rollout_id` and
  monotonic status transitions.
- Idempotent re-poll of terminal rollout returns the same record.

## Acceptance checklist (env author)

- [ ] `GET /health` 200 before job submit
- [ ] Tier declared (A/B/C) matches implemented routes
- [ ] `POST /rollout`, `GET /rollouts/{id}/state`, and `GET /rollouts/{id}` agree on terminal status and reward
- [ ] `checkpoint_schedule: {"mode": "per_llm_call"}` honored for Tier C
- [ ] Terminal record includes non-null `scheduled_checkpoints`
- [ ] `POST /rollouts/{id}/checkpoints` and `POST /rollouts/{parent}/resume` work without fallback to fresh rollout
- [ ] Terminal `RolloutRecord` schema validated
- [ ] Failed rollouts score as failures
- [ ] Resume from checkpoint reproduces state (smoke test)
- [ ] `react_system_prompt` overlay applied correctly
- [ ] Achievement ladder consistent on train + heldout
- [ ] `GET /compatibility?target=go_ex` reports the implemented tier and resume support
- [ ] SynthTunnel E2E with hosted submit (local dev)
- [ ] Evidence artifact: sample `checkpoint_frontier` row with real checkpoint ids

## Further reading

- Agent skill: `skills/gelo/SKILL.md` (checkpoint mining + debug checklist)
- Algorithm validity and checkpoint mining design are internal operator
  runbooks; public container authors should rely on this contract page and the
  hosted optimizer docs.
