---
name: gelo
description: Use when building, configuring, or debugging GELO (Go-Explore in prompt space) hosted jobs and GELO-compatible task containers — especially per-LLM-call checkpoint creation, async rollout contracts, achievement ladders, resume semantics, checkpoint retention/eviction, and the three-layer storage model (env container vs optimizer evidence vs hosted durable substrate). GELO is hosted-only in the public package; local execute stays in optimizers-beta.
---

# GELO Skill

Use this skill for **public** GELO work in `optimizers` (SDK, CLI, presets, container
contracts) and for authoring **GELO-compatible containers** that hosted jobs call over
HTTP.

**Product rule (locked):** GELO customer execution is **hosted-only**
(`HostedOptimizerClient.submit_gelo`, `synth-optimizers gelo submit`). Unlike GEPA,
there is no public `gelo run`. Engineers iterate locally via `optimizers-beta`
(`goex_local.sh`, `goex_serve.sh`) — internal, not the customer contract.

Normative hosted API/types: `GELO_HOSTED_SDK_CLI_SPEC.md`. Launch checklist:
`Jstack/.jstack/daily_notes/2026-06-09/gelo_blog_application_lanes_and_release_quality_plan.md`.

**Bundled author docs (HTML):** `src/synth_optimizers/docs/gelo/` — same markdown→HTML
pipeline as GEPA. Preview with `synth-optimizers gelo console` → `http://127.0.0.1:8767/docs/`.
Nav map: index, CLI, SDK, hosted jobs, containers/contract. This skill is the agent
runbook; prefer bundled docs for customer-facing copy.

## First Files

Load only what the question needs:

| Question | Read first |
|----------|------------|
| Checkpoint / async rollout contract | `optimizers-beta/evals/vending_bench/GO_EX_CONTAINER_REQUIREMENTS.md` §3–5 |
| Why miner needs per-call traces | `optimizers-beta/GO_EX_CHECKPOINT_MINING_DESIGN.md`, `GO_EX_CHECKPOINT_DATA_MINER_HANDOFF.md` |
| Optimizer knobs | `optimizers-beta/examples/*_goex_*_overlay.json`, `crates/synth_go_ex/src/config.rs` |
| Container route matrix | `optimizers-beta/goex_release.txt` §7 |
| Hosted storage boundaries | `optimizers-beta/HOSTED_OPTIMIZER_STORAGE_DESIGN.md` R4, R11 |
| Public SDK / bundled docs | `src/synth_optimizers/gelo.py`, `hosted.py`, `docs/gelo/`, `GELO_HOSTED_SDK_CLI_SPEC.md` |
| Env-specific | `optimizers-beta/dungeongrid/go_ex.md`, `seven_wonders/go_ex.md` |

## Mental Model

GELO is Go-Explore in **prompt space**:

1. **Full-scope lane** — fresh (or resumed) rollouts manufacture **checkpoints** at
   policy LLM decision points.
2. **Data miner** — labels checkpoints as forward-looking **near-misses** and
   **pre-achievement** exemplars (not “this rollout was about wood”).
3. **Theme lane** — partial rollouts **resume** from theme checkpoints; verifier scores
   subgoal progress.
4. **Consolidation lane** — merged prompts evaluated on **fresh** full rollouts;
   `best_base` promotes only on paired fresh evidence.

**Without per-LLM-call checkpoints, GELO degrades to “GEPA with extra steps.”** The
miner, themes, and resume branches all depend on mid-trajectory restart points with
achievement labels.

---

## Checkpoints After LLM Calls — Requirements

### Why per-LLM-call (not termination-only)

Go-Explore learns from **decision points**: states where the next few LLM calls could
reach a subgoal but the original trajectory missed it. Those states exist **mid-rollout**,
not only at episode end.

- **Cadence:** `full_rollout_checkpoint_cadence = per_llm_call` (optimizer config).
- **Lane:** schedule is sent only for `goex_lane = core_candidate_full` fresh full
  rollouts (`executor.rs`). Theme resume children **clear** `checkpoint_schedule` so
  they do not overwrite parent checkpoint ids.
- **Sweet spot:** per policy LLM turn (~6 turns × ~125 KB ≈ 750 KB/rollout on Crafter).
  Per env-step checkpointing explodes RAM (~1.25 GB/rollout) — **do not**.

### Optimizer → container request (full-scope fresh lane)

When cadence is `per_llm_call`, the optimizer sends:

```json
{
  "submission_mode": "async",
  "checkpoint_schedule": { "mode": "per_llm_call" }
}
```

**Container obligations:**

1. `POST /rollout` returns immediately: `status: "running"`, stable `rollout_id`.
2. Policy loop runs in a **background executor** (thread pool / async task).
3. **After each policy LLM call**, snapshot **true environment state** and append to
   `scheduled_checkpoints` (see schema below).
4. `GET /rollouts/{id}/state` — poll until terminal (`completed` | `failed` | `cancelled`).
5. `GET /rollouts/{id}` — return the **full terminal record** within ~30s of terminal
   `/state`. Optimizer **refuses `/state` fallback** for natural terminal fresh rollouts
   (no degraded partial records).

### `scheduled_checkpoints` schema (hard contract)

When `checkpoint_schedule.mode = per_llm_call`, the terminal rollout record **must**
include a non-null `scheduled_checkpoints` array. Optimizer validation
(`executor.rs::scheduled_checkpoints_from_response`) **errors** on any violation.

**Per checkpoint entry (required fields):**

| Field | Type | Rule |
|-------|------|------|
| `checkpoint_id` | string | Stable, unique within container; resumable via `POST .../resume` |
| `policy_llm_call_index` | int | 1-based or consistent index per env; one entry per policy LLM turn |
| `reward` | float | Numeric progress signal at this decision point |
| `achievements` | array | Milestone IDs unlocked **so far** (may be `[]` but key must exist) |

**Terminal record (required when per_llm_call):**

| Field | Rule |
|-------|------|
| `scheduled_checkpoints` | Non-null array; `len == policy_turn_count` when turn count known |
| `final_achievements` | Terminal milestone list |
| `summary`, `usage`, `turns`, `events`, `metadata` | Objects/arrays — **never JSON `null`** (serde default does not coerce null) |
| `outcome_reward` | Numeric terminal reward (or nested under `reward_info` / `summary` per env) |

**Achievement labels** are how the miner and theme machinery index state — not raw env
observations. Each env must define an **`achievement_ladder`** (overlay + `GET /metadata`)
and emit ladder IDs in `scheduled_checkpoints[].achievements` and `final_achievements`.

### Implementation pattern (container ReAct loop)

After each policy LLM step:

```python
checkpoint = {
    "checkpoint_id": f"{rollout_id}:{policy_llm_call_index}",
    "policy_llm_call_index": policy_llm_call_index,
    "reward": normalized_reward_at_this_point,
    "achievements": unlocked_milestone_ids_so_far,
}
scheduled_checkpoints.append(checkpoint)
if checkpoint_store:
    checkpoint_store(checkpoint_id, pickle.dumps(env_state), checkpoint)
```

**At episode end:** store a **terminal checkpoint** blob even if the last LLM call
already appended to `scheduled_checkpoints`. Resume children may strip
`checkpoint_schedule`; theme resume and `POST .../checkpoints` need a retrievable
terminal snapshot (VendingBench fix pattern in `react_loop.py`).

Reference implementations:

- Crafter: `synth-cookbooks-private/containers/crafter/synth_service_app.py` (`_llm_react_rollout`, ~per_llm_call block)
- VendingBench: `optimizers-beta/evals/vending_bench/agent/react_loop.py`
- Vending store: `optimizers-beta/evals/vending_bench/service/checkpoints.py`

### Optimizer-side consumption

1. Async poll retrieves full terminal JSON (must parse as `RolloutResponse`).
2. `scheduled_checkpoints` → `mid_rollout_checkpoints` on rollout evidence.
3. `import_search_evidence` registers **each** checkpoint on the run frontier (1→N).
4. Data miner workspace gets `new_checkpoints` + `rollout_traces` with per-call
   achievements and rewards.

If `scheduled_checkpoints` is missing or sterile (`achievements=[]`, `mid=0`), the miner
produces no themes — **plumbing success, algorithm failure**.

### Run-wide checkpoint budget (optimizer)

| Knob | Default | Purpose |
|------|---------|---------|
| `full_rollout_checkpoint_cadence` | `termination` | Set `per_llm_call` for full Go-Ex |
| `full_rollout_checkpoint_budget` | `1200` | Hard cap across parallel full-lane threads |

Budget is enforced via shared atomic counter (`executor.rs`). Exceeding budget on a
batch is a **hard error** — not silent truncation.

---

## Resume + Second Checkpoint Path

GELO uses **two** checkpoint mechanisms:

| Path | Who | When |
|------|-----|------|
| **Inline `scheduled_checkpoints`** | Container during async full rollout | Per-LLM-call frontier for data miner |
| **`POST /rollouts/{id}/checkpoints`** | Optimizer after rollout | Explicit branch points; theme resume parents |

**Resume** (`dispatch_kind = resume_rollout`):

```json
POST /rollouts/{parent_rollout_id}/resume
{
  "checkpoint_id": "<id from scheduled_checkpoints or POST checkpoints>",
  "policy": { "... candidate prompt ..." }
}
```

**Requirements:**

- Restore **true environment snapshot** (`checkpoint_restore_semantics: true_environment_snapshot`).
- Return a **new** child `rollout_id`.
- Continue policy from restored state.
- Child requests: optimizer clears `checkpoint_schedule` (null) — do not re-emit
  per-LLM schedule on resume unless explicitly designing a new full-lane child.

**Default:** `allow_resume_fallback_to_fresh = false` — missing parent checkpoint or
evicted blob is a **hard fail**, not a silent fresh re-run.

---

## Storage & Management — Three Layers

Do not conflate these. GELO bugs often come from fixing one layer while another evicts
or drops data.

### Layer 1 — Environment container (authority for resume bytes)

**Owns:** serialized env state blobs needed for `POST .../resume`.

| Env | Backend | Typical size |
|-----|---------|--------------|
| Crafter | In-memory pickles (`_ROLLOUT_LIVE_SNAPSHOTS`) | ~125 KB / checkpoint |
| DungeonGrid+ | `checkpoints.sqlite` | varies |
| VendingBench | In-memory `_CHECKPOINT_BLOBS` | **large** (full sim + email + inventory) |
| NLE | Normalized resumable state | varies |

**Retention knobs (container process):**

| Env var (Crafter example) | Default | Purpose |
|---------------------------|---------|---------|
| `CRAFTER_MAX_CHECKPOINT_SNAPSHOTS` | 1200 | Cap stored checkpoint blobs |
| `CRAFTER_MAX_ROLLOUT_RECORDS` | 512 | Cap rollout metadata entries |
| `CRAFTER_MAX_ROLLOUT_SNAPSHOTS` | 256 | Cap terminal rollout snapshots |
| `CRAFTER_MAX_LIVE_SNAPSHOTS` | 64 | Cap in-flight live snapshots |
| Vending `MAX_CHECKPOINT_SNAPSHOTS` | 500 | LRU eviction of pickle blobs |

**Rules:**

- Eviction of a checkpoint still referenced by an in-flight resume → **404** → run fails
  when `allow_resume_fallback_to_fresh=false`.
- **Multi-run on one container:** parallel GELO runs sharing one Crafter must scale caps
  (`ensure_crafter.sh`: `N_RUNS * per_run_budget + headroom`) or use **one container per
  run**.
- Long episodes (VendingBench): plan sqlite or aggressive caps; checkpoint RAM is often
  the bottleneck before optimizer disk.

**Container does not own:** optimizer theme state, proposer workspaces, or hosted S3
artifacts.

### Layer 2 — Optimizer run dir (evidence + frontier metadata)

**Owns:** checkpoint **metadata** and traces, not a second copy of full env pickles.

| Artifact | Contents |
|----------|----------|
| `artifacts/goex_checkpoint_frontier.json` | Registered checkpoint ids, rewards, achievements, theme assignments |
| Rollout evidence / `platform_workspace.sqlite` | Search measurements, `mid_rollout_checkpoints` JSON |
| `state/themes.json`, miner workspaces | `new_checkpoints`, `rollout_traces` for proposer |
| `artifacts/events.jsonl` | Spine: full_rollout → miner → verifier → consolidate |

**Pruning (optimizer RAM/disk, not container):**

| Knob | Purpose |
|------|---------|
| `frontier_prune_enabled` | Trim in-memory frontier growth (default true in config) |
| `frontier_prune_soft_cap` / `retain_tail` | Bound checkpoint frontier projection |
| `rollout_evidence_compact_*` | Compact old evidence while retaining tail |

**Hosted note:** `platform_workspace.sqlite` dominates optimizers-beta disk (~95% of
terminal run dirs). Storage phase S2 snapshots this to S3; terminal local GC follows
publish (R11).

### Layer 3 — Hosted durable substrate (customer observability)

**Owns:** run status, events, cursor, artifact indexes — **not** env-internal checkpoint
blobs (R4 in `HOSTED_OPTIMIZER_STORAGE_DESIGN.md`).

| Store | What |
|-------|------|
| Postgres | `optimizer_runs`, events, `finalize_state` |
| Redis | live cursor, SSE / goex-events tail |
| S3 | terminal artifacts, sqlite snapshots, manifest |

Public SDK reads Layer 3 via `HostedOptimizerClient` (`get_state`, `goex_events`,
`get_artifact`). **Resume still hits the user’s container URL** (or tunnel) for env
state — hosted storage does not substitute for Layer 1.

---

## Container Contract Tiers

Pick explicitly when authoring a new env (`GO_EX_CONTAINER_REQUIREMENTS.md`):

| Tier | Profile | Capability |
|------|---------|------------|
| **A** | `GELO_MINIMAL` | Fresh rollouts only — no themes, no resume |
| **B** | `GELO_CHECKPOINT_FULL` | Async + per-LLM snapshots + resume routes |
| **C** | B + metadata | `/metadata`, `/task_info`, `achievement_ladder`, `/compatibility` — **required for full flywheel** |

Full GELO needs **Tier C** for production hosted jobs with theme mining.

---

## Optimizer Overlay Alignment

Container alone is insufficient. Hosted `config_json` / overlay must include:

| Knob | Full GELO value |
|------|-----------------|
| `go_ex.full_rollout_checkpoint_cadence` | `per_llm_call` |
| `go_ex.full_rollout_checkpoint_budget` | 600–1200 (scale with env + parallelism) |
| `rollout_template.submission_mode` | `async` |
| `checkpoint_restore_semantics` | `true_environment_snapshot` |
| `allow_resume_fallback_to_fresh` | `false` |
| `resume_rollouts_per_parent` | > 0 for theme lane |
| `achievement_ladder` | env-specific milestone IDs |
| `proposer_system_prompt_suffix` | env-specific (prevent Crafter bleed in miner) |
| `container_connect_timeout_seconds` | 60+ |

Smoke overlays may use fewer rounds/seeds; **do not** disable per_llm_call on containers
advertised as GELO-full.

---

## Public SDK / CLI (hosted-only)

```python
from synth_optimizers.hosted import HostedOptimizerClient
from synth_optimizers.gelo import GeloPreset  # when landed

client = HostedOptimizerClient()
with client.open_synth_tunnel("http://127.0.0.1:8943") as tunnel:
    config = GeloPreset.crafter_smoke().materialize(container_tunnel=tunnel)
    resp = client.submit_gelo(config)
    record = client.wait_for_run(resp.run_id)
    board = client.get_state_slice(resp.run_id, "board")
```

```bash
synth-optimizers gelo submit --preset crafter_smoke --tunnel-url http://127.0.0.1:8943 --follow
```

**Container URL reachability:** the optimizers-beta worker must reach the container
(from Railway: use SynthTunnel or hosted pool — not `127.0.0.1` on the user's laptop
unless tunneled).

---

## Debugging Checklist — Checkpoint / Storage

When `theme_count=0`, `imported_empty` miner, or resume 404:

1. **Container terminal record** — `GET /rollouts/{id}`: is `scheduled_checkpoints` a
   non-empty array with `achievements` on at least some entries?
2. **Null collections** — any `summary: null` or `turns: null`? Fix to `{}` / `[]`.
3. **Async path** — was rollout `submission_mode=async` end-to-end? Sync rollouts may
   omit mid-call snapshots depending on env.
4. **Lane** — was item `core_candidate_full`? Resume/theme lanes won't populate miner
   the same way.
5. **Budget** — `full_rollout_checkpoint_budget` exhausted? Error should say so.
6. **Eviction** — Crafter/Vending caps too low for multi-round resume? Bump
   `CRAFTER_MAX_*` or isolate containers per run.
7. **Failed rollouts** — finite reward on `status=failed` must not count as acceptance
   evidence (NetHack 091805 class bug).
8. **Resume children** — terminal checkpoint stored at episode end? Resume 404 on
   `create_checkpoint` often means stripped schedule + no terminal blob.
9. **Hosted vs local** — repro checkpoint issues on tunnel + hosted submit before
   blaming algorithm; confirm container inherited policy API keys inside the container
   process.

**Healthy signals in `result_manifest.json`:**

- `checkpoint_frontier_count` > 0 after round 0 full lane
- Miner `new_checkpoints` with non-empty `achievements`
- `theme_count` small (2–5), not 10+ churn
- Resume theme rollouts use checkpoint ids present in frontier artifact

---

## Validation

Focused checks (do not run full GELO jobs unless the change requires it):

- Container Python: `python -m py_compile synth_service_app.py` (or env's service app).
- Optimizer Rust: `cargo check -p synth_go_ex` from `optimizers-beta`.
- Checkpoint smoke: env-specific scripts (e.g. `dungeongrid_plus_checkpoint_smoke.py`).
- Hosted path: `optimizers-beta/HOSTED_LOCAL_E2E.md` smoke via `HostedOptimizerClient`.
- Parse contract: one manual `GET /rollouts/{id}` jq inspection of `scheduled_checkpoints`.

Do not add test files unless the user explicitly asks.

---

## Public-Safe Guardrails

- Never print API keys; policy keys are loaded **inside** the container process.
- Document `checkpoint_restore_semantics` honestly per env (true snapshot vs normalized
  state vs request replay) — do not claim parity across NLE and Crafter.
- GELO hosted-only: public docs must not promise `gelo run`.
- Achievement ladders must be env-defined — no hard-coded Crafter achievements in generic
  miner prompts without overlay `achievement_ladder`.
- Checkpoint-forced wins are development evidence; publishable uplift requires
  fresh-start post-hoc heldout (see uplift soundness note in Jstack).
