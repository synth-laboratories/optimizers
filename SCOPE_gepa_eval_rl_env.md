# Scope: GEPA proposer eval = GEPA proposer RL env

Companion to [`HANDOFF_proposer_eval_container.md`](HANDOFF_proposer_eval_container.md)
and [`aug19_gepa.md`](aug19_gepa.md). Verified against code on 2026-08-19.

**Eval and RL env are one container.** Same MDP, same fixture, same reward.
The difference is a few HTTP routes on top of the existing GEPA `http_task`
surface — the same split `ContainerClient` already speaks for GEPA vs GELO.

Do not design two images, two contracts, or two observation schemas.

---

## One MDP

```
observation  =  GEPA cursor fixture (frontier, archive, per-seed scores, traces)
                + downstream container/task info
                = the Codex proposer workspace already written to disk
action       =  proposed candidate(s)  (payload: BTreeMap<String, String>)
reward       =  train exploration + train exploitation + eval uplift
                verifiable: the inner Banking77 rollouts actually run
reset        =  import a content-addressed checkpoint, fork a child run
step         =  one bounded proposer episode (propose → inner rollout → score)
```

A **task** is `(downstream container ref, checkpoint ref)`.
A **rollout / episode** is one bounded proposer round against that fixture.

Eval consumer: `POST /rollout` on a `task_id`, read terminal reward, done.
RL consumer: the same, plus checkpoint/resume so a trainer can fork, truncate,
and continue from the search state the way GELO already does on Crafter.

Two containers per episode either way: this container wraps the optimizer,
which drives the downstream Banking77 container. That is not an eval-vs-RL
difference.

---

## Route delta — the only product split

Outer container. Follow `evals/containers/images/banking77/image.toml`
(`contract = "http_task"`). Inner Banking77 is unchanged.

`ContainerClient` already has both columns
(`synth_optimizer_platform/src/http.rs`).

| Route | Eval | RL env | Notes |
|---|---|---|---|
| `GET /health` | yes | yes | |
| `GET /metadata` | yes | yes | advertise `optimizer_contracts.gepa` **and** a gelo-compat tier |
| `GET /program` | yes | yes | **this** container's program is the proposer workspace contract, not Banking77 `stage2_system` |
| `GET /task_info` | yes | yes | |
| `GET /taskset` | yes | yes | splits = fixture pools (`train` / `heldout`), not Banking77 rows |
| `POST /taskset/tasks` | yes | yes | `task_id` like `train:0` → content-addressed fixture. Same convention as live Banking77 |
| `POST /rollout` | yes | yes | one proposer episode; `submission_mode` sync or async |
| `GET /rollouts/{id}` | yes | yes | terminal record must carry the three reward terms + per-seed vectors |
| `GET /rollouts/{id}/state` | yes | yes | |
| `POST /rollouts/{id}/terminate` | yes | yes | already on the GEPA client |
| `POST /rollouts/{id}/checkpoints` | no | **yes** | pin a restorable search state mid-episode |
| `POST /rollouts/{parent}/resume_async` | no | **yes** | fork from that pin. Client method is `resume_rollout` → `resume_async`. GELO docs say `/resume`. **Do not invent a third name** — pick the client path and document it |
| `GET /compatibility?target=go_ex` | no | **yes** | GELO gating. Not a second contract |

That is the whole split. Reward schema, fixture identity, proposer attribution,
and the inner Banking77 session are shared.

Eval does not need resume because each episode is an independent fork from the
fixture. RL needs resume so a training step can continue or branch without
re-importing.

---

## Current truth — do not rediscover

Inherits the handoff's "already there" table. Extra facts for this scope:

| Thing | Where | Note |
|---|---|---|
| Client already speaks both surfaces | `http.rs` `taskset*` + `resume_rollout` + `create_rollout_checkpoint` | no new HTTP client |
| Proposer observation already exists | `codex_app_server.rs` `proposer_metadata_read_model` / `proposer_readme_read_model` | files under `state/` **are** the observation. Pack them; do not invent a second schema |
| Action already exists | `CandidateRecord.payload: BTreeMap<String, String>` | |
| Per-seed scores already exist | `RolloutScore { example_id, task_id, reward }` | exploitation input |
| Per-example Pareto already exists | `FrontierCellRecord` | |
| `generation_boundary` checkpoints are retained | `lib.rs` `complete_generation_boundary` → `record_checkpoint` (no compact) | **trap:** they are summaries (`checkpoint_snapshot_value`), not restorable cursors. Restore only reads `GEPA_CURSOR_CHECKPOINT_KIND` latest. Do not treat gen-boundary rows as fixtures |
| GELO `RLVR` plugin kind is a dead end for this | `gelo.py` `_GELO_FUTURE_PLUGIN_KINDS` | hosted GELO plugin is not this env. This env **is** the GEPA container with two extra routes |

Engine blockers are unchanged from the handoff. They block **both** consumers,
because both call `reset` = fork from a named checkpoint:

1. Cursor checkpoints compact to stubs every tick (`record_checkpoint_compacting_previous`).
2. No fork: `initialize_or_restore_cursor` is own-run + latest only.
3. No proposer identity on `CandidateRecord`.
4. `frontier.updated` is a snapshot, not a displacement.
5. Budgets are absolute, not delta-from-fork.

---

## What the outer `/program` and `/rollout` mean

Inner Banking77 `/program` stays `stage2_system` + seed candidate. Unchanged.

Outer (this container) `/program` describes the **proposer task**:

- mutable field = the candidate prompt payload the policy/proposer writes
- seed = parent payload at the fixture
- `task_id` selects which fixture
- overlay injects the proposed payload, then the optimizer runs the inner rollouts

`POST /rollout` body is a GEPA `RolloutRequest`: candidate overlay + task_id +
policy spec (here the policy **is** the proposer). Terminal `reward` is the
combined train-exploration + train-exploitation + eval-uplift scalar;
`objective_scores` carries the three terms separately; `reward_details` carries
per-`example_id` vectors so neither consumer has to re-derive them.

---

## First eval task

One image, one fixture, two arms. This is the smallest thing that is still
an eval (and, later, an RL env with two extra routes).

**Not** `factorybench/banking77`. That suite is a factory lifecycle campaign.
This task is an `http_task` image, same pattern as
`evals/containers/images/banking77/image.toml`.

### Identity

| Field | v0 value |
|---|---|
| Image id | `gepa-proposer` |
| `[[evals]]` id | `gepa-proposer/banking77-luna-effort` (packaging; can wait) |
| Downstream image | existing `banking77` (`target_id = "banking77_classify"`) |
| Fixture | `train:0` → one content-addressed cursor checkpoint |
| Arms | `gpt-5.6-luna` `reasoning_effort=low` vs `medium` |
| Variable | that one field. Nothing else. |

### What a task instance is

`(inner Banking77 container revision, pinned GEPA cursor checkpoint)`.

The outer taskset is **fixtures**, not Banking77 rows. v0 has one:

```
GET /taskset
→ { "splits": { "train": 1 } }

POST /taskset/tasks  {"task_ids": ["train:0"]}
→ { "task_id": "train:0", "split": "train", "seed": 0,
    "fixture_id": "sha256:…",
    "downstream": "banking77@<image digest>" }
```

Inner rows stay whatever the checkpoint already pinned (`train:0`…`train:99`).
Do not re-sample them at eval time.

### Fixture source (manufacture, then freeze)

Take the matrix **full** Banking77 search shape, not smoke
(`pipeline_matrix/matrix.toml` `[families.banking77.search]`):

| Knob | Value | Why |
|---|---|---|
| train / heldout rows | 100 / 16+ (cursor holds 100 train) | smoke 24/16 cannot fire exploration |
| `max_generations` | 2 | need a gen-1 frontier to fork from |
| `proposals_per_generation` | 6 | match the family, not smoke's 2 |
| `minibatch_size` | 20 | family default |
| pipeline | `sync_serial` | do not confound with flash |
| inner policy | whatever produced the fixture; pin the digest | must match at eval |
| proposer during manufacture | `gpt-5.6-luna` low is fine | only the **eval arms** must be the comparison |

Freeze the cursor at **generation-start of gen 2** (after gen 1 has been
accepted, before the next propose). That is `banking77@gen2-frontier≤7`
(seed + up to 6). Not a `generation_boundary` summary.

Name it by hash, not by that nickname. Nickname is a lookup.

### What one episode / `POST /rollout` is

Fork `train:0`. Run **one** proposer round (6 proposals). Score those
candidates on the inner minibatch already in the cursor, then score heldout
for eval uplift. Stop. Do not start gen 3.

```json
{
  "task_id": "train:0",
  "submission_mode": "async",
  "candidate": {},
  "policy": {
    "provider": "openai",
    "model": "gpt-5.6-luna",
    "reasoning_effort": "low"
  }
}
```

The outer `policy` **is** the proposer. The container invokes it; the eval
harness does not submit prompt text. `candidate` stays empty in v0 — the
proposer writes payloads. (RL later can submit payloads the same way a
policy would; that is not v0.)

Terminal record:

- `reward` — one scalar (see combiner below)
- `objective_scores` — `train_exploration`, `train_exploitation`, `eval_uplift`
- `reward_details` — per-`example_id` vectors for both arms' candidates
- `metadata.arm` — model + effort + config hash
- `usage` — proposer + inner rollouts, split

### Reward (same `example_id` set for both arms)

Inner ids = the fixture's minibatch/pareto pool, already in the cursor.

- **Train exploitation** — mean inner train reward of this episode's
  candidates, minus the same mean over the pre-fork archive. Dense.
- **Train exploration** — per train `example_id`, `max(0, new_max − prior_holder)`.
  Sum across ids. Sparse.
- **Eval uplift** — mean heldout of this episode's candidates, minus mean
  heldout of the pre-fork archive (same heldout `example_id` set when
  per-seed rows exist; otherwise scalar `heldout_reward`). Requires
  `skip_heldout=false` so the episode actually scores heldout.
- **Scalar `reward` for v0** — unweighted sum of the three terms.
  If that hides a real gap, look at the terms separately.

### Repeats

One rollout is not an eval. Luna-low heldout already moved 0.500 / 0.5625 /
0.625 on 16 rows (`aug19_gepa.md` §4). v0 is:

- same `task_id` (`train:0`)
- **isolated cache namespace per (arm, repeat)** — write this down; do not
  share cache across arms for the first comparison
- N ≥ 3 repeats per arm
- report mean ± spread of each term, not a single number

Pass: the two arms separate by more than that spread, and the gap is
explained (which seeds, which term). "Produces a number" is not a pass.

### Two processes, one eval

```
gepa-proposer :8080   ← the eval image (optimizer + fixture)
banking77     :8765   ← inner http_task, unchanged
```

The outer image starts or is given the inner URL. Credentials: Codex/ChatGPT
for the proposer, inner policy key as in the fixture run. Episode wall is
mostly the proposer (85%+ on this family). Budget ~2–5 min per episode at
full 100-row context, × 2 arms × 3 repeats.

### Files (evals)

v0 lives in this repo at `temp/gepa_proposer/` so we can iterate without
splitting into `evals` yet. Promote to `evals/containers/images/gepa-proposer/`
when the contract is stable.

```
temp/gepa_proposer/                    # http_task image, 3 Banking77 fixtures
containers/images/gepa-proposer/       # later
```

v0 driver can be curl/`ContainerRunner` against the image. The registry row
is packaging. Do not copy `suites/product/factory/factorybench/tasks/banking77`.

### v0 does not include

- the three RL routes (404 is fine; eval never calls them)
- a second fixture / heldout fixture split
- flash_evolve, hotpotqa, crafter
- submitting candidate text from outside
- the GELO `rlvr` plugin

### Image.toml sketch

```toml
contract = "http_task"
target_id = "gepa_proposer"
image_name = "evals-gepa-proposer"
dockerfile = "http_task.Dockerfile"
context = ".."
port = 8080
required_env = []          # Codex + inner policy keys; pin in extra_env later
pull = false

[extra_env]
SYNTH_CONTAINER_TARGET = "gepa_proposer"
```

Inner Banking77 stays its own catalog row. Nesting, not a new inner contract.

---

## Build order

Phases 1–4 stay in `optimizers`. Phase 5 is **one** image in `evals`.

1. **Pin + export/import checkpoints.** Exempt fork points from compaction.
   Content-address (`banking77@gen2-frontier6`). Provenance on the child.
   *Test: fork twice → byte-identical archive, frontier, row pools.*
2. **Proposer identity + arm id** on `CandidateRecord`, through resume and fork.
3. **Per-seed frontier displacement** records (new max, prior holder, delta).
4. **Delta budgets** from the fork point; idempotent resume (do not double-charge).
5. **The container** — `http_task` image implementing the shared column, then
   the two RL routes. Register in `catalog.toml` + `[[evals]]`. One
   `verify_gepa_contract`; RL routes gated by `/compatibility`.

Phase 1 is unblocked and does not wait on the resume-path name decision.

---

## Open decisions (do not block phase 1)

1. **Resume path name.** Client: `/rollouts/{parent}/resume_async`. GELO docs:
   `/rollouts/{parent}/resume`. Pick one; do not add a third.
2. **Cache namespace.** Shared = identical candidates not re-paid *or*
   re-sampled (lower variance, arm contamination). Isolated = the opposite.
   Write the choice down. Same policy for eval and RL.
3. **Episode horizon.** One proposer round vs N candidates. Eval and RL must
   use the same default so numbers transfer.
4. **Which checkpoints are fixtures.** Generation-start cursor snapshots, not
   `generation_boundary` summaries.

---

## Don't

- Don't build two containers, two observation schemas, or a Gym wrapper beside
  HTTP. The eval **is** the env with two extra routes.
- Don't fold this into the GELO `rlvr` plugin kind. That lane is explicitly
  not-yet-supported and is a different product.
- Don't use `generation_boundary` rows as fixtures. Retained ≠ restorable.
- Don't fixture the smoke shape.
- Don't let arms differ in anything but the proposer.
- Don't infer displacement by snapshot diff.
- Don't trust "already implemented" without measuring (`aug19_gepa.md` §1).
