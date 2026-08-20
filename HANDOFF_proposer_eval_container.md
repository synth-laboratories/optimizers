# Handoff: GEPA proposer-evaluation container

Owner: whoever picks this up. Repos: `optimizers` (engine work, phases 1–4),
`evals` (the container, phase 5), `synth-cookbooks-public` (fixture source runs).

Companion docs:
- [`aug19_gepa.md`](aug19_gepa.md) — 2026-08-19 FlashEvolve notes. Read its §7
  before running anything; several hours went into traps recorded there.
- [`SCOPE_gepa_eval_rl_env.md`](SCOPE_gepa_eval_rl_env.md) — eval and RL env are
  **one container**. Same MDP; RL adds checkpoint + resume routes. Do not split
  the image.

---

## Goal

Make a GEPA **task instance** be `(downstream container, optimizer state
checkpoint)`, where the action under evaluation is *"propose the next
candidates"* and the reward is what those candidates actually score when rolled
out against that container.

That is an RLVR environment whose policy is the **proposer**. Same fixture serves
two jobs: picking proposer models, and training Codex models (Nemotron 3.5) for
GEPA inference.

```
observation  =  optimizer state (frontier, archive, per-seed scores, traces)
                + downstream container/task info
action       =  proposed candidate(s)
reward       =  train exploration + train exploitation + eval uplift — verifiable, because
                the rollouts are really executed
```

## First experiment

`gpt-5.6-luna` on Banking77, **`reasoning_effort` low vs medium**, same fixture,
same downstream container, same seeds. Two arms, one variable.

This is deliberately the smallest question that exercises the whole path. If the
harness cannot separate low from medium on the same checkpoint, it cannot rank
models either.

**Size the fixture from the full Banking77 shape (100 train rows), not the smoke
shape.** See *Variance budget* — the smoke shape cannot resolve the exploration
term.

## Reward

Per-seed, not aggregate. Both terms are computed over the **same** `example_id`
set for every arm.

- **Exploration — new max downstream seed uplift.** For each `example_id`, did a
  candidate from this arm establish a new maximum, and by how much over the prior
  holder? Sparse, high-variance, measures genuine discovery.
- **Exploitation — mean across seeds vs. previous candidates.** Mean reward over
  `example_id`s for this arm's candidates, minus the same mean over the
  pre-fork candidates. Dense, smooth, good shaping signal.

Dense shaping plus sparse achievement is a sound RLVR pairing, but the sparse
term only works with enough seeds to fire.

## Current truth — do not rediscover

Verified by reading the code on 2026-08-19. `file:line` where it helps.

### Already there (better than expected)

| Thing | Where | Note |
|---|---|---|
| Checkpoint is a **self-contained portable search state** | `synth_gepa/src/lib.rs` `persist_gepa_run_state` | embeds train/minibatch/reflection/heldout **rows**, program, objective set, candidate archive, proposal queue, counters, cost, usage ledger |
| Forked run self-heals identity | same fn | `state.cursor.run_id = context.config.run.run_id` on every persist |
| **Per-seed rewards per candidate** | `RolloutScore { example_id, task_id, reward }`, on `CandidateRecord.minibatch_scores` / `.train_scores` | this is the exploitation term's entire input |
| **Per-example Pareto frontier** | `frontier_type = "per_example"`, `FrontierCellRecord` (`candidate_id`, `parent_candidate_id`, `objective`, `score`, `score_vector`, `rank`) | `synth_optimizer_platform/src/candidates.rs:136` |
| Pause/resume control surface | `synth_gepa/src/service.rs:1537` `POST /runs/{id}/pause`, `control_run` | |
| Resume-in-place | `restore_gepa_run_state` → `initialize_or_restore_cursor` | |
| Multi-objective machinery | `objective_directions`, `selection_objective`, dominance in `synth_optimizer_platform/src/scores.rs` | exists; Banking77 just declares one objective, so cost/latency is a **runner** gap not an engine gap |

### Blockers

**1. There is exactly one restorable checkpoint per run, and it moves.**
`persist_gepa_run_state` calls `record_checkpoint_compacting_previous`
(`workspace.rs:3262`), which replaces the prior snapshot with a
`storage_compacted_summary` stub — on every tick. Every earlier checkpoint is a
stub. **You cannot fork from an arbitrary point today.** This is the hard blocker
and phase 1 is mostly about it.

**2. No fork operation.** `initialize_or_restore_cursor(workspace, run_id)` reads
*own run_id* + *latest* only. There is no "run B starts from run A at sequence
N". Small change once (1) is fixed, since the state is already portable.

**3. Proposer identity is not on the candidate.** `CandidateRecord` has
`parent_id`, `source` (`"reflector:frontier_variation"`), `acceptance_metadata` —
nothing naming the proposer model/effort/config. **Without this there is no
reward signal at all**, because candidates cannot be grouped by arm.

**4. Frontier displacement is not recorded.** `frontier.updated` emits a
*snapshot* (`lib.rs:11012`), not a transition. "Candidate C took the max on seed
X from Y by delta Z" exists nowhere; you would have to diff consecutive
snapshots. Build it properly — it is also verbatim an audit item ("record
frontier transition events, including which candidate dominated which former
frontier members and why").

**5. Budgets are absolute, not relative to a fork point.** A forked cursor
inherits `rollout_count`, `cost_usd`, `generation`. "Run for a while from here"
needs delta budgets plus per-arm spend accounting separate from inherited totals.

## Build order

Phases 1–4 are in `optimizers`; phase 5 is the `evals` container.

1. **Checkpoint as a portable immutable fixture.** Exempt fork points from
   compaction (new checkpoint kind, or a retention/pin policy). Export/import
   across workspaces — the fixture must travel to wherever the container runs,
   not just fork in place. Content-address it so `banking77@gen2-frontier6` means
   the same bytes in six months. Record fork provenance (parent run_id +
   sequence) on the child.
   *Test: fork one checkpoint twice; assert both arms start with byte-identical
   candidate archive, frontier, and row pools.*
2. **Proposer identity + arm id on `CandidateRecord`**, propagated to frontier
   records and manifest export. Model, effort, backend, config hash, proposer
   round id.
   *Test: attribution survives both resume and fork.*
3. **Per-seed frontier displacement records** — new max, prior holder, delta,
   `example_id`. Serves the exploration term and the audit item.
4. **Delta budgets from the fork point** + per-arm spend accounting; determinism
   audit; **idempotent resume** (the audit's "completed evaluations must not be
   charged twice" is currently *untested*).
5. **The container** in `evals` (below).

## Container shape

`evals` is on branch `agent/workshop-evals-v04`. Follow the existing image
pattern, do not invent a second one:

- `containers/images/<id>/image.toml` — see `containers/images/banking77/image.toml`
  for the exemplar (`contract = "http_task"`, `dockerfile = "http_task.Dockerfile"`,
  `context = ".."`, `pull = false`)
- register the id in `containers/images/catalog.toml`
- add an `[[evals]]` row in `registry.toml` pointing at the suite's `eval.toml`
- serve via `ContainerRunner(image_id=..., catalog="evals/containers/images").serve()`
  or `synth-containers up <id> --port ...`

Mirror the existing GEPA container contract (`/health`, `/taskset`,
`/taskset/tasks`, `/rollout`, `verify_gepa_contract`) but where a **task** is a
fixture `(downstream container ref, checkpoint ref)` and a **rollout** is one
bounded proposer episode. The proposer-eval container wraps the optimizer, which
in turn drives the downstream Banking77 container — two containers per episode.

Task ids should carry the fixture identity, mirroring the downstream convention
(`train:0`, `test:0` — per-split indices; verified live via
`POST /taskset/tasks {"task_ids":["train:0"]}` →
`{"task_id":"train:0","split":"train","seed":0}`).

## Variance budget — read before designing the experiment

Measured 2026-08-19 on the Banking77 smoke shape (24 train / 16 heldout),
`gpt-5.6-luna` low effort, config byte-identical across runs:

| Quantity | Observed spread |
|---|---|
| heldout reward | 0.500 / 0.5625 / 0.625 — one row of 16 **is** 0.0625 |
| proposer round wall | 70.1 / 75.1 / 80.1 / 85.1 / 85.6 s |
| `gpt-5.4-nano` medium, same shape | 85.1 / 135.2 s |

**If the reward gap between two proposers is smaller than that, RLVR trains on
noise.** Consequences for the design:

- Hold the evaluation completely fixed across arms: same rows, same policy, same
  seeds. Row pools are already pinned in the cursor, so this is achievable.
- Decide the cache-namespace policy deliberately. Shared means identical
  candidates are not re-paid *or* re-sampled (lower variance, but arms can
  contaminate); isolated means the opposite. Pick one and write down why.
- Size the horizon so each arm produces several candidates, not one.
- Use the **full 100-row** Banking77 shape for fixtures. The exploration term
  needs enough seeds to fire at all.
- On the DAG families the proposer lane is **85–89% of wall clock** (measured:
  145–205s propose vs 11–25s rollout). Episode cost is dominated by the proposer,
  which is convenient — the thing you are measuring is the thing you pay for.

## Test commands

```bash
cd optimizers/rust
cargo test -p synth_gepa
cargo test -p synth_optimizer_platform
cargo check -p synth_gepa -p synth_optimizer_platform
```

Current baseline: 16 tests in `synth_gepa`, 27 in `synth_optimizer_platform`, all
green. `synth_optimizers_py` does **not** link under `cargo test` — pre-existing,
confirmed against a stashed tree; do not chase it.

Local engine build (the published `synth-optimizers==0.2.0` will not have any of
this):

```bash
cd optimizers && uv run --with maturin maturin build --release --out target/wheels
uv pip install --python <venv> --no-deps --reinstall --no-cache <path>/synth_optimizers-*.whl
```

`--no-cache` is **required** — same version + same filename means uv serves a
cached wheel and you will debug a fix that was never installed. See
`aug19_gepa.md` §7.

## Pass bar

1. One checkpoint forks N ways with byte-identical starting archive/frontier/rows.
2. Every candidate is attributable to the proposer arm that produced it, through
   resume and fork.
3. Per-seed displacement records let the exploration term be computed without
   diffing snapshots.
4. Resume does not re-charge or re-apply a completed evaluation.
5. **luna-low vs luna-medium on Banking77 separate by more than the measured
   noise floor**, with the gap explained, not just reported.

## Don't

- Don't build the fixture on the smoke shape. It cannot resolve the exploration
  term and no amount of re-running fixes that.
- Don't let the two arms differ in anything but the proposer. Row pools, policy,
  seeds, downstream container revision all pinned.
- Don't infer frontier displacement by diffing snapshots — record it.
- Don't trust "already implemented" without measuring. FlashEvolve shipped with a
  mode flag, config knobs, and docs claiming "highest throughput", and could not
  run two lanes concurrently at all. See `aug19_gepa.md` §1.
- Don't fold the exploratory lanes (rubric-aware, proxy, Combee/inference-aware,
  GEPA→XGBoost routing, RL proposer) into this. The Roam audit explicitly
  quarantines them behind interfaces/experiment flags.
