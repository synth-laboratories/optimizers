# Pickup: GameBench GEPA lever containers (2026-08-19 night)

Repo: `optimizers` on `backup/last72h-optimizer-platform-20260813`. **Uncommitted
(`temp/` is untracked). Do not commit unless asked.**

This continues [`SCOPE_gepa_custom_levers.md`](SCOPE_gepa_custom_levers.md) past the toy
stack. The toy (`temp/craftax_levers/`, a 3×3 wood/lava grid, **not** GameBench) proved
register-then-run on one game. This is the same contract over **four real GameBench
games × two modes**, with the engine-contract bugs that only show up at that scale
found and fixed.

Read in order, do not rediscover:

1. [`SCOPE_gepa_custom_levers.md`](SCOPE_gepa_custom_levers.md) — the contract.
2. [`temp/gamebench_levers/README.md`](temp/gamebench_levers/README.md) — **the
   authority for this work**: routes, rewards, world choices, every trap with its
   measurement.
3. [`HANDOFF_gepa_custom_levers.md`](HANDOFF_gepa_custom_levers.md) — the toy-stack
   predecessor. Its "not done" list is largely superseded by this file.

---

## What you are picking up

Eight targets: `sokoban` · `craftax` · `rogue` · `dungeongrid`, each in `code` and
`harness` mode.

| mode | search object | lever kind | protocol | entrypoint |
|---|---|---|---|---|
| `code` | the policy source | `policy_script` | `whole_file.v1` / `unified_diff.v1` | `act(obs) -> action` |
| `harness` | the whole agent loop | `harness_module` | `harness_restart.v1` | `run_episode(env, prompt, seed, max_steps, llm)` |

The harness seed is a **SpeedRunner actor** ([arXiv:2608.11338](https://arxiv.org/abs/2608.11338)):
the LLM picks a public skill by name, the skill expands to primitive `env.step`s with no
further model call. Optimizing it is harness search, not prompt search.

Proposer is GEPA's default Codex app-server. Do **not** swap in a chat-completions
"real proposer" — different experiment.

---

## Live state: everything below is measured, not projected

Both lanes search and climb. Runs are in `runs/gamebench_levers/<run_id>/`.

**Code lane** (3 generations × 2 proposals, minibatch 3, `cache=off`):

| target | seed | best (train) | heldout | run_id |
|---|---|---|---|---|
| sokoban | 0.0 | **0.775** | 0.808 | `gepa_sokoban_code_ec5bbc15` |
| rogue | 9.0 | **148.82** | 154.68 | `gepa_rogue_code_c312507d` |
| craftax | 0.0 | **1.667** | 0.667 | `gepa_craftax_code_4d59a2ff` |
| dungeongrid | 0.0 | **1.95** | — | `gepa_dungeongrid_code_9d5a66f7` |

**Harness lane** (2 × 2, minibatch 3) — all four re-run *after* the fixes below:

| target | seed | best (train) | heldout | run_id |
|---|---|---|---|---|
| rogue | 64.02 | **126.42** | 145.0 | `gepa_rogue_harness_adc9ef23` |
| craftax | 1.333 | **2.833** | 2.667 | `gepa_craftax_harness_49543338` |
| sokoban | 0.061 | **0.489** | 0.333 | `gepa_sokoban_harness_9d220d78` |
| dungeongrid | 1.99 | 1.99 (flat) | — | `gepa_dungeongrid_harness_888572eb` |

Two results worth carrying forward:

- **craftax/harness (2.833) beats craftax/code (1.667).** An LLM picking skills does
  better there than any Python policy the search wrote. Nobody predicted that.
- **sokoban/harness was flat at 0.061 until the isolation fix**, then climbed to 0.489
  on the same budget. A "the search found nothing" result was actually infrastructure.

Older run_ids in that directory are pre-fix and **stale** — ignore
`*_harness_fb23940e`, `*_harness_92d36315`, `*_harness_bcbaa795`,
`*_harness_9df335c0`, `*_harness_baa9ed9c`, `*_code_6af51ef3`, `*_code_e5fdbeb8`,
`*_code_e529e896`.

Tests: **36 passing**. Headroom gate: **4/4 pass**. No processes left running.

---

## What already landed (do not rebuild)

### `temp/gamebench_levers/` (~4.6k lines)

| path | what |
|---|---|
| `gamebench_levers/adapters/` | one per game: `make_session(seed) -> observation/step/score`, uniform obs |
| `gamebench_levers/seeds/` | weak seed code policy + SpeedRunner harness per game |
| `gamebench_levers/references/` | headroom-gate policies only — **never** handed to GEPA |
| `gamebench_levers/env_app.py` | `gamebench_env.v1` over HTTP, one game per process |
| `gamebench_levers/policy_code_app.py` | loads `act(obs)`, drives the env |
| `gamebench_levers/policy_harness_app.py` | loads the whole SpeedRunner script |
| `gamebench_levers/orchestrator_app.py` | GEPA plane A: `/program` `/candidates` `/rollout` `/asi` |
| `gamebench_levers/stack.py` | env + policy + supervisor + orchestrator; `PolicyPool` |
| `gate.py` | headroom gate |
| `run_gepa.py` | search driver |

### Engine-contract fixes (each cost real debugging; all have tests)

1. **A `candidate_id` always resolves to one policy.** GEPA sends the inline
   `lever_bundle` only on *some* rollouts. Falling back to the seed for an unknown id
   made one id score a row twice with two different policies; the engine aborted with
   `conflicting score vector material ... matched 2 distinct score payloads`.
   Registration is idempotent; an unseen id binds to the seed permanently.
2. **`/program` is the only thing the proposer sees** — not the env `/spec`. Craftax
   returned flat at *exactly* the seed score until the action space, glyph legend,
   recipe rules and a real sample observation were copied into the module constraints.
   That one change took it 0.0 → 1.667.
3. **Compile before applying.** Candidates are `compile()`-checked at register time,
   before anything is written or restarted, so the report reads
   `SyntaxError: unterminated string literal (detected at line 139)` with the offending
   line — not `restart_failed`. A non-compiling candidate never disturbs the running
   policy, so there is nothing to roll back.
4. **`episode_verdict.v1`** on every rollout disambiguates a 0.0 four ways:
   `apply_failed` / `no_env_steps` / `infra_error` / genuinely-scored-zero, each with a
   `fix_hint`. Before this they were indistinguishable, which is how sokoban got
   misdiagnosed as "bad strategy" when half its children never compiled.
5. **`per_candidate_worker` isolation** (default for harness). `serial_restart` silently
   degraded register-once/run-many into restart-per-rollout, because GEPA interleaves
   rollouts across candidates and one shared process restarts on every switch. Keyed
   LRU-bounded pool: 6 interleaved rollouts **65.5 s → 38.9 s**, spawns one-per-rollout
   → one-per-candidate.
6. **Infra errors split from policy errors** and retried at source (`llm.py`, 3 attempts
   + backoff). A dropped connection was landing as a real 0.0 in the baseline every
   proposal is compared against.
7. **Harness mode targets `harness_module` only.** With two target modules the Codex
   proposer can return null for one and the engine rejects the whole proposal
   (`proposal index=0 is not a valid proposal object: invalid type: null, expected a
   string`), killing the run. `system_prompt` stays advertised on `/program`.

### The `/asi` plane (added on request)

Typed side info still rides on the terminal record — the sensor reads
`actionable_side_info` there and moving it would break the loop — **and** is addressable:
`GET /asi/schemas`, `/asi/{rollout_id}`, `/asi/{rollout_id}/{schema_id}`,
`/asi?candidate_id=&task_id=&split=&schema_id=`. Every record carries `asi_ref`.

⚠️ **This deviates from the scope**, which says put ASI on the record and "not a new HTTP
verb / not a second proposer channel". The reason it was added: the engine stores the
record as an opaque `raw_response` blob, so reading a trace back meant re-parsing it —
and I misdiagnosed a run because of that. A test asserts the two views stay identical.
**Reconcile the scope text with this decision.**

---

## How to run

```bash
cd temp/gamebench_levers

PYTHONPATH=. uv run python gate.py                       # prove targets are searchable
PYTHONPATH=. uv run python run_gepa.py --game rogue --mode harness
PYTHONPATH=. uv run python run_gepa.py --all-code
PYTHONPATH=. uv run python -m gamebench_levers.stack --game rogue --mode code
PYTHONPATH=. uv run pytest tests/test_levers.py -q
```

Harness mode calls a model every turn; needs `OPENAI_API_KEY` or `OPENROUTER_API_KEY`
(`run_gepa.py` loads from `backend/.env.local`). There is no local stub — **do not skip
harness because keys are missing**. Rollout model used for all results above:
`gpt-4.1-nano`. Never print keys.

Useful flags: `--generations --proposals --minibatch --max-rollouts --max-train
--max-heldout --max-steps`. `start_stack(..., isolation="serial_restart", max_workers=N)`.

---

## Not done — in the order I would do it

1. **Repair turn on `compile_diagnostics`. Highest value by a distance.** The proposer
   emits invalid Python often on 5–8 KB whole-file rewrites — a raw newline inside a
   quoted string, a stray indent. It cost dungeongrid **half its budget** (2 of 4
   children were `SyntaxError`, lines 133 and 284) and rogue 2 of 4
   (`IndentationError` line 3). The diagnostic is now precise and sitting in ASI, but
   nothing feeds it back: every syntax error still burns a whole proposal slot. Feed it
   as a repair turn instead. This is most likely what stands between dungeongrid and a
   climb, and it lifts the other three too.
2. **Persist ASI.** `asi_store` is in-process, so `/asi` serves the live container only.
   Finished runs keep their copy inside each `rollout_traces/*.json` `raw_response`. A
   JSONL sidecar per run would make post-hoc analysis a lookup instead of a re-parse.
3. **DungeonGrid at its full split.** It ships 10 train / 10 heldout; I capped it to 5/3
   (`--max-train/--max-heldout`) so wall-clock matched the other games. Its numbers are
   cross-game comparable but are **not** the full split.
4. **Craftax code-lane overfit**: 1.667 train vs 0.667 heldout. The harness lane does not
   show this. Worth understanding before trusting craftax/code numbers.
5. **`unified_diff.v1` is implemented and tested but the proposer emits whole files in
   practice.** Nobody has confirmed a diff-shaped candidate end-to-end in a real search.
6. **FlashEvolve matrix cell never run.** Now unblocked: `per_candidate_worker` is what
   the scope requires before `flash_evolve` + harness opt is legal. Read the overlap
   ceiling in [`aug19_gepa.md`](aug19_gepa.md) before claiming a speedup.
7. **Code lane still reloads in-process per candidate switch.** Cheap (an import), so it
   was left alone — but it is the same thrash shape as #5 above if it ever gets costly.

---

## Traps (each one cost time)

- **One game per process, always.** Every GameBench task dir ships its own `gold_python`
  package; importing two games in one interpreter silently resolves to whichever loaded
  first. This is why the env is a subprocess — isolation is a side benefit, not the reason.
- **Craftax world choice is load-bearing.** Shipped `policy_dev_small` carries
  `densities: {tree: 0.16, water: 0.05}`; densities *scale* the vanilla generator, so it
  holds 1–3 trees across the whole map and zero coal/iron. `craftax_default` 48×48×9 has
  the resources but view radius 4 keeps reward flat at 0–1 achievements for hundreds of
  steps. This package seeds its own 9×9 room variants off `fixture_room`.
  **Craftax numbers here are not comparable to any other Craftax lane.**
- **Craftax `valid_actions()` returns the whole vocabulary, not the legal set.** Crafting
  needs the recipe met *and* the agent standing next to a table —
  `obs['state']['near_crafting_table']` reports that.
- **DungeonGrid's shipped composite rewards inaction** (counts remaining HP and unused
  steps), so a do-nothing policy scores well. This adapter scores progress only.
- **DungeonGrid advances `step_index` only on an applied, non-`end_turn` action**, so
  rejected actions loop forever; episodes are bounded by *attempts* too, and
  `last_applied` / `last_reject_reason` are in the obs.
- **`legal['directions']` in DungeonGrid always lists all four compass points** — it is
  not filtered by passability.
- **Sokoban `curriculum_easy` index 0 solves in one move.** Bank is `curriculum_medium`.
- **Minibatch pool must be strictly larger than `minibatch_size`**, or every proposal
  sees identical rows and lift measures nothing. Also `minibatch ⊆ reflection`.
- **Budget `max_total_rollouts` for heldout too**, or GEPA fails
  `gepa_terminal_heldout_not_evaluated`.
- **Never edit `gamebench_levers/` while a harness search is running.** The policy
  subprocess is respawned per candidate and re-reads seed/app source from disk; a
  mid-flight edit puts some candidates on old code and some on new, silently
  invalidating the comparison. Stop the run first.
- **Codex propose prints nothing.** Quiet stdout ≠ hung. Check `events.jsonl` for
  `to: "proposing"` and `codex app-server` processes.
- GameBench's own `craftax-singleplayer/containers/gepa_codepolicy` is the **older
  fused-apply shape** (no `/candidates`) and needs the Rust env at `:8098`. Different
  thing — do not confuse them.
- Killing a run with `pkill` skips `Stack.stop()`, leaving `~/tmp/gb_*` worker dirs.

## Scope deviations to reconcile

1. `/asi` as a dedicated plane (scope says record-only) — decided deliberately, above.
2. DungeonGrid reward departs from the shipped sweep composite (which rewards inaction).
3. Craftax uses generated room variants, not any shipped world.
4. DungeonGrid split capped to 5/3 for wall-clock parity.
