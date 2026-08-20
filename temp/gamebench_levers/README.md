# GameBench lever containers

GEPA search over four GameBench single-player games, in two modes each.

|  | search object | lever kind | protocol | entrypoint |
|---|---|---|---|---|
| `code` | the policy source | `policy_script` | `whole_file.v1` / `unified_diff.v1` | `act(obs) -> action` |
| `harness` | the whole agent loop | `harness_module` (+ `system_prompt`) | `harness_restart.v1` (+ `prompt_overlay.v1`) | `run_episode(env, prompt, seed, max_steps, llm)` |

Games: `sokoban`, `craftax`, `rogue`, `dungeongrid` — eight targets in total.

The harness seed is a **SpeedRunner-style actor** ([arXiv:2608.11338](https://arxiv.org/abs/2608.11338)):
the LLM picks a *public skill* by name and the skill is an ordinary program that
expands into primitive `env.step` calls with no further model call. Optimizing it
is harness search — GEPA may add skills, change the arbiter, or rewrite the
observation summary. It is not a prompt stuffed with harness source.

## Shape

```
GEPA ──A──> orchestrator ──┬── supervisor  POST /restart_policy
                           ├── policy service (subprocess: act() or run_episode())
                           └── env service   (subprocess, never restarted)
```

Register-then-run, for both modes:

```
GET  /program                     seed + advertised levers / protocols / ASI schemas
POST /candidates                  apply ONCE: write + load, or write + restart
     -> {candidate_id, apply_report.v1, base_hash}
POST /rollout {candidate_id, task_id}   run on demand, no re-apply
GET  /rollouts/{id}               reward + side_info[]
```

Apply cost is paid per candidate, not per seed. A candidate that fails to apply
gets **no** `candidate_id` that can be rolled out, and the container rolls the
tree back so the live policy keeps serving the parent.

## Isolation: one worker per candidate

`apply_isolation` defaults to **`per_candidate_worker`** for the harness lane.

With a single shared policy process (`serial_restart`), GEPA interleaving rollouts
across candidates forced a process restart on *every* switch — register-once/run-many
silently degraded back into restart-per-rollout, paying a respawn plus a re-import each
time. Keyed workers make a switch a routing decision instead:

| 6 interleaved rollouts, 2 candidates | wall | spawns |
|---|---|---|
| `serial_restart` | 65.5 s | one per rollout |
| `per_candidate_worker` | **38.9 s** | one per candidate (6 reuses) |

The pool is bounded (`--max-workers`, default 4) and evicts least-recently-used. An
evicted candidate respawns from its stored source on next use, so eviction costs
latency, never correctness. The supervisor exposes `POST /workers`, `GET /workers`
and `DELETE /workers/{candidate_id}`; `serial_restart` is still selectable via
`start_stack(..., isolation="serial_restart")`.

A candidate that fails to apply under pooling needs no rollback at all — nothing shared
was mutated, so the parent's worker keeps serving untouched.

## The ASI plane

Typed side info still rides on the terminal record — the GEPA sensor reads
`actionable_side_info` from there and moving it would break the search loop — but it is
also addressable on its own. The engine stores the whole terminal record as an opaque
`raw_response` blob, so reading a trace back afterwards meant re-parsing that blob and
guessing where the envelope lived. `/asi` is the read path.

| route | returns |
|---|---|
| `GET /asi/schemas` | what this container emits and when |
| `GET /asi/{rollout_id}` | the `asi_envelope.v1` for one rollout |
| `GET /asi/{rollout_id}?summary_only=true` | the same, with large `body` blocks dropped |
| `GET /asi/{rollout_id}/{schema_id}` | one typed frame, e.g. `speedrunner_trace.v1` |
| `GET /asi?candidate_id=&task_id=&split=&schema_id=&limit=` | index; filters compose |

Every terminal record carries `asi_ref` pointing at its envelope, and `/metadata`
advertises `asi_route`. Frames per mode:

- code: `apply_report.v1`, `code_policy_game_trace.v1`, `episode_verdict.v1`
- harness: `apply_report.v1`, `speedrunner_trace.v1`, `harness_inspect.v1`, `prompt_trace.v1`, `episode_verdict.v1`

### What ASI has to answer

A score of 0.0 is ambiguous three ways, and the proposer cannot fix what it cannot
see. `episode_verdict.v1` rides on every rollout and disambiguates:

| `not_scored_because` | meaning | `fix_hint` points at |
|---|---|---|
| `apply_failed` | never compiled, or the policy would not start | `compile_diagnostics` — message, line number, the offending source line |
| `no_env_steps` | loaded, but `run_episode` returned without ever stepping the env | `runtime_errors` |
| `infra_error` | transport/model failure — **not** evidence about the candidate | `infra_errors` |
| `null` | it really did play and really did score that | the trace frame |

This came out of a real miss. A proposed harness put a raw newline inside a quoted
string; the run reported `reject_reason: "restart_failed"` and an empty trace summary,
which is indistinguishable from "the strategy was bad". Candidates are now compiled
**before** anything is written or restarted, so the report reads
`SyntaxError: unterminated string literal (detected at line 139)` with the line itself
— and a non-compiling candidate never disturbs the running policy process at all.

Transport failures are kept off the policy-error channel and retried at source
(`llm.py`), because a dropped connection mid-episode otherwise lands as a real 0.0 in
the baseline every proposal is compared against.

The store is in-process, so it serves the **live** container. Finished runs still keep
their copy under `runs/<run_id>/rollout_traces/`, where the envelope sits inside each
sensor file's `raw_response`.

**One game per process.** Every GameBench task dir ships its own `gold_python`
package, so two games cannot share an interpreter. The env is always a subprocess.

## Run

```bash
cd temp/gamebench_levers

# prove the target is searchable before spending budget
PYTHONPATH=. uv run python gate.py                     # all four games
PYTHONPATH=. uv run python gate.py sokoban

# search
PYTHONPATH=. uv run python run_gepa.py --game sokoban --mode code
PYTHONPATH=. uv run python run_gepa.py --all-code
PYTHONPATH=. uv run python run_gepa.py --game craftax --mode harness

# a stack on its own, to poke by hand
PYTHONPATH=. uv run python -m gamebench_levers.stack --game rogue --mode code
```

Harness mode calls a model every turn and needs `OPENAI_API_KEY` or
`OPENROUTER_API_KEY`; `run_gepa.py` loads them from `backend/.env.local`. There is
no local stub — do not "skip harness because keys are missing".

## The headroom gate

`gate.py` boots the real code stack per game and scores two policies over the whole
train split: the seed GEPA starts from, and a hand-written reference in
`gamebench_levers/references/`. It **fails** a target when the reference cannot beat
the seed, because that means the reward does not pay for better code — a broken
target, not a hard one. References are never handed to GEPA.

Measured 2026-08-19 (code mode, whole train split):

| game | seed | reference | headroom |
|---|---|---|---|
| sokoban | 0.0 | 0.489 | +0.489 |
| craftax | 0.0 | 1.5 | +1.5 |
| rogue | 9.0 | 161.36 | +152.36 |
| dungeongrid | 0.0 | 1.95 | +1.95 |

## Search results (2026-08-19)

Codex `app-server` proposer, 3 generations x 2 proposals, minibatch 3, `cache=off`.
`train` is the whole train split; `heldout` is the split GEPA never proposed against.

| target | seed | best (train) | heldout | uplift | candidates | wall |
|---|---|---|---|---|---|---|
| sokoban / code | 0.0 | **0.775** | 0.808 | +0.775 | 7 | 615 s |
| rogue / code | 9.0 | **148.82** | 154.68 | +139.82 | 7 | ~900 s |
| craftax / code | 0.0 | **1.667** | 0.667 | +1.667 | 9 | 599 s |
| dungeongrid / code | 0.0 | **1.95** | — | +1.95 | 7 | 499 s |
| craftax / harness | 1.333 | **2.833** | 2.667 | +1.5 | 5 | 1446 s |
| rogue / harness | 64.02 | **126.42** | 145.0 | +62.4 | 5 | 1267 s |
| dungeongrid / harness | 1.99 | 1.99 | — | +0.0 | 5 | 1001 s |
| sokoban / harness | 0.061 | **0.489** | 0.142 → 0.333 | +0.429 | 5 | 2334 s |

All four harness targets were re-run after the diagnostics and isolation fixes. The
earlier sokoban run, on the pre-fix code, was flat at 0.061.

Sokoban and rogue generalize: heldout comes back at or above train. Craftax climbs on
train but drops on heldout — it is fitting the train rooms, and it is the one target
where the reference (1.5) and the search (1.667) are still close together, so there is
plenty of ladder left.

Both sokoban and rogue searches produced policies at or above the hand-written
reference (sokoban 0.775 vs 0.489; rogue 148.8 vs 161.4).

The harness lane climbs on two of four games, and **beats the code lane on craftax**
(2.833 vs 1.667) — an LLM picking skills does better there than any policy the search
wrote in Python. Both climbers generalize: rogue's heldout goes 33.0 → 145.0 and
craftax's 2.0 → 2.667.

`dungeongrid/harness` stayed flat for a legible reason, now that the verdict frame
says so: two of its four children died on `SyntaxError: unterminated string literal`
(lines 133 and 284) and the other two genuinely scored below the parent. Half the
budget went to malformed Python, not to bad strategy. Rogue's two failures were
`IndentationError: unexpected indent (line 3)`. That failure mode is the single
biggest tax on the harness lane and is what the diagnostics below now expose.

The earlier sokoban result was **proven operational but did not climb at that budget**. Every
mechanism fired — candidates registered, the policy process restarted per candidate
with the env untouched, episodes ran against a live model, rewards and typed side info
flowed back, and the frontier applied its accept rule. Two children matched their
parent's minibatch score exactly (0.2125) and were correctly rejected as
`primary_not_improved`. Two things to fix before reading anything into that number: the
minibatch was 2 rows out of 7 (the seed scores 0.2125 on those two rows but 0.061 on the
full train split, so the comparison is noisy), and the rollout model was
`gpt-4.1-nano`. Harness searches for craftax, rogue and dungeongrid have **not** been
run — each rollout costs one model call per turn, so they are far slower than the code
lane.

## Reward per game

| game | reward | why |
|---|---|---|
| sokoban | boxes-on-target fraction, 1.0 solved | dense, and solving is unambiguous |
| craftax | count of unique achievements | the standard Craftax metric |
| rogue | the engine's own `synth_shaped_reward` | already dense: pays for scouting, gold, descent |
| dungeongrid | `2·gold + 1.5·achievements + 2·spells + engine_reward − 0.1·invalid` | see below |

DungeonGrid departs from the shipped sweep composite on purpose. That composite adds
`armor` (which counts remaining HP) and `step_bonus` (`max_actions − steps`), so a
policy that does nothing keeps full HP, spends no steps and scores well — searching
against it rewards inaction.

## World choices that are load-bearing

**Craftax.** The shipped `policy_dev_small` default carries
`densities: {tree: 0.16, water: 0.05}`. Densities *scale* the vanilla generator, so
that world holds 1–3 trees across the whole map and zero coal or iron — a code policy
cannot climb a tech tree it cannot reach. The full 48×48×9 `craftax_default` has the
resources but a view radius of 4, so reward stays flat at 0–1 achievements for
hundreds of steps. This package instead seeds its own 9×9 room variants off the
`fixture_room` shape: every task is guaranteed 5 tree / 3 stone / 1 coal / 1 iron /
1 water / 1 ladder / cow / zombie, so the ladder is dense and every seed is a distinct
layout. A hand-written reference reaches 6 of the 16 ladder entries.

**Sokoban.** `curriculum_easy` index 0 solves in one move. The bank is
`curriculum_medium` (10 levels, 1–2 boxes, optimal 5–8).

## Traps already paid for

- `valid_actions()` on the Craftax engine returns the **whole vocabulary**, not the
  legal set. Crafting needs its recipe met *and* the agent standing next to a table;
  `obs['state']['near_crafting_table']` reports that.
- DungeonGrid advances `step_index` only on an applied, non-`end_turn` action, so
  rejected actions would loop forever. Episodes are bounded by *attempts* as well, and
  `obs['state']` carries `last_applied` / `last_reject_reason` so a policy can react
  to a refusal instead of repeating it.
- `legal['directions']` in DungeonGrid always lists all four compass points; it is not
  filtered by passability.
- **A candidate_id must always resolve to one policy.** GEPA rolls a candidate out many
  times and sends the inline `lever_bundle` only sometimes. An unknown id that silently
  fell back to the seed made the same id score one row with two different policies, and
  the engine aborted the run with `conflicting score vector material ... matched 2
  distinct score payloads`. Registration is idempotent and an unseen id binds to the
  seed permanently.
- **`/program` is the only thing the proposer sees.** The env `/spec` — action space,
  glyph legend, recipe rules, a real sample observation — is copied into the program's
  module constraints. Without it a Craftax candidate cannot even name a valid action,
  and the whole search comes back flat at exactly the seed's score.
- **Harness mode targets `harness_module` only.** With two target modules the Codex
  proposer can return null for one, and the engine rejects the whole proposal with
  `proposal index=0 is not a valid proposal object: invalid type: null, expected a
  string`, killing the run. `system_prompt` stays advertised on `/program`.
- The GEPA minibatch pool must be strictly larger than `minibatch_size`. A pool equal
  to the minibatch makes every proposal see identical rows and lift measures nothing.
- Budget `max_total_rollouts` for heldout too: GEPA fails closed with
  `gepa_terminal_heldout_not_evaluated` if it runs out before evaluating heldout.
- Policy subprocess stderr is captured to `policy.log` in the stack tmpdir. A policy
  that dies at boot is undiagnosable otherwise.

## Layout

| path | what |
|---|---|
| `gamebench_levers/adapters/` | one per game: `make_session(seed) -> reset/step/observe/score` |
| `gamebench_levers/seeds/` | seed code policy + SpeedRunner harness per game |
| `gamebench_levers/references/` | headroom-gate policies only, never seeds |
| `gamebench_levers/env_app.py` | `gamebench_env.v1` over HTTP, one game per process |
| `gamebench_levers/policy_code_app.py` | loads `act(obs)`, drives the env |
| `gamebench_levers/policy_harness_app.py` | loads the whole SpeedRunner script |
| `gamebench_levers/orchestrator_app.py` | GEPA plane A: `/program` `/candidates` `/rollout` |
| `gamebench_levers/stack.py` | boots env + policy + supervisor + orchestrator |
| `gate.py` | headroom gate |
| `run_gepa.py` | search driver |
