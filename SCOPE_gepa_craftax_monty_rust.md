# Scope: GEPA varies Monty Python policies on Rust Craftax

Companion to [`SCOPE_gepa_custom_levers.md`](SCOPE_gepa_custom_levers.md).
Grounded against GameBench + this repo on 2026-08-19.

GEPA searches a **Python policy file**. Monty **loads** it. GameBench
**Rust gold** is the env. The optimizer never imports the candidate and never
talks to `gold_rust` directly.

Do not invent a second env. Do not embed CPython in `gold_rust`. Do not reuse
GELO `forbid_code_policy_candidates`.

---

## Current truth — do not rediscover

| Thing | Where | Note |
|---|---|---|
| Python Craftax policies | `gamebench/tasks/craftax-singleplayer/policies/heuristic_baseline.py` | Entry `choose_actions`; returns `{"actions": [...], "policy_reason": ...}` |
| Isolated load | `gamebench/tasks/shared/codepolicy/policy_subprocess.py` | JSONL IPC; `engine=None`; no suite paths in the child |
| Sweep vs Rust | `scripts/run_policy_sweep.py --lane rust` | Python worker + `craftax_repl` JSONL or HTTP `:8098` |
| Rust gold HTTP | `gold_rust` `craftax_gold` / `shared/http_contract.md` | `/rollouts`, `/step`, `/readout`, `/event_log`, checkpoint |
| Monty policy spec | TTT `gold/monty.py` + Harbor `deterministic_tasks.md` | `{kind: "monty_python", module, entry}` — **not yet wired as Craftax GEPA load** |
| Monty on Craftax today | `monty_reward` in gold (numeric weights) | Reward dialect, **not** policy load |
| GELO Rust smoke | `craftax_gamebench_rust_smoke` | Prompt-only; `forbid_code_policy_candidates: true` |
| Toy GEPA code stack | `temp/craftax_levers` | Fake 3×3 grid, `act(obs)`, not GameBench |

Harbor already says Craftax policies are `monty_python` in
`adapters/harbor/bundles/craftax_singleplayer_gold/spec/deterministic_tasks.md`.
The live Craftax runner still `importlib`s `choose_actions` without a Monty
spec object. This scope is that missing load contract plus GEPA plane A.

---

## Product

```
[GEPA]  --A-->  [orchestrator]           GEPA http_task (this container)
                   | apply lever_bundle (whole_file / unified_diff)
                   | POST policy /load  {kind: monty_python, ...}
                   |
                   +--> [policy service]  IsolatedPolicyProcess (Monty source)
                   |         choose_actions(readout) → actions[]
                   |
                   +--> [env service]     gold_rust, stays up
                             POST /rollouts + /step + GET /readout
```

Same GEPA loop as [`SCOPE_gepa_custom_levers.md`](SCOPE_gepa_custom_levers.md)
**Candidate lifecycle**: register the policy (`POST /candidates` → write +
Monty `/load`), then run on demand (`POST /rollout` `{candidate_id, task_id}`
→ gold_rust episode). Do not `/load` per seed. Search object is the policy
source, not a prompt and not the Rust engine.

---

## Planes and hostnames

| Plane | Process | Hostname GEPA sees | Routes |
|---|---|---|---|
| **A** | Orchestrator | yes (the "container") | GEPA `/health` `/metadata` `/program` `/taskset` `/candidates` `/rollout` |
| **B-policy** | Monty policy service | no | load / act / episode / health |
| **B-env** | `gold_rust` | no | existing GameBench HTTP; do not gymnasium-rewrite |

Optimizer never speaks B. Apply is orchestrator-owned. Env has no candidate
bytes. Policy child never sees the Rust binary or task JSON paths
(`IsolatedPolicyProcess` already enforces this).

Keep GameBench env routes as they are. Do **not** wrap `gold_rust` into
toy `craftax_env.v1` (`POST /reset` + gymnasium `obs`). Craftax policies
already consume GameBench **readout** (`observation`, `observation_text`,
`valid_actions`). Signature of the lever is that readout, not the 3×3 toy.

Advertise env as:

```
env_id: gamebench.craftax-singleplayer
protocol_id: gamebench.craftax_gold_rust.v1
substrate: rust
```

Fail closed if a candidate claims `craftax_env.v1` (toy) or a ReAct prompt
kind.

---

## Monty load (inbound)

One advertised lever:

```json
{
  "lever_id": "policy_script",
  "kind": "policy_script",
  "protocol_id": "unified_diff.v1",
  "constraints": {
    "runtime": "python_source",
    "load": "monty_python",
    "entrypoint": "choose_actions",
    "signature": "gamebench.craftax_gold_rust.v1",
    "path": "policies/candidate.py",
    "module": "candidate",
    "isolation": "isolated_policy_process"
  }
}
```

`whole_file.v1` is the whole-code special case (seed + first apply).
`unified_diff.v1` is the default mutation dialect. Not `harness_restart.v1`:
this is optimize-anything on policy source. Reload is **re-exec the child**,
not restart Rust, not restart the orchestrator.

Monty spec the policy service stores after apply:

```json
{
  "kind": "monty_python",
  "module": "candidate",
  "entry": "choose_actions",
  "path": "policies/candidate.py"
}
```

Same shape as TTT `agent_0_policy`. Craftax entry stays `choose_actions`
(kwargs: `observation_text`, `session`, `valid_actions`, `engine=None`,
`readout`, `ply`). Do not silently alias TTT `choose_action(public, seed, ply)`
— wrap if a proposer emits that shape, but seed + `/program` pin
`choose_actions`.

`engine` is always `None` on this path. Policies that require
`engine.clone_for_sim()` fail load (`compile_ok: false`), not a host crash.

`sourced_python` is **not** this lever. Only advertise it if we later allow
helper modules the candidate imports. v0 is one file.

---

## Routes to expose

### Plane A — orchestrator (GEPA)

Unchanged names. New work is behind `/program` + `/rollout`.

| Route | v0 |
|---|---|
| `GET /health` | `{status, env: gold_rust health, policy: monty child ready}` |
| `GET /metadata` | `optimizer_contracts.gepa` + `env_protocol` + `apply_isolation: serial_reload` |
| `GET /program` | one mutable `policy_script` module; `side_info_schemas`; seed source |
| `GET /taskset` | `{taskset_id, splits: {train, heldout}}` |
| `POST /taskset/tasks` | `train:0` → seed from `policy_smoke_v1` (101–105 train; hold out 201+ or unused smoke seeds) |
| `POST /rollout` | apply + load + one Rust episode; sync ok if horizon is smoke (60 steps) |
| `GET /rollouts/{id}` | terminal record |

`/program` seed_candidate is the bytes of
`policies/heuristic_baseline.py` (hash-pinned). `base_hash` on diffs is
sha256 of the currently loaded source.

### Plane B — policy service

| Route | Body | Return |
|---|---|---|
| `GET /health` | | `{status, policy_kind: code_policy.v1, load: monty_python, pid, content_hash}` |
| `GET /metadata` | | lever_ids, entrypoint, env_protocol, isolation |
| `POST /load` | `{kind: monty_python, path, entry, source?}` | `{ok, compile_ok, error, content_hash}` |
| `POST /act` | `{observation_text, session, valid_actions, readout, ply}` | `{actions, policy_reason}` or error |
| `POST /episode` | `{env_url, seed, task_template, max_steps}` | reward + events + achievements |
| `POST /shutdown` | | stop child |

`POST /load` writes the applied file (if not already written by
orchestrator), kills the previous IsolatedPolicyProcess, spawns a new one,
waits `ready`. Failure → `apply_report.v1` `compile_ok: false`, reward 0.

v0: orchestrator may own the act-loop itself (`/act` per tick against
Rust `/step`) **or** call `POST /episode`. Prefer `/episode` on the policy
service so the GEPA container stays a thin apply+score shell. Either way
the loop is not a `harness_module`.

### Plane B — env (already exists, keep)

Do not add these on the GEPA hostname. Orchestrator/policy call:

| Route | Use |
|---|---|
| `GET /health` | env up; pid must not change across policy `/load` |
| `POST /rollouts` | `{task, seed}` → `rollout_id` |
| `GET /rollouts/{id}/readout` | Monty observation |
| `POST /rollouts/{id}/step` | `{action}` from `actions[0]` (v0: one action per decide) |
| `GET /rollouts/{id}/event_log` | ASI body |

Optional later: `craftax_repl` JSONL instead of HTTP for sweep throughput.
Same policy IPC. Do not require it for GEPA v0.

Checkpoint/restore stay GameBench-internal. GEPA v0 is fresh episodes.

---

## Apply on `POST /rollout`

1. Resolve `task_id` → seed + `policy_dev_template.json` (or suite task).
2. Read `lever_bundle.values.policy_script` (`whole_file.v1` or
   `unified_diff.v1`). Unknown `protocol_id` → fail closed.
3. Patch `policies/candidate.py`. Hash mismatch → `apply_failed`.
4. `POST` policy `/load` with Monty spec. Child replace. **Env pid unchanged.**
5. `POST` policy `/episode` (or act-loop). Rust episode to `max_steps`.
6. Score + `side_info[]`.

`apply_isolation`: `serial_reload` (one Monty child). Compatible with
`sync_serial`. Refuse `flash_evolve` until `per_candidate_worker` (one
child + one Rust rollout handle per in-flight candidate).

Not `serial_restart` of the policy HTTP process unless `/load` cannot
respawn the child in-process. IsolatedPolicyProcess already respawns;
keep the FastAPI policy service up.

---

## Reward and ASI

Scalar reward per GEPA rollout (one seed):

```
reward = unique_achievements / |CRAFTAX_ACHIEVEMENT_UNIVERSE|
```

Use `gamebench/.../shared/scoring.py` (`unique_achievement_reward` /
`achievement_success_score` with `episode_count=1`). Do not invent a
second Craftax score. Death/timeout still returns that fraction (0 if
none). Load/compile failure: reward 0.

`side_info[]`:

| schema_id | summary | body |
|---|---|---|
| `apply_report.v1` | protocol, patch_ok, compile_ok, content_hash, env_untouched | reject_reason |
| `code_policy_game_trace.v1` | ticks, achievements, death/stop reason, actions | truncated event_log + decide traces |
| `monty_load.v1` | kind, module, entry, isolation receipt | child pid, sandbox path omitted from proposer prompt |

Fold into existing `actionable_side_info`. Proposer `state/` gets
summaries, not full NEV.

---

## Seed and taskset

- Seed policy: `heuristic_baseline.py` (wood → table → wood pickaxe on the
  fixture room). Same file the Rust smoke already proves.
- Train: `policy_smoke_v1` seeds `[101, 102, 103, 104, 105]`, `max_steps=60`.
- Heldout: next seeds in the same template (e.g. `[201, 202]`) or a pinned
  subset of `policy_batch_default_v100`. Do not mix fixture-room smoke with
  procedural 48×48 in v0.
- `task_id` form: `train:0` → seed 101, same convention as Banking77 /
  toy levers.

Proof that is not a lie: seed policy on Rust smoke already records
collect_wood / place_table / make_wood_pickaxe. A noop candidate scores
~0. A broken file is `apply_failed`.

---

## What to build (v0 order)

1. **Monty loader for Craftax** in GameBench: `resolve_policy({kind: monty_python, ...})` wrapping `IsolatedPolicyProcess`. One function both the sweep runner and GEPA policy service call.
2. **Policy HTTP service** next to `containers/codepolicy/` (`policy_app.py`): `/load` `/act` `/episode` `/health`. Talks to `gold_rust` HTTP.
3. **GEPA orchestrator** (`gepa_app.py`): plane A routes; apply
   `whole_file.v1` / `unified_diff.v1`; never starts/stops Rust on apply.
4. **Image / run script**: start `craftax_gold :8098`, policy service,
   orchestrator. GEPA `container_url` = orchestrator only.
5. **`/program` advertisement** as above; GEPA preflight accepts
   `policy_script` + those protocols (engine already has `LeverKind::PolicyScript` + `lever_bundle`).
6. Smoke: overlay-less seed rollout on `train:0` > 0; unified_diff to noop → 0; restore seed → smoke achievements; env pid stable.

Implemented under `gamebench/tasks/craftax-singleplayer/containers/gepa_codepolicy/`.
Run: `python -m containers.gepa_codepolicy.stack` (GEPA `:19300`) or
`python -m containers.gepa_codepolicy.try_policies`.

Code lives in **gamebench** (`tasks/craftax-singleplayer/containers/gepa_codepolicy/`).
This repo adds a catalog/preset pointer and preflight only — not a second
Craftax engine.

Toy `temp/craftax_levers` stays the contract mock. Do not point GEPA at it
and call it GameBench.

---

## Out of scope (v0)

- Embedding Python in `gold_rust` / PyO3
- GELO hosted code-policy (`forbid_code_policy_candidates` stays on those presets)
- `harness_restart.v1` / ReAct on this container
- `runtime: rust_compile` `policy.rs`
- Passing `engine=` / world sim into the candidate
- FlashEvolve / `per_candidate_worker`
- Procedural 48×48 max-achievement as the first GEPA taskset
- Gymnasium `craftax_env.v1` adapter in front of gold_rust
- Optimizer-side SSH/`patch` into the container
- Changing `monty_reward` (that's scoring inside the engine, not the search object)

---

## Mapping onto the custom-levers spec

| Spec item | This container |
|---|---|
| Kind | `policy_script` |
| Protocol | `unified_diff.v1` default, `whole_file.v1` seed |
| Load | `monty_python` via IsolatedPolicyProcess |
| Env | existing `gold_rust` HTTP, protocol `gamebench.craftax_gold_rust.v1` |
| Restart | none on env; child respawn on `/load` |
| ASI | `code_policy_game_trace.v1` + `apply_report.v1` |
| GELO flags | off |
