# Scope: GEPA custom levers + side info

Companion to [`SCOPE_gepa_eval_rl_env.md`](SCOPE_gepa_eval_rl_env.md).
Grounded against code on 2026-08-19.

GEPA today searches **string prompt modules** (`BTreeMap<String, String>` against
`PromptProgram`). `LeverKind` already names `workspace_file`, `agents_md`,
`skill_md`, `action_policy`, etc., but those kinds only change cache identity
(`lever_bundle` vs `prompt_overlay`). Nothing applies a diff, restarts a policy
process, or types traces for a proposer.

This scope is the missing contract: containers advertise **how** to mutate
non-prompt levers, and **what** actionable side info they return so GEPA (and a
proposer) can act on it.

Do not invent a second optimizer. Same GEPA search loop. The container
**registers** a candidate (seed or applied lever), then **runs it on
demand** against tasks. Only the **apply** and **evidence** columns grow.

---

## Current truth — do not rediscover

| Thing | Where | Note |
|---|---|---|
| Candidates are string maps | `synth_gepa` `CandidateRecord.payload: BTreeMap<String, String>` | Whole-file text can already ride here; diffs and restart cannot |
| Program is prompt-shaped | `prompt_program.rs` `validate_for_gepa` | Mutable `candidate_field` + seed strings |
| Lever taxonomy exists | `levers.rs` `LeverKind` | Unused as apply protocol |
| Non-prompt only affects cache id | `synth_gepa` `rollout_materialization_identity` | No apply / restart |
| Craftax ReAct GEPA | catalog `crafter_container` | `react_system_prompt` only |
| Craftax Rust | GELO preset `craftax_gamebench_rust_smoke` | `required_candidate_kind: prompt`, `forbid_code_policy_candidates: true` |
| GEPA Monty Python on Rust Craftax | [`SCOPE_gepa_craftax_monty_rust.md`](SCOPE_gepa_craftax_monty_rust.md) | Search `policy_script`; load `monty_python`; eval `gold_rust`. Not GELO |
| Hosted GELO v1 | `docs/gelo/containers/20-contract.md` | `react_system_prompt` only |
| Rollout already has a junk drawer | `RolloutResult.trace` / `metadata` / `reward_details` | Untyped; proposer cannot rely on shape |
| Apply is fused into every `/rollout` | `synth_gepa` + toy/GameBench orch | Engine re-sends overlay/source per seed. Must become register-once (`POST /candidates`) then run-on-demand |

---

## One contract, two columns

```
lever protocol (inbound)   = how the optimizer mutates the running task
side info    (outbound)    = typed traces the proposer can act on
```

`GET /program` (or `/metadata`) advertises both. **Apply lives on register**,
not on every rollout. `POST /candidates` (or fused `/rollout` with an inline
bundle, prompt-only backward compat) carries a **typed apply**. `POST /rollout`
names an already-configured `candidate_id` plus a `task_id`. Terminal rollout
carries **typed side info**, not only a scalar reward.

Prompt overlay remains the default apply. Custom protocols are opt-in per
lever. Unknown protocol ids fail closed at preflight.

---

## Candidate lifecycle — register, then run on demand

This is the loop for **every** advertised lever (prompt, arbitrary code,
harness). Kind and protocol change what register does. They do not change
the shape.

```
optimizer
  GET  /program                         # seed + advertised levers / protocols / ASI schemas
  POST /candidates                      # INIT or LEVER: apply, configure, register
       body: lever_bundle | prompt overlay
       out:  candidate_id, apply_report.v1, base_hash
  POST /rollout                         # RUN on demand against the registered candidate
       body: { candidate_id, task_id }
       (repeat for every seed / minibatch row; do not re-apply)
  GET  /rollouts/{id}                   # END
       out:  scalar reward + side_info[]
  sensor / proposer / frontier
```

Candidates are **immutable** once registered. A lever produces a **new**
`candidate_id` (child of a parent), never a live mutate-in-place of `seed`.
The container holds the configured artifact (overlay, loaded module, or
restarted policy process) keyed by that id, and can evaluate it on any
task without the optimizer re-sending source.

### Why split register from rollout

GEPA already thinks this way: one candidate, many task rows. Today's
engine re-sends the full overlay on every `POST /rollout`. That is cheap
for a prompt string and **wrong** for compile/load and harness restart.

| If apply is fused into every rollout | What breaks |
|---|---|
| `whole_file.v1` / `unified_diff.v1` | Re-compile / re-`POST /load` per seed |
| `harness_restart.v1` | Restart the policy process per seed |
| Minibatch / eval split | Same candidate paid N apply costs |
| `apply_report.v1` | Mixed with episode traces; load failure looks like a bad rollout |

Plane B already has the split (`POST /load` or `/reload` or restart, then
`POST /episode`). Plane A must expose the same split. Optimizer never
speaks B.

### Plane A routes

Keep existing GEPA `/health` `/metadata` `/program` `/taskset` `/rollout`.
Add one register verb. Do not put plane-B `/load` on the GEPA hostname.

| Route | When | Body | Success | Failure |
|---|---|---|---|---|
| `GET /program` | once | — | seed text/source + lever ads | — |
| `POST /candidates` | init seed, and once per proposed child | `lever_bundle` (typed) or prompt overlay | `{candidate_id, parent_id?, apply_report, base_hash}` | `apply_failed`; **no** `candidate_id` that can be rolled out |
| `POST /rollout` | per task row | `{candidate_id, task_id}` — candidate already configured | terminal reward + `side_info[]` | env/episode error; candidate still registered |
| `GET /rollouts/{id}` | after | — | same terminal record | — |

**Init** = register the `/program` seed as `candidate_id=seed` (container
may do this at boot). **Lever** = register a child from a parent
`base_hash` + apply payload. **Rollout** = run. **End** = reward + ASI
for the sensor.

Backward compat: `POST /rollout` with an inline `lever_bundle` and no
`candidate_id` means register-if-needed then run **this one task**. Prompt
overlay containers may keep doing that forever. Code and harness
containers must still be able to register once and fan out seeds. Engine
v0 for custom protocols: register once, then `candidate_id` on every
rollout. Do not stringify diffs into `BTreeMap<String, String>`.

### ASCII — same outer loop, three inners

Optimizer always sees this. Kind only changes what the orchestrator does
behind `POST /candidates` vs `POST /rollout`.

```
  GEPA                                              orchestrator (plane A)
   |                                                     |
   |-- GET /program ------------------------------------>|  seed + lever ads
   |-- POST /candidates  lever_bundle ------------------>|  REGISTER (once)
   |<-- candidate_id, apply_report, base_hash -----------|
   |-- POST /rollout {candidate_id, task_0} ------------>|  RUN
   |<-- reward + side_info[] ----------------------------|
   |-- POST /rollout {candidate_id, task_1} ------------>|  RUN  (no re-apply)
   |<-- reward + side_info[] ----------------------------|
   '-- sensor / proposer / frontier
```

What happens **inside** on register vs run:

```
  PROMPT overlay.v1              CODE policy_script              HARNESS harness_restart.v1
  ─────────────────              ──────────────────              ─────────────────────────

  REGISTER                       REGISTER                        REGISTER
  orch stores overlay[c_id]      orch writes/patches source      orch writes/patches loop files
  policy POST /reload            policy POST /load               SIGTERM policy process
        (same PID, RAM)                (import / compile /             spawn policy' (NEW PID)
                                        monty child)                   wait /health
  env stays                      env stays                       env stays
  apply_report: overlay ok       apply_report: compile_ok        apply_report: health ok
                                 no candidate_id if load fails   no candidate_id if health fails

  ROLLOUT (per seed)             ROLLOUT (per seed)              ROLLOUT (per seed)
  policy POST /episode           policy POST /episode            policy' POST /episode
    overlay already set            module already loaded           new loop already up
  policy ──reset/step──► env     policy ──act/step──► env        policy' ──reset/step──► env
  ASI: prompt_trace              ASI: code_policy_game_trace     ASI: harness_v5_trace
       harness_v5_trace                (apply_report was               (apply_report was
                                       on REGISTER)                    on REGISTER)

  do NOT restart                 do NOT /load again              do NOT restart again
  re-overlay per seed is ok      re-compile per seed is wrong    restart per seed is wrong
  (cheap) but not required
```

Process topology after a successful register (env never dies):

```
  PROMPT                         CODE                            HARNESS
  ──────                         ────                            ───────

  [orch]                         [orch]                          [orch]
     |                              |                               |
     |  /reload (in-process)        |  /load (module or             |  restart
     v                              v   IsolatedPolicyProcess)      v
  [policy PID 1]                 [policy PID 1]                  [policy PID 2]   PID 1 gone
     |                              |   └─ monty child (opt)           |
     |  episode                     |  episode                         |  episode
     v                              v                                  v
  [env]  unchanged               [env]  unchanged                 [env]  unchanged
```

Today vs this (why code/harness cannot stay fused):

```
  TODAY (engine)                          THIS
  ──────────────                          ────
  for seed in minibatch:                  POST /candidates     # apply once
    POST /rollout {                         POST /rollout seed_0
      full overlay or source,               POST /rollout seed_1
      task_id                               POST /rollout seed_2
    }                                       ...
      → apply + load/restart + episode
        every time                        prompt MAY keep the fused /rollout
                                          code/harness MUST NOT
```

### What each kind does at each step

Same four steps. Different configure action.

#### 1. Prompt overlay (`system_prompt` / `text_prompt` / `user_prompt`)

Protocol: `prompt_overlay.v1`. No files. No process restart.

| Step | Container | Optimizer |
|---|---|---|
| Init | Register seed strings from `/program` as `seed` | Read `/program` |
| Lever | Store overlay map keyed by new `candidate_id`. Optionally `POST` policy `/reload` (in-process) | Send `{lever_id: text}` bundle |
| Rollout | `POST` policy `/episode` with that overlay (or already-reloaded process) | `{candidate_id, task_id}` only |
| End | `prompt_trace.v1` and/or `harness_v5_trace.v1` + scalar reward | Sensor copies ASI; proposer sees summaries |

Re-apply per seed is allowed (cheap) but not required. Prefer register
once so the engine's candidate id and the container's configured overlay
are the same object.

#### 2. Arbitrary code / optimize-anything (`policy_script`, `sourced_python`)

Protocol: `unified_diff.v1` (default) or `whole_file.v1`. Restart is **not**
implied unless `load` cannot hot-reload.

| Step | Container | Optimizer |
|---|---|---|
| Init | Seed source hash-pinned on disk; `POST` policy `/load` seed; register `seed` | Read `/program` constraints (`entrypoint`, `signature`, `load`) |
| Lever | Validate `base_hash`; write/patch; `POST` policy `/load` (import/compile). New `candidate_id` only if `compile_ok` | Send typed `lever_bundle.values` (`protocol_id`, `content`/`diff`, `base_hash`) — **not** a prompt field |
| Rollout | Episode / act-loop against the **already loaded** module. No second `/load` | `{candidate_id, task_id}` — many seeds, one load |
| End | `code_policy_game_trace.v1` per episode; `apply_report.v1` on **register** (and on rollout only if you fused) | Sensor: load errors are apply failures, not "the policy scored 0 and also compiled" mixed into one blob |

`policy_script` load/run contract: dedicated env entrypoint (`act` /
`choose_actions` / …). `sourced_python`: importlib module the running
service imports; no required env entrypoint. Do not stuff source into
`react_system_prompt`.

Craftax Monty: register = write temp file + IsolatedPolicyProcess
`monty_python` `/load`. Rollout = gold_rust episode with that child
already up. See [`SCOPE_gepa_craftax_monty_rust.md`](SCOPE_gepa_craftax_monty_rust.md).

#### 3. Harness (`harness_module`, optional `sourced_python` helpers)

Protocol: `harness_restart.v1`. Apply files, then restart the **policy
service**. Env stays up.

| Step | Container | Optimizer |
|---|---|---|
| Init | Seed harness files on disk; policy process serving the seed loop; register `seed` | Read `/program` (`paths[]`, `restart: process_restart`, `apply_isolation`) |
| Lever | Apply files; SIGTERM policy; spawn; wait `/health`. New `candidate_id` only if healthy. Roll back tree on failure | Same typed bundle as code, plus restart token |
| Rollout | `POST` policy `/episode` against the **new** process. No second restart | `{candidate_id, task_id}` — many seeds, one restart |
| End | `harness_v5_trace.v1` + scalar reward; `apply_report.v1` on register | Sensor as above |

`apply_isolation`:

- `serial_restart` — one policy process. Register swaps it. Rollouts of
  two candidates cannot overlap. **Refuse `flash_evolve`.**
- `per_candidate_worker` — register forks a worker keyed by
  `candidate_id`. Run-on-demand hits that worker. Required if overlap
  and harness opt are both on.

Do not restart per seed. Do not restart the env. Do not restart the
GEPA-facing orchestrator.

#### 4. Same loop, mixed program

A program may advertise a prompt lever **and** a `policy_script`. Each
register still produces one `candidate_id` whose payload is the full
`lever_bundle`. Rollouts run whatever that bundle configured. Unknown
`protocol_id` fails at **register** (preflight / apply), never mid-episode.

`workspace_file` / `agents_md` stay escape hatches. They still register
then run; they do not become the Craftax default.

### Engine (what `synth_gepa` must stop doing)

| Today | Needed |
|---|---|
| `POST /rollout` body is prompt overlay (`BTreeMap<String, String>`) every seed | Custom protocols: `POST /candidates` once, then `/rollout` with `candidate_id` |
| `LeverBundle::to_prompt_payload` drops non-string values | Keep `values` as objects (`protocol_id`, `content`/`diff`, `base_hash`) |
| Sensor copies `actionable_side_info` only | Container sets ASI; summaries in proposer `state/`; full traces `artifact_ref` |
| No protocol preflight | Unknown protocol fail closed before register; refuse `flash_evolve` + `serial_restart` |

Do not invent a second algorithm slug. Prompt overlay stays the default
path (inline `/rollout` still fine). This lifecycle is how custom levers
join that path.

---

## Lever protocol (inbound)

Each mutable lever declares:

| Field | Meaning |
|---|---|
| `lever_id` | Same as today's `candidate_field` |
| `kind` | **What it is** — load/run contract (`policy_script`, `sourced_python`, `harness_module`, …). Not the file extension |
| `protocol_id` | **How to mutate it** — `prompt_overlay.v1`, `unified_diff.v1`, `whole_file.v1`, `harness_restart.v1` |
| `apply_schema` | JSON Schema for the candidate value (not always a string) |
| `constraints` | Entrypoint, runtime, env protocol, paths, max bytes, restart budget |

`kind` and `protocol_id` are orthogonal. A `policy_script` may apply as
`whole_file.v1` or `unified_diff.v1`. A `harness_module` usually pairs with
`harness_restart.v1`. Do not encode “this is Python” or “this is a diff”
into `kind`.

### Dedicated kinds (add these)

Existing `LeverKind` is prompt-shaped plus a file dump (`workspace_file`,
`agents_md`, `skill_md`, vague `action_policy`). That is not enough for
optimize-anything or harness opt. Add **load/run kinds**. Keep
`workspace_file` as the fail-closed escape hatch, not the Craftax default.

| `kind` | Search object | Container load/run contract | Craftax / notes |
|---|---|---|---|
| `policy_script` | Source that **is** the policy | Load, then call a dedicated entrypoint compatible with the env protocol (`act` / `run` / `step`: obs → action). Runtime in constraints (`python_source` \| `rust_compile` \| `wasm`), not in the kind name | Code-policy Craftax. gepa-ai scheduling `_step` / kernel analogue |
| `sourced_python` | `.py` the **policy process imports** | `importlib` / exec into the running service (reload in-process). No required env entrypoint — helpers, encoders, parsers the harness imports | In-process Python the container sources. Use this when the file is a module, not the `act()` policy |
| `harness_module` | Agent **loop** (obs format, tool parse, termination, ReAct body) | Apply then **restart policy service**. Env stays. Isolation advertised on the bundle | Harness opt for React Craftax. gepa-ai ARC-AGI / DSPy-full-program analogue |
| `system_prompt` / `text_prompt` / `user_prompt` | Prompt text (already exist) | Overlay into the policy request | Default ReAct / Banking77 |
| `agents_md` / `skill_md` | Codex instruction files (already exist) | Sandbox workspace files; not the Craftax loop | FinQA / TBLite. Do **not** reuse for Craftax harness |
| `tool_policy` | Tool allow-list / schemas (already exist) | Policy-service tool surface | Optional Craftax ReAct tool filter |
| `workspace_file` | Generic path (already exist) | Write bytes, no run contract | Escape hatch only. Unknown run semantics fail closed if a dedicated kind exists |
| `action_policy` | **Do not advertise on new programs** | Parse alias of `policy_script` so old manifests still cache-identity | Was never a run contract. New containers use `policy_script` |

`policy_script` constraints (required to advertise):

```json
{
  "runtime": "python_source",
  "entrypoint": "act",
  "signature": "craftax_env.v1",
  "load": "import",
  "path": "policy.py"
}
```

| Constraint | Meaning |
|---|---|
| `runtime` | `python_source` (container sources / import) · `rust_compile` (`POST /load`) · `wasm` |
| `entrypoint` | Symbol the policy service calls: `act`, `run`, `step`, `Policy.act` |
| `signature` | Must equal the env `protocol_id` (`craftax_env.v1`). Fail closed on mismatch |
| `load` | `import` \| `compile` \| `subprocess` — matches policy-service `reload` |

`sourced_python` constraints: `import_name`, `path`, `reload: importlib`. If
the module **also** exposes `act` and is the policy, advertise
`policy_script` with `runtime: python_source` instead — one lever, one
kind.

`harness_module` constraints: `paths[]`, `restart: process_restart`,
`apply_isolation`. Default protocol `harness_restart.v1`.

Do **not** add language-named kinds (`python_file`, `rust_crate`) or
slice the harness into `observation_encoder` / `action_decoder` in v0.
Those are constraints or files inside `harness_module`. Do not add env or
reward-function kinds: the env is not the search object.

### Standard protocols (v0)

### Standard protocols (v0)

| `protocol_id` | Candidate payload | Container apply | Use |
|---|---|---|---|
| `prompt_overlay.v1` | `{lever_id: string}` | Inject into policy request | Banking77, ReAct Craftax |
| `whole_file.v1` | `{path, content, content_hash}` | Write file atomically | Seed Rust policy, small harness |
| `unified_diff.v1` | `{path, diff, base_hash}` | `patch` against `base_hash`; reject mismatch | Incremental code policy / harness |
| `harness_restart.v1` | `whole_file` or `unified_diff` **plus** restart token | Apply files, then restart **policy service**, then rollout | Harness opt when the running loop must reload |

`unified_diff.v1` is the default for code. `whole_file.v1` is the dedicated
special case when the search object is one (or a few) complete sources and
diffs are not worth the apply surface.

### Apply lifecycle (on register, not on every rollout)

See **Candidate lifecycle** above. Apply happens once per `candidate_id`:

1. Validate against advertised `apply_schema` and parent `base_hash`.
2. Apply (overlay / write / patch).
3. If protocol requires it: **load/compile** or **restart policy service**,
   wait `/health`. Fail **register** if the new process does not come up
   — do not issue a `candidate_id` that can be rolled out.
4. Return `apply_report.v1`. Rollouts against that id then only run the
   task.

Do not leave a half-applied tree if restart/load fails — roll back or
mark `apply_failed`. Optimizer does not shell into the container. Apply
is always container-owned. The wire only names protocol + payload.

### Special case A — whole-code updates

For Craftax **code policy** (and similar):

- One (or a small set of) `policy_script` levers, `protocol_id = whole_file.v1`
  or `unified_diff.v1`. Not `workspace_file`.
- Seed is the checked-in policy source (or a hash-pinned snapshot).
- Apply writes/patches, then the advertised `load` (`import` / `compile`).
- Entrypoint must type-check against `signature` = env protocol.
- Restart is **not** implied unless `load` cannot hot-reload.
  Advertise `reload: "per_candidate_load" | "in_process" | "process_restart"`.
  Load once per registered `candidate_id`, not per seed.

GEPA search object = policy source (or diffs against it). Not a ReAct prompt
with `forbid_code_policy_candidates`.

### Special case B — harness updates + policy service restart

For **harness opt** (React Craftax harness, Codex sandbox, tool loop):

- Levers are `harness_module` (the loop) plus optional `sourced_python`
  helpers. Not `agents_md` and not a bag of `workspace_file`.
- `protocol_id = harness_restart.v1` on the bundle (or on the lever that
  invalidates the running loop).
- After apply, container **restarts the policy service** (the process that
  serves the agent loop), then accepts the rollout.
- Restart is billed as apply cost, not as a successful search step if health
  fails.
- Concurrent rollouts against a restarting policy: container serializes apply
  or forks an isolated policy worker per candidate. Advertise
  `apply_isolation: "serial_restart" | "per_candidate_worker"`.

Do not overload `prompt_overlay.v1` by stuffing harness source into
`react_system_prompt`.

---

## Side info (outbound)

Reward stays the scalar GEPA maximizes. Side info is **typed, versioned,
actionable** evidence for the proposer and for later RL-on-proposer work.

Advertise on `/program` or `/metadata`:

```json
{
  "side_info_schemas": [
    {
      "schema_id": "code_policy_game_trace.v1",
      "when": "terminal_rollout",
      "purpose": "proposer_actionable"
    },
    {
      "schema_id": "harness_v5_trace.v1",
      "when": "terminal_rollout",
      "purpose": "proposer_actionable"
    }
  ]
}
```

Put payloads on the terminal record under a single envelope, not a new HTTP
verb:

```json
{
  "reward": 0.42,
  "side_info": [
    {
      "schema_id": "code_policy_game_trace.v1",
      "lever_ids": ["src/policy.rs"],
      "artifact_ref": "…",
      "summary": { "ticks": 120, "deaths": ["lava"], "compile_ok": true },
      "body": { }
    }
  ]
}
```

| `schema_id` (examples) | Actionable content |
|---|---|
| `code_policy_game_trace.v1` | Tick/action log, compile/load errors, achievement times — enough to propose a diff |
| `harness_v5_trace.v1` | Tool calls, loop state, restart timings, overlay vs harness split |
| `apply_report.v1` | Patch hunks applied, reject reason, `base_hash`, restart duration |
| `prompt_trace.v1` | Existing ReAct/LLM traces (today's `trace` field, now named) |

Rules:

- Side info is **not** the reward. Scoring stays numeric.
- Unknown `schema_id` is stored as an artifact; proposers skip it.
- Large bodies are artifact refs; `summary` stays small enough for the Codex
  workspace read model.
- Apply failures emit `apply_report.v1` with `compile_ok` / `patch_ok` /
  `restart_ok` so the proposer can fix the next candidate instead of seeing a
  generic rollout fail.

---

## Proposer / GEPA wiring

Keep `payload: BTreeMap<String, String>` only for `prompt_overlay.v1`.

For custom protocols, candidate records already have `lever_bundle` (`values:
BTreeMap<String, Value>`). Use that. Do not stringify diffs into prompt fields.

Preflight:

- Container advertises protocols ∩ optimizer supports them.
- Seed `base_hash` matches container snapshot.
- If any lever is `harness_restart.v1`, pipeline must tolerate apply latency
  and isolation mode.

Proposer workspace: include `apply_report` + `side_info.summary` in
`state/` the same way Banking77 traces are included today. Do not dump full
game traces into the prompt; link artifacts.

GELO Craftax Rust presets that set `forbid_code_policy_candidates` are a
**different algorithm**. This scope is GEPA. Do not silently reuse those
flags.

---

## Mapping onto gepa-ai `optimize_anything`

Canonical reference: [optimize_anything](https://gepa-ai.github.io/gepa/api/optimize_anything/optimize_anything/),
[Evaluator / ASI](https://gepa-ai.github.io/gepa/api/optimize_anything/Evaluator/),
[GEPAAdapter](https://gepa-ai.github.io/gepa/api/core/GEPAAdapter/),
[blog](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/).

gepa-ai's claim: if the artifact is text and quality is measurable, search it.
The evaluator returns `(score, ASI)`. ASI is first-class diagnostic feedback
(compiler errors, traces, images) for the proposer — not a second reward.
`make_reflective_dataset` turns trajectories into a small JSON dataset the
teacher LM reads. Candidates are still `str | dict[str, str]`. Three modes:
single-task, multi-task, generalization (`dataset` + `valset`). Our GEPA
taskset (train / heldout) is **generalization**. The ARC-AGI case in that
blog is harness search: seed a 10-line agent, evolve control flow + helpers
+ prompts as one text artifact.

| gepa-ai | This engine |
|---|---|
| `seed_candidate: str \| dict[str, str]` | `/program` seed registered as `candidate_id=seed` |
| proposed child candidate | `POST /candidates` with typed `lever_bundle` |
| `evaluator → (score, ASI)` | `POST /rollout` `{candidate_id, task_id}` → scalar `reward` + typed `side_info[]` |
| `oa.log()` / free-form ASI dict | versioned `schema_id` envelopes; large bodies are `artifact_ref` |
| `make_reflective_dataset` | Codex proposer workspace `state/` (already packs `actionable_side_info`) |
| `capture_traces=True` | always on terminal records; summaries only in the prompt |
| `objective` / `background` | `/program` objective + lever `constraints` |
| `dataset` / `valset` | taskset train / heldout seeds |
| DefaultAdapter | `prompt_overlay.v1` (today's Crafter ReAct) |
| code / policy as text artifact | `whole_file.v1` / `unified_diff.v1` (code-policy Craftax); load on register |
| DSPy full-program / ARC-AGI harness evolution | `harness_restart.v1` (React Craftax harness); restart on register |
| `batch_evaluator` | many `/rollout`s against one registered id (`async_pipelined` / `flash_evolve`) |

Do not wrap gepa-ai as a second engine. Same register → rollout → frontier
loop. The gap is: gepa-ai in-process `evaluate()` can `exec` the candidate;
our evaluator is a **container**. Apply, compile, and restart happen at
**register**, not host-side `exec` and not per seed.

The engine already has an ASI slot (`SensorFrame.actionable_side_info`) that
the Codex proposer workspace serializes. It is untyped junk-drawer today.
`side_info[]` is that slot with `schema_id`.

---

## Policy service + env HTTP (the missing split)

Today `crafter_container` is a **monolith**: ReAct loop + Craftax (JAX) live
in one process. GELO Crafter is the same shape (in-process policy loop, env
snapshots as pickles). Hosted GELO v1 and `craftax_gamebench_rust_smoke`
overlay `react_system_prompt` only (`forbid_code_policy_candidates: true`).

That monolith cannot:

- restart the agent loop without killing the env
- point a Rust code policy at the same Craftax env a ReAct policy uses
- mutate harness files and reload them independently of the env
- isolate concurrent candidates (FlashEvolve rollout lane) without cloning
  the whole container

Formalize **two HTTP planes**. GEPA still talks only to plane A.

```
Plane A  (existing GEPA contract)
  optimizer  -- /program /rollout /health -->  orchestrator (the "container")

Plane B  (new, internal to the image)
  policy service  -- reset / step / spec -->  env service
```

```
[GEPA] --A--> [orchestrator]
                 | apply candidate (overlay / write / patch)
                 | restart policy if protocol says so
                 | wait policy /health
                 |
                 +-- starts / owns --> [policy service]
                 |                      ReAct loop  OR  compiled code policy
                 |                      (harness lives here)
                 |
                 +-- starts / owns --> [env service]
                                        Craftax only: reset / step / obs
```

Optimizer never speaks plane B. Apply is orchestrator-owned. Env has no
prompts, no LLM, no candidate overlay.

### Env service — formalize

Gymnasium-over-HTTP. One Craftax implementation, shared by ReAct and code
policy. No policy imports this as a Python module; HTTP only.

| Route | Body | Return |
|---|---|---|
| `GET /health` | | `{status, env_id, version}` |
| `GET /spec` | | action space, observation space, achievement ids, max horizon |
| `POST /reset` | `{seed, max_steps?}` | `{obs, info}` |
| `POST /step` | `{action}` | `{obs, reward, terminated, truncated, info}` |

v0 may omit env-side checkpoint/resume (that's GELO Layer 1). GEPA v0 is
fresh episodes per rollout. If we later want GELO on the same env, add
`POST /checkpoint` + `POST /restore` as an additive env capability — do not
put snapshots on the policy service.

`info` must be rich enough to build `code_policy_game_trace.v1` /
`harness_v5_trace.v1` summaries: tick, inventory, achievements unlocked this
step, death cause, lava/water flags. Reward stays the Craftax scalar.

Advertise `env_id` + `protocol_id = craftax_env.v1` on orchestrator
`/metadata` so GEPA preflight can refuse a policy built for a different env.

### Policy service — formalize

Long-lived process. Talks to env over plane B. Two variants, **one envelope**.

Common:

| Route | Purpose |
|---|---|
| `GET /health` | process up; after restart this is the ready gate |
| `GET /metadata` | `policy_kind`, `reload`, `lever_ids`, `env_protocol` |
| `POST /shutdown` | orchestrator-owned teardown |

`policy_kind`: `react_http.v1` | `code_policy.v1`.
`reload`: `prompt_overlay` | `in_process` | `per_candidate_load` | `process_restart`.
(`per_candidate_load` = once per registered id. Do not advertise `per_rollout_compile`.)
`env_protocol`: `craftax_env.v1`.

**Variant 1 — ReAct (HTTP between policy and env).** Today's in-process
Crafter loop, split. The policy service owns the harness: observation
formatting, tool/action parse, LLM call, retry, termination. It calls env
`/reset` + `/step` until terminal, then returns an episode record to the
orchestrator.

| Route | Purpose |
|---|---|
| `POST /episode` | `{env_url, seed, prompt_overlay?, max_steps, policy}` → episode result + `harness_v5_trace.v1` |
| `POST /reload` | apply `prompt_overlay.v1` without process restart |

Episode loop stays **in the policy service**, not the orchestrator. Harness
opt mutates that loop. If the orchestrator owned the loop, harness files
would be orchestrator source and restart would mean restarting the GEPA
container — wrong process.

`policy` on `/episode` is the rollout policy spec (model, base_url).
Credentials still do not cross into GEPA's proposer; they stay container-
local as today.

**Variant 2 — code policy.** Search object is policy source (`policy.rs` /
`policy.py`). No LLM per tick unless the code itself calls one.

| Route | Purpose |
|---|---|
| `POST /load` | compile or import the applied source; `{compile_ok, error}` |
| `POST /act` | `{obs} → {action}` (orchestrator or a thin runner drives the episode) |
| `POST /episode` | optional: policy drives env itself, same as ReAct |

v0 can let a thin runner inside the orchestrator call `/act` in a loop —
code-policy harness is not the search object. `/load` failure is
`apply_report.v1` with `compile_ok: false`, reward 0, not an engine crash.
gepa-ai's evaluator rule: never raise on one example; return a failed score
plus ASI. Same here.

### Orchestrator — what GEPA sees

Unchanged GEPA discovery routes (`/program`, `/taskset`, `/health`, …).
Apply moves to `POST /candidates`. `POST /rollout` evaluates a registered
id.

**Register** (`POST /candidates`):

1. Validate vs advertised `apply_schema` / `base_hash`.
2. Apply (overlay / write / patch) into the policy service's file tree.
3. If `prompt_overlay.v1`: `POST /reload` on policy (no process restart).
4. If `whole_file.v1` / `unified_diff.v1` on a code policy: `POST /load`
   (compile/import). Advertise `reload: per_candidate_load` (once per
   registered id, not per seed).
5. If `harness_restart.v1`: SIGTERM policy, spawn new process, wait
   `/health`, fail closed as `apply_failed` if it does not come up. Roll
   back the tree. Env process stays up.
6. Return `{candidate_id, apply_report}`. Do not run an episode here.

**Run on demand** (`POST /rollout` with `candidate_id` + `task_id`):

7. `POST /episode` (or act-loop) against the env using the already
   configured policy process / overlay / loaded module.
8. Return `reward` + `side_info[]`.

`apply_isolation` (already in this spec):

- `serial_restart` — one policy process, rollouts queue behind restart.
  Fine for `sync_serial`. **Incompatible with FlashEvolve overlap** if
  restart is on the rollout critical path.
- `per_candidate_worker` — fork a policy worker (and optionally an env)
  per in-flight candidate. Required if `flash_evolve` + harness opt are
  both on. See [`aug19_gepa.md`](aug19_gepa.md): overlap needs real
  concurrent lane execution, not just concurrent leases.

---

## Two Craftax products, two search modes

Same env. Different policy service. Different advertised levers.

### A. Optimize-anything (any advertised text lever)

gepa-ai DefaultAdapter / code-as-text. **Do not restart** unless the lever
says so.

| Container | Kind | Search object | Protocol | ASI |
|---|---|---|---|---|
| Craftax ReAct, split | `system_prompt` | `react_system_prompt` | `prompt_overlay.v1` | `prompt_trace.v1` + `harness_v5_trace.v1` summary |
| Craftax code policy | `policy_script` | `policy.py` / `policy.rs` with `act` | `whole_file.v1` or `unified_diff.v1` | `code_policy_game_trace.v1` + `apply_report.v1` |

Code-policy seed is the checked-in policy source (hash-pinned). Register
writes and `/load`s once; each seed is a `/rollout` against that
`candidate_id`. Traces come back per episode. This is the Craftax analogue
of gepa-ai evolving a scheduling `_step` or a CUDA kernel: the candidate
**is** the code, ASI is compile (on register) + tick diagnostics (on
rollout).

GELO `forbid_code_policy_candidates` stays off this path.

### B. Harness code opt (React Craftax)

gepa-ai ARC-AGI / DSPy-full-program analogue: evolve the **loop**, not the
prompt stuffed with harness source.

Levers: `harness_module` (loop) + optional `sourced_python` helpers.
`protocol_id = harness_restart.v1`. After apply, orchestrator restarts
**policy service only**, env stays. Concurrent rollouts need
`per_candidate_worker`.

ASI: `harness_v5_trace.v1` — tool calls, parse failures, loop-state,
restart duration, overlay vs harness split — plus `apply_report.v1`.

Do not implement harness opt by putting harness source into
`react_system_prompt`.

---

## FlashEvolve constraints (from `aug19_gepa.md`)

Lane overlap hides at most `min(propose_busy, rollout_busy)`. Implications
for these containers:

- **Code policy** rollouts are compile + ticks, no per-tick LLM. Rollout
  lane is short. FlashEvolve will not look fast unless many seeds run in
  parallel (`workers.rollout`) or the proposer is cheap. Measure the
  ceiling before claiming speedup. Same trap as Banking77 smoke.
- **Harness opt + `serial_restart`** serializes the rollout lane on
  process boot. That **destroys** the overlap the 2026-08-19 worker pool
  was built to provide. Harness opt under `flash_evolve` requires
  `per_candidate_worker` (or refuse the combo at preflight).
- ReAct episodes stay **async** `/rollout`. Code-policy may be sync if
  compile+episode is short; still prefer async so the lane executor can
  overlap.
- Crafter cells in the 3×3 matrix have never been executed. First proof
  of these modes is a smoke shape, not a wall-clock claim.
- Duplicate-enqueue / `async_lane_work_already_queued` still applies:
  a restart that returns `apply_failed` must not re-enqueue as a new
  prerequisite forever.

---

## What to formalize vs build

### Formalize (contracts, fail-closed)

1. `craftax_env.v1` — env routes, observation/action JSON, `info` fields
   required for traces.
2. Policy-service envelope — `policy_kind`, `reload`, `/health` ready
   gate, `/episode` result schema shared by ReAct and code policy.
3. Lever advertisement on `/program` — `protocol_id`, `apply_schema`,
   `constraints` (paths, max bytes, restart budget), `apply_isolation`.
4. `side_info[]` JSON Schema for `code_policy_game_trace.v1`,
   `harness_v5_trace.v1`, `apply_report.v1`. Fold into existing
   `actionable_side_info` on `SensorFrame`; do not add a second proposer
   channel.
5. Preflight: optimizer ∩ container protocols; refuse
   `flash_evolve` + `serial_restart`; refuse unknown `protocol_id`.
6. OpenAPI for plane B (env + policy). Plane A stays
   `synth_optimizers.gepa.v2` plus additive fields.

### Build (v0, in order)

1. Split today's Crafter monolith: env process + ReAct policy service +
   orchestrator that still speaks GEPA `/rollout`. Prove prompt overlay
   still matches `crafter_container` reward on a few seeds.
2. Code-policy Craftax: same env, `whole_file.v1` + `/load` compile,
   `code_policy_game_trace.v1`. Seed = pinned `policy.rs` (or Python
   equivalent). GEPA preflight + `lever_bundle` through `/rollout`.
3. Typed `side_info[]` on terminal records → proposer `state/` summaries
   (engine already serializes `actionable_side_info`).
4. Harness opt on the ReAct policy tree: `unified_diff.v1` +
   `harness_restart.v1`, `serial_restart` first, then
   `per_candidate_worker` if FlashEvolve is in scope.
5. Only then: `flash_evolve` matrix cell. Know the overlap ceiling
   first.

Prompt overlay stays the default. Optimize-anything is "any advertised
text lever." Harness restart is the dedicated special case.

---

## Out of scope (v0)

- A new optimizer algorithm slug
- Hosted GELO code-policy (still prompt-only unless GELO adopts this contract)
- Optimizer-side `patch` / SSH into the container
- Arbitrary Python import of user harnesses on the host
- Changing the Banking77 prompt path
- Env-side GELO resume (`/checkpoint` `/restore`) — additive later
- Wrapping the public `gepa` PyPI engine inside this optimizer
- Putting plane-B routes on the GEPA-facing hostname

---

## v0 slice

1. Advertise `protocol_id` + `side_info_schemas` on `/program` (additive;
   prompt containers omit them and keep today's overlay).
2. Formalize `craftax_env.v1` + policy-service envelope; split ReAct
   Crafter so policy↔env is HTTP.
3. Implement `unified_diff.v1` + `whole_file.v1` apply on one code-policy
   container (Craftax code policy is the motivating task).
4. Implement `harness_restart.v1` on the ReAct Craftax policy service
   (harness files, restart policy only).
5. Envelope `side_info[]` on terminal rollouts; ship
   `code_policy_game_trace.v1` and `harness_v5_trace.v1` as the first two
   schemas; fold into `actionable_side_info`.
6. GEPA: validate protocol at preflight; store `lever_bundle` values;
   `POST /candidates` once per child; `POST /rollout` with `candidate_id`
   per task; surface side-info summaries to the proposer; refuse
   `flash_evolve` + `serial_restart`. Prompt overlay may keep fused
   inline `/rollout`.

Prompt overlay stays the default. Custom protocols are explicit special
cases, not a second GEPA.
