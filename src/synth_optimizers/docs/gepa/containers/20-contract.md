# Container contract

If your task can return a numeric reward for a prompt, it can be a GEPA container. You
implement the contract with the `synth-containers` package; GEPA talks to it over HTTP. The
contract is HTTP + JSON, so the language is up to you (the catalog ships Python, Rust, and
TypeScript).

Canonical definitions are exposed by the public `synth-containers` package and
its OpenAPI contract artifact. The GEPA contract version is
`synth_optimizers.gepa.v2`.

## Authoring with the `Container` class

```python
from synth_containers import Container

container = Container("banking77")

# Declare the prompt program: which modules exist, which are mutable, their seed text.
container.program(...)        # -> PromptProgram (modules, target_modules, seed_candidate)
container.taskset(...)        # -> dataset splits + rows behind each seed/task id
container.rollout(run_one)    # -> your callback: (candidate, task) -> reward

app = container.fastapi()     # a FastAPI app with every contract route registered
# or:
with container.serve() as handle:
    url = handle.connection().url   # hand this to GEPA's [container].url
```

`Container.serve()` returns a `ContainerHandle` (`.connection()`, `.health()`, `.logs()`,
`.down()`; context-manager friendly). `handle.connection()` yields a `ContainerConnection`
whose `.url` is exactly what you put in `[container]` / `GepaConfig(container=...)`.

## HTTP routes

The full surface a container exposes (from `synth_containers.sdk`):

| Route | Method | Purpose |
|-------|--------|---------|
| `/health` | GET | Liveness; returns `{status, contract_version}`. |
| `/metadata` (alias `/info`) | GET | Runtime metadata + `optimizer_contracts.gepa` version. |
| `/task_info` | GET | Task definition & capabilities for a seed/split/family. |
| `/program` | GET | The `PromptProgram`: modules, `target_modules`, `seed_candidate`. |
| `/dataset` | GET | Split names and row counts. |
| `/dataset/rows` | POST | Rows for `{split, seeds[], filters}`. |
| `/rollout` (alias `/rollouts`) | POST | Run a candidate on a task; sync or async. |
| `/rollouts/{id}` | GET | Poll an async rollout's result. |
| `/rollouts/{id}/state` | GET | Detailed lifecycle state. |
| `/rollouts/{id}/terminate` | POST | Cancel an async rollout. |

## The prompt program

`/program` returns a `PromptProgram` (version `prompt_program.v1`):

- **`modules`** — every `PromptModule`: `module_id`, `role` (`system`/`user`), `content`
  (template), `mutable`, `candidate_field`, `template_variables`.
- **`target_modules`** — the `TargetModule`s GEPA may mutate: `module_id`,
  `candidate_field` (the key GEPA writes), `objective` (e.g. `outcome_reward`).
- **`seed_candidate`** — `dict[str, str]` of baseline text per mutable field, e.g.
  `{"stage2_system": "Classify the query into exactly one Banking77 intent..."}`.
- **`rollout_overlay_schema`** — how candidate fields are injected into a rollout request.

GEPA reads this, optimizes only the `target_modules`, and sends new values back in the
`candidate` field of each rollout request.

## Rollouts

A `RolloutRequest` (strict, see `http_models.RolloutRequestModel`) carries the
`candidate` (`{candidate_field: new_text}`), a `task_id`/`seed`, a `submission_mode`
(`sync`|`async`), and a `policy` spec (`provider`, `model`, `base_url`, `credential_mode`,
`tool_call_style`, … — **no raw credentials**). Your rollout callback runs the task and
returns a `RolloutResult`:

```python
RolloutResult(
    rollout_id=...,
    task_id=...,
    reward=<float>,              # the scalar GEPA optimizes
    status="completed",
    success_status="succeeded",
    objective_scores=[ObjectiveScore(objective="accuracy", value=...)],  # optional, multi-metric
    reward_details={...}, summary={...}, usage={...}, trace={...}, metadata={...},
)
```

`reward` is the signal GEPA maximizes. `objective_scores` lets a container report multiple
objectives for Pareto selection. `trace`/`usage` flow into the run evidence and the
[board](#/cli).

### Sync vs async

- **Sync** (`submission_mode="sync"`) — POST `/rollout` blocks and returns the completed
  `RolloutResponse` (HTTP 200). Simplest; fine for fast tasks.
- **Async** (`submission_mode="async"`) — POST `/rollout` returns immediately
  (HTTP 202, `status="running"`); GEPA polls `/rollouts/{id}` until terminal, and may
  `POST /rollouts/{id}/terminate`. Use for slow rollouts (live episodes, tool agents) so
  many can run concurrently — this is what the pipelined / Flash Evolve
  [modes](#/algorithms/pipeline-modes) exploit.

## Minimal checklist

1. Expose `/program` with your mutable modules + seed text.
2. Expose a dataset (`/dataset`, `/dataset/rows`) with train/heldout splits.
3. Implement `/rollout`: apply the `candidate`, run the task, return a `reward`.
4. Advertise `optimizer_contracts.gepa = "synth_optimizers.gepa.v2"` in `/metadata`.

That's the whole contract. Point `[container].url` at it and run `gepa run` as usual.
