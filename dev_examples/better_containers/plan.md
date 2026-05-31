# Better Containers Plan

## Goal

Make Synth task containers elegant to author while keeping the container contract
algorithm-neutral. A container should expose business logic, datasets, rollout
execution, rewards, traces, and optional mutable-program metadata without naming
or depending on a specific optimizer such as GEPA.

The optimizer side can decide that a given generic container surface is enough
for a particular algorithm. The container side should not advertise
algorithm-specific compatibility by name.

## Current Problem

The Banking77 GEPA dev example currently hand-wires a FastAPI app around task
logic. That makes a simple prompt-optimization task responsible for too many
details:

- HTTP route declarations.
- Metadata shape.
- Dataset and row payload conventions.
- Prompt-program payload conventions.
- Rollout timeout behavior.
- Reward and usage response shape.
- Optimizer-specific route advertising.

The `synth-containers` package has useful typed pieces already, but the authoring
experience is still too low-level. The next layer should feel more like an SDK:
define a container, attach required methods, decorate business logic, and let the
library project it to HTTP.

## Ownership Boundary

`containers` should own:

- Stable runtime and task vocabulary.
- Stable HTTP routes for task execution and data access.
- Typed request and response models.
- Capability metadata and route hints.
- An ergonomic authoring API for wrapping business logic.
- Optional beta surfaces that are still algorithm-neutral.

`optimizers` should own:

- GEPA-specific compatibility checks.
- GEPA contract versions.
- Interpretation of candidate-program fields for GEPA.
- Algorithm-specific validation and error messages.
- Algorithm-specific docs and examples.

## Route Boundary

Stable generic routes:

```text
GET  /health
GET  /metadata
GET  /task_info
GET  /dataset
POST /dataset/rows
POST /rollout
```

These are container substrate routes. They should not be branded by a consumer or
optimizer.

Beta generic routes:

```text
GET /beta/program
```

The program route is useful for prompt optimizers, but its exact shape is still
evolving. Keep it under `/beta` until the mutable-program contract is stable
enough to promote.

Avoid routes and metadata like:

```text
/gepa/program
/optimizer/gepa
metadata.optimizer_contracts.gepa
GEPA_OPTIMIZER_CONTRACT_VERSION
```

## Capability Metadata

Containers can advertise generic capability and route hints:

```json
{
  "capabilities": {
    "rollout_modes": ["blocking"],
    "route_hints": {
      "dataset_routes": ["/dataset", "/dataset/rows"],
      "rollout_routes": ["/rollout"],
      "program_routes": ["/beta/program"]
    },
    "metadata": {
      "candidate_program": {
        "status": "beta",
        "route": "/beta/program"
      }
    }
  }
}
```

GEPA can then decide in `optimizers` that a container with `/dataset`,
`/dataset/rows`, `/rollout`, and `/beta/program` is GEPA-compatible.

## SDK-Style Authoring API

Target shape:

```python
from synth_containers import Container

container = Container(
    id="banking77_gepa_dev",
    name="Banking77 dev container",
)

@container.task_info
def task_info():
    return ...

@container.dataset
def dataset():
    return ...

@container.dataset_rows
def dataset_rows(split: str, seeds: list[int]):
    return ...

@container.beta.program
def program():
    return ...

@container.rollout(timeout_seconds=25)
async def rollout(ctx):
    row = ctx.dataset_row
    candidate = ctx.candidate
    return ...

app = container.fastapi()
```

The decorator layer should register intent. It should not create a second
semantic model. Internally, the SDK should still project through the canonical
typed objects and HTTP payload formatters.

## Lifecycle SDK

We do not yet have a clear first-class lifecycle API in `synth-containers`.
Current code exposes HTTP models/adapters, clients, capability metadata, and
compatibility projections. The missing layer is a Modal-like local runner that
can serve a container, return a URL, and shut it down cleanly.

Target shape:

```python
from synth_containers import Container

container = Container(id="banking77", name="Banking77")

# ... decorators register task_info, taskset, rollout, program ...

handle = container.serve()
try:
    print(handle.url)
finally:
    handle.down()
```

Preferred GEPA script shape:

```python
from synth_optimizers import OptimizerRun
from synth_optimizers.gepa import GepaConfig

with container.serve() as handle:
    config = GepaConfig(
        container=handle.connection(),
        taskset=...,
        program=None,
        objectives=None,
        policy=None,
        proposer=...,
    )
    result = OptimizerRun(config).execute()
```

Manual serve/down shape:

```python
handle = container.serve()
try:
    config = GepaConfig(
        container=handle.connection(),
        taskset=...,
        program=None,
        objectives=None,
        policy=None,
        proposer=...,
    )
    result = OptimizerRun(config).execute()
finally:
    handle.down()
```

The context manager is just the safer spelling of the same lifecycle.

Equivalent lower-level runner:

```python
from synth_containers import ContainerRunner

with ContainerRunner.from_app(app).serve() as handle:
    url = handle.url
```

The important boundary: optimizers should receive a `ContainerConnection` or URL,
not a launch command. `synth-containers` should own local process management,
port selection, health polling, logs, and cleanup.

### Lifecycle Objects

Proposed public objects:

```python
class Container:
    def fastapi(self) -> FastAPI: ...
    def serve(self, *, host: str = "127.0.0.1", port: int | None = None) -> ContainerHandle: ...

class ContainerRunner:
    @classmethod
    def from_container(cls, container: Container) -> ContainerRunner: ...
    @classmethod
    def from_app(cls, app: FastAPI) -> ContainerRunner: ...
    @classmethod
    def from_command(cls, command: list[str], *, cwd: str | None = None, env: dict[str, str] | None = None) -> ContainerRunner: ...
    def serve(self, *, host: str = "127.0.0.1", port: int | None = None) -> ContainerHandle: ...

class ContainerHandle:
    url: str
    host: str
    port: int
    def connection(self) -> ContainerConnection: ...
    def health(self) -> dict: ...
    def logs(self) -> Iterable[str]: ...
    def down(self) -> None: ...
    def __enter__(self) -> ContainerHandle: ...
    def __exit__(self, *exc: object) -> None: ...

class ContainerConnection:
    url: str
```

`ContainerHandle` owns lifecycle. `ContainerConnection` is the optimizer-facing
value.

### Lifecycle Semantics

- `serve()` allocates a port when none is provided.
- `serve()` starts the app or command and waits for `/health`.
- `serve()` returns a `ContainerHandle` only after the container is reachable.
- `handle.connection()` returns the stable URL-only value for optimizers.
- `handle.down()` stops only the process or runtime that the handle started.
- Context manager exit calls `down()` automatically.
- `down()` should be idempotent.
- Logs should be available for debugging without mixing container stdout into the
  optimizer's structured progress by default.
- If the runner attaches to an already-running URL, `down()` should not kill it
  unless the handle explicitly owns it.

### Naming

Use `serve()` / `down()` as the friendly local lifecycle verbs. `start()` /
`stop()` can be aliases if useful, but examples should prefer:

```python
with container.serve() as handle:
    ...
```

This mirrors the desired Modal-like experience while keeping the local behavior
plain and predictable.

## Non-Blocking Rollout Mode

Current dev examples mostly implement blocking `/rollout`: the request does the
full model call or verifier work and returns only after the rollout is complete.
The substrate already has a richer shape:

```text
POST /rollout                  submit rollout
POST /rollouts                 submit rollout alias
GET  /rollouts/{rollout_id}    fetch completed/current rollout record
GET  /rollouts/{rollout_id}/state
GET  /rollouts/{rollout_id}/summary
POST /rollouts/{rollout_id}/terminate
```

The wire model already supports:

```json
{"submission_mode": "sync" | "async"}
```

And Rust GEPA already has transport support:

```toml
[gepa]
rollout_submission_mode = "async"
rollout_poll_interval_ms = 250
rollout_async_timeout_seconds = 600
```

This is separate from GEPA's optimizer pipeline mode. `rollout_submission_mode`
chooses the container wire protocol. `gepa.pipeline.mode` chooses optimizer
orchestration.

### Target SDK Behavior

Container authoring should let the same rollout function support both sync and
async without hand-writing route plumbing:

```python
@container.rollout(timeout_seconds=60, modes=["sync", "async"])
async def rollout(ctx):
    ...
```

For `submission_mode="sync"`:

- Execute rollout immediately.
- Return terminal rollout record.

For `submission_mode="async"`:

- Create a rollout id if one was not supplied.
- Store queued/running state.
- Return a non-terminal rollout record quickly.
- Execute work in a background task/thread/process.
- Let callers poll `/rollouts/{rollout_id}` or `/rollouts/{rollout_id}/state`.

### Migration Plan

1. Keep blocking `/rollout` as the default for simple examples and compatibility.
2. Add the Container SDK runtime store needed for async rollouts:
   queued/running/completed/failed records keyed by rollout id.
3. Teach `Container` / `ContainerRunner` to expose the existing `/rollouts/*`
   routes automatically.
4. Update Banking77, TBLite, and MiniGrid containers to use the SDK route layer
   instead of hand-written `@app.post("/rollout")` routes.
5. Set example GEPA configs to:

```toml
[gepa]
rollout_submission_mode = "async"
rollout_poll_interval_ms = 250
rollout_async_timeout_seconds = 600
```

6. Keep `sync` smoke profiles for debugging, since a blocking request is still
   easier to reason about when the container logic itself is failing.

### Open Questions

- Should all SDK-authored containers default to async-capable, or should authors
  explicitly opt in with `modes=["async"]`?
- Should async execution use `asyncio.create_task`, a thread pool, or a process
  pool by default for local containers?
- How should logs and exceptions from async rollout workers be exposed on
  `GET /rollouts/{rollout_id}`?
- Should GEPA terminate timed-out async rollouts through
  `POST /rollouts/{rollout_id}/terminate` before marking them failed?

## Banking77 Migration Sketch

Keep `Banking77ClassificationTask` as the business-logic owner:

- `labels_and_rows()`
- `row_for_seed()`
- `predict_label()`
- `run_rollout()`
- prompt construction and scoring helpers

Move HTTP concerns out of the example:

- Replace direct `FastAPI(...)` route declarations with `Container(...)`.
- Replace `container_metadata()` with typed runtime metadata or container
  constructor fields.
- Replace `program_payload()` with a beta program registration.
- Keep `/dataset`, `/dataset/rows`, and `/rollout` as stable routes.
- Expose `/beta/program` as the only beta route required by GEPA.

## Cleanup Plan

1. In `containers`, remove GEPA-specific exports and helper names.
2. Move GEPA route compatibility docs to `optimizers`.
3. Change the generic HTTP adapter so it does not auto-advertise
   `metadata.optimizer_contracts.gepa`.
4. Add generic beta route hints for candidate-program support.
5. Add the `Container` authoring facade.
6. Add the `ContainerRunner` / `ContainerHandle` lifecycle facade.
7. Migrate Banking77 to the facade as the golden example.
8. Update GEPA in `optimizers` to validate the generic route/capability surface.

## Follow-Up Pushes

- Move the public vocabulary from `dataset`, `/dataset/rows`, `seed`,
  `seed_id`, and `seeds` to `taskset`, `/taskset/tasks`, `task`, `task_id`,
  and `task_ids` in a separate taskset migration push. Track that scope in
  `tasks.md`.

## Open Questions

- Should `/beta/program` return the current `PromptProgram` shape unchanged, or
  should it be renamed before promotion to avoid prompt-only assumptions?
- Should the stable route eventually be `/program` or `/v1/program`?
- Should `Container.fastapi()` be the only transport builder initially, or should
  the authoring API be transport-independent from day one?
- Should dataset rows be typed as generic `TaskInstance` objects or remain public
  JSON objects with required stable identities?
