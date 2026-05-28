# synth_gepa

GEPA algorithm crate.

This crate owns search state and optimization semantics. It consumes platform
services but does not perform raw filesystem, HTTP, SQLite, or subprocess work
except through `synth_optimizer_platform` traits and structs.

## Responsibilities

- Candidate registry and lineage.
- Prompt candidate validation against the container prompt program.
- Seed candidate registration.
- Rollout candidate overlays.
- Proposer request and response schemas.
- Codex proposer orchestration through the platform proposer boundary.
- Minibatch, full-train, and heldout scheduling.
- Pareto/frontier logic.
- Acceptance criteria.
- Stopper logic.
- Final best-candidate selection.

## Core Types

### Candidate State

- `CandidateId`: stable id derived from normalized candidate payload.
- `CandidateRecord`: payload, parent ids, source, status, and lineage.
- `CandidateRegistry`: append-only registry plus lookup by id.
- `CandidateEvaluation`: rollout results over a seed pool.
- `CandidateOverlay`: platform overlay applied to container rollout requests.

### Seed Pools

- `SeedPool`: named pool such as `minibatch`, `train`, `heldout`.
- `SeedBatch`: concrete seeds and rows for one evaluation action.
- `Scheduler`: chooses the next work item from GEPA state.

### Proposer

- `ProposerRequest`: parent candidates, reflective evidence, target modules,
  budget, and config.
- `ProposerResponse`: candidate proposals plus usage and raw trace.
- `ProposedCandidate`: candidate payload and rationale.
- `ProposerBackend`: `codex_app_server`. Unknown backends, including the old
  `local_process_json` fallback, fail during config validation.
- `CodexAppServerWorkspace`: run-local files under artifacts that Codex reads
  and updates by writing `proposal/manifest.json`. The workspace contract uses
  `gepa_workspace_proposal_v3` plus `state/algorithm_read_model.json`,
  `state/candidates.json`, `state/rollouts.json`, evidence frames, links, and
  the parent payload so the proposer can inspect actual search evidence.

### Frontier And Selection

- `ParetoFront`: per-example or aggregate candidate frontier.
- `FrontierMember`: candidate id plus objective vector.
- `AcceptanceDecision`: accept, reject, or defer with reason.
- `StopDecision`: continue or stop with terminal reason.

### Engine

- `GepaEngine`: deterministic state machine.
- `GepaState`: candidates, evaluations, frontier, usage, and terminal status.
- `GepaStep`: next action requested by the engine.
- `GepaOutcome`: final best candidate and artifact refs.

### Runtime Pipeline

The GEPA crate keeps rollout transport separate from optimizer pipeline mode.
`gepa.rollout_submission_mode` chooses the container wire protocol
(`sync` blocking POST or `async` submit/poll/fetch). `gepa.pipeline.mode`
chooses orchestration:

- `sync_serial` is the default correctness baseline. It runs the durable
  one-transition tick state machine already used by `/worker/tick`.
- `async_pipelined` is the FlashEvolve-style runtime mode. Stages declare
  work, while the runtime records queue, worker, scheduling, and pool-version
  state in the GEPA cursor. Phase 1 supports the `full` staleness policy and
  three lanes: `propose`, `rollout`, and `evaluate`.

`async_pipelined` runs through the same durable `/worker/tick` driver as the
baseline: each tick consumes finished work first or schedules one new runtime
transition. `run-next` remains a loop over those ticks. Rollout transport stays
separate and is still selected by `gepa.rollout_submission_mode`.

## State Machine

```text
Init
  -> DiscoverProgram
  -> RegisterSeedCandidate
  -> EvaluateSeed
  -> ReflectAndPropose
  -> EvaluateMinibatch
  -> AcceptOrReject
  -> EvaluateFullTrain
  -> UpdateFrontier
  -> StopCheck
  -> HeldoutEvaluation
  -> Finalize
```

The engine should be able to pause after each step once durable state is added.
The current public examples execute in-process, but the state vocabulary should
not prevent future pause/resume.

## Service Mode

`synth_gepa::service` owns the first standing optimizer service for the public
package. It is intentionally small and blocking:

- `GET /health`
- `GET /status`
- `POST /runs`
- `POST /worker/run-next`
- `POST /worker/tick`
- `POST /worker/recover`

`POST /runs` submits a durable workspace run request and, by default, starts a
background worker that claims the request and executes GEPA with
`execute_gepa_from_toml`. The service database stores the queue lifecycle; the
GEPA run itself writes its normal per-run artifacts and `workspace.sqlite`.
On success, the service request imports the per-run result summary fields so a
standing service can show the manifest path, workspace DB path, best candidate,
cost, usage, and result JSON without opening every run workspace.
Workers requeue expired leases before claiming, heartbeat the run-request lease
while GEPA executes, and only terminal-write a request while the same lease is
still current. `/worker/recover` exposes service restart/recovery scans for
expired run requests and expired per-run optimizer jobs; `/worker/tick` is the
idempotent service-worker entrypoint used by the first local ticker.
Candidate snapshots and rollout/sensor frames are persisted during execution,
not only at finalize time, so service status can inspect live GEPA progress from
the per-run `workspace.sqlite`.
State transitions are persisted live too. Service-run executions receive a
cancellation source tied to the run-request row; when that row becomes
`cancelled`, GEPA stops at the next cancellation check and the worker records a
cancelled terminal state with a synthetic `cancel_requested` transition in the
per-run workspace.

## Acceptance Inputs

Acceptance reads only typed evidence:

- candidate id
- parent candidate id
- minibatch objective values
- baseline objective values
- full-train objective values when scheduled
- configured acceptance rule
- budget and stopper state

It must not inspect raw model text outside the proposer response schema.

## Non-Goals

- No direct OpenAI, Codex, or app-server HTTP implementation.
- No container-specific rules in the algorithm crate.
- No TOML parsing.
- No result manifest writing.
- No private GEPA runtime copy.
