# synth_optimizer_platform

Shared platform crate for public optimizers.

This crate owns durable IO boundaries and stable user-facing contracts. It does
not know GEPA search policy. It gives algorithms a small set of typed services
for config, containers, cache, events, manifests, usage, and artifacts.

## Responsibilities

- Load the frozen TOML shape:
  - `[run]`
  - `[container]`
  - `[dataset]`
  - `[candidate]`
  - `[seed_candidate]`
  - `[policy]`
  - `[proposer]`
  - `[gepa]`
  - `[cache]`
- Validate config without silently defaulting risky fields.
- Discover and verify the `synth_optimizers.gepa.v1` container contract.
- Fetch prompt-program metadata and dataset rows from `synth-containers`.
- Submit rollouts to `/rollout` and preserve existing container routes.
- Manage SQLite request/response cache entries.
- Manage SQLite workspace state, including durable service-mode run requests,
  leases, run records, candidates, candidate graph rows, rollout jobs, sensor
  frames, cache profiles, cache accesses, event stream rows, usage ledger rows,
  stopper/budget state rows, checkpoint rows, artifacts, and manifests.
- Write `events.jsonl`, `events.normalized.jsonl`, candidate registry, frontier,
  cache profile, run registry, and result manifests.
- Normalize events by stripping timestamps, generated ids, local artifact roots,
  volatile session ids, and host-specific ports.
- Track usage and cost at the external-call boundary.

## Core Types

### Config

- `SynthOptimizerConfig`: whole TOML document.
- `RunConfig`: run id, output root, deterministic seed.
- `ContainerConfig`: container URL or managed local process command.
- `DatasetConfig`: split names, seed pools, filters.
- `CandidateConfig`: target module declarations and candidate shape.
- `PolicyConfig`: student policy model route.
- `ProposerConfig`: Codex local-process proposer settings.
- `CacheConfig`: mode, path, namespace, size profile.

### Container

- `ContainerClient`: blocking HTTP client for public container routes.
- `ContainerContract`: discovered metadata under
  `metadata.optimizer_contracts.gepa`.
- `ManagedContainerProcess`: optional local process wrapper for cookbook
  containers. On Unix it launches containers in a process group and uses a
  bounded terminate/kill cleanup path so cancellation cannot block forever
  waiting on a child process.

### Prompt Program

- `PromptProgram`: program id, modules, target modules, seed candidate.
- `PromptModule`: module id, role, content, mutability, template variables.
- `TargetModule`: GEPA target declaration.
- `PromptCandidatePayload`: candidate field map.
- `CandidateOverlay`: rollout overlay applied to a candidate.

### Cache And Events

- `RequestCache`: SQLite request/response cache.
- `CacheMode`: `off`, `readwrite`, `readonly`.
- `CacheProfile`: stats and external-call counts for the run manifest.
- `CacheProfileRecord`: durable per-run cache summary row for `workspace.sqlite`.
- `CacheAccessRecord`: ordered cache hit/miss/write row keyed by namespace,
  boundary, request hash, and response hash.
- `EventWriter`: append-only JSONL writer.
- `EventStreamRecord`: typed SQLite mirror of each emitted JSONL event.
- `NormalizedEventFeed`: deterministic event feed for parity comparison.
- `UsageLedgerRecord`: typed token/cost/call-count row for external call
  boundaries.
- `StopperStateRecord`: typed budget/stopper snapshot for rollout count,
  remaining rollout budget, cost, candidate, stage, and terminal stop status.
- `CheckpointRecord`: typed resumable-boundary snapshot for run state,
  generation, best candidate, candidates, frontier, usage, cost, and rollout
  progress.

### Candidate Graph

- `CandidatePayloadRecord`: current full lever bundle/payload for a candidate.
- `CandidateDeltaRecord`: parent-to-child mutation, target levers, changed
  fields, before/after payloads, and rationale.
- `AcceptanceDecisionRecord`: seed, accepted, rejected, deferred, pending, or
  observed decision row with stage and reward context.
- `FrontierCellRecord`: current frontier membership under a split/objective.
- `PlanLinkRecord`: typed lineage edge across candidates, payloads, deltas,
  sensor frames, decisions, and frontier cells.

### Artifacts

- `ArtifactPaths`: run workspace path resolver.
- `ArtifactRef`: path plus kind, sha256, and public retention class.
- `GepaRunResult`: final result object exported to Python.
- `RunRegistry`: append-only JSONL index of run/workspace start and finish
  records under the configured output root.

### Workspace

- `WorkspaceStore`: SQLite source of truth for run status, service run
  requests, state history, candidates, optimizer jobs, rollout jobs, sensor
  frames, candidate payloads, candidate deltas, acceptance decisions, frontier
  cells, plan links, cache profiles, cache accesses, rollout execution records,
  event stream events, usage ledger rows, stopper states, rollout events,
  checkpoints, objectives, scores, trace annotations, evidence frames,
  verifier jobs, subagent invocations, operations, resource leases, projection
  freshness, invariant reports, artifact refs, and manifests.
- `WorkspaceStatus`: service-style read model for standing optimizer status
  pages and future HTTP `GET /runs` endpoints.
- `WorkspaceRunRequestStatus`: durable submit/claim/start/complete/fail/cancel
  state for queued optimization requests, plus completed-run summary fields
  imported from the per-run GEPA result.
- `WorkspaceStore::heartbeat_run_request`: extends a worker-owned run-request
  lease and refreshes its resource leases while service execution is active.
- Lease-guarded run-request start/result/terminal helpers keep stale workers
  from overwriting a request after recovery or reassignment.
- `WorkspaceStore::recover_expired_run_requests`: requeues expired leased or
  running requests so local service restart can recover work.
- `WorkspaceStore::record_state_transition`: persists optimizer lifecycle rows
  as they happen.
- `WorkspaceStore::record_run_cancelled`: marks a per-run workspace as
  cancelled and appends a synthetic `cancel_requested` transition when a
  service request cancellation stops execution.

Run-request claiming skips queued requests whose container URL or cache
namespace conflicts with a non-expired leased/running request. That gives the
local service a conservative first concurrency policy while still allowing
different containers/cache namespaces to run independently.

Trace/evidence persistence is additive. Raw sensor frames stay intact; the
workspace also derives deterministic local `TraceAnnotation`, `EvidenceFrame`,
`VerifierJob`, and `SubagentInvocation` rows so service status can show
annotation, verification, and summarization work explicitly before a future
Codex app-server backend owns those jobs.

Score persistence is separate from sensor persistence. Each `SensorFrame`
produces typed `ObjectiveSpec` and `ScoreRecord` rows so objective aggregation,
frontier selection, dashboards, and cached replay can inspect scores without
parsing candidate JSON sidecars.

Candidate graph persistence is separate from the summary candidate row.
`candidate_payloads` stores the current full payload/lever bundle, `candidate_deltas`
stores parent-to-child mutations, `acceptance_decisions` records GEPA's decision
stage, `frontier_cells` records current train frontier membership, and
`plan_links` ties those rows back to candidates and sensor frames.

Cache ledger persistence is separate from the cache payload store.
`cache_profiles` stores the compact per-run hit/miss/write summary used by the
manifest, while `cache_accesses` stores each ordered cache hit, miss, and write
by namespace and external boundary. This lets readonly replay, cached reruns,
and parity debugging prove which boundaries were actually served from cache
without persisting raw request/response payloads in the run workspace.

Event stream persistence mirrors `events.jsonl` into `event_stream_events`.
Each emitted event gets a sequence number, event type, message, timestamp,
fields JSON, and full event JSON. JSONL remains the release/parity artifact;
SQLite rows make service status, recovery inspection, and future streaming APIs
queryable without reparsing files.

Usage ledger persistence stores accounting at external-call boundaries.
`usage_ledger` rows record boundary, source, candidate/stage, model/provider,
call count, prompt/completion/total tokens, cost, raw usage JSON, and metadata.
Rollout rows are derived from sensor frames; proposer rows are derived from
proposer generation completions.

Stopper persistence stores explicit budget and terminal snapshots.
`stopper_states` rows record sequence number, status, reason, generation,
candidate, evaluation stage, rollout count, max/remaining rollout budget, cost,
max cost, cost-budget enablement, and whether the optimizer observed a budget
stop.

Checkpoint persistence stores resumable run-boundary snapshots.
`checkpoints` rows record checkpoint kind, status, run state, generation,
candidate, best candidate, candidate/frontier counts, rollout count, usage,
cost, and a snapshot JSON payload with candidate registry and frontier state.

Rollout execution persistence is separate from rollout queue persistence.
`rollout_jobs` records scheduling/completion of work, `rollouts` records the
execution observed for one sensor frame, and `rollout_events` records derived
timeline facts such as rollout completion and trace digest observation.

Service operation/resource persistence makes the standing optimizer queue
inspectable. Submit, claim, start, terminal, and recovery actions write
`OperationRecord` rows. Claims also write active `ResourceLeaseRecord` rows for
the container URL and cache namespace; terminal requests release those leases,
and recovery marks expired leases explicitly.

Projection and invariant health are persisted too. `ProjectionFreshnessRecord`
rows compare source tables with derived read-model tables, while
`InvariantReport` rows capture count mismatches across candidate graph rows,
cache profile/access rows, event-stream/state-transition rows,
usage-ledger/external-call rows, stopper/terminal rows, checkpoint/terminal
rows, rollout/sensor rows, missing manifests, unreleased leases, and other
repairable workspace problems with typed violations and repair hints.
`workspace status` refreshes these rows before
returning its read model, so stale old workspaces are visible instead of
silently passing.

Runtime overrides are intentionally narrow: `SYNTH_OPTIMIZERS_RUN_ID`,
`SYNTH_OPTIMIZERS_OUTPUT_DIR`, `SYNTH_OPTIMIZERS_CACHE_MODE`,
`SYNTH_OPTIMIZERS_CACHE_PATH`, `SYNTH_OPTIMIZERS_CACHE_NAMESPACE`, and
`SYNTH_OPTIMIZERS_PROPOSER_BACKEND`. They let each public cookbook produce
fresh, cached, and readonly replay artifacts from one frozen TOML config.

## Error Vocabulary

Platform errors must be typed enough for users to act:

- `Config`: invalid or unsupported user config.
- `Container`: missing route, bad status, invalid JSON, or missing GEPA contract.
- `CacheMiss`: readonly replay cannot continue because a boundary was not cached.
- `EventCompare`: normalized feeds differ.
- `Proposer`: proposer process or schema failure.
- `Io`, `Json`, `Toml`, `Sqlite`, `Http`: source-level failures with paths or
  request context.

## Cache Boundary

Cache keys include:

- namespace
- boundary kind: `container.rollout`, `container.dataset_rows`, `proposer.codex`
- normalized request payload
- schema version

Cache values store the full response payload needed to replay without launching
the original external dependency.

Readonly miss behavior:

```text
if mode == readonly and key missing:
    raise CacheMiss(namespace, key)
```

No fallback live call is allowed in readonly mode.
