# synth-optimizers

Public optimizer tooling for Synth cookbook users.

The first public milestone is a GEPA vertical slice with four first-class
examples: Banking77, TBLite, code-review, and Crafter. It is intentionally
smaller than the private optimizer stack:

- public boundary: `synth-containers`
- optimizer core: Rust crates under `rust/`
- Python surface: thin PyO3 wrapper under `src/synth_optimizers/`
- proposer backend: Codex app-server local process first
- release status: prerelease; the public examples need accepted fresh, cached,
  and readonly replay runs before `0.1.0` can pass the release checklist/tag
  step

## Directory Map

```text
packages/synth-optimizers/
  rust/
    crates/
      synth_optimizer_platform/   # shared platform primitives and IO boundaries
      synth_gepa/                 # GEPA algorithm state machine
      synth_optimizers_py/        # PyO3 module and CLI bridge
  src/synth_optimizers/           # importable Python wrapper only
```

Cookbook assets live outside the package at:

```text
cookbooks/optimizers/gepa/
  configs/
  banking77_container/
  tblite_container/
  code_review_container/
  crafter_container/
```

## Public API

Python:

```python
from synth_optimizers import GepaRun

result = GepaRun.from_toml("gepa.toml").execute()
print(result.best_candidate)
```

CLI:

```bash
synth-optimizers gepa run --config gepa.toml
synth-optimizers gepa service --db service.sqlite --bind 127.0.0.1:8879
synth-optimizers gepa run-next --db service.sqlite
synth-optimizers events replay --events runs/latest/events.jsonl
synth-optimizers events compare --left runs/a/events.normalized.jsonl --right runs/b/events.normalized.jsonl
synth-optimizers workspace status --db runs/latest/workspace.sqlite
synth-optimizers workspace submit --db service.sqlite --config gepa.toml
synth-optimizers workspace claim --db service.sqlite --lease-id worker-1
synth-optimizers workspace start --db service.sqlite --request-id runreq_...
synth-optimizers workspace complete --db service.sqlite --request-id runreq_...
synth-optimizers workspace cancel --db service.sqlite --request-id runreq_...
synth-optimizers workspace recover --db service.sqlite
```

Result fields:

- `best_candidate`
- `manifest_path`
- `event_feed_path`
- `normalized_event_feed_path`
- `cache_profile_path`
- `candidate_registry_path`
- `frontier_path`
- `run_registry_path`
- `workspace_db_path`
- `artifact_refs`
- `cost_usd`
- `usage`
- `state_history`

Errors are raised as `SynthOptimizerError` subclasses with stable
`error_code` class attributes. Public callers can catch specific cases such as
`CacheMissError`, `ContainerContractError`, `ConfigError`, and `ProposerError`.

Workspace inspection:

```python
from synth_optimizers import workspace_status

status = workspace_status("runs/latest/workspace.sqlite")
print(status["runs"][0]["rollout_job_status_counts"])
print(status["runs"][0]["rollout_event_type_counts"])
print(status["runs"][0]["score_objective_counts"])
print(status["runs"][0]["evidence_frame_kind_counts"])
print(status["runs"][0]["verifier_job_status_counts"])
print(status["runs"][0]["cache_access_action_counts"])
print(status["runs"][0]["cache_access_boundary_counts"])
print(status["runs"][0]["event_stream_event_type_counts"])
print(status["runs"][0]["usage_ledger_boundary_counts"])
print(status["runs"][0]["projection_freshness"])
print(status["runs"][0]["invariant_report"])
```

Service-mode queue substrate:

```python
from synth_optimizers import (
    workspace_claim_next_run_request,
    workspace_complete_run_request,
    workspace_start_run_request,
    workspace_status,
    workspace_submit_run_request,
)

request = workspace_submit_run_request("service.sqlite", "gepa.toml")
claimed = workspace_claim_next_run_request("service.sqlite", "worker-lease-1")
workspace_start_run_request("service.sqlite", claimed["request_id"])
workspace_complete_run_request("service.sqlite", claimed["request_id"])
status = workspace_status("service.sqlite")
print(status["run_request_status_counts"])
print(status["operation_status_counts"])
print(status["resource_lease_status_counts"])
print(status["projection_status_counts"])
print(status["invariant_status_counts"])
```

Standing GEPA service:

```bash
synth-optimizers gepa service \
  --db service.sqlite \
  --bind 127.0.0.1:8879 \
  --worker-id local-gepa

curl -s -X POST http://127.0.0.1:8879/runs \
  -H 'Content-Type: application/json' \
  -d '{"config_path":"gepa.toml","priority":5,"auto_start":true}'

curl -s http://127.0.0.1:8879/status
```

Endpoints:

- `GET /health`: process liveness.
- `GET /status`: workspace status and queued/completed request counts.
- `POST /runs`: submit a TOML config as a durable run request. By default this
  auto-starts a background worker in the service process.
- `POST /worker/run-next`: claim and execute the next queued request
  synchronously.
- `POST /worker/recover`: requeue expired leased/running requests.

The service queue records config path, run id, standing container URL, cache
namespace, priority, lease ownership, lifecycle timestamps, and terminal
error/cancel payloads. Completed requests also import the per-run summary:
`run_workspace_db_path`, `result_manifest_path`, `best_candidate_id`,
`cost_usd`, `usage`, and the full GEPA result JSON. Per-run artifacts still
live under the configured run directory, including that run's own
`workspace.sqlite`.

Claiming enforces a local concurrency policy: a queued request will not be
claimed while another non-expired request with the same container URL or cache
namespace is leased/running. Expired leases are requeued by
`workspace recover`, service startup, `gepa run-next`, or `POST /worker/recover`.

During GEPA execution, candidate records and rollout/sensor rows are persisted
incrementally into the per-run `workspace.sqlite`, so status readers can see
live candidate and rollout progress before the final manifest is written.
Candidate payloads, parent-to-child deltas, acceptance decisions, frontier
cells, and plan links are persisted as separate rows; users should not need to
parse `candidate_registry.json` to inspect lineage or why a candidate is in the
frontier.
Cache profiles and cache accesses are persisted as separate rows too. The
workspace records every cache hit, miss, and write by namespace and boundary
(`container.program`, `container.dataset_rows`, `container.rollout`, and
`proposer.codex` today), while the run-level cache profile remains the compact
manifest artifact.
The JSONL event feed is mirrored into typed `event_stream_events` rows with
sequence numbers, event types, messages, timestamps, fields, and the full event
payload. Status views can therefore answer live/debug questions from SQLite
without parsing `events.jsonl`, while normalized JSONL remains the parity
comparison artifact.
Usage accounting is mirrored into typed `usage_ledger` rows at the external
call boundary. Rollout usage rows are linked to sensor frames, and proposer
usage rows are linked to proposer generations, so token/cost totals can be
audited without scraping the final manifest.
Stopper and budget state are mirrored into typed `stopper_states` rows. Each
snapshot records rollout count, remaining rollout budget, cost, max cost,
candidate, stage, generation, and whether a budget stop or terminal completion
was observed.
Checkpoint snapshots are mirrored into typed `checkpoints` rows at resumable
boundaries. Each checkpoint records the run state, generation, candidate,
best candidate, candidate/frontier counts, rollout count, usage, cost, and a
snapshot payload containing candidate registry and frontier state.
Optimizer state transitions are persisted incrementally as well. Service-run
GEPA executions also poll the service run-request row; cancelling that request
causes the active GEPA loop to stop and marks the service request and per-run
workspace as cancelled with a synthetic `cancel_requested` state transition.
Local container processes launched by the optimizer are cleaned up with a
bounded process-group shutdown path so cancellation does not leave the worker
blocked indefinitely.

`workspace status` also refreshes projection and invariant health rows. Per-run
projection freshness records check that derived tables such as
`candidate_payloads`, `candidate_deltas`, `acceptance_decisions`,
`frontier_cells`, `plan_links`, `cache_profiles`, `cache_accesses`,
`event_stream_events`, `usage_ledger`, `stopper_states`, `checkpoints`,
`rollouts`, `scores`, `trace_annotations`, `evidence_frames`, verifier jobs,
subagent invocations, and manifests cover their source rows. Invariant reports
summarize missing or stale derived state with repair hints instead of hiding
read-model drift inside JSON sidecars.

## Frozen V1 Config Shape

The public TOML shape is sectioned by durable nouns:

```toml
[run]
[container]
[dataset]
[candidate]
[seed_candidate]
[policy]
[proposer]
[gepa]
[cache]
```

Algorithm internals should not leak into user config unless a field is stable,
safe, and directly tied to user intent. GEPA-specific scheduling, frontier, and
acceptance details remain owned by `synth_gepa`.

`[gepa]` owns hard run limits and pre-dispatch reservation estimates. Rollout
limits are hard by default. `max_cost_usd = 0.0` means no cost limit; when cost,
token, or time limits are set, proposer and rollout estimate fields must be
positive so the runtime can reject spending work before dispatch instead of
falling back to an implicit zero estimate.

`max_total_rollouts` remains the legacy single rollout cap. For runs that need
heldout reporting after train optimization, set `max_train_rollouts` and
`max_heldout_rollouts`; the optimizer stops proposing/training at the train
budget, then evaluates heldout against the separate heldout budget. When either
split budget is set, the runtime reservation cap is `max_train_rollouts +
max_heldout_rollouts` with unspecified split values falling back to
`max_total_rollouts`.

Rollout transport and optimizer pipeline mode are separate axes. Transport is
controlled by `gepa.rollout_submission_mode = "sync" | "async"`. Pipeline mode
is controlled by the nested `gepa.pipeline` table:

```toml
[gepa.pipeline]
mode = "sync_serial"        # default, correctness baseline
staleness_policy = "full"   # guarded/reflective are reserved for later phases
max_in_flight_candidates = 8

[gepa.pipeline.workers]
propose = 1
rollout = 8
evaluate = 1
```

`async_pipelined` is the durable queue-worker mode for overlapping proposer,
rollout, and evaluate lanes with pool-version staleness handling.

## Boundary Rules

- `synth-containers` is the only runtime task boundary.
- No arbitrary user Python imports from the optimizer core.
- No private backend services.
- No Go-EX or MIPROv2 surfaces in this slice.
- No Docker or local-process proposer fallback mode.
- `codex_app_server` launches a real local `codex app-server`; unsupported
  proposer backends fail during config validation.
- Cache modes are exactly `off`, `readwrite`, and `readonly`.

For cookbook acceptance runs, the loader accepts narrow runtime overrides such
as `SYNTH_OPTIMIZERS_RUN_ID`, `SYNTH_OPTIMIZERS_CACHE_MODE`, and
`SYNTH_OPTIMIZERS_PROPOSER_BACKEND`. These do not change the frozen TOML
section shape; they only let the same config produce separate fresh, cached, and
readonly replay artifact directories.

## Implementation Order

1. Define contracts and directory responsibilities in markdown.
2. Implement the platform crate around config, container HTTP, cache, events,
   manifests, and artifacts.
3. Implement the GEPA crate around candidate state, scheduling, proposer
   requests, selection, acceptance, and stopping.
4. Expose the PyO3 module and thin Python wrapper.
5. Add the Banking77, TBLite, code-review, and Crafter cookbooks and validate
   fresh, cached, and readonly replay behavior.
