# Public GEPA Validation Audit

Date: 2026-05-20

Scope: public `synth-optimizers` GEPA v1 vertical slice in
`synth-cookbooks-public`, including Banking77 plus public-safe TBLite,
code-review, and Crafter fixtures. This document records validation evidence;
it is not a release acceptance packet unless a row is explicitly labeled
`FULL E2E ACCEPTANCE`. This is not a PyPI publish, `0.1.0` release tag, or prod
promotion record.

## Acceptance Boundary

Acceptance tests are full feature end-to-end runs. A run is acceptance evidence
only when it exercises the actual runtime rails being claimed: policy execution,
verifier/subagent execution when applicable, objective scoring, cache/replay,
artifacts, event feeds, workspace state, and failure behavior.

Fixture runs, deterministic containers, contract checks, cache probes, service
lifecycle probes, projection checks, and schema checks are pre-acceptance
validation. They are useful, but they must not be cited as acceptance for
TBLite, code-review, Crafter, verifiers/subagents, multi-objective GEPA, or
long-running service behavior.

## Status

Full feature end-to-end acceptance is not complete for TBLite, code-review, or
Crafter. The current evidence proves a substantial public Rust/PyO3 validation
surface: Banking77 scalar GEPA vertical-slice behavior, deterministic public
fixtures for TBLite/code-review/Crafter, fresh/cached/readonly replay, normalized
event comparisons, SQLite workspace state, service queue behavior, cache
profiles, event-stream rows, usage-ledger rows, stopper/checkpoint rows,
projection/invariant health, and standing HTTP service mechanics.

Banking77 is the closest current vertical slice: it exercises the public
container contract, scalar reward, cache, event stream, manifest, chart, and
Codex proposer path. It does not prove verifier/subagent, annotator, terminator,
or multi-objective platform acceptance.

TBLite, code-review, and Crafter are contract fixtures only. They do not prove
their Python-style full feature flows because they do not run the real task
policy actors, verifiers/subagents, multi-objective scoring, private Docker/task
environments, or production-equivalent artifacts required by those examples.

Current acceptance matrix:

| Surface | Current status | Meaning |
| --- | --- | --- |
| Banking77 scalar GEPA | VALIDATED | Public scalar vertical slice runs, but this is not acceptance for verifier/subagent, annotator, terminator, or multi-objective behavior. |
| TBLite | NOT ACCEPTED | Current public run is a deterministic contract fixture, not a real TBLite policy/verifier/task run. |
| Code-review | NOT ACCEPTED | Current public run is a deterministic contract fixture, not a Codex review workspace with structured `review.json` and gold-label scoring. |
| Crafter | NOT ACCEPTED | Current public run is a deterministic contract fixture, not a real environment rollout with achievement scoring, wasted-effort verifier, and multi-objective GEPA. |

There are currently no rows in this document labeled `FULL E2E ACCEPTANCE`.

Known caveat: the worktree also contains broad pre-existing cookbook and MIPRO
deletions outside this GEPA slice. This audit covers only the public GEPA
package, `synth-containers` contract additions, and GEPA cookbooks under
`cookbooks/optimizers/gepa`.

## Validation Audit

| Requirement | Status | Evidence |
| --- | --- | --- |
| Major subdirectories were sketched in markdown before code was filled in. | PROVED | `packages/synth-optimizers/README.md`, `packages/synth-optimizers/rust/README.md`, `packages/synth-optimizers/rust/crates/*/README.md`, `packages/synth-optimizers/src/synth_optimizers/README.md`, `cookbooks/optimizers/gepa/README.md`, `cookbooks/optimizers/gepa/banking77_container/README.md`, `cookbooks/optimizers/gepa/tblite_container/README.md`, `cookbooks/optimizers/gepa/code_review_container/README.md`, `cookbooks/optimizers/gepa/crafter_container/README.md` |
| `synth-containers` advertises GEPA contract `synth_optimizers.gepa.v1` while preserving existing routes. | PROVED | `packages/synth-containers/src/synth_containers/prompt_programs.py`, `packages/synth-containers/src/synth_containers/http_adapter.py`, `packages/synth-containers/docs/gepa_optimizer_contract.md`, `packages/synth-containers/openapi/container-contract-v1.yaml` |
| Public prompt-program types: `PromptProgram`, `PromptModule`, mutable field ids, target modules, seed candidates, rollout overlays. | PROVED | `packages/synth-containers/src/synth_containers/prompt_programs.py`; Rust mirror in `packages/synth-optimizers/rust/crates/synth_optimizer_platform/src/prompt_program.rs` |
| Rust workspace with `synth_optimizer_platform`, `synth_gepa`, and `synth_optimizers_py`. | PROVED | `packages/synth-optimizers/Cargo.toml`, `packages/synth-optimizers/rust/crates/*/README.md`, crate source under `packages/synth-optimizers/rust/crates/` |
| Python wrapper stays thin with `GepaRun.from_toml("gepa.toml").execute()`. | PROVED | `packages/synth-optimizers/src/synth_optimizers/__init__.py`, `packages/synth-optimizers/src/synth_optimizers/cli.py`, `packages/synth-optimizers/rust/crates/synth_optimizers_py/src/lib.rs` |
| Result fields include best candidate, artifact paths, cache profile, cost, usage, registry/frontier evidence, workspace DB, and optimizer state history. | PROVED | `packages/synth-optimizers/rust/crates/synth_optimizer_platform/src/artifacts.rs`; live `result_manifest.json` includes `best_candidate`, `manifest_path`, `event_feed_path`, `normalized_event_feed_path`, `cache_profile_path`, `candidate_registry_path`, `frontier_path`, `run_registry_path`, `workspace_db_path`, `artifact_refs`, `cost_usd`, `usage`, and `state_history` |
| CLI exposes `synth-optimizers gepa run --config`, `events replay`, and `events compare`. | PROVED | `packages/synth-optimizers/src/synth_optimizers/cli.py` |
| Frozen TOML v1 shape uses `[run]`, `[container]`, `[dataset]`, `[candidate]`, `[seed_candidate]`, `[policy]`, `[proposer]`, `[gepa]`, `[cache]`. | PROVED | `packages/synth-optimizers/rust/crates/synth_optimizer_platform/src/config.rs`, `cookbooks/optimizers/gepa/banking77_container/gepa.toml` |
| Platform crate owns config, errors, HTTP container client, run registry, SQLite workspace store, SQLite cache, event JSONL, normalized compare, manifests, cost/usage, artifact refs, typed failures, jobs, levers, sensors, candidate graph rows, cache profile/access rows, typed event-stream rows, usage-ledger rows, stopper/budget state rows, checkpoint rows, rollout records/events, scores/objectives, evidence/trace annotations, operations/resource leases, projection freshness, invariant reports, and optimizer state transitions. | PROVED | `synth_optimizer_platform/src/{config,error,http,registry,workspace,cache,events,artifacts,failures,jobs,levers,sensors,candidates,rollouts,scores,evidence,operations,resources,projections,invariants,state_machine,usage,stopper,checkpoints}.rs`; live `workspace.sqlite`, `run_registry.jsonl`, `cache_profile.json`, event feeds, manifest, `state_history`, `lever_bundle`, `sensor_frames`, `candidate_payloads`, `candidate_deltas`, `acceptance_decisions`, `frontier_cells`, `plan_links`, `cache_profiles`, `cache_accesses`, `event_stream_events`, `usage_ledger`, `stopper_states`, `checkpoints`, `rollouts`, `rollout_events`, `objectives`, `scores`, `trace_annotations`, `evidence_frames`, `verifier_jobs`, `subagent_invocations`, `operations`, `resource_leases`, `projection_freshness`, and `invariant_reports` |
| GEPA crate owns candidate registry, prompt candidate validation, proposer schemas, Codex proposer orchestration, minibatch/full-train/heldout scheduling, frontier, acceptance, durable stopper snapshots, checkpoint snapshots, best selection, lever bundle materialization, sensor frame capture, and state-machine event emission. | PROVED | `packages/synth-optimizers/rust/crates/synth_gepa/src/lib.rs`, `packages/synth-optimizers/rust/crates/synth_gepa/src/codex_app_server.rs`; latest accepted candidates are Banking77 `gepa_706a0fa37ef3`, TBLite `gepa_53d671c830f8`, code-review `gepa_51538598b796`, and Crafter `gepa_837aedcf7438` |
| Codex app-server proposer v1 supports `execution_mode = "local_process"` plus sandbox, approval, reasoning, auth-copy, env, timeout, model config. | PROVED | `cookbooks/optimizers/gepa/banking77_container/gepa.toml`, `synth_optimizer_platform/src/config.rs`, `synth_gepa/src/codex_app_server.rs`; live Banking77 audit under `/tmp/synth-gepa-live-codex-20260520/banking77_live_codex_audit` recorded two `proposer.completed` events with `backend=codex_app_server`, proposer workspaces for generations 0 and 1, completed state, and passing workspace health |
| Cache modes are `off`, `readwrite`, `readonly`; readonly miss is typed; normalized feeds strip volatile run fields. | PROVED | `synth_optimizer_platform/src/cache.rs`, `synth_optimizer_platform/src/events.rs`, `synth_optimizers_py/src/lib.rs`; uncached readonly probe raised `CacheMissError synth_optimizer_cache_miss` |
| Banking77 cookbook includes one public container and one public TOML config. | PROVED | `cookbooks/optimizers/gepa/banking77_container/synth_service_app.py`, `cookbooks/optimizers/gepa/banking77_container/gepa.toml` |
| TBLite public cookbook fixture preserves the private TBLite `starting_prompt` lever without private Harbor/Docker services. | PRE-ACCEPTANCE | `cookbooks/optimizers/gepa/tblite_container/synth_service_app.py`, `cookbooks/optimizers/gepa/tblite_container/gepa.toml`; current fresh/cached/readonly artifacts under `/tmp/synth-gepa-workspace-20260520/tblite`. This is a contract fixture, not full TBLite E2E acceptance. |
| Code-review public cookbook fixture preserves the private reviewer-guidance levers without private Codex review workspaces. | PRE-ACCEPTANCE | `cookbooks/optimizers/gepa/code_review_container/synth_service_app.py`, `cookbooks/optimizers/gepa/code_review_container/gepa.toml`; current fresh/cached/readonly artifacts under `/tmp/synth-gepa-workspace-20260520/code_review`. This is a contract fixture, not full code-review E2E acceptance. |
| Crafter public cookbook fixture preserves a ReAct/action-policy prompt shape and prompt-call trace schema without private Crafter/Go-EX services. | PRE-ACCEPTANCE | `cookbooks/optimizers/gepa/crafter_container/synth_service_app.py`, `cookbooks/optimizers/gepa/crafter_container/gepa.toml`; current fresh/cached/readonly artifacts under `/tmp/synth-gepa-workspace-20260520/crafter`. This is a contract fixture, not full Crafter E2E acceptance. |
| Fresh readwrite run writes required artifacts. | PROVED | Current Banking77, TBLite, code-review, and Crafter fresh directories under `/tmp/synth-gepa-status-20260520` each contain `result_manifest.json`, `events.jsonl`, `events.normalized.jsonl`, `cache_profile.json`, `best_candidate.json`, `candidate_registry.json`, `frontier.json`, and `workspace.sqlite` |
| Immediate cached rerun makes no new policy/proposer/rollout external cache writes. | PROVED | Final cached cache profiles: Banking77 `45 hits, 0 misses, 0 writes`; TBLite `15 hits, 0 misses, 0 writes`; code-review `18 hits, 0 misses, 0 writes`; Crafter `18 hits, 0 misses, 0 writes` |
| Readonly replay succeeds when fully cached. | PROVED | Final readonly cache profiles: Banking77 `45 hits, 0 misses, 0 writes`; TBLite `15 hits, 0 misses, 0 writes`; code-review `18 hits, 0 misses, 0 writes`; Crafter `18 hits, 0 misses, 0 writes` |
| `events compare` reports normalized parity between original, cached, and readonly runs. | PROVED | Banking77, TBLite, code-review, and Crafter fresh-vs-cached and fresh-vs-readonly compare commands returned `normalized event feeds match` |
| Optimizer state machine records run lifecycle from created to terminal state. | PROVED | Current `workspace.sqlite` files under `/tmp/synth-gepa-status-20260520` have `optimizer_state_history` rows: Banking77 40, TBLite 19, code-review 19, Crafter 19; manifests end in `completed` |
| Rollout observations are captured as sensor frames. | PROVED | Current `workspace.sqlite` files under `/tmp/synth-gepa-status-20260520` have `sensor_frames` rows: Banking77 40, TBLite 11, code-review 14, Crafter 14, with rollout jobs persisted 1:1 |
| Candidate payloads, candidate deltas, acceptance decisions, frontier cells, and plan links are first-class workspace rows. | PROVED | Final current-schema rerun under `/tmp/synth-gepa-final-current-20260520` completed Banking77, TBLite, code-review, and Crafter fresh/cached/readonly runs. Fresh workspaces contain candidate graph rows: Banking77 `candidate_payloads=5`, `candidate_deltas=4`, `acceptance_decisions=5`, `frontier_cells=5`, `plan_links=63`; TBLite `2/1/2/1/18`; code-review `2/1/2/1/21`; Crafter `2/1/2/1/21`. Each fresh run returned `projection_status_counts={fresh: 21}`, `invariant_status_counts={pass: 2}`, zero invariant violations, cached writes `0`, readonly writes `0`, and normalized event comparisons true. |
| Cache profiles and cache accesses are first-class workspace rows. | PROVED | Focused Banking77 cache-ledger smoke under `/tmp/synth-gepa-cache-ledger-20260520/banking77_cache_ledger_smoke/workspace.sqlite` produced `cache_profiles=1`, `cache_accesses=78`, action counts `{hit: 12, miss: 33, write: 33}`, and boundary counts `{container.dataset_rows: 4, container.program: 2, container.rollout: 68, proposer.codex: 4}`. Four-example rerun under `/tmp/synth-gepa-cache-ledger-final-20260520` produced fresh workspace cache row counts: Banking77 `1/78`, TBLite `1/27`, code-review `1/32`, Crafter `1/32`; cached and readonly writes remained `0`, normalized comparisons returned true, and cache profile/access projections were fresh. |
| Event JSONL is mirrored into typed workspace event-stream rows. | PROVED | Focused Banking77 event-stream smoke under `/tmp/synth-gepa-event-stream-20260520/banking77_event_stream_smoke/workspace.sqlite` produced `event_stream_events=67`, `optimizer.state.transitioned=40`, and `state_transitions=40`; `projection_status_counts={fresh: 18}` and invariants passed. Four-example rerun under `/tmp/synth-gepa-event-stream-final-20260520` produced fresh event-stream row counts: Banking77 `67`, TBLite `33`, code-review `33`, Crafter `33`; every fresh workspace had `event_stream_events_cover_state_transitions` fresh and zero invariant violations. |
| Usage ledger rows account for rollout and proposer external-call usage. | PROVED | Focused Banking77 usage-ledger smoke under `/tmp/synth-gepa-usage-ledger-20260520/banking77_usage_ledger_smoke/workspace.sqlite` produced `usage_ledger=42`, with `container.rollout=40`, `proposer.codex=2`, and `usage_ledger_from_external_calls` fresh `42/42`. Four-example rerun under `/tmp/synth-gepa-final-current-20260520` produced fresh usage rows: Banking77 `42`, TBLite `12`, code-review `15`, Crafter `15`; each run had usage boundary counts matching rollout sensors plus proposer completions and zero invariant violations. |
| Stopper and budget state snapshots are first-class workspace rows. | PROVED | Focused Banking77 stopper smoke under `/tmp/synth-gepa-stopper-20260520/banking77_stopper_smoke/workspace.sqlite` produced 13 `stopper_states` with status counts `{within_budget: 12, completed: 1}`, a fresh `stopper_state_from_terminal_run` projection `1/13`, `projection_status_counts={fresh: 20}`, `invariant_status_counts={pass: 2}`, and zero invariant violations. Four-example rerun under `/tmp/synth-gepa-final-current-20260520` produced fresh stopper rows: Banking77 `13`, TBLite `6`, code-review `6`, Crafter `6`; every fresh run had terminal stopper projection fresh, cached writes `0`, readonly writes `0`, and normalized event comparisons true. |
| Checkpoint snapshots are first-class workspace rows. | PROVED | Focused Banking77 checkpoint smoke under `/tmp/synth-gepa-checkpoint-20260520/banking77_checkpoint_smoke/workspace.sqlite` produced 6 `checkpoints` across `candidate_registry`, `evaluation_boundary`, `generation_boundary`, `pre_heldout`, and `terminal`; `checkpoint_from_terminal_run` was fresh `1/6`, `projection_status_counts={fresh: 21}`, `invariant_status_counts={pass: 2}`, and invariant violations were `0`. Four-example rerun under `/tmp/synth-gepa-final-current-20260520` produced fresh checkpoint rows: Banking77 `6`, TBLite `5`, code-review `5`, Crafter `5`; every fresh run had terminal checkpoint projection fresh, cached writes `0`, readonly writes `0`, and normalized event comparisons true. |
| Rollout execution and rollout event timelines are first-class workspace rows. | PROVED | Fresh Crafter rollout smoke under `/tmp/synth-gepa-rollout-20260520/crafter_rollout_smoke/workspace.sqlite` has 14 `rollouts` and 28 `rollout_events`; `workspace status` returned `rollout_status_counts={completed: 14}`, `rollout_event_type_counts={rollout_observed: 14, trace_digest_observed: 14}`, and stage counts matching score/rollout job stages. A SQLite probe found `sum(rollouts.event_count)=56` and sample trace events with call site `crafter.react_policy`. |
| Scores and objectives are first-class workspace rows. | PROVED | Fresh Banking77 score smoke under `/tmp/synth-gepa-score-20260520/banking77_score_smoke/workspace.sqlite` has 1 objective row (`outcome_reward`, `maximize`, `mean`, `per_split_then_overall`) and 40 score rows; `workspace status` returned `score_objective_counts={outcome_reward: 40}` and score stage counts matching rollout stages. |
| Trace annotation, evidence, verifier, and subagent rows are first-class workspace records. | PRE-ACCEPTANCE | Fresh Banking77 evidence smoke under `/tmp/synth-gepa-evidence-20260520/banking77_evidence_smoke/workspace.sqlite` has 40 `trace_annotations`, 80 `evidence_frames`, 40 `verifier_jobs`, 40 `subagent_invocations`, and 120 `optimizer_jobs`; Crafter evidence smoke under `/tmp/synth-gepa-evidence-20260520/crafter_evidence_smoke/workspace.sqlite` has 14/28/14/14/42 respectively, with `llm_request_count_sum=14`, `tool_call_count_sum=14`, and call site `crafter.react_policy`. These rows prove workspace shape only; they do not prove live SubagentService/verifier acceptance. |
| Workspace status read model exposes service-style run/job/rollout/evidence state. | PROVED | `synth-optimizers workspace status --db /tmp/synth-gepa-rollout-20260520/crafter_rollout_smoke/workspace.sqlite` returned run state `completed`, latest transition `completed`, rollout job and rollout execution stage/status counts, rollout event type counts, score stage counts, candidate status counts, optimizer job kind/status counts, trace annotation status counts, evidence kind counts, verifier/subagent status counts, artifact/manifest counts, and usage. |
| Workspace run-request queue records service-mode submit/claim/start/terminal lifecycle. | PROVED | Queue smoke under `/tmp/synth-gepa-service-queue-20260520-kf2lxqzq/service.sqlite` submitted Banking77 and TBLite configs, claimed Banking77 with a lease, marked it running then completed, cancelled TBLite, and `workspace status` returned `run_request_status_counts={cancelled: 1, completed: 1}`. |
| Service operations and resource leases are first-class workspace rows. | PROVED | Operation/resource smoke under `/var/folders/7b/96hbyfld35zflr_plpgvrdxm0000gn/T/synth-gepa-ops-20260520-h76ihhye/service.sqlite` submitted three requests, claimed two, started/completed one, cancelled one, recovered one expired lease, and returned `operation_status_counts={completed: 9}`, `resource_lease_status_counts={released: 2, expired: 2}`, `resource_lease_kind_counts={container_url: 2, cache_namespace: 2}`; SQLite grouped operations as submit=3, claim=2, start=1, complete=1, cancel=1, recover=1. |
| Projection freshness and invariant reports are first-class workspace health rows. | PROVED | Final current-schema rerun under `/tmp/synth-gepa-final-current-20260520` returned `projection_status_counts={fresh: 21}`, `invariant_status_counts={pass: 2}`, and zero invariant violations for Banking77, TBLite, code-review, and Crafter fresh workspaces. The 21 global fresh projections include per-run usage ledger from external calls, event-stream events covering state transitions, stopper state from terminal run, checkpoint from terminal run, cache accesses from cache profile, cache profile from terminal run, candidate payloads, candidate deltas, acceptance decisions, plan links, frontier cells, rollout jobs, rollouts, rollout events, scores, trace annotations, evidence frames, verifier jobs, subagent invocations, terminal manifests, plus the workspace run-request operation projection. Service queue projection smoke under `/tmp/synth-gepa-projection-20260520/service.sqlite` returned fresh run-request submit-operation projection and passing workspace invariant report. |
| Current projection/invariant schema passes all four public fresh/cached/readonly examples. | PRE-ACCEPTANCE | Deterministic current-schema rerun under `/tmp/synth-gepa-final-current-20260520` completed Banking77, TBLite, code-review, and Crafter fresh/cached/readonly runs. Each fresh run returned `projection_status_counts={fresh: 21}`, per-run projection count `20`, `invariant_status_counts={pass: 2}`, zero invariant violations, cached rerun writes `0`, readonly writes `0`, and fresh-vs-cached plus fresh-vs-readonly normalized event comparisons returned `true`. Per-example status files are under `/tmp/synth-gepa-final-current-20260520/*/status_fresh.json`; this is deterministic validation, not four-example acceptance. |
| Standing GEPA service starts a run through HTTP and executes it through the Rust GEPA path. | PROVED | Service smoke under `/tmp/synth-gepa-service-http-20260520` started `synth-optimizers gepa service --db /tmp/synth-gepa-service-http-20260520/service.sqlite --bind 127.0.0.1:8899`, `GET /health` returned `ok`, `POST /runs` submitted Crafter with `auto_start=true`, service status recorded the request as `completed`, and the per-run workspace ended `completed` with 14 rollout jobs/sensor frames. |
| Completed service requests import per-run result summary fields into the service DB. | PROVED | Service-worker summary smoke under `/var/folders/7b/96hbyfld35zflr_plpgvrdxm0000gn/T/synth-gepa-service-summary-cancel-20260520-a721x5b0/summary/service.sqlite` completed request `runreq_7a74ca08efb64502a034f7b7bb0258b2`; the request row includes `run_workspace_db_path`, `result_manifest_path`, `best_candidate_id=gepa_366035051adb`, `cost_usd=0.0`, `usage`, and full `result` JSON matching the per-run manifest. |
| Service queue enforces same-container/cache-namespace concurrency and recovers expired leases. | PROVED | Recovery/concurrency smoke under `/tmp/synth-gepa-recovery-concurrency-20260520-zawwbtc3/service.sqlite` submitted two Crafter requests, blocked the second claim while the first was leased, claimed the second after first completion, requeued an expired Banking77 lease through `workspace recover`, and requeued an expired TBLite lease through `POST /worker/recover`; final counts were `{cancelled: 2, completed: 1, queued: 1}`. |
| GEPA persists candidates and rollout/sensor rows incrementally while the service request is running. | PROVED | Live polling smoke under `/tmp/synth-gepa-incremental-20260520-agkn4_pe` observed request `runreq_28d5de4640de4c06b45097ce06a78742` in `running` state while `/tmp/synth-gepa-incremental-20260520-agkn4_pe/runs/banking77_incremental/workspace.sqlite` already had 3 candidates, 21 rollout jobs, and 21 sensor frames; final workspace ended completed with 5 candidates, 40 rollout jobs, and 40 sensor frames. |
| GEPA persists optimizer state transitions incrementally and observes service cancellation. | PROVED | Cancellation smoke under `/tmp/synth-gepa-state-cancel-20260520-v_098aar` observed request `runreq_5b9fbca019ac48efaa76edf6dda22baa` in `running` state while per-run workspace already had 4 state transitions and state `rollout_running`; `workspace cancel` caused `gepa run-next` to return `run request cancelled`, service request ended `cancelled`, and per-run workspace ended `cancelled` with 8 state transitions and 5 rollout/sensor rows. Follow-up cancellation-exit smoke under `/var/folders/7b/96hbyfld35zflr_plpgvrdxm0000gn/T/synth-gepa-cancel-exit-20260520-c86emxri` proved latest transition `to_state=cancelled`, `trigger=cancel_requested`, `details.source=service_request_cancelled`. |
| Service-run cancellation does not leave the worker blocked in local container cleanup. | PROVED | Cancellation-exit smoke under `/var/folders/7b/96hbyfld35zflr_plpgvrdxm0000gn/T/synth-gepa-cancel-exit-20260520-c86emxri` cancelled request `runreq_3fa8869b43ac4ebcacf220c89dad51be`; `synth_optimizers.cli gepa run-next` returned within the 30s communicate timeout with `run request cancelled`, and no temp container remained on the generated port. |
| Public examples avoid same-port startup races for concurrent cached/readonly probes. | PROVED | `synth_optimizer_platform/src/process.rs` holds an OS-level per-container-URL run lock; parallel TBLite cached/readonly lock probe completed without bind errors |
| Docker/Colima proposer mode. | FOLLOW-UP | Planned after Banking77 local-process mode; not part of this milestone. |
| Private Harbor-backed TBLite and live Codex reviewer rollouts. | REQUIRED FOR FULL E2E ACCEPTANCE | The public fixtures prove the optimizer contract and replay story only. Full TBLite/code-review acceptance requires real policy/runtime, verifier/scoring, artifact, cache/replay, and failure rails. |
| PyPI publish, `0.1.0` tag, prod promotion. | FOLLOW-UP | Blocked on launch checklist and evidence packet; not performed here. |
| Private backend services, arbitrary user Python imports, Go-EX, MIPROv2. | OUT OF SCOPE | No public surface is exposed for this GEPA slice; the Python package exports only GEPA run/result, event helpers, and typed errors. |

## Pre-Acceptance Evidence

Current workspace-status output root for all four examples:
`/tmp/synth-gepa-status-20260520`

Current checkpoint/stopper/usage-ledger/event-stream/cache-ledger/
candidate-graph/projection/invariant schema rerun root:
`/tmp/synth-gepa-final-current-20260520`

Banking77:

- Fresh run: `banking77_status_fresh`
- Accepted candidate: `gepa_706a0fa37ef3`
- Train reward: `1.0`
- Heldout reward: `1.0`
- Current binary emitted 11 accepted-candidate sensor frames and persisted
  40 total rollout job/sensor-frame rows plus 40 state transitions.
- Fresh cache profile: 33 entries, 12 hits, 33 misses, 33 writes
- Cached rerun: 45 hits, 0 misses, 0 writes
- Readonly replay: 45 hits, 0 misses, 0 writes
- Workspace tables: 1 run, 5 candidates, 5 candidate payloads,
  4 candidate deltas, 5 acceptance decisions, 5 frontier cells, 63 plan links,
  1 cache profile, 78 cache accesses, 67 event stream events, 42 usage ledger
  rows, 13 stopper rows, 6 checkpoint rows,
  40 rollout jobs, 40 sensor frames, 40 state transitions, 7 artifact refs,
  1 manifest.
- Fresh-vs-cached and fresh-vs-readonly normalized feeds matched.
- Current-schema rerun under `/tmp/synth-gepa-final-current-20260520`
  ended with `projection_status_counts={fresh: 21}`,
  `invariant_status_counts={pass: 2}`, zero invariant violations, cached
  rerun `45 hits, 0 writes`, readonly replay `45 hits, 0 writes`, and both
  normalized comparisons true.
  Cache access action counts were `{hit: 12, miss: 33, write: 33}`.
  Event stream type counts included `optimizer.state.transitioned=40`.
  Usage ledger boundary counts were `{container.rollout: 40, proposer.codex: 2}`.
  Stopper status counts were `{within_budget: 12, completed: 1}`.
  Checkpoint kind counts were `candidate_registry=1`,
  `evaluation_boundary=1`, `generation_boundary=2`, `pre_heldout=1`,
  `terminal=1`.

TBLite:

- Fresh run: `tblite_status_fresh`
- Accepted candidate: `gepa_53d671c830f8`
- Train reward: `0.8214285714285715`
- Heldout reward: `0.8541666666666667`
- Current binary emitted 8 accepted-candidate sensor frames and persisted
  11 total rollout job/sensor-frame rows plus 19 state transitions.
- Fresh cache profile: 12 entries, 3 hits, 12 misses, 12 writes
- Cached rerun: 15 hits, 0 misses, 0 writes
- Readonly replay: 15 hits, 0 misses, 0 writes
- Workspace tables: 1 run, 2 candidates, 2 candidate payloads,
  1 candidate delta, 2 acceptance decisions, 1 frontier cell, 18 plan links,
  1 cache profile, 27 cache accesses, 33 event stream events, 12 usage ledger
  rows, 6 stopper rows, 5 checkpoint rows,
  11 rollout jobs, 11 sensor frames, 19 state transitions, 7 artifact refs,
  1 manifest.
- Fresh-vs-cached and fresh-vs-readonly normalized feeds matched.
- Current-schema rerun under `/tmp/synth-gepa-final-current-20260520`
  ended with `projection_status_counts={fresh: 21}`,
  `invariant_status_counts={pass: 2}`, zero invariant violations, cached
  rerun `15 hits, 0 writes`, readonly replay `15 hits, 0 writes`, and both
  normalized comparisons true.
  Cache access action counts were `{hit: 3, miss: 12, write: 12}`.
  Event stream type counts included `optimizer.state.transitioned=19`.
  Usage ledger boundary counts were `{container.rollout: 11, proposer.codex: 1}`.
  Stopper status counts were `{within_budget: 5, completed: 1}`.
  Checkpoint kind counts were `candidate_registry=1`,
  `evaluation_boundary=1`, `generation_boundary=1`, `pre_heldout=1`,
  `terminal=1`.

Code-review:

- Fresh run: `code_review_status_fresh`
- Accepted candidate: `gepa_51538598b796`
- Train reward: `0.6009615384615384`
- Heldout reward: `0.5608974358974359`
- Current binary emitted 10 accepted-candidate sensor frames and persisted
  14 total rollout job/sensor-frame rows plus 19 state transitions.
- Fresh cache profile: 14 entries, 4 hits, 14 misses, 14 writes
- Cached rerun: 18 hits, 0 misses, 0 writes
- Readonly replay: 18 hits, 0 misses, 0 writes
- Workspace tables: 1 run, 2 candidates, 2 candidate payloads,
  1 candidate delta, 2 acceptance decisions, 1 frontier cell, 21 plan links,
  1 cache profile, 32 cache accesses, 33 event stream events, 15 usage ledger
  rows, 6 stopper rows, 5 checkpoint rows,
  14 rollout jobs, 14 sensor frames, 19 state transitions, 7 artifact refs,
  1 manifest.
- Fresh-vs-cached and fresh-vs-readonly normalized feeds matched.
- Current-schema rerun under `/tmp/synth-gepa-final-current-20260520`
  ended with `projection_status_counts={fresh: 21}`,
  `invariant_status_counts={pass: 2}`, zero invariant violations, cached
  rerun `18 hits, 0 writes`, readonly replay `18 hits, 0 writes`, and both
  normalized comparisons true.
  Cache access action counts were `{hit: 4, miss: 14, write: 14}`.
  Event stream type counts included `optimizer.state.transitioned=19`.
  Usage ledger boundary counts were `{container.rollout: 14, proposer.codex: 1}`.
  Stopper status counts were `{within_budget: 5, completed: 1}`.
  Checkpoint kind counts were `candidate_registry=1`,
  `evaluation_boundary=1`, `generation_boundary=1`, `pre_heldout=1`,
  `terminal=1`.

Crafter:

- Fresh run: `crafter_status_fresh`
- Accepted candidate: `gepa_837aedcf7438`
- Train reward: `0.9545454545454546`
- Heldout reward: `1.0`
- Usage: 1 proposer call, 14 rollout calls, 1695 prompt tokens,
  84 completion tokens, 1779 total tokens
- Fresh cache profile: 14 entries, 4 hits, 14 misses, 14 writes
- Cached rerun: 18 hits, 0 misses, 0 writes
- Readonly replay: 18 hits, 0 misses, 0 writes
- Accepted candidate includes a `lever_bundle` for `react_system_prompt`.
- Accepted candidate includes 10 `sensor_frames` across minibatch,
  full-train, and heldout stages.
- Workspace tables: 1 run, 2 candidates, 2 candidate payloads,
  1 candidate delta, 2 acceptance decisions, 1 frontier cell, 21 plan links,
  1 cache profile, 32 cache accesses, 33 event stream events, 15 usage ledger
  rows, 6 stopper rows, 5 checkpoint rows,
  14 rollout jobs, 14 sensor frames, 19 state transitions, 7 artifact refs,
  1 manifest.
- Result manifest includes 19 optimizer state transitions ending in
  `completed`.
- Fresh-vs-cached and fresh-vs-readonly normalized feeds matched.
- Current-schema rerun under `/tmp/synth-gepa-final-current-20260520`
  ended with `projection_status_counts={fresh: 21}`,
  `invariant_status_counts={pass: 2}`, zero invariant violations, cached
  rerun `18 hits, 0 writes`, readonly replay `18 hits, 0 writes`, and both
  normalized comparisons true.
  Cache access action counts were `{hit: 4, miss: 14, write: 14}`.
  Event stream type counts included `optimizer.state.transitioned=19`.
  Usage ledger boundary counts were `{container.rollout: 14, proposer.codex: 1}`.
  Stopper status counts were `{within_budget: 5, completed: 1}`.
  Checkpoint kind counts were `candidate_registry=1`,
  `evaluation_boundary=1`, `generation_boundary=1`, `pre_heldout=1`,
  `terminal=1`.

Shared run registry:

- Current deterministic runs write per-example registries under
  `/tmp/synth-gepa-status-20260520/<example>/run_registry.jsonl`.
- Prior lock-probe evidence remains under
  `/tmp/synth-gepa-firstclass-final.eEStdM/run_registry.jsonl`.

Typed readonly miss:

- Final uncached readonly namespace probe raised:
  `CacheMissError synth_optimizer_cache_miss`

Container run lock probe:

- Parallel TBLite `tblite_lock_probe_cached` and
  `tblite_lock_probe_readonly` completed against the same configured port and
  cache namespace with no address-in-use bind error.

## Validation Run

Final build/static validation passed after the four-example pre-acceptance
validation pass, the Rust platform noun/state-machine update, SQLite workspace
persistence, and workspace status read model:

```bash
cargo fmt --manifest-path packages/synth-optimizers/Cargo.toml --all --check
cargo check --manifest-path packages/synth-optimizers/Cargo.toml --workspace
cargo clippy --manifest-path packages/synth-optimizers/Cargo.toml --workspace -- -D warnings
uv run --group dev maturin develop --manifest-path rust/crates/synth_optimizers_py/Cargo.toml
uv run --project packages/synth-optimizers --group dev python -m py_compile packages/synth-optimizers/src/synth_optimizers/__init__.py packages/synth-optimizers/src/synth_optimizers/cli.py cookbooks/optimizers/gepa/banking77_container/synth_service_app.py cookbooks/optimizers/gepa/tblite_container/synth_service_app.py cookbooks/optimizers/gepa/code_review_container/synth_service_app.py cookbooks/optimizers/gepa/crafter_container/synth_service_app.py
uv run --project packages/synth-optimizers --group dev ruff check packages/synth-optimizers/src cookbooks/optimizers/gepa/banking77_container/synth_service_app.py cookbooks/optimizers/gepa/tblite_container/synth_service_app.py cookbooks/optimizers/gepa/code_review_container/synth_service_app.py cookbooks/optimizers/gepa/crafter_container/synth_service_app.py
uv run --project packages/synth-optimizers --group dev ty check packages/synth-optimizers/src
git diff --check
```

## GEPA Async Rollout Mode Banking77 Smoke

Public Rust GEPA gained opt-in async rollout submission on 2026-05-21 while
keeping sync as the default. The optimizer plans the same durable rollout jobs
in both modes; runtime dispatch selects either blocking `/rollout` or async
`/rollout` plus `/rollouts/{id}/state` polling and final
`/rollouts/{id}` fetch. Rollout cache keys normalize away `submission_mode`, so
readonly replay can cross between sync and async dispatch shapes.

- Banking77 root: `/tmp/synth-gepa-banking77-async-20260520_231846`
- Banking77 run: `banking77_async_20260520_231846`
- Config: 24 train seeds, 12 heldout seeds, `max_generations=4`,
  `proposals_per_generation=2`, `minibatch_size=16`,
  `max_total_rollouts=400`, `rollout_submission_mode="async"`,
  `rollout_poll_interval_ms=100`, async timeout `600s`.
- Driver: `gepa run --config
  /tmp/synth-gepa-banking77-async-20260520_231846/config.toml` with
  `SYNTH_OPTIMIZERS_MAX_CONCURRENT_ROLLOUTS=16` and live OpenAI Banking77
  policy/proposer.
- Wall time: `real 108.81s`.
- Terminal result: completed, best `gepa_14456234ec77`, train
  `23/24 = 95.83%`, heldout `11/12 = 91.67%`.
- Baseline seed: train `21/24 = 87.50%`, heldout `11/12 = 91.67%`.
- Interpretation: async Banking77 confirmed high performance and clear train
  lift (`+8.33pp`); heldout was flat because all six heldout-evaluated
  candidates tied at `91.67%`.
- Usage/cost: 328 rollouts, 4 proposer calls, 44,048 total tokens
  (`41,795` prompt, `2,253` completion), recorded cost `$0.0000` because the
  public container records token usage but not provider price.
- Async evidence: the stored config has
  `gepa.rollout_submission_mode="async"`; all 10 rollout optimizer jobs contain
  async dispatch requests and zero contain sync dispatch requests.
- Batch evidence: rollout runtime effects covered seed full-train `24`,
  candidate minibatch `16 + 32 + 32 + 32`, candidate full-train
  `48 + 24 + 24 + 24`, and heldout `72`.
- Cache evidence: rollout cache has `248/248` distinct entries and
  `0` cached rollout requests contain `submission_mode`.
- Workspace health: `workspace status` produced `invariant_status_counts={
  pass: 2 }`, `projection_status_counts={ fresh: 34 }`, and zero invariant
  violations.

Static gates after the async rollout changes:

```bash
cargo fmt --check
cargo check --workspace
cargo clippy --workspace -- -D warnings
python -m compileall -q src ../../cookbooks/optimizers/gepa/banking77_container/synth_service_app.py
uv run --group dev ruff check src ../../cookbooks/optimizers/gepa/banking77_container/synth_service_app.py
uv run --group dev ty check src
uv run --group dev maturin develop
git diff --check
```

Behavioral validation evidence includes the original status root
`/tmp/synth-gepa-status-20260520` and the latest final current-schema
rerun root `/tmp/synth-gepa-final-current-20260520`:

```bash
synth-optimizers gepa run --config cookbooks/optimizers/gepa/banking77_container/gepa.toml
synth-optimizers gepa run --config cookbooks/optimizers/gepa/tblite_container/gepa.toml
synth-optimizers gepa run --config cookbooks/optimizers/gepa/code_review_container/gepa.toml
synth-optimizers gepa run --config cookbooks/optimizers/gepa/crafter_container/gepa.toml
synth-optimizers events compare --left <fresh>/events.normalized.jsonl --right <cached>/events.normalized.jsonl
synth-optimizers events compare --left <fresh>/events.normalized.jsonl --right <readonly>/events.normalized.jsonl
synth-optimizers workspace status --db <fresh>/workspace.sqlite
synth-optimizers workspace submit --db /tmp/synth-gepa-service-queue-20260520-kf2lxqzq/service.sqlite --config cookbooks/optimizers/gepa/banking77_container/gepa.toml --priority 5
synth-optimizers workspace claim --db /tmp/synth-gepa-service-queue-20260520-kf2lxqzq/service.sqlite --lease-id worker-lease-1 --worker-id worker-1
synth-optimizers workspace start --db /tmp/synth-gepa-service-queue-20260520-kf2lxqzq/service.sqlite --request-id runreq_87b5b97b2a774aebad2940bd45d0006c
synth-optimizers workspace complete --db /tmp/synth-gepa-service-queue-20260520-kf2lxqzq/service.sqlite --request-id runreq_87b5b97b2a774aebad2940bd45d0006c
synth-optimizers workspace cancel --db /tmp/synth-gepa-service-queue-20260520-kf2lxqzq/service.sqlite --request-id runreq_0a0c0eb686e14c458bdc27abd85848da --reason superseded
synth-optimizers gepa service --db /tmp/synth-gepa-service-http-20260520/service.sqlite --bind 127.0.0.1:8899 --worker-id service-smoke --lease-seconds 120
curl -s http://127.0.0.1:8899/health
curl -s -X POST http://127.0.0.1:8899/runs -H 'Content-Type: application/json' -d '{"config_path":"cookbooks/optimizers/gepa/crafter_container/gepa.toml","priority":3,"auto_start":true}'
curl -s http://127.0.0.1:8899/status
synth-optimizers workspace status --db /tmp/synth-gepa-service-http-20260520/runs/crafter_service_http/workspace.sqlite
synth-optimizers workspace recover --db /tmp/synth-gepa-recovery-concurrency-20260520-zawwbtc3/service.sqlite
curl -s -X POST http://127.0.0.1:8901/worker/recover -H 'Content-Type: application/json' -d '{}'
operation/resource smoke /var/folders/7b/96hbyfld35zflr_plpgvrdxm0000gn/T/synth-gepa-ops-20260520-h76ihhye/service.sqlite -> operations completed=9, resource leases released=2 expired=2
candidate graph smoke /tmp/synth-gepa-candidate-graph-20260520/banking77_candidate_graph_smoke/workspace.sqlite -> candidate_payloads=5, candidate_deltas=4, acceptance_decisions=5, frontier_cells=5, plan_links=63, projection_status_counts={fresh: 15}, invariant_status_counts={pass: 2}, invariant violations=0
cache ledger smoke /tmp/synth-gepa-cache-ledger-20260520/banking77_cache_ledger_smoke/workspace.sqlite -> cache_profiles=1, cache_accesses=78, action counts hit=12 miss=33 write=33, boundary counts container.dataset_rows=4 container.program=2 container.rollout=68 proposer.codex=4, projection_status_counts={fresh: 17}, invariant_status_counts={pass: 2}, invariant violations=0
event stream smoke /tmp/synth-gepa-event-stream-20260520/banking77_event_stream_smoke/workspace.sqlite -> event_stream_events=67, optimizer.state.transitioned=40, state_transitions=40, projection_status_counts={fresh: 18}, invariant_status_counts={pass: 2}, invariant violations=0
usage ledger smoke /tmp/synth-gepa-usage-ledger-20260520/banking77_usage_ledger_smoke/workspace.sqlite -> usage_ledger=42, container.rollout=40, proposer.codex=2, usage_ledger_from_external_calls fresh 42/42, projection_status_counts={fresh: 19}, invariant_status_counts={pass: 2}, invariant violations=0
stopper smoke /tmp/synth-gepa-stopper-20260520/banking77_stopper_smoke/workspace.sqlite -> stopper_states=13, status counts within_budget=12 completed=1, stopper_state_from_terminal_run fresh 1/13, projection_status_counts={fresh: 20}, invariant_status_counts={pass: 2}, invariant violations=0
checkpoint smoke /tmp/synth-gepa-checkpoint-20260520/banking77_checkpoint_smoke/workspace.sqlite -> checkpoints=6, kind counts candidate_registry=1 evaluation_boundary=1 generation_boundary=2 pre_heldout=1 terminal=1, checkpoint_from_terminal_run fresh 1/6, projection_status_counts={fresh: 21}, invariant_status_counts={pass: 2}, invariant violations=0
service projection/invariant smoke /tmp/synth-gepa-projection-20260520/service.sqlite -> run_request_submit_operations fresh 2/2, workspace invariant pass
current-schema four-example deterministic validation /tmp/synth-gepa-final-current-20260520 -> Banking77, TBLite, code-review, Crafter fresh/cached/readonly completed; cached writes=0; readonly writes=0; fresh-vs-cached and fresh-vs-readonly normalized comparisons true; every fresh run projection_status_counts={fresh: 21}, per-run projection count=20, invariant_status_counts={pass: 2}, invariant violations=0; fresh checkpoint/stopper/usage/event/cache row counts Banking77=6/13/42/67/1/78 TBLite=5/6/12/33/1/27 code-review=5/6/15/33/1/32 Crafter=5/6/15/33/1/32
live Codex app-server audit /tmp/synth-gepa-live-codex-20260520/banking77_live_codex_audit -> completed Banking77 with backend=codex_app_server, best=gepa_c46bfac0c13c, proposer.completed=2, proposer.codex usage rows=2, checkpoint rows=6, stopper rows=13, projection_status_counts={fresh: 21}, invariant_status_counts={pass: 2}, invariant violations=0
rollout smoke /tmp/synth-gepa-rollout-20260520/crafter_rollout_smoke/workspace.sqlite -> 14 rollouts, 28 rollout_events, rollout_event_type_counts={rollout_observed: 14, trace_digest_observed: 14}
score smoke /tmp/synth-gepa-score-20260520/banking77_score_smoke/workspace.sqlite -> 1 objective, 40 scores, score_objective_counts={outcome_reward: 40}
evidence smoke /tmp/synth-gepa-evidence-20260520/banking77_evidence_smoke/workspace.sqlite -> 40 trace annotations, 80 evidence frames, 40 verifier jobs, 40 subagent invocations, 120 optimizer jobs
evidence smoke /tmp/synth-gepa-evidence-20260520/crafter_evidence_smoke/workspace.sqlite -> 14 trace annotations, 28 evidence frames, 14 verifier jobs, 14 subagent invocations, 42 optimizer jobs, llm_request_count_sum=14, tool_call_count_sum=14
live poll /tmp/synth-gepa-incremental-20260520-agkn4_pe/runs/banking77_incremental/workspace.sqlite while service request was running -> 3 candidates, 21 rollout jobs, 21 sensor frames
cancel smoke /tmp/synth-gepa-state-cancel-20260520-v_098aar -> running request had 4 live state transitions, then ended cancelled after workspace cancel
service summary smoke /var/folders/7b/96hbyfld35zflr_plpgvrdxm0000gn/T/synth-gepa-service-summary-cancel-20260520-a721x5b0/summary/service.sqlite -> completed request imported run DB path, result manifest path, best candidate, usage, and result JSON
cancellation-exit smoke /var/folders/7b/96hbyfld35zflr_plpgvrdxm0000gn/T/synth-gepa-cancel-exit-20260520-c86emxri/service.sqlite -> worker returned run request cancelled; per-run latest transition is cancel_requested -> cancelled
readonly uncached probe -> CacheMissError synth_optimizer_cache_miss
parallel TBLite cached/readonly lock probe -> no bind error
```

## Restart Tick Acceptance Smoke

One-job GEPA tick restart smoke, failure terminalization, cancellation
terminalization, run-next parity, and direct-result parity were revalidated on
2026-05-20 using only the public CLI and SQLite surfaces. No public CLI names,
SQLite schema, result fields, manifest fields, or checkpoint kind changed.

Primary one-process-per-tick TBLite run:

- Root: `/tmp/synth-gepa-restart-1779320201`
- Run: `tblite_restart_1779320201`
- Driver: `workspace submit`, then 57 separate `gepa tick --db` invocations.
- Environment: legacy deterministic proposer fallback plus unique run id, output
  dir, cache path, and cache namespace under `/tmp`. That fallback is now
  unsupported; new acceptance runs should use `codex_app_server`.
- Actions advanced one transition per tick through claim, start, setup, plan,
  execute, consume, checkpoint, and terminalize.
- Final service status: `{completed: 1}`.
- Latest `gepa_cursor`: `phase=completed`, no pending job/effect/reservation,
  `terminal_summary=true`.
- Runtime idempotency: optimizer jobs `14/14 distinct job_id`; runtime effects
  `14/14 distinct runtime_effect_id` and `14/14 distinct idempotency_key`.
- Rollout stages were distinct: seed full-train `3`, candidate minibatch `3`,
  candidate full-train `3`, heldout `4`.
- Registry rows: exactly `started`, `finished`.
- Manifest exists at
  `/tmp/synth-gepa-restart-1779320201/runs/tblite_restart_1779320201/result_manifest.json`.
- Workspace invariant report: `pass`.

Terminal probes:

- Root: `/tmp/synth-gepa-terminal-1779320578`
- Failure probe used a temporary `/tmp` TOML with proposer command
  `["/bin/false"]`; final service status `{failed: 1}`.
- Failure cursor: `phase=failed`, no pending job/effect/reservation,
  `error_summary=true`; registry rows exactly `started`, `failed`; failure
  manifest exists.
- Failed proposer state: optimizer jobs included `proposer failed=1`; runtime
  effects included `proposer failed=1`; seed rollout jobs remained completed.
- Cancellation probe submitted a fresh service run, ticked through setup,
  cancelled with `workspace cancel`, then one more `gepa tick --db` returned
  `terminalize_run`; final service status `{cancelled: 1}`.
- Cancellation cursor: `phase=cancelled`, no pending job/effect/reservation,
  `error_summary=true`; registry rows exactly `started`, `cancelled`;
  cancellation manifest exists.

Run-next and parity probes:

- `gepa run-next --db` smoke root:
  `/tmp/synth-gepa-run-next-1779320619`; final service status `{completed: 1}`,
  latest cursor `completed`, workspace invariants `pass`.
- `run-next` result keys matched its terminal manifest keys exactly:
  `artifact_refs`, `best_candidate`, `cache_profile_path`,
  `candidate_registry_path`, `cost_usd`, `event_feed_path`, `frontier_path`,
  `manifest_path`, `normalized_event_feed_path`, `run_registry_path`,
  `score_chart_path`, `state_history`, `usage`, `workspace_db_path`.
- Direct `gepa run --json` parity root:
  `/tmp/synth-gepa-direct-1779320682`; direct result keys matched direct
  manifest keys and the `run-next` key set above.

Static gates after the restart/failure/cancel fixes:

```bash
cargo fmt --check
cargo check --workspace
cargo clippy --workspace -- -D warnings
python -m compileall -q src
uv run --group dev ruff check src
uv run --group dev ty check src
uv run --group dev maturin develop
git diff --check
```

## Banking77 Full Tick Smoke

Full public Banking77 GEPA service/tick smoke was run on 2026-05-21 after the
restart acceptance fixes, using the public `banking77_container/gepa.toml`,
live OpenAI policy rollouts, and a fresh `/tmp` service DB/output/cache
namespace.

- Root: `/tmp/synth-gepa-banking77-1779322771`
- Run: `banking77_tick_1779322771`
- Driver: `workspace submit`, then 673 separate `gepa tick --db`
  invocations.
- Final service status: `{completed: 1}`.
- Latest `gepa_cursor`: `phase=completed`, no pending job/effect/reservation,
  `terminal_summary=true`, `error_summary=false`.
- Registry rows: exactly `started`, `finished`.
- Manifest exists at
  `/tmp/synth-gepa-banking77-1779322771/runs/banking77_tick_1779322771/result_manifest.json`.
- Workspace invariant reports: `pass=2`, invariant violations `0`.
- Idempotency checks: optimizer jobs `850/850 distinct job_id`; runtime
  effects `214/214 distinct runtime_effect_id` and `214/214 distinct
  idempotency_key`.
- Terminal best candidate: `gepa_d7ca6f89f619`, status `accepted`, train
  `22/24 = 91.67%`, heldout `11/12 = 91.67%`.
- Seed baseline: `gepa_112df7436bc4`, train `22/24 = 91.67%`, heldout
  `11/12 = 91.67%`.
- Interpretation: high performance is confirmed on the public Banking77 slice;
  measurable lift is not shown by this small default split because accepted
  candidates tied the seed and rejected candidates were worse on full-train.

This run also hardened terminal heldout selection: the incumbent is kept on
heldout ties unless a challenger is better on the train score vector.

## GEPA Parallel Rollout Batch Smoke

Public Rust GEPA rollout planning was hardened on 2026-05-21 to restore the
private platform behavior where stage rollouts fan out in one bounded batch:
rows inside an evaluation run concurrently, multiple proposal candidates are
batched together for minibatch/full-train, and heldout candidates are batched
together. Runtime concurrency is controlled by `GEPA_ROLLOUT_CONCURRENCY` or
the compatibility alias `SYNTH_OPTIMIZERS_MAX_CONCURRENT_ROLLOUTS`.

- Banking77 root: `/tmp/synth-gepa-banking77-parallel-smoke-20260520-213522`
- Banking77 run: `banking77_parallel_smoke_20260520_213522`
- Driver: `workspace submit`, then `gepa run-next --db` with the legacy
  deterministic proposer fallback and `SYNTH_OPTIMIZERS_MAX_CONCURRENT_ROLLOUTS=16`.
  That fallback is now unsupported; new acceptance runs should use
  `codex_app_server`.
- Wall time: `real 90.68s`; final service status `{completed: 1}`; latest
  cursor `completed`; manifest exists.
- Workspace DB evidence: two candidate-minibatch `rollout_batch` jobs, each
  with `candidate_ids` for both proposals and `rollout_count=16`; `example_refs`
  are round-robin by candidate/example (`A train:0`, `B train:0`, ...).
- Banking77 idempotency/status: optimizer jobs `210/210 distinct job_id`;
  runtime effects `6/6 distinct runtime_effect_id` and `6/6 distinct
  idempotency_key`; sensor frames `68`; no SQLite lock errors.

Heldout grouping was separately exercised with TBLite:

- TBLite root: `/tmp/synth-gepa-tblite-parallel-smoke-20260520-213740`
- TBLite run: `tblite_parallel_smoke_20260520_213740`
- Wall time: `real 35.81s`; final service status `{completed: 1}`; latest
  cursor `completed`.
- Workspace DB evidence: one heldout `rollout_batch` job with
  `candidate_ids=["gepa_05c38a291174","gepa_e3b95432cec3"]`,
  `rollout_count=4`, and round-robin `example_refs` across both candidates;
  optimizer jobs `44/44 distinct job_id`; runtime effects `5/5 distinct
  runtime_effect_id` and `5/5 distinct idempotency_key`; sensor frames `13`.

Static gates after the batch rollout changes:

```bash
cargo fmt --check
cargo check --workspace
cargo clippy --workspace -- -D warnings
python -m compileall -q src
uv run --group dev ruff check src
uv run --group dev ty check src
uv run --group dev maturin develop
git diff --check
```
