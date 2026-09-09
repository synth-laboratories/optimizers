# Container-first queue-native CISPO: architecture and engineering handoff

Initial benchmark: 2026-09-02\
Engineering handoff updated: 2026-09-03\
Reframed against Synth Style and the Tito data plane: 2026-09-03

## Queue architecture reference

These diagrams describe the actual asynchronous topology of SLIME, MiLeS, and
Prime-RL before mapping those ideas onto the proposed Harbor + Tinker system.
They are intentionally not normalized into one linear pipeline: each shows its
own producer, buffers, fan-out/fan-in, feedback, and retry paths.

SLIME and MiLeS perform reward computation inside the active generation task
before placing a completed group in their output buffer. Prime-RL receives a
verifier episode through the dispatcher output queue. Harbor needs an additional
durable agent-to-verifier boundary because its agent and verifier are distinct
Docker workloads.

### SLIME fully-async rollout

Reference: [`slime/rollout/fully_async_rollout.py`](https://github.com/THUDM/slime/blob/main/slime/rollout/fully_async_rollout.py)

```text
                         SLIME FULLY-ASYNC PIPELINE
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PRODUCER LOOP                                  │
│                                                                             │
│  ┌──────────────────┐       ┌────────────────────────────────────────────┐  │
│  │ Prompt/data      │       │ Persistent background worker              │  │
│  │ buffer           │──────▶│                                            │  │
│  │                  │       │ collect finished tasks                    │  │
│  │ aborted groups ◀─┼───────│ top up available concurrency              │  │
│  └──────────────────┘       │ stop top-up when output queue is full     │  │
│                             └────────────────┬───────────────────────────┘  │
│                                              │                              │
│                 ┌────────────────────────────┼─────────────────────────┐    │
│                 │        ACTIVE GROUP TASKS  │                         │    │
│                 │  ┌─────────┐  ┌─────────┐  │  ┌─────────┐           │    │
│                 │  │ Group 1 │  │ Group 2 │ ...│ Group N │           │    │
│                 │  │ samples │  │ samples │  │  │ samples │           │    │
│                 │  │   + RM  │  │   + RM  │  │  │   + RM  │           │    │
│                 │  └────┬────┘  └────┬────┘  │  └────┬────┘           │    │
│                 └───────┼────────────┼────────┼───────┼────────────────┘    │
│                         └────────────┴────────┴───────┘                     │
│                                              │                              │
│                              completed, non-aborted groups                  │
│                                              ▼                              │
│                  ┌─────────────────────────────────────────┐                │
│                  │ COMPLETED-GROUP OUTPUT_QUEUE            │                │
│                  │ persistent across training iterations   │                │
│                  │ qsize gate applies producer backpressure│                │
│                  └────────────────────┬────────────────────┘                │
└───────────────────────────────────────┼─────────────────────────────────────┘
                                        │ consume exact batch requirement
                                        ▼
                           ┌────────────────────────┐
                           │ Training batch assembly│
                           └────────────┬───────────┘
                                        ▼
                              ┌──────────────────┐
                              │ Policy training  │
                              └─────────┬────────┘
                                        │ new weights
                                        ▼
                         ┌────────────────────────────┐
                         │ Rollout inference engines  │
                         └────────────────────────────┘

       Producer remains alive while batch assembly and training execute.
```

The characteristic shape is one persistent producer, many active group tasks,
and a completed-group queue that survives across training iterations. Aborted
groups return sideways to the input buffer.

### MiLeS fully-async rollout

References:

- [`miles/rollout/fully_async_rollout.py`](https://github.com/radixark/miles/blob/main/miles/rollout/fully_async_rollout.py)
- [`miles/rollout/submission_scheduler.py`](https://github.com/radixark/miles/blob/main/miles/rollout/submission_scheduler.py)
- [`miles/rollout/fully_async_data_buffer.py`](https://github.com/radixark/miles/blob/main/miles/rollout/fully_async_data_buffer.py)

```text
                          MiLeS FULLY-ASYNC PIPELINE

 ┌────────────────┐        ┌──────────────────────────────────────┐
 │ Prompt groups  │───────▶│ SAMPLE-BACKFILL SUBMISSION SCHEDULER│
 └───────▲────────┘        │ credits represent completed samples │
         │                 │ rather than completed whole groups  │
         │ retry           └───────────┬──────────────────────────┘
         │                             │ open/continue groups
         │                             ▼
         │       ┌─────────────────────────────────────────────────────┐
         │       │                  ACTIVE GROUPS                      │
         │       │  Group A                 Group B                    │
         │       │  ┌────┬────┬────┐         ┌────┬────┬────┐         │
         │       │  │ A0 │ A1 │... │         │ B0 │ B1 │... │         │
         │       │  └─┬──┴─┬──┴─┬──┘         └─┬──┴─┬──┴─┬──┘         │
         │       │    └────┴────┴────┐    ┌────┴────┴────┘            │
         │       └────────────────────┼────┼───────────────────────────┘
         │                            ▼    ▼
         │                ┌────────────────────────┐
         │                │ Generation semaphore   │
         │                │ waiting → generating   │
         │                │ tool calls / multistep │
         │                └────────────┬───────────┘
         │                             │ each completion returns credit
         │                             ├───────────────────────────────┐
         │                             ▼                               │
         │                ┌────────────────────────┐                   │
         │                │ Reward computation     │                   │
         │                │ individual or group RM │                   │
         │                └────────────┬───────────┘                   │
         │                             │ complete group                │
         │                             ▼                               │
         │       ┌─────────────────────────────────────────────────┐   │
         │       │ BOUNDED ASYNC DATA BUFFER                       │   │
         └───────┤ put: reject/retry aborted or filtered groups    │   │
                 │ full: producer blocks on asyncio.Condition      │   │
                 │ get: calculate current policy staleness         │   │
                 │ stale: drop or recycle                          │   │
                 └──────────────────────┬──────────────────────────┘   │
                                        │ fresh completed groups       │
                                        ▼                              │
                              ┌─────────────────────┐                  │
                              │ Training batch      │                  │
                              └──────────┬──────────┘                  │
                                         ▼                             │
                              ┌─────────────────────┐                  │
                              │ Policy update       │──────────────────┘
                              └─────────────────────┘       current version
```

MiLeS adds sample-completion backfill, a genuinely bounded buffer, and a hard
staleness check when training retrieves a group.

### Prime-RL orchestrator

References:

- [`src/prime_rl/orchestrator/dispatcher.py`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/orchestrator/dispatcher.py)
- [`src/prime_rl/orchestrator/orchestrator.py`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/orchestrator/orchestrator.py)
- [`src/prime_rl/orchestrator/train_sink.py`](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/src/prime_rl/orchestrator/train_sink.py)

```text
                           PRIME-RL ORCHESTRATOR

                                  CONTROL
       ┌───────────────────┐                       ┌──────────────────┐
       │ Progress/policy   │◀─────────────────────▶│ Weight watcher   │
       │ version           │                       │ inference update │
       └─────────┬─────────┘                       └──────────────────┘
                 │ dispatch gate / staleness
                 ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                              DISPATCHER                                    │
│  ┌────────────────┐     ┌────────────────┐      ┌───────────────────────┐  │
│  │ Task/curriculum│────▶│ Group states   │─────▶│ Shared capacity       │  │
│  │ source         │     │ remaining work │      │ permits               │  │
│  └────────────────┘     └────────────────┘      └───────────┬───────────┘  │
│                                                            ▼              │
│       ┌──────────────────────────────────────────────────────────────┐     │
│       │                     IN-FLIGHT ATTEMPTS                       │     │
│       │  ┌─────────┐  ┌─────────┐  ┌─────────┐       ┌─────────┐    │     │
│       │  │ Train A0│  │ Train A1│  │ Train B0│  ...  │ Eval E0 │    │     │
│       │  └────┬────┘  └────┬────┘  └────┬────┘       └────┬────┘    │     │
│       └───────┼────────────┼────────────┼──────────────────┼─────────┘     │
│               └────────────┴────────────┴──────────────────┘               │
│                                      │                                     │
│       every attempt emits exactly one episode, failure, or cancellation    │
│                                      ▼                                     │
│                   ┌──────────────────────────────────┐                     │
│                   │ BOUNDED dispatcher.out_q         │                     │
│                   │ asyncio.Queue[DispatchResult]    │                     │
│                   └────────────────┬─────────────────┘                     │
└────────────────────────────────────┼───────────────────────────────────────┘
                                     ▼
                           ┌────────────────────┐
                           │ Orchestrator router│
                           └──────┬───────┬─────┘
                                  │       │
                         train    │       │ eval
                                  ▼       ▼
             ┌────────────────────────┐  ┌──────────────────┐
             │ TRAIN SINK             │  │ EVAL SINK        │
             │ pending_groups         │  │ pending results  │
             │ pending_failures       │  │ evaluation       │
             │ pending_cancellations  │  │ aggregation      │
             │ pending_batch/tokens   │  └──────────────────┘
             └───────────┬────────────┘
                         │ group accounting complete
                         ▼
             ┌────────────────────────┐
             │ Episode finalization   │
             │ Group advantages       │
             │ Curriculum admission   │
             │ Hard staleness sweep   │
             └───────────┬────────────┘
                         ▼
             ┌────────────────────────┐
             │ Packed TrainBatch      │
             └───────────┬────────────┘
                         ▼
             ┌────────────────────────┐
             │ Trainer                │──────▶ progress / weight watcher
             └────────────────────────┘
```

Prime-RL's important accounting invariant is that every admitted attempt reaches
`out_q` exactly once, including failures and cancellations. Its train sink is a
stateful fan-in system, not a barrier over successful futures.

### Proposed Harbor + Tinker topology

This design combines Prime-RL's per-attempt dispatch and terminal accounting,
MiLeS's sample backfill and dequeue-time staleness checks, and SLIME's persistent
production. It adds a durable agent-to-verifier handoff for Harbor.

```text
                     HARBOR + TINKER QUEUE TOPOLOGY

                                 CONTROL PLANE
 ┌──────────────────────────────────────────────────────────────────────────┐
 │   ┌────────────┐       ┌───────────────────┐       ┌──────────────────┐  │
 │   │ Task/seed  │──────▶│ Group registry    │◀─────▶│ Policy registry  │  │
 │   │ source     │       │ A: 17/20 scored   │       │ current: v18     │  │
 │   └────────────┘       │ B:  8/20 scored   │       │ allowed: v17–18  │  │
 │                        │ C:  0/20 scored   │       └────────▲─────────┘  │
 │                        └─────────▲─────────┘                │            │
 │                                  │                  ┌───────┴────────┐   │
 │    ┌─────────────────────────────┴───────────────┐  │ Policy watcher │   │
 │    │ Per-attempt scheduler                       │  └────────────────┘   │
 │    │ continue open groups; enforce staleness     │                       │
 │    └──────────────────────┬──────────────────────┘                       │
 └───────────────────────────┼──────────────────────────────────────────────┘
                             ▼
                 ┌────────────────────────┐
                 │ ROLLOUT_QUEUE          │◀───────────────┐
                 │ A17 A18 A19            │                │
                 │ B08 ... B19; C00 ...   │                │
                 └────────────┬───────────┘                │
                              ▼                            │
                    ┌────────────────────┐                 │
                    │ Capacity broker    │                 │
                    │ 30 OrbStack slots  │                 │
                    │ rollout + scoring  │                 │
                    └──────┬───────┬─────┘                 │
              rollout slots│       │verifier slots         │
                           ▼       ▼                       │
          ┌────────────────────┐  ┌────────────────────┐   │
          │ ROLLOUT WORKER POOL│  │ SCORING WORKER POOL│   │
          │ ┌────┐ ┌────┐     │  │ ┌────┐ ┌────┐     │   │
          │ │ R1 │ │ R2 │ ... │  │ │ S1 │ │ S2 │ ... │   │
          │ └─┬──┘ └─┬──┘     │  │ └─▲──┘ └─▲──┘     │   │
          └───┼───────┼────────┘  └───┼──────┼────────┘   │
              └───────┴──────┐        │      │            │
                              ▼        │      │            │
       ┌──────────────────────────────┐│      │            │
       │ DURABLE ARTIFACT STORE       ││      │            │
       │ workspace snapshot/lease     ││      │            │
       │ patch + sealed v5 trace      ││      │            │
       │ tokens/logprobs + TPS        ││      │            │
       │ behavior-policy span         ││      │            │
       └──────────────┬───────────────┘│      │            │
                      │ ScoreBundle ref│      │            │
                      ▼                │      │            │
              ┌───────────────────┐    │      │            │
              │ SCORE_QUEUE       ├────┘      │            │
              │ A03 B01 A07 ...   │           │            │
              └───────────────────┘           │            │
                                              │            │
                                      verifier results     │
                                              ▼            │
                               ┌──────────────────────────┐ │
                               │ SCORED-RESULT QUEUE      │ │
                               │ episode / failure /      │ │
                               │ cancellation             │ │
                               └────────────┬─────────────┘ │
                                            │               │
                 ┌──────────────────────────┼────────────┐  │
                 ▼                          ▼            ▼  │
          ┌─────────────┐            ┌─────────────┐  ┌─────────────┐
          │ Group A sink│            │ Group B sink│  │ Group C sink│
          │ 20/20 ready │            │ 12/20       │  │ 3/20        │
          │ CISPO adv.  │            │ pending     │  │ pending     │
          └──────┬──────┘            └─────────────┘  └─────────────┘
                 │ complete + admitted + fresh
                 ▼
          ┌───────────────────┐          stale/rejected ───┘
          │ TRAIN_READY_QUEUE │
          │ bounded: 1–2      │
          └─────────┬─────────┘
                    ▼
              ┌─────────────┐        ┌───────────────────┐
              │ Tinker      │───────▶│ POLICY_PUBLISH_Q  │
              │ CISPO train │        │ v18 → v19         │
              └─────────────┘        └─────────┬─────────┘
                                              └──▶ Policy registry

  rollout failure ────────▶ RETRY_QUEUE ──────────────────────────────┘
  verifier failure ───────▶ SCORE_QUEUE retry
  stale group ────────────▶ regenerate using the current policy
  terminal/consumed ──────▶ labelled artifact and container cleanup

  ┌────────────────────────────────────────────────────────────────────────┐
  │ LIFECYCLE LEDGER + METRICS BUS                                         │
  │ All queues and workers publish transitions, queue/service time, policy │
  │ version, staleness, v5 trace IDs, TPS, cost, retries, and cleanup.      │
  └────────────────────────────────────────────────────────────────────────┘
```

The durable `ScoreBundle` is the key Harbor-specific boundary. It lets an agent
finish and release rollout capacity while its verifier waits independently.
Without this boundary, agent execution and verification remain one fused
operation and a score queue cannot produce genuine pipeline overlap.

### Interpretation of the existing benchmark

The sync/Q2/Q4 measurements below describe the current grouped-future runner,
not the proposed queue-native implementation. The runner overlaps pre-submitted
groups but still waits for every future in the next group before group
finalization and training. Therefore its measured speedups and slowdowns should
be retained as historical evidence, but must not be treated as a measurement of
the architecture above.

## Engineering handoff: container-first CISPO

### Objective

Implement container-independent reinforcement learning with the same
container-independence that GEPA already has. The executor receives a container
connection, validates a declared contract, and operates only through that
contract. It must not know whether the container is Banking77, HealthBench,
Craftax, TBLite, or a future task.

CISPO is the first preset of that executor, not its definition. Synth Style 04
(general foundations, targeted affordances) makes the distinction load-bearing
rather than cosmetic: the queue engine, container contract, handshake, evidence
rules, and checkpoint catalog are plane-level concerns that GSPO, PPO, SAO, or a
distillation preset must reuse without a fork.

The first end-to-end milestone is a small, real CISPO run for each of:

- Banking77
- HealthBench
- Craftax
- TBLite

Two multi-policy targets follow: the cooperative turn-based GameBench Rust
DungeonGrid party, and the concurrent real-time competitive RuneBench Runite
Race. The same executor and configuration schema must reach both without an
environment branch.

The provider is a targeted affordance, not the foundation. Every run in this
milestone uses Tinker for sampling and training with the policy/training model
`openai/gpt-oss-20b`, and the resolved identity appears in the receipt; the
policy-binding abstraction must accept a second provider without touching the
queue engine, the contract, or the catalog. Every environment rollout and reward
must be owned by a container. Direct task execution or task-specific reward calculation in
the optimizer does not satisfy the milestone.

### Relationship to the Tito data plane

`~/GitHub/training` already implements an RL data plane. `tito` serves both
public OpenAI wires and owns token capture; `tito_orchestrator` runs
TODO/INFLIGHT/DONE/TRAIN queues with a curriculum and Modal, Daytona, and local
Docker sandbox providers; `tito_train` composes an `AlgorithmPlan` whose presets
already include `cispo`, `cispo_climb`, `gspo`, `ppo`, `sao`, and
`multi_teacher_opd`; `tito.schemas` defines `InferenceCallV2`, `RewardRecordV1`,
`RolloutReceiptV2`, `GroupPin`, `BehaviorFingerprint`, `TaskSpec`, and
`MixedGroupError`.

This work is deliberately built in `synth-optimizers` anyway. The reasons are
specific: GEPA's container-independence, the `ContainerClient` and
`GepaOptimizerContract` precedent, the Tinker provider, the hosted and local
optimizer services, and the durable job store all live here, and this milestone
is defined by container independence rather than by sandbox ownership. Tito owns
its own sandboxes and assumes it launches the environment; the contract in this
document talks to a container that declares what it can do.

Accepting two planes is a real cost, and the mitigation is compatibility rather
than optimism. Where Tito has already solved a problem, this design adopts its
shape and its names so a later reconciliation is a mapping and not a rewrite:

| Concern | Tito primitive adopted here |
|---|---|
| Per-call durable record | `InferenceCallV2`: one immutable record per proxied model call, persisted before any trajectory flattening |
| Token provenance | Tokens and logprobs come from under the public wire, never from detokenize/retokenize of wire JSON |
| Turn stitching | Strict-prefix merge with branch fork and segment sealing |
| Renderer identity | `behavior_fingerprint` over the pinned renderer package, config, and tokenizer |
| Group identity | `GroupPin` plus a mixing key, and `MixedGroupError` on violation |
| Reward evidence | `RewardRecordV1` and the sealed Trace V5 digest |
| Algorithm shape | `AlgorithmPlan` dimensions, with CISPO as a preset |
| Queue shape | Bounded durable queues with a lag filter at the train boundary |

Divergence from these shapes is allowed only with a recorded reason. Anything
this document adds that Tito lacks — the declared container contract and
endpoint surface, the handshake and probe, joint-episode topology with teams and
pinned opponents, horizon quiescence and settlement, and the checkpoint lineage
catalog with policy sets and match sets — is written to be portable into Tito
rather than to depend on anything specific to this repository.

### Algorithm plan, not a CISPO engine

The executor runs an immutable plan, hashed at startup and recorded in the run
manifest and in every group pin. `algorithm = "cispo"` is a preset expansion,
never a branch in the engine. The dimensions, matching the plane Tito already
validates:

| Dimension | Meaning | CISPO preset value |
|---|---|---|
| `rollout` | episode origin, cardinality, readiness, grouping | `task_reset`, 4-8, `group_complete`, `task` |
| `scorers` | auxiliary roles evaluated alongside the actor | `old_actor` |
| `credit` | reward to per-sample advantage | `length_weighted_leave_one_out` |
| `objective` | policy loss and its clipping | `cispo`, token granularity, `eps_low = 1.0`, `eps_high = 4.0` |
| `correction` | off-policy handling | `staleness_drop` |
| `reducer` | loss aggregation | `branch_aware_root_mean` |
| `schedule` | weight mode, policy span count, packing | `sync_pin`, `policy_span_count = 1` |
| `context_views` | which conversation view each learner sees | `actor` |

One measured discrepancy is now on the record. This repository's existing
`group_advantages` is mean-centering, which is the plan's `group_mean` credit
and not the `length_weighted_leave_one_out` that Tito's `cispo` preset names.
At equal segment lengths the two differ by exactly `n/(n-1)`, with identical
signs and ordering, and the zero-advantage skip verdict is identical in every
case, which is why the existing runs behaved sensibly. The normalized legacy
variant has no exact plan equivalent: it divides by an unbiased deviation plus
`1e-6`, where the plan's standardized estimator divides by the population
deviation and returns exact zero on a tie. Reproducing the old normalized
numbers bit-for-bit needs a new credit kind in both planes; adopting the plan
definition is the recommendation, and either way that choice belongs in the
plan rather than in the executor.

Consequences the implementation must honor:

- The plan hash is part of group identity. A group whose members were produced
  under different plan hashes is rejected.
- Zero-advantage skipping, staleness bounds, and packing are plan fields, not
  executor constants.
- A second preset must be reachable by configuration alone. If adding GSPO or a
  distillation objective requires touching the queue engine, the container
  client, the handshake, or the catalog, the separation has failed and the
  refactor is part of this work rather than a follow-up.

### GEPA parity is the design rule

GEPA validates `metadata.optimizer_contracts.gepa`, discovers declared routes,
and executes through the generic `ContainerClient`. CISPO must follow the same
pattern with a stricter on-policy evidence contract:

| Concern | GEPA | Required CISPO design |
|---|---|---|
| Environment selection | Container connection | Container connection |
| Compatibility | Declared GEPA contract | Declared CISPO contract |
| Task discovery | Container taskset | Container taskset |
| Execution | Generic rollout client | Generic queued rollout client |
| Reward | Container result | Container-authoritative reward receipt |
| Trace | Optimization evidence | Exact trainable Trace V5 policy spans |
| Environment branches | None | None |

Changing from one compliant environment to another must require changing the
container connection and task selection only. Environment image IDs, targets,
harnesses, trace parsers, and reward logic are deployment/container concerns,
not CISPO algorithm configuration.

### Declared contract

The container advertises a versioned CISPO contract in `/metadata`. Routes are
declared rather than guessed or hard-coded:

```json
{
  "metadata": {
    "optimizer_contracts": {
      "cispo": {
        "version": "synth_optimizers.cispo.v1",
        "health_route": "/health",
        "capabilities_route": "/training/capabilities",
        "handshake_route": "/training/handshake",
        "taskset_route": "/taskset",
        "taskset_tasks_route": "/taskset/tasks",
        "topology_route": "/topologies/{topology_id}",
        "policy_bind_route": "/policy-configs",
        "policy_set_bind_route": "/policy-sets",
        "rollout_route": "/rollout",
        "rollout_state_route": "/rollouts/{rollout_id}",
        "rollout_events_route": "/rollouts/{rollout_id}/events",
        "rollout_renew_route": "/rollouts/{rollout_id}/renew",
        "rollout_finalize_route": "/rollouts/{rollout_id}/finalize",
        "rollout_terminate_route": "/rollouts/{rollout_id}/terminate",
        "trace_route": "/rollouts/{rollout_id}/trace",
        "artifacts_route": "/rollouts/{rollout_id}/artifacts",
        "reward_route": "/reward"
      }
    }
  }
}
```

Required surface, and the one thing each route must make true:

| Declared key | Method | Must return | Exists because |
|---|---|---|---|
| `health_route` | GET | liveness, container version, image digest | receipts name the exact build |
| `capabilities_route` | GET | hashed capability document: lifecycle, evidence, reward authority, topology, horizon, renderer profile, advertised concurrency | fail-closed preflight before any paid request |
| `handshake_route` | POST | per-clause verdicts, stated obligations, task digests, `handshake_id`, `agreement_digest`, expiry, clock skew | the container confirms it can honor this run before the run costs anything |
| `taskset_route` | GET | taskset ID, version, declared splits | discovery instead of configuration |
| `taskset_tasks_route` | GET | one row per requested ID, duplicate-free, each row naming its `topology_ref` | deterministic lookup; a family with n12/n18/n24 variants resolves topology per task, not per container |
| `topology_route` | GET | full instance roster, teams, channels, turn and actuation model, minimum viable roster | binding a topology the executor never infers |
| `policy_bind_route` | POST | `config_id`, the resolved renderer profile, and sampler readiness | on-policy binding without embedded credentials |
| `policy_set_bind_route` | POST | one atomic binding for every instance in a joint episode, trainable and pinned-opponent alike | no episode may start with a half-bound roster |
| `rollout_route` | POST | 202, `rollout_id`, lease expiry, and the accepted correlation echo | idempotent asynchronous submission |
| `rollout_state_route` | GET | state, lease expiry, per-instance liveness | polling and straggler detection |
| `rollout_events_route` | GET/SSE | ordered events with a monotone resumable cursor | cheap progress and restart recovery |
| `rollout_renew_route` | POST | new lease expiry | hour-scale episodes must not depend on an open HTTP request |
| `rollout_finalize_route` | POST | horizon-clipped state snapshot plus the quiescence attestation | the reward must describe the horizon, not whatever was still running later |
| `rollout_terminate_route` | POST | terminal cancellation, exactly once | declared straggler and cancellation policy |
| `trace_route` | GET | sealed Trace V5 inline, or a reference plus digest | trainable evidence |
| `artifacts_route` | GET | artifact inventory with digests and fetch handles | multi-hundred-megabyte recordings must not transit the job store inline |
| `reward_route` | GET/POST | receipt bound to rollout ID and trace digest, per-team channels, horizon/clipping/settlement fields | container-authoritative reward |

The executor calls only declared routes. A container may add or rename any of
them; it may not omit a mandatory one and it may not expect the executor to
guess a path that is absent from `/metadata`.

The existing hashed `training.rollout.capabilities.v1` preflight in
`src/synth_optimizers/training.py` is the starting point. Extend and wire it into
the local executor rather than creating task profiles. Persist the complete
capability response and hash in every run receipt so the exact execution
contract can be audited later.

### Startup handshake and readiness agreement

Reading a container's advertisement is discovery, not agreement. A container
that publishes a compliant contract can still be unable to honor this
particular run: its concurrency may be lower than the requested group size, its
lease TTL shorter than the horizon, its renderer profile a different build, its
taskset rows changed since the config was written, its clock skewed against the
horizon the reward will be read at. Each of those produces a run that starts
successfully and wastes provider spend before failing, or worse, trains on
evidence that was never valid.

So before the executor creates a training session or issues one paid provider
request, it completes a two-sided handshake and, where supported, one unpaid
probe episode. Nothing is negotiated after training starts.

Order, all of it before any spend:

1. `GET health_route` — build identity, container version, image digest.
2. `GET /metadata` — contract version and the declared route table.
3. `GET capabilities_route` — capability document and its content hash.
4. `POST handshake_route` — the executor's requirement document.
5. Per-clause verdict, obligations, and agreement digest come back.
6. Renderer-profile equality is checked against the profile the training
   session will use, which is resolvable locally from the model identity.
7. One probe episode exercises the full evidence path at zero provider cost.
8. Only then: create the Tinker session, save and catalog the baseline
   checkpoint, bind the policy set, and admit real attempts.

The executor sends what it needs, in full:

```json
{
  "schema_version": "cispo.handshake.v1",
  "run_id": "run_id",
  "optimizer": {"name": "synth_optimizers.cispo", "version": "0.2.20"},
  "policy": {
    "provider": "tinker",
    "model_id": "openai/gpt-oss-20b",
    "transport": "message_in_capture_out | tokens_in_tokens_out"
  },
  "renderer_profile": {"profile_id": "renderers.gpt-oss.low.v1", "config_digest": "sha256:..."},
  "requirements": ["contract.routes", "evidence.behavior_logprobs", "reward.horizon_quiescence"],
  "topology": {
    "expected_topology_id": "runite-race-4x6",
    "trainable_teams": ["terra"],
    "partial_roster": "drop_instance"
  },
  "run_plan": {
    "group_size": 8,
    "groups_per_step": 1,
    "max_execution_slots": 8,
    "maximum_policy_lag": 1,
    "target_train_updates": 10,
    "expected_horizon_seconds": 5400
  },
  "taskset": {"taskset_id": "taskset_id", "split": "train", "task_ids": ["..."]},
  "clock": {"executor_time": "RFC3339", "monotonic_source": "CLOCK_MONOTONIC"}
}
```

The container answers per clause, never with a bare boolean:

```json
{
  "schema_version": "cispo.handshake.v1",
  "handshake_id": "hs_immutable_id",
  "accepted": false,
  "clauses": [
    {"clause_id": "evidence.behavior_logprobs", "verdict": "accepted"},
    {"clause_id": "lifecycle.lease_renewal", "verdict": "accepted", "note": "max ttl 900s"},
    {"clause_id": "lifecycle.concurrency", "verdict": "degraded", "reason": "30 leases available, 8 requested per group, 2 groups in flight exceeds pool"},
    {"clause_id": "reward.horizon_quiescence", "verdict": "rejected", "reason": "cannot kill agent-authored background processes"},
    {"clause_id": "evidence.tito", "verdict": "unsupported"}
  ],
  "obligations": {
    "max_concurrency": 30,
    "lease_ttl_seconds": 900,
    "deferred_scoring": true,
    "quiescence": false,
    "settlement_window_seconds": 150,
    "horizon": {"horizon_kind": "wall_clock", "value": 5400, "time_dilation": 4.0}
  },
  "taskset_resolution": [
    {"task_id": "task_id", "content_digest": "sha256:...", "topology_ref": "runite-race-4x6"}
  ],
  "capability_hash": "sha256:...",
  "agreement_digest": "sha256:...",
  "expires_at": "RFC3339",
  "clock": {"container_time": "RFC3339", "measured_skew_seconds": 0.4}
}
```

Rules that make the handshake load-bearing rather than decorative:

- Verdicts are `accepted`, `degraded`, `rejected`, or `unsupported`, each with a
  reason. A rejected mandatory clause stops the run immediately, before session
  creation, and the receipt names the clause list. A rejected or unsupported
  optional clause records the fallback the run will use. A degraded clause is
  accepted only if the executor can satisfy it by lowering its own run plan, and
  the lowered plan is re-handshaked rather than assumed.
- `agreement_digest` binds both documents plus the capability hash, the renderer
  profile, the resolved task digests, and the obligations. Every rollout carries
  `handshake_id`, and the container must refuse any attempt whose handshake is
  absent, expired, revoked, or whose agreement digest does not match. A run
  cannot drift out from under its own agreement.
- The handshake expires. Renewal re-reads the capability document and fails
  closed on any change, exactly as the original preflight does. The container
  may revoke a handshake when it degrades; the executor must then stop admitting
  new attempts, finish or cancel in-flight ones, and re-handshake before
  resuming.
- Restart recovery re-handshakes before re-admitting queued, active, scored, or
  train-ready work, and refuses to resume against a different agreement digest.
- For a wall-clock horizon, both sides record their time and the measured skew.
  Skew beyond the declared tolerance is a rejected clause, because the horizon
  is the instant the reward is read.

The probe episode is where claims become evidence. The container declares a
`probe` policy-binding kind that returns deterministic canned generations
carrying well-formed but explicitly synthetic evidence, so the whole path can be
walked without a provider request:

- One probe attempt must exercise submit, state, events, lease renewal, trace,
  reward, finalize, and terminate, plus an idempotent resubmit of the same key
  that yields the same logical attempt, plus one cancellation.
- The executor validates shape, not quality: span field completeness, token and
  logprob length agreement, mask presence, renderer-profile stamping, prefix
  consistency across two turns, reward bound to rollout ID and trace digest,
  monotone event cursor, exactly one terminal result, and the quiescence
  attestation when quiescence was accepted.
- Probe episodes are marked non-trainable and can never enter a group or a
  batch. A container that returns probe evidence indistinguishable from real
  evidence fails conformance.
- When `probe` is unsupported, exactly one real paid canary attempt is allowed
  instead. It is also marked non-trainable, and its cost is recorded as
  handshake overhead rather than training spend.

Minimum clause set the handshake must resolve:

| Group | Clauses |
|---|---|
| Contract | `contract.version`, `contract.routes` |
| Discovery | `discovery.taskset`, `discovery.task_digests`, `discovery.topology` |
| Policy | `policy.binding_transport`, `policy.renderer_profile_match`, `policy.revision_immutability`, `policy.no_embedded_credentials`, `policy.session_scoped_origin` |
| Lifecycle | `lifecycle.idempotency`, `lifecycle.lease_renewal`, `lifecycle.cancellation`, `lifecycle.concurrency`, `lifecycle.exactly_one_terminal`, `lifecycle.pause_resume` |
| Evidence | `evidence.trace_v5`, `evidence.behavior_logprobs`, `evidence.strict_prefix`, `evidence.masking`, `evidence.wire_objects`, `evidence.artifact_reference`, `evidence.tito` |
| Reward | `reward.authority`, `reward.binding_digest`, `reward.horizon_quiescence`, `reward.settlement_window`, `reward.channels` |
| Recovery | `recovery.restart`, `recovery.stale_discard` |
| Topology | `topology.roster`, `topology.channels`, `topology.minimum_roster`, `topology.opponent_pinning` |

The canonical ids live in `src/synth_optimizers/contracts/rl_clauses.py`; that
module is authoritative and this table follows it. Every route in the surface
above must be *declared* even where the corresponding behavior is an optional
clause: route presence and behavior support are separate questions, and a
container that cannot serve artifacts by reference still declares the route it
would serve them on.

The clause list is generic. No clause names a task, a harness, or an
environment, and a container may accept every clause without knowing which
optimizer asked.

### Mandatory container requirements

The executor must fail during preflight, before any paid sampling or training,
unless all mandatory requirements are present.

#### Task discovery

- Versioned taskset and task lookup routes.
- Stable, non-empty task identity for every row.
- Declared train and evaluation splits.
- Deterministic lookup by task ID.
- Duplicate-free responses with one returned row for each requested ID.

The executor may optionally verify an expected task or task-family identity,
but it must not contain a hard-coded allowlist.

#### Rollout lifecycle

- Asynchronous submission and polling.
- Idempotent rollout/attempt IDs.
- Cancellation and terminal failure reporting.
- Heartbeats or expiring renewable leases.
- Advertised maximum concurrency.
- Preservation of opaque correlation metadata supplied by CISPO:
  `run_id`, `group_id`, `sample_index`, `seed`, `policy_revision`, and, for a
  joint episode, `agent_instance_id`, `team_id`, and `policy_set_revision_id`.
- Exactly one terminal result per accepted attempt: episode, failure, or
  cancellation.
- A declared episode horizon and a declared wall-clock ceiling. An episode that
  runs on real time rather than a step budget must advertise its horizon
  (`horizon_kind = "wall_clock" | "steps" | "env_ticks"`), its value, and any
  environment time dilation, so leases and queue timeouts are derived rather
  than guessed.
- Leases sized for the advertised horizon, renewed by heartbeat. An episode
  measured in hours must not depend on an HTTP request staying open.
- Quiescence at the horizon before any scoring. The container must stop every
  agent-authored background process, loop, or scheduled program it allowed the
  policy to create, and must attest that no environment mutation occurred
  between the horizon and the scored read. An environment that cannot quiesce
  must instead expose a horizon-clipped state snapshot taken at the horizon.

The container does not need to understand group advantages. It only needs to
round-trip correlation fields and execute each requested attempt exactly once.

#### Policy binding

- Accept a versioned external sampler binding or another explicitly advertised
  on-policy transport.
- Accept a session-scoped sampler origin rather than a global one. The bound
  base URL carries the per-attempt request identity in its path, so stitching is
  a URL parse and a leaked credential cannot cross rollouts. The credential
  still names the group, sample, wire, policy kind, and pinned policy revision;
  the harness never sees those fields and never chooses them.
- Support the Tinker sampler connection without embedding raw credentials in a
  rollout request.
- Record the behavior-policy revision on every trainable model call.
- Make a policy revision dispatchable only after the corresponding sampler is
  ready.
- Keep a rollout's behavior-policy identity immutable after admission.
- Bind every agent instance in a joint episode individually, so one episode may
  carry several distinct behavior revisions at once, each pinned per instance.
- Accept non-trainable opponent bindings: an instance may be driven by a frozen
  checkpoint, an external provider model, or a scripted baseline. The container
  must report which instances are trainable and which are not, and must return
  trainable evidence only for the trainable ones while still recording the
  others' identity for reproducibility.

The container owns its policy harness. The CISPO executor must never select
`react`, `mini_swe`, Harbor, Craftax, or another harness by name.

#### Group identity and mixing rejection

A group is the unit of comparison, so anything that changes what a sample means
must be identical across its members. Every group carries an immutable pin, and
the executor rejects a group whose members disagree on any pinned field. This is
Tito's `GroupPin` mixing key extended for the container contract and for joint
episodes:

```json
{
  "group_pin": {
    "group_id": "group_id",
    "run_id": "run_id",
    "algorithm_plan_hash": "sha256:...",
    "behavior_fingerprint": "sha256:...",
    "policy_revision": 17,
    "policy_set_revision_id": "party-set-20",
    "match_set_revision_id": "match-set-0007",
    "wire_api": "chat_completions | responses",
    "policy_kind": "declared by the container",
    "model_family": "gpt_oss",
    "sampling_transport": "message_in_capture_out | tokens_in_tokens_out",
    "container_image_digest": "sha256:...",
    "container_contract_hash": "sha256:...",
    "handshake_agreement_digest": "sha256:...",
    "topology_id": "runite-race-4x6",
    "task_family": "seed/scenario family",
    "cardinality": 8
  }
}
```

Mixing any of those fields inside one group is a rejected group, not a warning
and not a silently averaged batch. `policy_span_count` is 1 for this milestone:
a single group may not straddle two published policy revisions. Episodes from
different containers, different images, different wires, or different plans are
different datasets that happen to share an optimizer.

#### Token authority: renderer profile, TiTo, and logprob validity

Exactly one party renders tokens for a given policy revision, and the trainer
must be able to prove it was the same renderer that produced the training
tokens. This is the single highest-risk seam in the whole design: a renderer
disagreement produces a run that looks healthy, trains on plausible tokens, and
optimizes nothing.

The renderer profile is a first-class pinned identity, not a version string:

```json
{
  "renderer_profile": {
    "profile_id": "renderers.gpt-oss.low.v1",
    "package": "renderers",
    "package_version": "0.1.11",
    "config_digest": "sha256:...",
    "tokenizer_id": "openai/gpt-oss-20b",
    "tokenizer_digest": "sha256:...",
    "stop_token_ids": [200002, 199999],
    "modalities": ["text"],
    "add_generation_prompt": true
  }
}
```

It appears in the capability response, in the policy-binding response, in the
checkpoint record's `compatibility` block, and on every trainable span. The
binding's profile must equal the training session's profile. A mismatch is a
preflight failure before any paid request, and a mismatch discovered at
evaluation time is an evidence failure rather than a warning.

The wire is pinned, and it is not always chat completions. Production coding
agents speak `POST /v1/responses`; ReAct-style and mini-SWE-style harnesses
speak `POST /v1/chat/completions`. Both are first-class, the container declares
which it uses, and the group pin carries it. Flattening Responses output items
into chat messages and training on the result is prohibited, as is presenting a
chat-completions trajectory as a Responses distribution: they are two datasets,
not one. The original wire objects are persisted alongside the token evidence,
because the wire object is the semantic record and the tokens are the training
record; neither substitutes for the other.

Two transports are allowed, and the baseline is the one that keeps containers
out of the tokenizer business:

- **Mandatory baseline — message-in, capture-out.** The container's harness
  makes an ordinary chat-completions call against its bound sampler endpoint.
  The renderer-owning side renders messages to token IDs, samples, and returns
  the assistant text together with exact prompt token IDs, generation token
  IDs, per-token behavior logprobs, the sampled mask, the renderer profile, and
  the behavior policy revision. Existing harnesses need no token awareness at
  all, which is why mini-SWE, OpenCode-style, and ReAct harnesses can be made
  conformant without being rewritten.
- **Optional declared capability — `tokens_in_tokens_out`.** The container
  sends `prompt_token_ids` and receives `generation_token_ids` with logprobs;
  no text round-trip is authoritative. Required for any harness that already
  speaks tokens, and required to declare the identical renderer profile, which
  the executor verifies for equality. TiTo must never become the route by which
  a second renderer enters the run.

Under either transport the container declares which it uses, and the receipt
records it. A container that declares TiTo and returns text-derived tokens, or
declares the baseline and re-tokenizes locally, fails conformance.

Multi-turn stitching follows the strict-prefix rule, and it is checkable.
Two calls concatenate into one trainer sequence only when the next request is a
byte-for-byte token prefix of the previous prompt-plus-generation. Tool loops
normally stitch. Context compaction, summarization, chat-template rewrite,
subagent dispatch, or any other history rewrite does not: it forks a branch with
`branch_id` and `parent_branch_id`, seals the prior segment, and the next turn
is a full render. Never retokenize new text onto old IDs.

When a fork keeps only the prefix of an earlier assistant generation, those
tokens remain in the prompt as context and are entirely loss-masked. Once the
original sample is severed they are not on-policy under the reconstructed
prompt, however plausible they look. An unexplained prefix divergence — a
divergence with no branch record and no declared compaction — is an evidence
failure.

Behavior logprobs are accepted only when all of the following hold:

- Length equals the generated token count exactly.
- No value is a provider sentinel. `-9999.0` is both a missing-evidence marker
  and a lower-bound clamp in the vLLM path, so its presence can never prove a
  real logprob was returned; the same applies to NaN and infinities.
- The vector is not identically zero across a span.
- They came from the pinned behavior policy at sampling time, in the same
  forward pass that produced the tokens. A later recomputation is prohibited
  even when it would be numerically close.
- The span records its finish reason — renderer stop token, length cap, or
  container abort — and the stop token IDs the renderer declared. A
  length-truncated tail has different training semantics from a stopped one and
  must not be silently pooled with it.

Prompt-budget behavior is declared, not improvised. The container or the
renderer-owning side declares one policy for an overlong rendered prompt —
refuse the attempt, truncate under a stated rule, or compact under a stated
rule — and the span carries the resulting provenance. Silently dropping middle
messages without recording it is prohibited; the existing TBLite gateway's
compaction record is the minimum bar.

Loss masks derive from the renderer's own sampled mask intersected with policy
authorship, and the convention is fixed rather than per-container:

| Token class | Mask |
|---|---|
| system, developer, user, and template structure | 0 |
| tool observations and environment steps | 0 |
| harness-generated or deterministic compaction | 0 |
| another agent instance's or an opponent's message tokens | 0 |
| verifier, rubric-judge, and reward-model text | 0 |
| assistant text sampled by the policy | 1 |
| assistant reasoning sampled by the policy | 1 |
| assistant-generated function-call syntax | 1 |
| assistant-generated function arguments | 1 |

The distinction that matters most: if the policy itself samples a summarization
or compaction call, those generated tokens are trainable. If the harness rewrites
context deterministically, they are not. When a renderer distinguishes content
tokens from structural ones, the trace records both masks so a later reader can
tell which convention a batch used. For multimodal profiles the placeholder
ranges must be recorded so masks remain exact around non-text spans.

#### Joint-episode topology, teams, and opponents

The number of agent instances, their roles, their team membership, and the
reward relation between teams are container-declared facts. The executor binds
what is declared and must not infer topology from a task name, an agent count,
or a role string.

The container declares, in its capability response:

```json
{
  "topology": {
    "topology_id": "runite-race-4x6",
    "turn_model": "sequential | concurrent_realtime",
    "actuation_model": "direct_action | deferred_program",
    "agent_instances": [
      {"agent_instance_id": "terra_c", "role_id": "miner", "policy_type_id": "miner", "team_id": "terra", "trainable": true},
      {"agent_instance_id": "terra_f", "role_id": "scout", "policy_type_id": "scout", "team_id": "terra", "trainable": true},
      {"agent_instance_id": "gemini37_a", "role_id": "miner", "policy_type_id": "opponent", "team_id": "gemini37", "trainable": false}
    ],
    "teams": [{"team_id": "terra", "trainable": true}, {"team_id": "gemini37", "trainable": false}],
    "reward_relation": "cooperative | competitive_rank | competitive_margin | mixed",
    "communication_channels": [
      {"channel_id": "team_pm", "scope": "intra_team", "trainable_for_author": true},
      {"channel_id": "public", "scope": "cross_team", "trainable_for_author": true}
    ],
    "horizon": {"horizon_kind": "wall_clock", "value": 5400, "time_dilation": 4.0}
  }
}
```

A concurrent real-time topology is a first-class case, not a degenerate
sequential one. When `turn_model = "concurrent_realtime"`:

- Every agent instance runs its own independent call stream, and instances
  sample simultaneously. The executor must not serialize a joint episode into a
  global turn order.
- Trace evidence orders actions by environment tick or environment timestamp,
  not by a turn index, and per-instance streams must be individually monotone.
- No single environment effect may be attributed to two agent instances.

When `actuation_model = "deferred_program"` — the policy emits code, a script,
or a standing loop whose effects continue after the sampling call returns — the
following are mandatory:

- Each trainable span declares the interval of environment effect it authored,
  so reward is attributable to the span that caused it.
- Programs authored by the policy are owned by the container and are killed at
  the horizon by the quiescence requirement above. A program still mutating the
  environment after the horizon is an evidence failure, not extra reward.
- A span whose authored effects are known to extend past the horizon and are
  neither quiesced nor clipped must be refused, not masked into the batch.

Communication between agent instances is observation, never free reward:

- Another instance's message tokens are never trainable for a receiving
  instance, on any channel, intra-team or cross-team.
- Non-trainable opponent instances contribute no trainable spans at all, while
  their identity, model, and revision are still recorded.
- The container declares its channels, and the executor validates channel
  completeness against the declaration. A silently dropped channel is an
  evidence failure: a topology whose declared cross-team channel returns no
  messages for an entire episode must fail rather than train on a truncated
  observation history.

Competitive reward relations change what a group is:

- A group may only contain joint episodes that share the same topology ID, the
  same scenario/seed family, and the same immutable opponent set. Episodes with
  different opponents are not comparable samples and must not share a group.
- `competitive_rank` reward is a declared ordering with a declared tie policy.
  The container returns the raw per-team measure and the resolved rank; the
  executor computes group-relative advantages from the declared channel it was
  configured to optimize, and records which channel that was.
- The trainee's advantage is computed across episodes, never across teams
  inside one episode. A within-episode comparison between a trainee team and a
  frozen opponent team is not a CISPO group.

Partial rosters must be an explicit, declared, receipted policy rather than an
accident. A joint episode with twenty-four instances will lose instances. The
container declares a minimum viable roster per team and per topology, and the
run configuration declares the disposition:

- `refuse` — any missing instance fails the episode.
- `drop_instance` — the episode trains on surviving instances; the dead
  instance's absence, death time, and last live tick are recorded, and its
  team's reward is still attributed.
- `refuse_team` — a team below its minimum viable roster is excluded while other
  teams' episodes remain valid.

A missing per-instance trajectory that is not covered by the declared
disposition remains a terminal evidence failure. Silently training on
twenty-three of twenty-four instances is prohibited.

#### Checkpoint catalog and policy lineage

Every materialized model checkpoint must be registered as a durable,
addressable artifact. This includes the imported baseline, every component
checkpoint produced at a published training update/round, intermediate
checkpoints retained by policy, and staged or orphaned checkpoints produced by
a partially failed multi-policy update. A successfully materialized checkpoint
may not exist only as a path printed in a log.

The catalog must distinguish Tinker's two artifact roles explicitly:

- `sampler_weights` is the immutable artifact used to create a sampling client
  for rollout or evaluation.
- `training_state` is the resumable LoRA/training artifact produced by
  `save_state` and used to continue training.

A sampler-weight path must never be presented as resumable training weights,
and a training-state path must not be assumed to be directly sampleable. Every
published policy revision must resolve to the appropriate sampler artifact;
every resumable revision must separately resolve to its training-state
artifact when one exists.

Use an append-only catalog record equivalent to:

```json
{
  "schema_version": "cispo.checkpoint.v1",
  "checkpoint_id": "ckpt_immutable_id",
  "run_id": "run_id",
  "update_id": "update_0004",
  "train_call_ids": ["provider_train_request_id"],
  "parameter_group_id": "elf_policy",
  "policy_type_ids": ["elf"],
  "policy_revision_id": "elf_policy@4",
  "parent_checkpoint_id": "elf_policy@3",
  "base_model": "openai/gpt-oss-20b",
  "artifacts": {
    "sampler_weights": {"ref": "provider_ref", "digest": "sha256:..."},
    "training_state": {"ref": "provider_ref", "digest": "sha256:..."}
  },
  "publication_status": "staged|published|orphaned|superseded",
  "policy_set_revision_ids": ["party-set-20"],
  "training_evidence": {
    "groups": ["group_id"],
    "examples": 0,
    "tokens": 0,
    "provider_cost": 0.0
  },
  "compatibility": {
    "renderer_profile": "renderer_id",
    "tokenizer": "tokenizer_id",
    "container_contract_hash": "sha256:..."
  },
  "created_at": "RFC3339 timestamp"
}
```

A competitive topology needs one more immutable record. A match-set revision
pins every instance binding in an episode: the trainee policy-set revision plus
each non-trainable opponent's frozen checkpoint ID, external model identity, or
scripted-baseline identity. Rollouts, groups, and evaluations reference the
match-set revision, because a reward earned against one opponent set is not
comparable to a reward earned against another. Resolving an opponent as
`latest`, or as a provider model alias that can change under the run, is
prohibited for the same reason a trainee `latest` is prohibited.

A published revision stays loaded while anything is still sampling from it. A
revision may be retired only when its active-attempt count reaches zero;
unloading a revision an in-flight attempt is still using would make that
attempt's behavior identity unverifiable and its evidence unusable. The catalog
records load, ready, active-count, and retire transitions alongside the
checkpoint record, and publication marks a revision ready only after its sampler
artifact is materialized and health-checked.

Checkpoint records are immutable. Policy-set records reference component
`checkpoint_id` values, and evaluation records reference the exact
`checkpoint_id`, `policy_set_revision_id`, or `match_set_revision_id` they
evaluated. Later evaluations
are append-only relations rather than mutations of the original checkpoint
record.

The run manifest must list every checkpoint it created, while the checkpoint
catalog provides the reverse lookup from checkpoint to producing run, update,
policy type, parameter group, parent, provider requests, and policy-set
publications. The evaluation entrypoint must accept a stable checkpoint or
policy-set ID and resolve its immutable sampler artifact without requiring a
user to copy a provider path from logs. Human aliases such as `baseline`,
`latest-published`, and `best:<metric>` are optional mutable pointers and must
resolve to immutable IDs in the evaluation receipt.

#### Trainable Trace V5 evidence

Every successful rollout must seal a Trace V5 document with a common training
view equivalent to:

```text
TrainableEpisode
  rollout_id
  task_id
  seed
  policy_revision
  outcome
    reward
    objective_scores
    terminal_status
  segments[]
    model_call_id
    policy_revision
    prompt_token_ids[]
    response_token_ids[]
    behavior_logprobs[]
    loss_mask[]
    compaction/provenance metadata
  usage
    calls
    prompt_tokens
    completion_tokens
    sampling_seconds
    tokens_per_second
    provider_request_ids[]
```

For every trainable segment:

- Token and logprob lengths must agree.
- Logprobs must come from the pinned behavior policy, not a later forward pass.
- Loss masks must exclude environment, tool, verifier, rubric-judge, and other
  non-policy tokens.
- The trace must identify the renderer/profile used to produce the tokens.
- Mixed policy revisions within an episode must either be represented exactly
  per segment or refused by the selected staleness policy.
- Missing or malformed trainable evidence is a terminal evidence failure, not a
  zero-reward trajectory.
- In a joint episode every segment additionally carries `agent_instance_id`,
  `role_id`, `policy_type_id`, `parameter_group_id`, `team_id`, the component
  policy revision, and the policy-set revision, plus the environment tick or
  timestamp at which the resulting action took effect.
- Spans authored by another agent instance, an opponent instance, a verifier, or
  a rubric judge are recorded as untrainable context with their author
  identified. Foreign authorship must be explicit, not implied by a zero mask.
- A trace bundle that is too large to inline is stored by reference with a
  digest, and the reference must resolve for the retention life of the run. A
  twenty-four-box episode producing gigabytes of recordings must not force the
  trace through the job store inline.

The shared Prime renderer/Trace V5 layer performs normalization. The optimizer
must not contain environment-specific event extraction.

#### Reward authority

- Versioned, container-authoritative reward or objective-score receipt.
- Reward bound to the exact rollout ID and sealed trace digest.
- Finite numeric result; zero must remain distinguishable from absent reward.
- Idempotent final scoring and explicit missing-evidence failure.
- Stable evaluation-plan/reward-calculator identity.
- Optional deferred or asynchronous scoring advertised as a capability rather
  than inferred from the task name.
- Scoring at the declared horizon. The receipt records the horizon, the actual
  time of the scored read, whether horizon clipping was applied, and the
  quiescence attestation. Environment activity after the horizon must never
  reach the reward. Scoring a run late and crediting post-horizon progress is a
  reward-integrity failure that silently rewrites the ranking.
- A declared settlement window for environments whose scored state lags the
  authoritative state. When a container's readable state trails its own writes,
  the reward contract states the lag bound and the receipt states which
  post-horizon settlement was credited, so two reads of the same episode cannot
  produce two rewards.
- For a competitive topology, one reward channel per team, plus the resolved
  relation. Absolute measure and rank are both recorded; the optimized channel
  is named in the receipt.

Craftax may finalize native environment signals quickly. HealthBench may run a
container-owned rubric judge. TBLite may run a deferred verifier against a
workspace artifact. These differences remain behind the same reward contract.

If HealthBench uses a distinct judge model, that judge must be identified in
the reward receipt and its spans must be excluded from policy training. The
policy being sampled and trained for this milestone remains
`openai/gpt-oss-20b`.

#### Queue and recovery safety

- Advertised concurrency must meet the requested minimum.
- Retrying an idempotency key cannot create a second logical attempt.
- Active work must have a recoverable lease.
- Stale queued work can be discarded before execution.
- Staleness is checked again when a complete group leaves the train-ready
  queue; this dequeue check is the hard training guarantee.
- Retained workspaces, sessions, and artifacts have explicit release or
  retention semantics.
- Leases, heartbeats, and queue timeouts are derived from the container's
  advertised horizon. An hour-scale joint episode is a normal case; a queue that
  assumes minute-scale attempts will declare healthy work dead.
- A declared straggler policy: an episode exceeding its horizon plus a declared
  grace period is cancelled and replaced, and the replacement is recorded as
  such rather than silently changing group membership.
- Post-horizon quiescence and artifact collection are part of the attempt's
  lease, not work performed after the attempt is considered complete.

### Lifecycle controls and offline parity

Synth Style 02 asks for online/offline parity with pause and resume, and treats
lifecycle controls as first-class rather than as operational afterthoughts. A
long-horizon RL run needs them for ordinary reasons: provider quota, a container
redeploy, a cost ceiling reached mid-round, an operator who wants to inspect a
round before paying for the next one.

Four controls, each first-class in the API and each with defined semantics at
every queue boundary:

- **Pause.** Stop admitting new attempts. In-flight attempts keep their leases
  and run to a terminal result; scoring, validation, and catalog registration
  continue. Training does not start a new step. A paused run is a legal resting
  state, not a degraded one.
- **Drain.** Pause, then let every in-flight attempt finish and every complete
  group train, then stop. Partial groups are recorded as abandoned with their
  membership, so their cost is attributable.
- **Resume.** Re-handshake first, verify the agreement digest, re-verify the
  capability hash, then re-admit work. Resuming under a changed contract,
  container image, plan hash, or renderer fingerprint is refused; that is a new
  run with a lineage edge to the old one, not a continuation.
- **Stop.** Cancel in-flight attempts through the declared terminate route,
  release leases and workspaces, and leave the catalog and receipts complete.

Offline parity is the other half, and it is what makes algorithm iteration
affordable. Every per-call record, reward receipt, and sealed trace is durable,
so the same plan must be runnable against stored evidence with no live container
and no provider sampling:

- **Replay mode** consumes stored episodes by run, group, or selector and
  executes the plan's credit, objective, correction, and reducer dimensions
  exactly as an online run would. Same code path, no rollout production.
- Replay is explicitly off-policy. It may not publish a policy revision that is
  later presented as an on-policy result, and every replay-derived update is
  marked in the catalog with its source run and the staleness it accepted.
- A replay of an online run's stored evidence must reproduce that run's
  advantages and batch composition bit-for-bit, given the same plan hash. This
  is the cheapest regression test the system has, and it should gate changes to
  credit, reducer, and masking code.
- Development and production differ only in the container connection, the
  provider binding, and the spend ceiling. No plan field, contract clause, or
  evidence rule may be development-only.

### Optional capabilities and fallback behavior

The following improve throughput but are not universal compatibility
requirements:

- Deferred scoring.
- Durable agent-to-verifier artifact handoff.
- Asynchronous score execution.
- SSE or WebSocket streaming.
- Environment checkpoint/resume.
- Live frames or provisional rewards.

A compliant container without deferred scoring remains usable. Its rollout
returns a sealed trace and materialized reward together; the score queue then
validates and admits the receipt. A staged container may instead transition
`running → awaiting_score → completed`, allowing rollout capacity to be released
before scoring begins.

### Generic configuration surface

The same configuration schema is used for all containers:

```toml
schema_version = "cispo.container.v1"

[container]
url = "http://127.0.0.1:8080"
headers = {}
# auth_bearer_env = "CONTAINER_TOKEN"

[taskset]
train_split = "train"
evaluation_split = "heldout"
train_ids = []
evaluation_ids = []

[model]
provider = "tinker"
id = "openai/gpt-oss-20b"
rank = 8

[plan]
# CISPO is a preset expansion, hashed into the run manifest and every group pin.
preset = "cispo"
# Dimension overrides are explicit; there is no algorithm branch in the engine.
# credit = "length_weighted_leave_one_out_standardized"
# correction = "staleness_drop"
# reducer = "branch_aware_root_mean"

[lifecycle]
# Pause, drain, resume and stop are first-class controls, not signals.
resume_requires_rehandshake = true

[offline]
# Replay the same plan against stored evidence: no container, no provider spend.
mode = "off"           # off | replay
# source_run_ids = []

[cispo]
group_size = 8
groups_per_step = 1
target_train_updates = 1
maximum_sampled_groups = 5
eps_clip = 1.0
eps_clip_high = 4.0

[pipeline]
mode = "async_queued"
max_execution_slots = 8
rollout_queue_capacity = 16
score_queue_capacity = 8
train_ready_capacity = 2
maximum_policy_lag = 1
rollout_retries = 2
score_retries = 1

[topology]
# Container-declared topology accepted by ID; the executor never defines one.
expected_topology_id = "runite-race-4x6"
trainable_teams = ["terra"]
partial_roster = "drop_instance"   # refuse | drop_instance | refuse_team
same_policy_reduction = "token_weighted_mean"

[topology.policy_types]
# Declared policy type -> parameter group. Roles come from the container.
miner = "miner_policy"
scout = "scout_policy"

[opponents]
# Every non-trainable instance resolves to an immutable, pinned identity.
match_set_revision = "match-set-0007"
allow_alias_resolution = false

[reward]
optimized_channel = "team_rank"    # declared by the container's reward contract
horizon_grace_seconds = 120

[evaluation]
paired = true
baseline_samples = 4
trained_samples = 4
fixed_match_set = true

[artifacts]
checkpoint_every_published_update = true
retain_training_state = true
catalog = "runs/checkpoints.jsonl"
```

There must be no environment, harness, renderer-selection, or reward-mode field
in the CISPO algorithm section. A generic launcher may resolve a local command,
image, or pool target into the container URL before execution, just as GEPA
does; the executor itself receives only the connection and declared contract.

### Interaction map

#### Ownership stack

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ synth-optimizers · CISPO executor + queue engine                             │
│   knows: declared contract, queues, advantages, Tinker sessions, catalog     │
│   never knows: task, harness, renderer, reward rule, image, topology shape   │
├──────────────────────────────────────────────────────────────────────────────┤
│ rust/synth_optimizer_platform · ContainerClient + CispoOptimizerContract     │
│   validates contract version and every declared absolute route               │
│   typed TrainableEpisode / reward receipt / capability hash                  │
├──────────────────────────────────────────────────────────────────────────────┤
│ synth-containers · shared container platform layer                           │
│   /metadata  /training/capabilities  /taskset  /policy-configs  /rollout     │
│   /rollouts/{id}[/events|/terminate|/trace]  /reward                         │
│   policy binding · lifecycle · leases · trace sealing · quiescence · scoring │
├──────────────────────────────────────────────────────────────────────────────┤
│ environment implementation — one per family, declares its own capabilities   │
│  banking77 │ healthbench │ craftax │ tblite │ dungeongrid_gold │ runite race │
├──────────────────────────────────────────────────────────────────────────────┤
│ execution substrate                                                          │
│  in-process │ harbor leases + docker compose │ rust engine │ 24 agent boxes  │
└──────────────────────────────────────────────────────────────────────────────┘
     ▲                                                                    ▲
     │ the only run-to-run difference is the container URL + task ids     │
     └────────────────────────────────────────────────────────────────────┘
```

#### One attempt, end to end

```text
 CISPO executor         container platform      sampler gateway      Tinker
      │                        │                      │                │
 (0)  │─ GET /health ─────────▶│                      │                │
      │─ GET /metadata ───────▶│                      │                │
      │◀ contract + routes ────│                      │                │
      │─ GET /training/…  ────▶│                      │                │
      │◀ capabilities+topology │                      │                │
      │  hash & persist        │                      │                │
      │─ GET /taskset/tasks ──▶│                      │                │
      │◀ task rows + digests ──│                      │                │
      │                        │                      │                │
 (0a) │─ POST /training/hand- ▶│ evaluate every       │                │
      │  shake {requirements,  │ clause against its   │                │
      │   renderer_profile,    │ own build            │                │
      │   run_plan, topology,  │                      │                │
      │   task_ids, clock}     │                      │                │
      │◀ clause verdicts,      │ accepted / degraded  │                │
      │  obligations, task     │ rejected / unsupport │                │
      │  digests, handshake_id,│                      │                │
      │  agreement_digest,     │                      │                │
      │  expiry, clock skew    │                      │                │
      │                        │                      │                │
      │  mandatory clause rejected ⇒ STOP here. no session, no spend.  │
      │  degraded ⇒ lower run plan, re-handshake, never assume.        │
      │  renderer profile ≠ trainer profile ⇒ STOP here.               │
      │                        │                      │                │
 (0b) │─ POST /rollout (probe)▶│ probe binding:       │                │
      │   binding_kind=probe   │ deterministic canned │                │
      │◀ full path: state,     │ generations, marked  │                │
      │  events, renew, trace, │ synthetic, never     │                │
      │  reward, finalize,     │ trainable            │                │
      │  terminate, replay,    │                      │                │
      │  cancel                │                      │                │
      │  validate SHAPE only: field completeness, lengths, masks,      │
      │  prefix consistency, reward↔digest binding, one terminal       │
      │                        │                      │                │
 (1)  │─ create session ───────┼──────────────────────┼───────────────▶│
      │─ save sampler_weights ─┼──────────────────────┼───────────────▶│
      │◀ ref + digest ─────────┼──────────────────────┼────────────────│
      │  catalog.register(ckpt_0, status=published)   │                │
      │─ bind revision ──────────────────────────────▶│ route=rev, im- │
      │                        │                      │ mutable once   │
      │                        │                      │ registered     │
 (2)  │─ POST /policy-configs ▶│ external sampler     │                │
      │◀ config_id ────────────│ binding, no creds    │                │
      │─ POST /rollout ───────▶│ idempotency_key,     │                │
      │   {run,group,sample,   │ correlation kept     │                │
      │    seed,revision,      │                      │                │
      │    agent_instance,     │                      │                │
      │    team,policy_set}    │                      │                │
      │◀ 202 rollout_id + lease│                      │                │
      │                        │                      │                │
 (3)  │                        │ harness step ───────▶│ sample(rev)───▶│
      │                        │                      │◀ tokens+logp ──│
      │                        │◀ text + capture ─────│ span: ids,     │
      │                        │  (loop per step)     │ logprobs, mask,│
      │                        │                      │ revision, tick │
      │                        │                      │                │
 (4)  │─ GET /rollouts/{id} ──▶│ running →            │                │
      │◀ state + heartbeat ────│ awaiting_score →     │                │
      │  (poll / events / SSE) │ completed            │                │
      │                        │ horizon: quiesce all │                │
      │                        │ policy-authored jobs │                │
      │                        │                      │                │
 (5)  │─ GET /…/trace ────────▶│ sealed Trace V5      │                │
      │◀ episode or ref+digest │ (by reference when   │                │
      │─ GET /reward ─────────▶│  large)              │                │
      │◀ receipt: reward bound │ container-authorit-  │                │
      │  to rollout+digest,    │ ative; zero ≠ absent │                │
      │  horizon, clipping,    │                      │                │
      │  settlement, quiescence│                      │                │
      │                        │                      │                │
 (6)  │  validate → admit to scored queue → complete group             │
      │  advantages → recheck staleness at train dequeue               │
      │─ train_step(cispo.slime.v1) ──────────────────┼───────────────▶│
      │─ save_weights (sampler) / save_state (resume) ┼───────────────▶│
      │◀ refs + digests ──────────────────────────────┼────────────────│
      │  catalog.register(ckpt_n) → publish policy set → bind new route│
      └────────────────────────── next group samples on rev n ─────────┘
```

#### Queue engine — where pipeline-RL lives

```text
        task rows                                     ┌───────────────────┐
            │                                         │ policy registry   │
            ▼                                         │ rev n published   │
 ┌─────────────────────┐  per-sample dispatch         │ rev n+1 staged    │
 │ ROLLOUT QUEUE       │  (never whole groups)        └─────────┬─────────┘
 │ bounded, durable    │                                        │ bind
 │ lease + heartbeat   │──┐                                     ▼
 └─────────────────────┘  │                          ┌────────────────────┐
            ▲             │  POST /rollout           │ sampler gateway    │
            │ retry same  ├─────────────────────────▶│ route = revision   │
            │ idempotency │                          │ immutable per route│
            │ key ⇒ same  │                          └─────────┬──────────┘
            │ attempt     │                                    │ sample
 ┌──────────┴──────────┐  │  container attempts                ▼
 │ open groups (many)  │  │  ┌────────┬────────┬────────┐  ┌────────┐
 │ prefer completing   │◀─┘  │ a0     │ a1     │ …  aN  │  │ Tinker │
 │ the oldest open one │     └───┬────┴───┬────┴───┬────┘  └────────┘
 └──────────┬──────────┘         │        │        │
            │                    ▼        ▼        ▼
            │              ┌──────────────────────────┐
            │              │ SCORE QUEUE              │  deferred verifier,
            │              │ awaiting_score attempts  │  rubric judge, or
            │              │ (capability, not a guess)│  native env reward
            │              └────────────┬─────────────┘
            │                           ▼
            │              ┌──────────────────────────┐
            │              │ SCORED-RESULT QUEUE      │  validate: tokens =
            │              │ episode+receipt admitted │  logprobs, masks,
            │              │ absent reward ⇒ failure  │  revision, digest
            │              └────────────┬─────────────┘
            │                           ▼
            │              ┌──────────────────────────┐
            └─────────────▶│ TRAIN-READY QUEUE        │ complete groups only
   replace skipped or      │ bounded (capacity = lag) │ same topology, seed
   stale groups            └────────────┬─────────────┘ family, match set
                                         │
                        ┌────────────────▼─────────────────┐
                        │ DEQUEUE GATE — hard guarantee    │
                        │ recheck staleness now, not at    │
                        │ submit; stale ⇒ discard/recycle  │
                        └────────────────┬─────────────────┘
                                         ▼
                        ┌──────────────────────────────────┐
                        │ group advantages → packed batch  │
                        │ per parameter group              │
                        └────────────────┬─────────────────┘
                                         ▼
                        ┌──────────────────────────────────┐
                        │ Tinker train_step → save → cata- │
                        │ log → atomic policy-set publish  │
                        └──────────────────────────────────┘

  rollout production never stops while scoring, training, checkpointing and
  publication run. queue_depth-1 ≤ max_staleness is enforced at startup.
```

#### Joint episode → parameter groups → atomic publish

```text
 container declares topology (executor binds it, never names it)

   agent instance     role      policy type    parameter group     trainable
   ────────────────   ───────   ────────────   ────────────────    ─────────
   terra_a…terra_e    miner     miner          miner_policy         yes
   terra_f            scout     scout          scout_policy         yes
   gemini37_*         miner     opponent       —  (pinned ckpt)     no
   grok46_*  opus5_*  …         opponent       —  (pinned ckpt)     no

 ONE JOINT EPISODE                                pinned by ONE match set
 ┌──────────────────────────────────────────┐     ┌───────────────────────┐
 │ concurrent_realtime, horizon 5400 s @ 4× │     │ trainee policy set    │
 │                                          │     │  miner_policy@n       │
 │  terra_a ▸▸▸▸▸ spans (tick intervals)    │     │  scout_policy@n       │
 │  terra_b ▸▸▸▸▸                           │     │ opponents (frozen)    │
 │  …                                       │     │  gemini37 = ckpt_x    │
 │  terra_f ▸▸ chat-heavy, ore-light        │     │  grok46   = ckpt_y    │
 │  gemini37_a ▪▪▪ untrainable context      │     │  opus5    = ckpt_z    │
 │  public/PM channels ▪▪▪ received = ctx   │     └───────────────────────┘
 └───────────────┬──────────────────────────┘
                 │ one team measure + resolved rank, read AT the horizon
                 ▼
        ┌────────────────────┐
        │ group-relative     │  group = episodes sharing topology + seed
        │ team advantage     │  family + match set. never a within-episode
        └─────────┬──────────┘  comparison between teams.
                  │ fan out, receipted reduction (not token-count accident)
        ┌─────────┴─────────┐
        ▼                   ▼
 ┌─────────────┐     ┌─────────────┐
 │ miner batch │     │ scout batch │   spans filtered by parameter_group_id
 │ a…e spans   │     │ f spans     │   foreign-authored spans excluded
 └──────┬──────┘     └──────┬──────┘
        ▼                   ▼
 ┌─────────────┐     ┌─────────────┐
 │ Tinker      │     │ Tinker      │   may train concurrently
 │ session M   │     │ session S   │
 └──────┬──────┘     └──────┬──────┘
        │ staged             │ staged
        └─────────┬─────────┘
                  ▼
        ┌──────────────────────────────────────────┐
        │ ATOMIC POLICY-SET PUBLISH                │ rollouts see neither
        │ both components or neither               │ staged component until
        │ one-sided failure ⇒ prior set stays live │ both land
        │ orphaned component still catalogued      │
        └──────────────────┬───────────────────────┘
                           ▼
        ┌──────────────────────────────────────────┐
        │ catalog: ckpt records + lineage edges +  │ eval resolves by
        │ policy-set + match-set revisions         │ immutable ID only
        └──────────────────────────────────────────┘
```

#### Substrate — what one attempt actually costs

```text
 POST /rollout (one attempt)
        │
        ├─▶ banking77 / healthbench   in-process env, one or few model calls
        │                             reward: exact match or container rubric
        │
        ├─▶ craftax                   env process + ReAct loop, native reward
        │
        ├─▶ tblite ─── harbor lease ──▶ docker compose project per attempt
        │              (bounded)       ┌──────────────┐   ┌──────────────┐
        │                              │ agent box    │──▶│ verifier box │
        │                              │ mini-SWE     │   │ deferred     │
        │                              │ workspace ───┼──▶│ scoring      │
        │                              └──────────────┘   └──────────────┘
        │                              port-scoped workspace cache; cleanup
        │                              by exact synth.parent label
        │
        ├─▶ dungeongrid ──────────────▶ rust DungeonGridSession, sequential
        │                              turns, 4 instances, party return
        │
        └─▶ runite race ──────────────▶ 24 agent boxes + world server @ 4×
                                       ┌────────────────────────────────┐
                                       │ world engine (2 rocks, ticks)  │
                                       └───────┬────────────────────────┘
                                       ┌───────┴────────┬───────────────┐
                                       │ 6 boxes/team × 4 teams         │
                                       │ each: harness + client + logs  │
                                       └───────┬────────────────────────┘
                                       horizon ⇒ quiesce every box, then
                                       watcher sample + verifier, clipped;
                                       100–175 MB recordings pulled in
                                       parallel, trace stored by reference

 every branch above is a container/deployment concern. the executor sees
 only: declared routes, declared capabilities, attempts, episodes, receipts.
```

### Implementation work

#### 1. Shared optimizer-platform contract

- Add a typed `CispoOptimizerContract` beside `GepaOptimizerContract` in
  `rust/crates/synth_optimizer_platform/src/container_contract.rs`.
- Validate the contract version and every declared absolute route.
- Extend `ContainerClient` with typed capabilities, rollout state, events,
  trace, reward, and termination methods using the declared routes.
- Preserve bearer/header behavior and transient HTTP retry behavior already
  used by GEPA.
- Add typed validation for `TrainableEpisode` and reward receipts.

#### 2. Container capability preflight

- Extend `TrainingRolloutRequirement` and
  `training.rollout.capabilities.v1` with Trace V5, behavior-logprob,
  policy-revision, idempotency, lifecycle, and reward-authority requirements.
- Stop requiring the caller to hard-code a specific task ID; discover it and
  persist it, with an optional expected-identity assertion.
- Add the `cispo.handshake.v1` requirement document, per-clause verdict
  handling, agreement digest, expiry and renewal, revocation handling, and the
  probe-episode shape validator. No session creation or provider request may
  precede acceptance.
- Keep the content hash fail-closed. A changed capability response invalidates
  the prior preflight and the handshake built on it.
- Validate the declared topology, communication channels, horizon, actuation
  model, reward relation, and minimum viable roster in the same hashed
  capability response.
- Validate the declared renderer profile against the training session's
  renderer profile, including package version, config digest, tokenizer digest,
  and stop token IDs, and record the sampling transport (message-in or TiTo).
- Make local and hosted CISPO use the same validation logic.

#### 3. Generic persistent queue engine

- Implement bounded rollout, score, scored-result, and train-ready queues.
- Dispatch individual samples, not whole groups.
- Maintain multiple partially filled groups and prefer completing open groups.
- Persist queue transitions and leases in the existing job store rather than
  relying on `ThreadPoolExecutor` internals.
- Emit exactly one episode/failure/cancellation result per accepted attempt.
- Keep rollout production alive while scoring, training, checkpointing, and
  policy publication run.
- Recheck policy staleness at train dequeue.
- Derive leases, heartbeats, and timeouts from the container's advertised
  horizon, and implement the declared straggler cancel-and-replace policy.
- Treat horizon quiescence and artifact collection as in-lease work.

#### 4. Generic container RL executor, with CISPO as its first preset

- Introduce the immutable algorithm plan and its preset expansion, hash it into
  the run manifest and every group pin, and route zero-advantage skipping,
  staleness bounds, and packing through plan fields rather than constants.
- Replace direct task sampling and reward calculation in
  `src/synth_optimizers/cispo_executor.py` with the container client.
- Remove the embedded Banking77 label extraction and exact-match calculation.
- Retire the TBLite-specific runner as an experiment wrapper or convert it into
  a thin generic-config launcher; no algorithm logic should remain there.
- Build training batches solely from validated `TrainableEpisode` segments and
  container reward receipts.
- Pack multiple mixed groups into each provider training step where configured,
  while retaining per-group CISPO advantages.
- Publish a sampler revision after a completed training step/round and make it
  available to subsequent container rollouts.
- Bind container-declared topologies generically: agent instance to policy type
  to parameter group, trainable and non-trainable instances, teams, declared
  communication channels, and the declared reward relation. Support both
  `sequential` and `concurrent_realtime` turn models and both `direct_action`
  and `deferred_program` actuation without an environment branch.
- Enforce group comparability on topology ID, seed family, and match-set
  revision, and apply the configured same-policy reduction from the receipt.
- Own the renderer for the baseline transport, enforce the strict-prefix rule
  with branch forking and segment sealing, and reject sentinel or malformed
  behavior logprobs before a batch is assembled.
- Persist one immutable per-call record before any trajectory flattening, keep
  the original wire objects beside the token evidence, and support both the
  chat-completions and responses wires as pinned first-class paths.
- Build and enforce the group pin: reject a group mixing plan hash, behavior
  fingerprint, policy revision, wire, policy kind, model family, transport,
  container image digest, contract hash, agreement digest, topology, or task
  family.
- Implement pause, drain, resume, and stop with defined semantics at every queue
  boundary, and make resume re-handshake before re-admitting work.
- Implement replay mode: run the same plan against stored evidence with no
  container and no provider spend, marked off-policy in the catalog.
- Pack multiple groups per provider step and save the sampler artifact once per
  published round rather than once per group, at the operational floor the
  workspace requires: three groups per step and no more than fifteen steps per
  round unless the plan states otherwise.

#### 5. Container-side conformance

- Add `optimizer_contracts.cispo` advertisement and the hashed training
  capability response to the shared container platform.
- Implement the handshake endpoint once in the shared platform layer: evaluate
  each clause against the running build, return obligations and task digests,
  issue and expire `handshake_id`, enforce it on every attempt, and support
  revocation. Each environment contributes only its own clause answers.
- Implement the `probe` policy-binding kind generically, returning deterministic
  synthetic evidence that is explicitly marked non-trainable.
- Implement the common policy-binding, rollout, trace, reward, and cancellation
  behavior once in `synth-containers`.
- Make each target provide only its existing environment implementation and
  declared capabilities.
- Implement staged scoring generically where supported. TBLite uses it for the
  agent/verifier workspace handoff; other tasks may use the same mechanism.
- Keep task-specific tests and reward logic in the container repository.

#### 6. Conformance tests

Create a reusable CISPO container conformance suite. Test at least:

- A one-call synthetic classification container.
- A multi-turn environment-reward container.
- A synthetic joint-episode container with four agent instances mapped onto
  two shared policy parameter groups.
- A deferred-verifier container.
- A rubric-scored container whose judge spans are not trainable.
- Idempotent retry after a lost HTTP response.
- Cancellation and expired-lease recovery.
- A missing-logprobs refusal before training.
- A zero reward accepted as scored.
- A missing reward rejected as evidence failure.
- A stale group discarded at train dequeue.
- Restart recovery with queued, active, scored, and train-ready items.
- Checkpoint-catalog recovery after process interruption, including an
  unpublished staged component checkpoint.
- Evaluation lookup by immutable component checkpoint ID and by multi-policy
  policy-set revision ID.
- A synthetic concurrent real-time container with two teams, one trainable and
  one pinned non-trainable, a rank reward channel, and simultaneous per-instance
  call streams.
- A deferred-program container whose policy-authored loop keeps mutating state
  after its call returns: quiesced and clipped at the horizon it scores
  correctly, and un-quiesced it fails as an evidence failure rather than
  reporting inflated reward.
- A container declaring a cross-team channel that returns no messages, refused
  as a dropped-channel evidence failure.
- A joint episode missing one instance trajectory: refused under `refuse`,
  trained with a recorded absence under `drop_instance`.
- Two groups whose only difference is the opponent match-set revision, rejected
  as one group.
- An opponent binding that attempts alias or `latest` resolution, refused.
- A renderer-profile mismatch between the bound sampler and the training
  session, refused at preflight before any paid request.
- A TiTo container and a message-in container reaching byte-identical prompt
  token IDs for the same task row and renderer profile.
- A multi-turn episode whose second turn re-renders instead of bridging,
  refused as an unexplained prefix divergence, and the same episode with a
  declared compaction accepted with provenance.
- Sentinel, NaN, all-zero, and length-mismatched logprob vectors, each refused.
- A length-truncated generation distinguished from a stop-token generation in
  the span record.
- An overlong prompt handled by each declared policy: refuse, truncate, compact.
- A handshake rejecting one mandatory clause: the run stops before any session
  is created and no provider request is issued.
- A handshake returning `degraded` concurrency: the executor lowers its run plan
  and re-handshakes rather than proceeding on the original plan.
- A rollout submitted with an absent, expired, revoked, or mismatched
  `handshake_id`, refused by the container in each case.
- A capability document that changes after acceptance, invalidating the
  handshake on renewal.
- A container revoking a handshake mid-run: new attempts stop, in-flight
  attempts finish or cancel, and resumption requires a fresh handshake.
- Restart recovery refusing to resume queued work under a different agreement
  digest.
- Clock skew beyond tolerance on a wall-clock horizon, returned as a rejected
  clause.
- A probe episode walking the full path at zero provider cost, and a probe
  episode whose evidence is indistinguishable from real evidence, refused.
- A per-model-family renderer golden: wire input through the adapter and pinned
  renderer to token IDs, compared exactly against known-good direct template
  IDs, including parse of tools, reasoning, and stop.
- A group mixing each pinned field in turn, rejected in every case.
- A responses-wire trajectory flattened into chat messages, refused rather than
  trained.
- A tool loop that stitches under the strict-prefix rule, and a compaction that
  forks a branch, seals the prior segment, and loss-masks the retained prefix.
- A policy-sampled summarization whose tokens are trainable, beside a
  harness-deterministic compaction whose tokens are not.
- Pause, drain, resume, and stop at each queue boundary, including a resume
  refused because the contract, image, plan hash, or renderer fingerprint
  changed.
- A replay of a stored online run reproducing its advantages and batch
  composition bit-for-bit under the same plan hash.
- Retiring a policy revision while an attempt is still sampling from it,
  refused until the active count reaches zero.
- A second preset (GSPO or a distillation objective) reaching a training step
  through configuration alone, with no engine, contract, queue, or catalog
  change.

The generic conformance suite must not select behavior by task name.

#### 7. Checkpoint catalog and evaluation resolver

- Add typed checkpoint, policy revision, policy-set revision, match-set
  revision, lineage-edge, and evaluation-binding records to the durable
  artifact/job store.
- Register the baseline before rollout admission and register each component
  save result before policy publication. Publication must fail closed if any
  component checkpoint is absent from the catalog.
- Materialize one sampler artifact and, when configured, one resumable training
  state per parameter group after each published update/round. Do not save once
  per individual group when several groups are packed into the same update.
- Record failed save attempts and catalog successfully created but unpublished
  components as staged/orphaned artifacts with an explicit retention policy.
- Add a generic resolver used by both rollout and evaluation entrypoints:
  immutable checkpoint ID resolves one policy; immutable policy-set revision
  resolves every component policy as an atomic team.
- Add list/describe operations indexed by producing run, policy type, parameter
  group, update, parent checkpoint, publication status, and evaluation metric.
- Make evaluation receipts persist the requested selector, its immutable
  resolution, and the exact provider sampler references actually loaded.
- Verify artifact existence and digest before an eval starts; missing or
  role-incompatible components are evidence failures rather than silent
  fallback to `latest`.

### Rulings the implementation settled

Six parallel work streams built against this document and each returned the
places it was underdetermined. These are the rulings, recorded here so the next
reader does not re-litigate them.

**Leases have two clocks, and only one of them is a lease.** A heartbeat TTL
keeps an attempt alive; a straggler deadline fixed at grant decides when it has
run too long. Heartbeats never move the deadline, or a heartbeating straggler
is immortal. The straggler deadline covers the horizon times its dilation plus
the quiescence and artifact-collection budgets plus the declared grace. Where a
container declares a grace and the run configuration also carries one, the
container's declaration wins; the configured value is a fallback for a
container that declares none.

**A unit horizon must declare its conversion.** A `steps` or `env_ticks`
horizon carries no duration, so `seconds_per_unit` is required and its absence
is a refusal rather than a default of one second per unit. A container's lease
TTL is its own advertised obligation and is not derived from the horizon.

**Only admission may be refused, never executed work.** The rollout queue and
the open-group bound apply backpressure by refusing admission. Downstream
fullness throttles dispatch instead: refusing an attempt that already ran would
break exactly-one-terminal-result. Straggler replacements and lease-expiry
recovery bypass the admission bound, because they re-enter work that already
left the pipeline.

**Recycling returns slots, not tasks.** A group discarded at the dequeue gate
for staleness returns its slots — sample index, task id, seed, original
idempotency key — and the executor re-admits them under a fresh pin. The queue
engine may not mint a task identity.

**A group that can never complete is reported, not silently dropped.** A
straggler cancelled with no replacement budget leaves a group that cannot fill.
The engine reports it; disposal is the caller's, and automatic discard must be
a policy field if it is ever wanted.

**Drain cancels what it can never run.** Attempts admitted but never dispatched
are cancelled with a recorded reason distinguishing them from terminate-routed
cancellations, so receipts stay complete and cost stays attributable.

**Checkpoint records are immutable; their status is a relation.** A record
carries its registration-time publication status and policy-set memberships;
the effective values derive from an append-only event log. The run receipt
serializes the derived view. This is the only reading under which "records are
immutable" and "publication_status is a record field" are both true.

**Superseded checkpoints remain evaluable.** Published and superseded resolve;
staged resolves only when explicitly allowed; orphaned never does. Otherwise a
paired baseline-versus-trained comparison across rounds stops resolving the
moment a newer round supersedes the baseline.

**Ambiguous selectors are refused, not guessed.** `latest-published` with
published checkpoints in several parameter groups has no single answer, so it
raises rather than picking one; qualify it with a run, parameter group, or
policy type. `best:<metric>` maximizes unless the metric declares a direction.

**Two identities for a revision, carried together.** The queue counts policy
revisions as integers; the catalog names them as text. A group pin carries
both, so the bridge is a field rather than a lookup convention.

### Decisions taken after the first implementation pass

**Clock skew is its own clause.** `lifecycle.clock_skew`, and it is conditional
rather than optional: a `steps` or `env_ticks` horizon reads no wall clock, so
the clause does not apply, which is a different statement from a container
declining it. An optional clause may be declined; a conditional one may not be
declined where it applies. The requirement document names conditional clauses
only when their condition holds.

**A mandatory clause may be satisfied by a declared substitute.**
`CLAUSE_SUBSTITUTES` names which substitute answers which clause — a container
that cannot quiesce may clip its state to the horizon instead. That is neither
a rejection, which stops the run, nor a degradation, which implies a run-plan
dimension to lower and clipping has none. The executor must acknowledge the
substitute it will run under in `accept_degraded`, a field of the requirement
document, and the run receipt records it as a fallback. An acknowledgement
never rewrites a clause the container already accepted, and a substitute nobody
declared is refused.

**Credit follows the plan, not the old executor.** The plane uses the plan's
`length_weighted_leave_one_out` and its standardized variant. No
legacy-compatible credit kind is added: the two differ by exactly `n/(n-1)`
with identical signs, ordering, and skip verdicts, so behavior is materially
unchanged, and a second permanently maintained estimator buys only the ability
to reproduce old normalized numbers exactly. Runs before this change are
reproducible from their own receipts, not from this code path.

### What building the container half revealed

The container side was built against the optimizer client rather than against
this document, which is the only way the two halves were ever going to agree.
Four things surfaced that reading the note alone would not have found.

**Three executor bugs, each invisible from one side.** The capability parser
read the horizon magnitude under one spelling and never read the declared
`seconds_per_unit` at all, so a step or tick horizon parsed cleanly and then
raised at the first lease sizing — a parse bug wearing a queue bug's clothes.
The unanswered-clause check demanded a verdict for every mandatory clause
without asking whether it applied, so a container that correctly omitted the
conditional skew clause would have been refused before spend. And a call could
not declare its author while a segment could, so two independent
implementations re-derived authorship from role and policy type.

**The route surface collides with what a container already serves.** Ten of the
seventeen canonical paths already exist on the reference app with blocking,
GEPA-era semantics, and a blocking rollout result is not a lease-bearing 202.
A container therefore mounts the CISPO surface under its own prefix; the
contract permits it because the executor calls only declared routes, and the
canonical table stays the declaration.

**Existing runtime machinery is close but not sufficient**, and the gaps are
worth recording rather than papering over:

- The target runtime's only entry point is one-shot and synchronous, with no
  submit/poll pair, no cancellation, no quiescence, and no horizon snapshot.
  A rollout runtime is a new protocol, not a subtype of it.
- The token capture model holds ids and logprobs and nothing else the training
  record needs — no sampled mask, finish reason, stop tokens, renderer
  fingerprint, behavior revision, branch provenance, or joint-episode identity.
  Those ride in a single reserved metadata namespace until the capture models
  grow the fields.
- Capture provenance is coarser than the contract's: only provider-observed
  capture maps to engine metadata, so harness-observed, imported, and
  retokenized capture all collapse to untrainable. That is the safe direction
  to be wrong in.
- Binding minting stamps the current time, so a rebuilt trace sealed a
  different digest each time. A digest a reward binds to may not depend on when
  it was computed; the minting path needs the creation time passed in.
- The log sealer requires a closed log, so it cannot be the digest a deferred
  reward binds to. Reward binds to the sealed trace document's own digest.

**A capability document has one author.** Two builders existed briefly, one on
each side of the handshake. They were bridged rather than merged: the container
publishes the document, and everything else parses that document and treats its
hash as authoritative. Two builders that agree today are two builders that
disagree later.

### The milestone policy and the TBLite harness disagree

TBLite is the target the original benchmark ran, and `openai/gpt-oss-20b` is the
policy every acceptance gate names. Under this contract those two cannot
currently be combined, and the reason is worth stating precisely.

The mini-SWE harness rewrites `stored_content` for `openai/gpt-oss-*` before
feeding history back to the model. So turn `k+1`'s prompt is not an extension of
turn `k`'s prompt-plus-generation: the assistant's own earlier text has been
changed underneath it. The container refuses that rather than mislabelling it,
which is right -- a rewrite is not a compaction, and calling it one would put a
false provenance record in the evidence.

But refusing is not the end of the analysis. This document already says that
"context compaction, summarization, chat-template rewrite, subagent dispatch, or
any other history rewrite" forks a branch and seals the prior segment. A
harness-authored content rewrite is exactly that: not a compaction, but a
history rewrite, and the branch mechanism exists for the whole class. The honest
treatment is a branch fork whose rule names what actually happened -- a harness
content rewrite, authored by the harness and therefore never trainable -- rather
than either a silent stitch or a refusal.

The consequence is real and should be accepted rather than engineered around: a
harness that rewrites history every turn produces one sealed segment per turn.
Each turn's own generation is still trainable, and long stitched sequences are
not available. That is a smaller training signal, and it is the true one. If
long sequences matter more than the rewrite does, the fix belongs in the
harness -- stop rewriting stored content -- not in the evidence rules.

Three options, in the order I would take them: stop the rewrite in the harness
for this policy; failing that, fork a branch per rewrite with an honest rule
name and accept per-turn segments; failing both, run the TBLite gate on a policy
whose harness does not rewrite, and say in the receipt that the milestone policy
was not the one measured.

### What the containers could not declare

Five images were built against this contract. Each was asked to declare only
what it can honestly declare, and the refusals are more informative than the
acceptances.

**DungeonGrid cannot declare a party communication channel at all, and that
blocks its row of the evidence matrix.** The Rust engine's HTTP wire has no
message verb: `action_from_string` has no `message` branch and
`legal_action_strings` never offers one, so no policy playing that wire can
author a party message. `DungeonGridAction::Message` and `apply_message` exist
in the engine core and are reachable only in-process. Because a declared
channel that returns nothing for a whole episode is a dropped-channel evidence
failure, the container declares no channel rather than one it cannot fill, and
says so on its health route and in its reward receipt. The carrying half is
written and tested against real engine state: a delivered message is emitted as
untrainable context with its author declared, and its text sits in the
receiving seat's prompt where the mask is zero, so it is untrainable twice
over. The fix is upstream: add a message verb to the engine wire and emit it
from the legal-action list when communication is enabled. Until then, criterion
9 of the MARL gate is demonstrable and the party-communication half of the
evidence matrix row is not.

**Two declarations are too narrow, with concrete shapes proposed.**
`PolicyFacts.wire_api` is scalar, and a binding is refused when its wire
differs, so an image that genuinely serves both wires can advertise only one --
DungeonGrid already runs policies over both. It should be a set plus a default:
`wire_apis: tuple[str, ...]` with `default_wire_api: str`, and membership
rather than equality at binding. `pinned_identity` is an untyped string, so a
non-trainable instance cannot say whether it is a frozen checkpoint, an
external model, or a scripted baseline; today that has to be smuggled into a
string prefix. It should be typed: `PinnedIdentity(kind, identity, revision)`
over the three declared kinds. Neither blocks the MARL gate as fixtured -- one
wire suffices, and a four-trainable roster pins nobody -- but the second is
reached the moment an opponent appears, which is the competitive topology.

**Tokenizer identity belongs to the deployment, not the image.** Banking77 and
Craftax capture the sampler's token ids rather than rendering their own, which
is better training evidence and means neither can declare a tokenizer. Both
fail closed: absent the declaration the target is not installed and all
seventeen routes answer a typed 501. On a multi-turn container a wrong
tokenizer identity is worse than no identity.

**A container cannot receipt the same-policy reduction it was trained under.**
DungeonGrid publishes per-instance token counts and per-episode segments, so a
reduction is computable, but the contract has no field in which the container
can record which one the executor applied. The rule that a chatty role must not
dominate by token count is therefore auditable only from the executor's side.

**One image found a second stop condition hiding inside a horizon.** Craftax
counts policy calls as its horizon while the engine independently limits
environment ticks. Folding the two into one number would have made the horizon
mean two different things; it declares the horizon and reports the engine limit
separately on the receipt.

### Success criteria

#### Common gates for all acceptance runs

Every Banking77, HealthBench, Craftax, TBLite, and GameBench DungeonGrid MARL
smoke run must satisfy all of the following:

1. The run is launched through the same generic CISPO entrypoint and schema.
2. The only environment selection is its container connection/task selection.
3. Preflight validates and persists `optimizer_contracts.cispo`, the capability
   response, and its hash before any paid provider request.
3a. The two-sided handshake is accepted, its agreement digest recorded, and the
   probe or single canary path validated before the Tinker session exists. Every
   attempt in the run carries that `handshake_id`.
4. Policy sampling and training use Tinker with
   `openai/gpt-oss-20b`; the resolved model identity appears in the receipt.
5. All environment execution occurs through container rollout endpoints.
6. The reward used for CISPO comes from the container reward receipt and is
   bound to the sealed Trace V5 digest.
7. At least one non-zero-advantage group reaches an actual Tinker CISPO training
   call. Zero-advantage groups may be skipped and replaced up to the configured
   maximum sampled-group bound.
8. Training produces a new sampleable policy revision/checkpoint distinct from
   the baseline revision.
9. A paired baseline/trained evaluation completes on the same task IDs or
   seeds. Uplift is measured but is not required for this contract smoke test.
10. Every successful training trajectory has exact token IDs, behavior
    logprobs, loss masks, policy revision, provider request IDs, token counts,
    latency, and TPS in its evidence.
11. Queue receipts show real `queued → active → queued-for-score → scored →
    train-ready → consumed` transitions without a whole-group future barrier.
12. No accepted attempt is lost or counted twice; failures and cancellations
    close group accounting explicitly.
13. The run terminates within its configured wall-clock, rollout, token, and
    cost caps.
14. Run-labelled temporary containers, workspace snapshots, volumes, and
    dangling image layers are cleaned after completion while reusable images
    and sealed receipts/traces remain.
15. The baseline and every published update/round have immutable checkpoint
    catalog records. Every recorded sampler artifact can be loaded for an eval,
    and every retained training-state artifact can be loaded for resume using
    its distinct artifact role.

#### Per-container evidence matrix

| Container | Required environment evidence | Required scoring evidence | Required trainable policy evidence |
|---|---|---|---|
| Banking77 | Container task row and policy prediction | Container exact-match reward receipt | Single policy response span with tokens/logprobs |
| HealthBench | Container prompt/answer episode | Container rubric receipt with judge identity and criteria | Answer-policy spans only; judge spans excluded |
| Craftax | Multi-step environment transitions and terminal episode | Container-native environment reward/achievement receipt | All selected ReAct policy-call spans with revisions |
| TBLite | Agent workspace artifact, patch, and mini-SWE episode | Container verifier receipt bound to workspace/trace digest | All selected agent policy-call spans, including compaction provenance |
| GameBench DungeonGrid MARL | Rust `dungeongrid_gold` joint episode, resolved four-agent roster, active-agent turn sequence, and party communication | One container-native party return bound to the complete joint episode | Four actor traces routed into exactly two policy batches: both elf actors to the elf parameter group and both barbarian actors to the barbarian parameter group |
| RuneBench Runite Race | Concurrent real-time joint episode, resolved twenty-four-instance four-team roster, declared horizon with time dilation, per-instance tick streams, cross-team and intra-team channels, and horizon quiescence attestation | Per-team measure and resolved rank read at the horizon, clipped, with settlement window and no post-horizon mutation | Trainee-team spans only, each carrying agent instance, role, policy type, parameter group, team, component and policy-set revision, authored-effect tick interval, and the receipted same-policy reduction; opponent and received-message spans untrainable |

#### GameBench Rust DungeonGrid MARL gate

Use the real GameBench Rust environment at
`gamebench/tasks/dungeongrid-multiplayer/gold_rust`, exercised through its
container service. Add a deterministic acceptance fixture derived from a
checked-in multiplayer scenario with this party declaration:

```json
{
  "hero_roles": ["elf", "elf", "barbarian", "barbarian"]
}
```

The Rust engine resolves those entries in order to `agent_0`, `agent_1`,
`agent_2`, and `agent_3`. The container must declare the exact resolved roster
and the optimizer must bind it as follows, without any DungeonGrid-, elf-, or
barbarian-specific branch in the generic executor:

| Agent instance | Policy type | Parameter group | Tinker training session |
|---|---|---|---|
| `agent_0` | `elf` | `elf_policy` | Elf session initialized from `openai/gpt-oss-20b` |
| `agent_1` | `elf` | `elf_policy` | Same elf session as `agent_0` |
| `agent_2` | `barbarian` | `barbarian_policy` | Barbarian session initialized from `openai/gpt-oss-20b` |
| `agent_3` | `barbarian` | `barbarian_policy` | Same barbarian session as `agent_2` |

This environment is deterministic and turn-based. A joint rollout is one
complete Rust `DungeonGridSession`; on each turn the container routes the
active agent's observation to the policy bound to that agent's declared policy
type. Party messages from other actors are context, never trainable tokens for
the receiving policy.

The MARL smoke run passes only when all of the following are demonstrated in
machine-checkable receipts:

1. Preflight resolves four agent instances, two policy types, two parameter
   groups, one cooperative team, `turn_model = "sequential"`, and a shared
   party-return reward channel.
2. Each rollout is pinned to one immutable policy-set revision containing both
   the elf and barbarian behavior revisions. Mixing independently resolved
   `latest` revisions is rejected.
3. A group contains complete joint episodes from the same scenario/seed family,
   topology, and behavior policy-set revision; individual actor trajectories
   are never grouped as if they were independent episodes.
4. CISPO computes one group-relative team advantage per joint episode and fans
   it out to both parameter groups.
5. The elf batch contains only trainable spans generated by `agent_0` and
   `agent_1`; the barbarian batch contains only spans from `agent_2` and
   `agent_3`. Every span carries `agent_instance_id`, `policy_type_id`,
   `parameter_group_id`, component policy revision, and policy-set revision.
6. Same-policy actor contributions use the configured, receipted reduction;
   they are not implicitly flattened with other roles or weighted accidentally
   by another agent's token count.
7. The two Tinker policy batches may train concurrently, but rollout workers
   cannot observe either staged revision until both training operations finish
   and one new policy-set revision is atomically published. A one-sided train
   failure leaves the prior policy set active. The successfully created staged
   component is still catalogued as `orphaned` or `staged`, never lost.
8. At least one update changes both sampleable component revisions. A paired
   baseline/trained evaluation then runs on identical held-out DungeonGrid
   seeds and the same fixed four-agent roster. Uplift is recorded but is not a
   contract-pass requirement. The evaluation selects the team by immutable
   policy-set revision and records the exact elf and barbarian component
   checkpoint IDs it resolved.
9. Trace V5 evidence reconciles the Rust active-agent turn order with policy
   request IDs, exact tokens, behavior logprobs, train masks, per-policy token
   counts, latency, TPS, and cost. No action may be attributed to two agents or
   two parameter groups.
10. The generic single-agent acceptance targets still pass through the same
    executor as one-agent, one-policy policy sets.

#### RuneBench Runite Race competitive multi-population gate

The DungeonGrid gate covers one cooperative team on a deterministic
turn-based engine. The Runite Race covers the opposite corner: twenty-four
concurrent agent instances, four teams, a rank-only reward, real time at 4x
dilation, a ninety-minute horizon, cross-team public chat, and an actuation
model where the policy writes scripts that keep running after the call returns.
Reference run:
`runite-race-split-4x6-4x-grok46-vs-gemini37flash-or-vs-gpt56terra-or-vs-opus5-or-20260901-233345`
(24 sessions, 4,925 steps, 2,783 messages, 223 deaths, 45 ore).

This gate is staged after the four single-agent targets and DungeonGrid. It is
the acceptance case for concurrent real-time competitive topologies. Train one
team; the other three are pinned non-trainable opponents.

The gate passes only when all of the following hold in machine-checkable
receipts:

1. Preflight resolves the declared topology: twenty-four agent instances, the
   per-team roster including its role split, four teams with exactly one
   trainable, `turn_model = "concurrent_realtime"`,
   `actuation_model = "deferred_program"`, the declared communication channels,
   and `horizon_kind = "wall_clock"` with its value and time dilation. No
   RuneBench-, ore-, miner-, or scout-specific branch exists in the executor.
2. Every rollout pins one immutable match-set revision: the trainee policy-set
   revision plus each opponent's frozen identity. Independently resolved
   `latest` opponents, or a floating external model alias, are rejected.
3. A group contains only joint episodes sharing topology ID, scenario/seed
   family, and match-set revision. Per-team results from one episode are never
   grouped as independent samples.
4. The reward receipt names the optimized channel, records each team's absolute
   measure and resolved rank, states the horizon, the scored-read time, whether
   clipping was applied, the credited settlement window, and the quiescence
   attestation. A run scored after the horizon without clipping fails the gate.
   The reference run demonstrates why: a scored read twenty-six minutes late
   credited one team twelve post-horizon ore and inverted the ranking.
5. Quiescence is demonstrated: the container kills every policy-authored
   background program at the horizon, and the receipt shows zero environment
   mutation between the horizon and the scored read. An episode where a
   policy-authored loop outlived its own session and kept earning reward is an
   evidence failure.
6. Trainable spans cover only the trainee team's instances. Opponent spans,
   cross-team public messages received, and intra-team messages authored by
   other instances are recorded as untrainable context with authorship
   identified.
7. Role asymmetry does not distort the gradient. When a low-throughput role
   emits most of the team's tokens, the receipted same-policy reduction shows
   the applied normalization; a role's share of the update must not be an
   accident of its token count. The reference run is the worst case: one
   level-1 scout produced 103 of 124 public messages and zero ore.
8. Per-span attribution reconciles against environment ticks: each trainable
   span carries the tick interval of the effect it authored, per-instance
   streams are monotone, and no environment effect is attributed to two
   instances.
9. The declared partial-roster disposition is exercised and receipted. The
   reference run lost one session at 15:35 and one box was unreachable at
   collection, yielding twenty-three of twenty-four trajectories; the gate
   requires that outcome to be either a declared `drop_instance` with the
   absence recorded, or a refusal, never a silent twenty-three-instance batch.
10. At least one non-zero-advantage group reaches a real training call and
    produces new sampleable component revisions for every trainee parameter
    group, published atomically as one policy-set revision.
11. A paired baseline/trained evaluation runs on identical held-out
    scenario/seed sets against the identical pinned match set, selected by
    immutable policy-set and match-set IDs. Uplift is recorded, not required.
12. Queue evidence shows hour-scale leases with heartbeats, at least one
    straggler cancelled and replaced under the declared grace policy, and
    trace bundles stored by reference with digests rather than inlined.

#### Universality gate

The implementation is not complete merely because these six real targets run.
Add a synthetic container that was not named in executor code. If it advertises
the same contract and returns valid trainable evidence, the same binary and
configuration schema must complete a CISPO smoke run without code changes.

Automated checks should fail if the queue engine or generic CISPO executor
contains literal task dispatch on `banking77`, `healthbench`, `craftax`,
`tblite`, `dungeongrid`, `elf`, `barbarian`, `runite`, `runebench`, `miner`,
`scout`, `harbor`, `mini_swe`, `opencode`, or `react`.

### Required run artifacts

Each smoke run must leave a self-contained receipt directory containing:

- Effective redacted configuration and the expanded algorithm plan with its
  plan hash.
- The group pin for every group, and any rejected-group records with the field
  that caused the rejection.
- Lifecycle transition log: pause, drain, resume, stop, with the re-handshake
  performed at each resume.
- For a replay run: source run IDs, the accepted staleness, and the
  bit-for-bit comparison against the original run's advantages.
- Container metadata, contract, capability response, and capability hash.
- The handshake pair: requirement document, per-clause verdicts, obligations,
  resolved task digests, `handshake_id`, `agreement_digest`, expiry, measured
  clock skew, every renewal, and any revocation.
- Probe or canary validation record, marked non-trainable, with its cost.
- Container/image digest and relevant repository commits.
- Baseline policy revision and trained policy revision, or the complete
  baseline/trained policy-set manifests for a multi-policy run.
- An append-only checkpoint catalog containing the baseline and every
  materialized intermediate, staged, published, orphaned, and final component
  checkpoint retained by the run.
- Separate immutable sampler-weight and resumable training-state references,
  with digests and retention status, for each checkpoint where available.
- Checkpoint lineage edges linking parent checkpoint, producing run/update,
  policy type, parameter group, provider train/save requests, and every
  policy-set publication that contains it.
- Evaluation manifests that reference immutable checkpoint or policy-set IDs
  and record the resolved component checkpoint IDs.
- The resolved topology, its declared communication channels, the trainable and
  non-trainable instance rosters, and the applied partial-roster disposition.
- The match-set manifest naming every opponent's pinned identity, for every
  group and every evaluation.
- Horizon, scored-read time, clipping decision, credited settlement window, and
  quiescence attestation for each episode.
- Per-instance liveness ledger: admitted, last live tick, terminal status, and
  whether its evidence entered the batch.
- Per-team reward channels with absolute measure and resolved rank, and the
  optimized channel.
- Renderer profile, sampling transport, prompt-budget policy, and the count of
  spans carrying compaction provenance.
- Queue transition journal and aggregate queue metrics.
- Group membership, rewards, advantages, staleness, and skip decisions.
- Provider usage, training-token counts, cost, and request IDs.
- Sampling TPS by call and weighted aggregate TPS.
- Container reward receipts.
- Sealed Trace V5 references or bundles.
- Paired baseline/trained evaluation rows and summary.
- Cleanup receipt listing exactly what was removed and retained.

### Prohibited shortcuts

The following do not satisfy this plan:

- Calling Tinker sampling directly from CISPO instead of through the bound
  container policy.
- Calculating Banking77 accuracy, HealthBench rubric results, Craftax rewards,
  or TBLite verifier results inside `synth-optimizers`.
- Adding task-name switches to the executor or queue engine.
- Branching the engine on the algorithm name, or making CISPO's dimensions
  implicit constants instead of plan fields.
- Flattening one wire's trajectory into another's, or training a policy of one
  wire on rollouts collected through the other.
- Retokenizing new text onto previously captured token IDs, or stitching two
  calls that are not a byte-for-byte prefix.
- Discarding the original wire objects once tokens are captured.
- Publishing a replay-derived revision as an on-policy result, or retiring a
  policy revision with active attempts still sampling from it.
- Treating pause, resume, and offline replay as operational scripts rather than
  contract-level behavior.
- Treating thread-pool futures as the persistent queues.
- Recomputing behavior logprobs after the rollout.
- Training from assistant text without exact tokens, masks, and behavior
  logprobs.
- Keeping a checkpoint only as an unstructured provider path in console output
  or conflating sampler weights with resumable training state.
- Counting absent or failed rewards as zero.
- Scoring an episode from state read after its horizon without quiescence and
  clipping, or letting policy-authored background programs keep earning reward
  past the horizon.
- Training on spans authored by another agent instance, an opponent, a verifier,
  or a judge.
- Grouping episodes played against different or unpinned opponent sets, or
  resolving an opponent as `latest` or a floating provider alias.
- Serializing a declared concurrent real-time topology into a global turn order,
  or inferring topology from agent count, role names, or the task name.
- Silently training on a partial roster.
- Starting a training session, saving a baseline checkpoint, or issuing any paid
  provider request before the handshake is accepted and the probe path
  validated.
- Treating a bare boolean acceptance, or a successful capability GET, as the
  handshake.
- Admitting attempts without a live `handshake_id`, or continuing under an
  expired, revoked, or digest-mismatched agreement.
- Negotiating capability, concurrency, horizon, or renderer identity after
  training has started.
- Letting probe or canary evidence enter a group, a batch, or a training-spend
  total.
- Rendering or tokenizing on the container side while claiming the baseline
  message-in transport, or introducing a second renderer through TiTo.
- Accepting provider sentinel, NaN, all-zero, or length-mismatched logprobs.
- Re-rendering an accumulated message list between turns instead of extending
  the previous turn's rendered sequence, or dropping messages to fit a prompt
  budget without recording the compaction.
- Claiming the existing Q2/Q4 grouped-future benchmark demonstrates the new
  queue-native architecture.

## Setup

- Model: `openai/gpt-oss-20b`, sampled and trained through Tinker.
- Harness: Harbor TBLite with mini-SWE compaction.
- Training: CISPO, five update steps, 20 rollouts per group, 100 rollouts per arm.
- Local capacity: 30 Harbor leases on OrbStack.
- Evaluation seeds: disabled for this throughput comparison.

| Arm | Workers | Queue depth | Maximum staleness |
|---|---:|---:|---:|
| Sync | 20 | 1 | 0 |
| Async Q2 | 30 | 2 | 1 |
| Async Q4 | 30 | 4 | 3 |

## Results

| Metric | Sync | Async Q2 | Async Q4 |
|---|---:|---:|---:|
| Wall time | 521.7 s | 633.5 s | 387.5 s |
| Rollouts/minute | 11.50 | 9.47 | 15.48 |
| Sampling completion TPS | 47.1 | 33.8 | 52.3 |
| Reward across training rollouts | 38/100 | 34/100 | 29/100 |
| Observed staleness | 0, 0, 0, 0, 0 | 0, 1, 1, 1, 1 | 0, 1, 2, 3, 3 |
| Skipped zero-advantage updates | 1 | 1 | 2 |

Async Q4 delivered the strongest measured throughput: 1.35x the sync rate and
25.7% lower wall time. It also had the greatest policy lag and the lowest
training-rollout reward.

Async Q2 bounded lag to one update and retained more reward than Q4, but did not
improve throughput in this run. Its provider sampling rate fell to 33.8 TPS,
versus 47.1 TPS for sync and 52.3 TPS for Q4. Several transient Tinker/Cloudflare
502 polling errors recovered automatically, and one group took 332.5 seconds.
Training/checkpoint time was not the bottleneck: Q2 spent 32.7 seconds there,
versus 34.0 seconds for sync.

The reward comparison is directional, not a paired quality evaluation: the
three arms used independent stochastic rollouts, and no common baseline/trained
evaluation seeds were run.

## Model calls per rollout

One call is one model sampling request made by mini-SWE. The number varies with
how many observe/reason/command iterations the policy takes before finishing.

| Arm | Total calls | Mean/rollout | Median | P90 | Range |
|---|---:|---:|---:|---:|---:|
| Sync | 701 | 7.01 | 6 | 16 | 1–35 |
| Async Q2 | 623 | 6.23 | 5 | 12 | 1–28 |
| Async Q4 | 731 | 7.31 | 6 | 16 | 1–32 |

Across all three arms, the weighted mean was 6.85 calls per rollout
(2,055 calls over 300 rollouts).

## Receipts

- `runs/tblite-cispo-orbstack30-sync-5x20-20260902/summary.json`
- `runs/tblite-cispo-orbstack30-asyncq2-5x20-20260902/summary.json`
- `runs/tblite-cispo-orbstack30-asyncq4-5x20-20260902/summary.json`

## Ten-train-call parallel rerun (2026-09-03)

The three arms ran concurrently against distinct Harbor TBLite platform
instances. Each used a port-scoped host workspace, a distinct
`SYNTH_PLATFORM_ID`, 20 rollouts per group, and five local rollout workers (15
simultaneously configured workers overall).

| Arm | Groups | Train calls | Rollouts | Time to target | Rollouts/min | Updates/min | Weighted TPS | Calls/rollout | Train completion tokens | Mean/max staleness | Directional reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Sync | 12 | 10 | 240 | 2,782.9 s | 5.17 | 0.216 | 46.46 | 6.93 | 262,551 | 0.0 / 0 | 99/240 (41.25%) |
| Async Q2 | 13 | 10 | 260 | 2,746.8 s | 5.68 | 0.218 | 45.03 | 6.75 | 233,008 | 0.9 / 1 | 108/260 (41.54%) |
| Async Q4 | 12 | 10 | 240 | 2,660.5 s | 5.41 | 0.226 | 46.54 | 6.90 | 281,012 | 2.4 / 3 | 96/240 (40.00%) |

Q4 reached ten train calls 4.4% faster than sync; Q2 was 1.3% faster. These
reward rates cover different training-rollout mixes, not a paired held-out
evaluation, so they are not evidence of model uplift. Q2 needed 13 groups
because three zero-variance groups were skipped; sync and Q4 skipped two each.

The earlier multi-instance failures came from all platform ports sharing
`~/.synth-containers/work/harbor-tblite`. The launcher now scopes that path by
host port, eliminating rollout-ID races and missing verifier rewards. OrbStack
remained healthy for the full sustained run.

`--cleanup-docker` now removes only nested containers with the exact arm's
`synth.parent` label, the exact platform container for the selected port, that
port-scoped rollout workspace cache, and dangling image layers. Reusable tagged
TBLite task images and unrelated containers are preserved. The completed rerun
reclaimed 37 GB of rollout workspaces; the audit found zero relevant leftover
containers and zero dangling images.

Rerun receipts:

- `runs/tblite-cispo-sync-train10-parallel-r6-20260903/summary.json`
- `runs/tblite-cispo-asyncq2-train10-parallel-r6-20260903/summary.json`
- `runs/tblite-cispo-asyncq4-train10-parallel-r6-20260903/summary.json`
