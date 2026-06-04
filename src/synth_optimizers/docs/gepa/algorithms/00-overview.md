# Algorithms overview

GEPA is the public optimizer today; future algorithms plug into the same platform
contract (the container HTTP contract + the run/evidence model). This section covers how
GEPA actually searches, the **pipeline modes** that schedule that search, and the
**proposer + policy** layer that generates and rolls out candidates.

## GEPA — reflective prompt evolution

GEPA optimizes the **mutable prompt modules** of a task (declared by the container's
`/program` route) without touching weights. One generation, in the abstract:

1. **Propose** — the reflective proposer (Codex) reads recent rollout evidence and writes
   a new candidate: edited text for the target modules.
2. **Rollout** — the candidate runs against the task container on the minibatch (and, for
   survivors, the Pareto/heldout pools). The container returns a reward per task.
3. **Evaluate** — scores are attributed to the candidate; objectives are computed.
4. **Select** — GEPA keeps a **Pareto frontier** of candidates across tasks/objectives
   rather than a single best, so a candidate that wins on a hard slice survives even if
   another wins on average. Heldout evidence gates what is reported.

This repeats until the budget is exhausted. The frontier, every rollout, and the
acceptance decision are written to the run dir as inspectable evidence (events, registry,
manifest) — the same data the [board](#/cli) renders.

## The pools

A run partitions task ids into pools (`GepaTaskPools`): **minibatch** (cheap per-generation
signal), **pareto** (the frontier evaluation set), **reflection** (examples shown to the
proposer), and **heldout** (never proposed against; used only to gate reported results).
Keeping these distinct is what prevents the proposer from overfitting to the tasks it sees.

## What varies between runs

- **[Pipeline mode](#/algorithms/pipeline-modes)** — how propose/rollout/evaluate are
  scheduled: strictly serial, asynchronously pipelined, or fully overlapped (Flash Evolve).
- **[Proposer & policy](#/algorithms/proposer-and-policies)** — which reflective model
  writes candidates (and under which auth), and which policy type runs the rollouts
  (`dag`, `react`, or `codex`).
- **Objectives & acceptance** — the metric(s) a container reports and how the Pareto
  acceptance criterion decides whether a challenger is kept.
