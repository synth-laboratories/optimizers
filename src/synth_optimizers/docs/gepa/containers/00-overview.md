# Containers overview

A **container** is the scored environment GEPA optimizes against. It owns the task: the
dataset, the policy that runs a rollout, and the reward. GEPA only ever talks to it over
HTTP — it proposes new prompt text, asks the container to roll it out on some task ids, and
reads back a reward. This boundary is what keeps GEPA task-agnostic: the same optimizer
drives Banking77 classification, a Craftax survival agent, or a pytest-verified coding
task without knowing anything about them.

```
GEPA  ──HTTP──►  Container
  proposes candidate ──►  /rollout (candidate, task_id)
                       ◄──  reward, trace, usage
```

The contract is defined by the public optimizer HTTP task contract. You author a
container with the `Container` class; it hands GEPA a URL via
`handle.connection()`.

## What a container provides

- **A prompt program** (`/program`) — the modules it exposes, which are **mutable**
  (`target_modules`), and the **seed candidate** (baseline prompt text). GEPA only edits
  the mutable fields.
- **A dataset** (`/dataset`, `/dataset/rows`) — named splits (train / heldout) and the
  rows behind each seed/task id.
- **A rollout endpoint** (`/rollout`) — run one candidate on one task and return a reward
  (sync) or a handle to poll (async).
- **Metadata** (`/metadata`, `/task_info`) — capabilities and the optimizer contract
  version it speaks (`synth_optimizers.gepa.v2`).

## Two ways to think about it

- **Using** a container — pick one from the [catalog](#/containers/catalog), launch it,
  point a `gepa.toml` at its URL, and run. Most users start here.
- **Authoring** a container — implement the [contract](#/containers/contract) for your own
  task. If your task can produce a numeric reward for a prompt, it can be a container.

## Language-agnostic

The contract is HTTP + JSON, so a container can be written in any language. The catalog
ships Banking77 in **Python, Rust, and TypeScript** as functionally identical proof that
the boundary — not the language — is what matters.

Next: the [catalog](#/containers/catalog) of shipped containers, then the
[contract](#/containers/contract) for authoring your own.
