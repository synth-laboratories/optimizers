---
name: gepa
description: Use when running, configuring, debugging, extending, or adapting public Rust GEPA in synth-optimizers, including GEPA TOML profiles, Codex app-server proposer auth, proposer workspace manifests, task_info-guided prompt optimization, rollout budgets, frontier/heldout interpretation, and GEPA-compatible cookbook containers.
---

# Rust GEPA Skill

Use this skill for public GEPA cookbook work in `synth-cookbooks-public`.
The goal is a reproducible prompt-optimization run driven by TOML config,
HTTP task containers, and a Codex app-server proposer that inspects actual
run evidence before writing candidate prompts.

## First Files

Load only the files needed for the question:

- `synth-cookbooks-public/cookbooks/optimizers/gepa/README.md` for cookbook usage and container
  contract.
- `README.md` for CLI/Python API and config shape.
- `rust/crates/synth_gepa/README.md` for algorithm
  behavior and workspace semantics.
- Container-local `gepa.toml`, `run_profiles/*.toml`, and
  `run_fresh_gepa.sh` for the task being run.
- Run artifacts under `synth-cookbooks-public/cookbooks/optimizers/gepa/runs/<run_id>/` when
  debugging behavior.

## Mental Model

GEPA optimizes mutable prompt fields in a container-declared prompt program:

1. The container exposes task metadata, dataset rows, a seed prompt program,
   and a rollout route.
2. Rust GEPA evaluates the seed on train seeds.
3. GEPA materializes a proposer workspace with candidate payloads, scores,
   rollouts, failure summaries, task info, prompt guidelines, and schema docs.
4. Codex app-server inspects the workspace and writes
   `proposal/manifest.json`.
5. GEPA registers proposed candidates, evaluates minibatches/full train, updates
   frontier state, and finally evaluates selected candidates on heldout.

The proposer should not guess from generic benchmark knowledge alone. It should
use `/task_info`, actual rollout wins/losses, candidate deltas, and verifier
evidence to write task-specific but generalizing prompt updates.

## One-Command Cookbook Runs

Run from the container directory:

```bash
cd synth-cookbooks-public/cookbooks/optimizers/gepa/banking77_container
bash run_fresh_gepa.sh --profile long
```

Common public examples:

```bash
cd synth-cookbooks-public/cookbooks/optimizers/gepa/banking77_container && bash run_fresh_gepa.sh --profile long
cd synth-cookbooks-public/cookbooks/optimizers/gepa/hotpotqa_container && bash run_fresh_gepa.sh --profile long
cd synth-cookbooks-public/cookbooks/optimizers/gepa/tblite_container && bash run_fresh_gepa.sh --profile long
cd synth-cookbooks-public/cookbooks/optimizers/gepa/crafter_container && bash run_fresh_gepa.sh --profile long
cd synth-cookbooks-public/cookbooks/optimizers/gepa/minigrid_container && bash run_fresh_gepa.sh --profile long
```

Use:

```bash
bash run_fresh_gepa.sh --list
```

to inspect available profiles for that container.

## Auth Model

Full quickstarts (OpenAI API key, OpenRouter proposer, ChatGPT subscription,
policy BYOK boundaries): **`README.md` → Authentication and models**.

GEPA has two credential boundaries:

- **Policy** — container env or Synth proxy (`credential_mode = byok`). Optimizer
  never sends raw keys on rollout HTTP.
- **Proposer** — host Codex app-server with run-local `CODEX_HOME`.

### Public cookbook default (OpenAI API key)

Reproducible public runs should use `auth_mode = "api_key"`, not an implicit host
`~/.codex` login.

```bash
export OPENAI_API_KEY="..."
export SYNTH_OPTIMIZERS_TERMINAL=1   # live usage total / policy / proposer
```

```toml
[proposer]
backend = "codex_app_server"
execution_mode = "local_process"
provider = "openai"
model = "gpt-5.4-nano"
auth_mode = "api_key"
api_key_env = "OPENAI_API_KEY"
copy_host_auth = false
sandbox_mode = "workspace-write"
approval_policy = "never"
timeout_seconds = 900
```

Rust GEPA creates a run-local `.codex_api_key_home` with only the configured key.

### OpenRouter proposer

```toml
[proposer]
provider = "openrouter"
auth_mode = "api_key"
api_key_env = "OPENROUTER_API_KEY"
copy_host_auth = false
model = "x-ai/grok-4.3"
```

Policy rollouts may still use OpenAI (`OPENAI_API_KEY` in the container).

### ChatGPT subscription proposer

For subscription models (`gpt-5.4-mini`, etc.), use OAuth — not a Platform API key.

1. `codex auth login` or [opencode-openai-codex-auth](https://github.com/numman-ali/opencode-openai-codex-auth)
2. Set explicit `codex_home` (no silent host fallback)

```toml
[proposer]
auth_mode = "chatgpt"
codex_home = "~/.codex"
copy_host_auth = true
model = "gpt-5.4-mini"
```

Do not combine `auth_mode = "chatgpt"` with `api_key_env`. Legacy `auth_mode = "host"`
maps to `chatgpt`. Allowed models are enforced at config validation time.

### Do not recommend

- Direct DeepSeek API through Codex (Responses wire mismatch) — use OpenRouter or
  wait for adapter support.
- `copy_host_auth = true` without explicit `codex_home` in public cookbooks.

## TOML Sections

The public v1 config shape is sectioned by durable nouns:

- `[run]`: run id, output directory, seed.
- `[container]`: standing `url` or local `command` and `cwd`.
- `[dataset]`: train/heldout splits, seeds, optional filters or sampler hints.
- `[candidate]`: mutable target modules.
- `[seed_candidate]`: baseline payload matching `/program` mutable fields.
- `[policy]`: rollout policy provider/model/API key env.
- `[proposer]`: Codex app-server backend, model, auth, sandbox, timeout.
- `[gepa]`: generations, proposal count, minibatch size, rollout/cost/time
  budgets, transport, pipeline, adaptive concurrency.
- `[cache]`: off/readwrite/readonly cache behavior.

Prefer changing TOML profiles over passing command-line flags. Profiles should
make the run readable: container, dataset sizes/seeds, models, generations,
minibatch size, budgets, concurrency, and timeouts should be visible in the
resolved run log.

## Container Contract

GEPA requires an HTTP task container with:

- `GET /health`
- `GET /metadata`
- `GET /task_info`
- `GET /program`
- `GET /dataset`
- `POST /dataset/rows`
- `POST /rollout`

Optional async routes may be used, but sync blocking rollout is enough for most
public examples.

`/task_info` matters. It should tell the proposer:

- what task is being solved
- what the policy input looks like
- what outputs/actions/patches are valid
- how scoring works
- what constitutes overfitting
- what task-specific prompt strategies are promising or invalid

If proposals look generic, soft, or overfit, inspect `/task_info` and rollout
evidence before blaming GEPA search.

## Proposer Workspace

For each generation, inspect:

```text
runs/<run_id>/proposer_workspaces/generation_000/
  README.md
  prompting_best_practices.md
  proposal/PROPOSAL_SCHEMA.md
  proposal/manifest.json
  state/proposer_metadata.json
  state/task_info.json
  state/program_contract.json
  state/parent_payload.json
  state/candidates.json
  state/candidate_deltas.json
  state/proposer_failure_summary.json
  state/proposer_repair_hints.json
  state/proposer_examples.json
  state/rollouts.json
  state/scores.json
  state/evidence_frames.json
```

`proposal/manifest.json` must be strict JSON with:

```json
{
  "schema_version": "gepa_workspace_proposal_v3",
  "critique": "...",
  "evidence": {
    "reviewed_files": ["..."],
    "candidate_comparison": "...",
    "failure_patterns": ["..."],
    "winning_patterns": ["..."],
    "example_ids_used": ["..."]
  },
  "rationale": "...",
  "proposals": []
}
```

If the proposer fails, inspect the manifest and schema first. A good manifest
should cite files reviewed, concrete failure clusters, and distinct candidate
strategies. It should not be a mild paraphrase of the seed prompt unless the
run intentionally asks for a conservative control.

## Prompt Proposal Standards

Strong GEPA proposals should:

- Target the task's main failure clusters, not generic prompt polish.
- Use `/task_info` plus observed wins/losses to infer task semantics.
- Be ambitious: structural rewrites, decision procedures, boundary taxonomies,
  few-shot examples when valid, conflict precedence, action gating, verifier
  rubrics, or role/task decomposition.
- Stay general. Closed-output classification may use label boundary examples;
  open-output QA/coding/agent tasks should avoid copying literal train answers
  into reusable prompts.
- Preserve output contracts and mutable payload keys.
- Produce distinct candidates rather than near-duplicates.

Weak proposal signs:

- Only restates "return exactly one label" or "be concise".
- Adds examples from train data that are literal memorization for open-output
  tasks.
- Ignores the actual rollout failure summaries.
- Changes non-mutable fields or omits parent payload keys.
- All candidates use the same strategy.

## Interpreting Logs

Important log lines:

- `seed ... train=...`: baseline train score.
- `frontier ... size=... (+N/-M net ...)`: Pareto/frontier update.
- `coverage=train X/Y rows`: rows solved by at least one frontier candidate.
- `best_seeds=...`: rows solved by the current best candidate.
- `candidate ... minibatch=... parent=...`: minibatch comparison.
- `accepted ... primary_improvement`: candidate advanced after full-train
  evaluation.
- `deferred ... insufficient budget`: candidate looked promising but budget
  prevented full-train evaluation.
- `heldout ...`: final generalization check.
- `baseline -> best diff`: prompt delta that actually won heldout.

Do not confuse frontier coverage with score. Coverage is the fraction of train
seeds solved by at least one candidate in the Pareto frontier. Best score is
the aggregate score of the selected best candidate.

## Throughput And Slow Runs

GEPA logs section throughput:

```text
rollout section done stage=candidate_full_train rollouts=200 wall=15.62s throughput=12.80/s cache=80/200 tokens=0.329M jobs=4
```

For one-call classifier/QA tasks, low throughput usually means policy provider
latency, too-small chunking, or container serialization. For live environment
tasks such as Crafter or MiniGrid, a single rollout can contain many policy
calls and environment steps, so rollout/sec will naturally be much lower.

When debugging slow rollouts:

- Check whether the route handler blocks an async event loop.
- Compare `rollout_workers`, chunk size, and provider concurrency.
- Look for timeout/retry loops and provider 429/overloaded responses.
- Use adaptive rollout concurrency from TOML when provider limits are unknown.
- Lower live-env `max_turns` for smoke runs.
- Treat warnings as diagnostics, not proof of deadlock.

## Budgets And Profiles

Use small profiles for plumbing:

- `smoke`: confirms container starts, proposer auth works, schema is valid.
- `default`: moderate run for quick signal.
- `long`: enough generations/budget to observe real movement.
- `throughput`: isolates rollout throughput and provider behavior.

For better signal, increase train/heldout size and minibatch size together.
Tiny train sets can overfit or make improvements noisy. Bigger heldout catches
candidate memorization. For closed-output tasks, random balanced sampling is
usually better than contiguous seed windows.

## Debugging Checklist

When a run fails:

1. Read the final error and `result_manifest.json` if present.
2. Inspect `events.normalized.jsonl` around the failure.
3. If the failure is proposer-related, inspect
   `proposer_workspaces/generation_*/proposal/manifest.json` and
   `proposal/PROPOSAL_SCHEMA.md`.
4. If rollouts are slow or stuck, check route logs, section throughput,
   container timeouts, and whether the handler is sync/blocking-safe.
5. If proposals are poor, inspect `state/task_info.json`,
   `state/proposer_failure_summary.json`, `state/proposer_examples.json`,
   `state/rollouts.json`, and `prompting_best_practices.md`.
6. If disk budget fails, clear ignored run artifacts under
   `synth-cookbooks-public/cookbooks/optimizers/gepa/runs/` after verifying no needed run is active.
7. If cache behavior is confusing, inspect `cache_profile.json` and the
   workspace SQLite status.

## Validation

Use focused validation:

- Rust change: `cargo check -p synth_gepa` from
  `rust`.
- Runner shell change: `bash -n run_fresh_gepa.sh`.
- Container Python change: `python -m py_compile synth_service_app.py`.
- End-to-end cookbook change: `bash run_fresh_gepa.sh --profile smoke`.

Do not add test files unless the user explicitly asks for tests. Do not run
long GEPA profiles as validation unless the user asks or the change directly
requires it.

## Public-Safe Guardrails

- Never print or commit API keys.
- Use API-key proposer auth in public docs.
- Keep run artifacts ignored unless intentionally curated.
- Do not depend on private backend services.
- Every public container should use a real verifier or real environment.
- Keep container-specific heuristics in container metadata and task prompts,
  not hard-coded into the GEPA algorithm.

