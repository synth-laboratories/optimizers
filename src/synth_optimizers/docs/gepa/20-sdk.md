# SDK reference

The Python SDK exposes the same run that the CLI drives. You build a `GepaConfig`
(or load one from TOML), wrap it in a runner, and call `.execute()`.
The same `GepaConfig` is also submit-ready for hosted jobs through
`HostedOptimizerClient.submit(config)`.

```python
from synth_optimizers import GepaConfig, GepaRun, OptimizerRun  # noqa
```

Everything below is importable from the top-level `synth_optimizers` package.

## Quickstart

```python
from synth_containers import Container
from synth_optimizers import GepaConfig, GepaTaskPools, OptimizerRun, TasksetSelection

container = Container("my-task")

with container.serve() as handle:
    result = OptimizerRun(
        GepaConfig(
            container=handle.connection(),
            taskset=TasksetSelection(train_ids=["train:0", "train:1"], heldout_ids=["test:100"]),
            task_pools=GepaTaskPools(
                pareto=["train:0"],
                minibatch=["train:0"],
                reflection=["train:0", "train:1"],
                heldout=["test:100"],
            ),
            program=None,
            objectives=None,
            policy=None,
        )
    ).execute()

print(result.best_candidate)
print("cost: unknown" if result.cost_usd is None else f"cost: ${result.cost_usd:.2f}")
```

## Runners: `OptimizerRun` vs `GepaRun`

Two equivalent ways to execute a config:

```python
# Generic optimizer runner — takes any OptimizerConfig (GepaConfig is one):
from synth_optimizers import OptimizerRun, GepaConfig
result = OptimizerRun(GepaConfig(...)).execute()

# GEPA-specific runner — same result, plus a TOML loader:
from synth_optimizers import GepaRun
result = GepaRun(GepaConfig(...)).execute()
result = GepaRun.from_toml("gepa.toml").execute()   # load CLI-style TOML directly
```

`OptimizerConfig` is a `Protocol` with a single `execute()` method; `OptimizerRun` is the
generic runner over it. `GepaRun` is the GEPA-specialized runner and adds
`GepaRun.from_toml(path)`. Both `.execute()` calls return a `GepaRunResult`.

Hosted submit uses the shared hosted client:

```python
from synth_optimizers import GepaConfig, HostedOptimizerClient

client = HostedOptimizerClient()
submit = client.submit(GepaConfig.from_toml("gepa.toml"))
record = client.wait_for_run(submit.run_id)
```

### `GepaRunResult`

The result object carries the best candidate, the frontier, and usage/cost. Common fields:

```python
result.best_candidate     # the winning prompt set
result.cost_usd           # total run cost
```

Inspect the run dir (events, registry, manifest) for the full evidence trail — the same
data the [board](#/cli) renders.

## Config schema

A `GepaConfig` mirrors the TOML sections. The minimal TOML:

```toml
[container]
url = "http://127.0.0.1:8765"
command = ["uv", "run", "python", "banking77_container/synth_service_app.py", "--port", "8765"]

[candidate]
target_modules = ["stage2_system"]

[seed_candidate]
stage2_system = "Classify the query into exactly one Banking77 intent. Return only the label."

[dataset]
train_seeds = [0, 1, 2, 3, 4, 5, 6, 7]
heldout_seeds = [100, 101, 102, 103]
```

Load it with `GepaRun.from_toml("gepa.toml")`, or build the equivalent objects directly.
These are the `GepaConfig` constructor arguments (matching the quickstart above):

| `GepaConfig` argument | SDK type | Purpose |
|-----------------------|----------|---------|
| `container=` | container connection | Which scored environment to talk to (`handle.connection()` — URL + optional launch command). |
| `taskset=` | `TasksetSelection` | Train and heldout task-id selection. |
| `task_pools=` | `GepaTaskPools` | How those task ids split into the pareto / minibatch / reflection / heldout pools. |
| `program=` | `PromptProgram` | Which prompt modules to optimize; `None` derives it from the container's `/program`. |
| `policy=` | `PolicyConfig` | The model that runs rollouts inside the container; `None` uses the container/TOML policy. |
| `objectives=` | `ObjectiveConfig` | Scoring objectives and acceptance; `None` uses the container default. |

The `[proposer]` and `[output]` TOML sections map to `ProposerConfig` / `OutputConfig`; the
`[seed_candidate]` section sets the starting prompt text per module. Pass `None` for
`program` / `policy` / `objectives` to derive them from the container and defaults.

Useful config types exported from `synth_optimizers`: `GepaConfig`, `GepaTaskPools`,
`TasksetSelection`, `PolicyConfig`, `PolicyType`, `ProposerConfig`, `ProposerPromptConfig`,
`ObjectiveConfig`, `OutputConfig`, `BudgetConfig`, `GepaBudgetConfig`, `CacheConfig`,
`RunSettings`, `GepaStalenessPolicy`.

## Authentication and models

Policy models run inside your task container; the reflective proposer runs Codex on the
host (or in Docker). Rollout requests never carry proposer keys.

API-key proposer (run-local Codex home, does not touch your host `~/.codex`):

```toml
[policy]
provider = "openai"
model = "gpt-4.1-nano"
api_key_env = "OPENAI_API_KEY"

[proposer]
backend = "codex_app_server"
runtime_substrate = "local"
provider = "openai"
auth_mode = "api_key"
api_key_env = "OPENAI_API_KEY"
copy_host_auth = false
model = "gpt-5.4-nano"
sandbox_mode = "workspace-write"
approval_policy = "never"
timeout_seconds = 900
```

- **ChatGPT subscription proposer** — `auth_mode = "chatgpt"` with a required `codex_home`
  (OAuth via the Codex CLI). Proposer usage is $0; policy rollouts still bill normally.
- **OpenRouter / DeepSeek** — set `[proposer].provider` accordingly; policy can stay on OpenAI.
- **Docker proposer** — `runtime_substrate = "docker"` with `[proposer.docker].image`.
- **Preflight validation** — missing keys, missing `codex_home`/`auth.json`, or disallowed
  ChatGPT models fail before any rollout starts.

## Observability

The SDK also exposes the same board/observability projection that `gepa board` renders:

```python
from synth_optimizers import RunBoard, RunStatus, RunState, project_run_events
```

These read a run's `events.jsonl` / registry into typed status objects — handy for custom
dashboards (the run board served by `gepa console` is built on exactly this surface).
