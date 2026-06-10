<h1 align="center">synth-optimizers</h1>
<p align="center">A shared optimizer platform for AI applications — starting with open-source GEPA on a language-agnostic task contract.</p>
<p align="center">
<a href="https://pypi.org/project/synth-optimizers/">PyPI</a> ·
<a href="https://github.com/synth-laboratories/containers">Containers</a> ·
<a href="https://github.com/synth-laboratories/synth-cookbooks-public/tree/main/cookbooks/optimizers/gepa">Cookbooks</a> ·
<a href="docs/hosted-optimizers.md">Hosted jobs</a>
</p>

`synth-optimizers` provides a shared Rust optimizer core for running search
algorithms against any task exposed through the
[`synth-containers`](https://github.com/synth-laboratories/containers) HTTP
contract.

- **Shared platform core** — reusable Rust machinery for container I/O, workspaces, cache profiles, budgets, telemetry, failure handling, and replayable evidence.
- **Algorithm layer** — [GEPA](https://gepa-ai.github.io/gepa/) is the first public optimizer; future algorithms can plug into the same platform contract.
- **GEPA runs today** — configure GEPA with TOML or `GepaConfig`; it proposes prompt changes, rolls them out, scores them, keeps a Pareto frontier, and emits inspectable run evidence.

## Supported algorithms

| Algorithm | Status | In this repo | Paper & docs |
| --- | --- | --- | --- |
| **GEPA** — reflective prompt evolution | Supported | [`rust/crates/synth_gepa/`](rust/crates/synth_gepa/) (Rust engine + service), [`src/synth_optimizers/gepa.py`](src/synth_optimizers/gepa.py) (Python API), [`skills/gepa/SKILL.md`](skills/gepa/SKILL.md) (agent runbook) | [Paper](https://arxiv.org/abs/2507.19457) · [gepa-ai docs](https://gepa-ai.github.io/gepa/) · bundled HTML via `gepa console` |
| **GELO** — Go-Explore in prompt space (hosted) | Hosted submit | [`src/synth_optimizers/gelo.py`](src/synth_optimizers/gelo.py), [`skills/gelo/SKILL.md`](skills/gelo/SKILL.md), [`GELO_HOSTED_SDK_CLI_SPEC.md`](GELO_HOSTED_SDK_CLI_SPEC.md) | Bundled HTML via `gelo console` — [`src/synth_optimizers/docs/gelo/`](src/synth_optimizers/docs/gelo/) |

The shared [`synth_optimizer_platform`](rust/crates/synth_optimizer_platform/)
crate is the substrate for optimizer implementations; GEPA is the first public
local algorithm; GELO is hosted-only in the public package (execution in optimizers-beta).
Hosted GEPA/GELO submission is covered in [`docs/hosted-optimizers.md`](docs/hosted-optimizers.md).

## Install

```bash
pip install synth-optimizers
# or
uv add synth-optimizers
```

Latest daily dev build:

```bash
pip install --pre \
  synth-containers==0.2.0.dev202605312141 \
  synth-optimizers==0.2.0.dev202606010003

uv add --prerelease allow \
  synth-containers==0.2.0.dev202605312141 \
  synth-optimizers==0.2.0.dev202606010003
```

Install [`uv`](https://github.com/astral-sh/uv) for local development and editable installs.

## Local development

Pair this repo with [`synth-containers`](https://github.com/synth-laboratories/containers)
at matching dev versions, then sync and install editable:

```bash
cd optimizers
uv sync --group dev
uv pip install -e ../containers
uv pip install -e .
uv run maturin develop --manifest-path rust/crates/synth_optimizers_py/Cargo.toml
```

## Quickstart

A run is defined by TOML (or `GepaConfig`): which **container** to talk to, which prompt
modules to optimize, and how to score them.

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

```python
from synth_containers import Container
from synth_optimizers import GepaConfig, GepaRun, GepaTaskPools, OptimizerRun, TasksetSelection

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
print(f"cost: ${result.cost_usd:.2f}")
```

Or load TOML directly: `GepaRun.from_toml("gepa.toml").execute()`.

CLI:

```bash
synth-optimizers gepa run --config gepa.toml
synth-optimizers gepa service --db service.sqlite
synth-optimizers events compare --left a.jsonl --right b.jsonl
```

Runnable task examples: [GEPA cookbooks](https://github.com/synth-laboratories/synth-cookbooks-public/tree/main/cookbooks/optimizers/gepa)
(Banking77, HotpotQA, MiniGrid, TBLite, Crafter).

<details>
<summary><strong>Authentication and models</strong></summary>

Policy models run inside your task container; the reflective proposer runs Codex on the
host (or in Docker). Rollout requests never carry proposer keys.

Default OpenAI API key setup:

```bash
export OPENAI_API_KEY="sk-..."
export SYNTH_OPTIMIZERS_TERMINAL=1   # optional: live usage in the terminal
```

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

OpenRouter proposer (`provider = "openrouter"`, `api_key_env = "OPENROUTER_API_KEY"`) —
policy can stay on OpenAI. See [skills/gepa/SKILL.md](skills/gepa/SKILL.md) for full TOML.

</details>

## Features

- **OpenAI API key proposer** — run-local Codex home; does not use your host `~/.codex` login.
- **OpenRouter proposer** — provider-aware Codex config and base URL; OpenRouter works for policy rollouts too.
- **ChatGPT subscription proposer** — `auth_mode = "chatgpt"` with required `codex_home` (OAuth via [Codex CLI](https://github.com/openai/codex) or [opencode-openai-codex-auth](https://github.com/numman-ali/opencode-openai-codex-auth)); models include `gpt-5.4-mini`, `gpt-5.3-codex`, `gpt-5.3-codex-spark`, `gpt-5.5`; proposer usage is $0, policy rollouts still bill normally.
- **Live usage** — `SYNTH_OPTIMIZERS_TERMINAL=1` prints running token and cost splits (`usage total=… policy=… proposer=…`).
- **Docker proposer** — `runtime_substrate = "docker"` with `[proposer.docker].image`; workspaces stage under `~/.cache/synth-gepa-docker-workspaces/`, sync back, then cleanup; image: `docker/codex-gepa-proposer/Dockerfile`.
- **Gemini and other policy providers** — supported on the policy side via `[policy].provider`, `base_url`, and container env keys; proposer stays Codex.
- **DeepSeek direct** — `provider = "deepseek"` with `backend = "deepseek_chat"` runs the proposer through DeepSeek Chat Completions; OpenRouter DeepSeek slugs remain supported through `provider = "openrouter"`.
- **Preflight validation** — missing keys, missing `codex_home`/`auth.json`, or disallowed ChatGPT models fail before rollouts start.

Agent docs: [skills/gepa/SKILL.md](skills/gepa/SKILL.md).

## Links

- [GEPA docs (gepa-ai)](https://gepa-ai.github.io/gepa/) — algorithm overview, case studies, and adapter guides
- [GEPA paper](https://arxiv.org/abs/2507.19457) — *GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning*
- [Cookbooks](https://github.com/synth-laboratories/synth-cookbooks-public/tree/main/cookbooks/optimizers/gepa) — runnable GEPA examples
- [synth-containers](https://github.com/synth-laboratories/containers) — the task contract
- [uv](https://github.com/astral-sh/uv) — Python package and project manager
- [GEPA service OpenAPI](rust/crates/synth_gepa/openapi/gepa-service-v1.yaml)

## License

Apache-2.0
