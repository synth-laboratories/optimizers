<h1 align="center">synth-optimizers</h1>
<p align="center">Prompt-program optimizers for production agents — open-source GEPA on a language-agnostic task contract.</p>
<p align="center">
<a href="https://pypi.org/project/synth-optimizers/">PyPI</a> ·
<a href="https://github.com/synth-laboratories/containers">Containers</a> ·
<a href="https://github.com/synth-laboratories/synth-cookbooks-public/tree/main/cookbooks/optimizers/gepa">Cookbooks</a>
</p>

`synth-optimizers` runs [GEPA](https://arxiv.org/abs/2507.19457) — reflective prompt
evolution — against any task exposed through the
[`synth-containers`](https://github.com/synth-laboratories/containers) HTTP contract.
Point it at a container and a TOML config; it proposes prompt changes, rolls them out,
scores them, keeps a Pareto frontier, and returns a deployable candidate with replayable
evidence.

- **Container boundary** — the optimizer only speaks HTTP. It never imports your task code or sees your model credentials.
- **Inspectable runs** — every candidate carries its prompt diff, per-seed scores, rollout traces, cache profile, and usage.
- **Rust core, thin Python** — the GEPA state machine is Rust (PyO3); the Python surface is just `GepaRun` and a CLI.

## Install

```bash
pip install synth-optimizers
# or
uv add synth-optimizers
```

## Local development

This repo pairs with [`synth-containers`](https://github.com/synth-laboratories/containers)
at `synth-containers==0.2.0.dev20260531` / `synth-optimizers==0.2.0.dev20260531`.

Install both editable checkouts with `uv` (sibling repos under your workspace):

```bash
cd optimizers
uv sync --group dev
uv pip install -e ../containers
uv pip install -e .
uv run maturin develop --manifest-path rust/crates/synth_optimizers_py/Cargo.toml
```

Verify the installed paths and versions:

```bash
uv run python -c "import importlib.metadata as m, synth_containers, synth_optimizers; print(synth_containers.__file__); print(synth_optimizers.__file__); print(m.version('synth-containers')); print(synth_optimizers.__version__)"
```

Most of `dev_examples/` stays local-only. The Better GEPA acceptance harness is
tracked for merge validation:

```bash
cd dev_examples/better_gepa
python run_acceptance.py --profile openai_baseline --mode cost_stop
python run_acceptance.py --profile openai_baseline_docker --mode cost_stop
```

See [dev_examples/better_gepa/acceptance_usage_termination.md](dev_examples/better_gepa/acceptance_usage_termination.md)
for profile/env requirements. Other cookbook dev scripts (`dev_examples/banking77/`,
etc.) remain on disk but are not committed.

## Quickstart

A run is defined entirely by one TOML: which **container** to talk to, which prompt
modules to optimize, and how to score them. The optimizer launches (or connects to)
the container, then only speaks HTTP to it.

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

GEPA can only optimize against a **live container** — it speaks HTTP, nothing else.
The same `url` you put in `[container]` is the one to confirm is up via `GET /health`
before spending a single rollout:

```python
from synth_containers import Container
from synth_optimizers import GepaConfig, OptimizerRun, TasksetSelection

container = Container("my-task")

# Register task_info, program, dataset, task_rows, and rollout callbacks here.

with container.serve() as handle:
    config = GepaConfig(
        container=handle.connection(),
        taskset=TasksetSelection(train_ids=["train:0", "train:1"], heldout_ids=["test:100"]),
        program=None,
        objectives=None,
        policy=None,
    )
    result = OptimizerRun(config).execute()

print(result.best_candidate)
print(f"cost:       ${result.cost_usd:.2f}")
print(f"frontier:   {result.frontier_path}")
print(f"score plot: {result.score_chart_path}")
print(f"events:     {result.event_feed_path}")
```

`GepaRun.from_toml("gepa.toml").execute()` remains as a compatibility shim, but
new examples should use `OptimizerRun(GepaConfig(...)).execute()` with a
`ContainerConnection` from `synth-containers`.

Or from the CLI:

```bash
synth-optimizers gepa run --config gepa.toml        # one-shot optimization
synth-optimizers gepa service --db service.sqlite   # standing HTTP service
synth-optimizers events compare --left a.jsonl --right b.jsonl
```

A run needs a container URL and a TOML config. The
[GEPA cookbooks](https://github.com/synth-laboratories/synth-cookbooks-public/tree/main/cookbooks/optimizers/gepa)
have runnable examples across task shapes: Banking77, HotpotQA, MiniGrid, TBLite, and Crafter.

## Authentication and models

GEPA has **two independent inference boundaries**. Policy credentials stay in the
container (or Synth inference proxy). Proposer credentials stay on the host in a
run-local Codex `CODEX_HOME` bundle started by Rust GEPA.

| Boundary | Who calls the model | Where secrets live |
|----------|---------------------|--------------------|
| **Policy** (rollouts) | Your task container | Container env or proxy (`credential_mode = byok`) |
| **Proposer** (prompt edits) | Codex app-server (`runtime_substrate = "local"` or `"docker"`) | Host env + run-local `CODEX_HOME` |

The optimizer never embeds API keys in rollout HTTP requests. Proposer keys never
leave the host process.

Set `SYNTH_OPTIMIZERS_TERMINAL=1` to print live usage during a run
(`usage total=… policy=… proposer=… cost=$…`).

### OpenAI API key (default — cookbooks and CI)

Use the same key for policy rollouts and the Codex proposer. GEPA builds an isolated
run-local Codex home and does **not** read your host `~/.codex` login.

```bash
export OPENAI_API_KEY="sk-..."
export SYNTH_OPTIMIZERS_TERMINAL=1
```

```toml
[policy]
provider = "openai"
model = "gpt-4.1-nano"
api_key_env = "OPENAI_API_KEY"

[proposer]
backend = "codex_app_server"
runtime_substrate = "local"
execution_mode = "local_process"
provider = "openai"
auth_mode = "api_key"
api_key_env = "OPENAI_API_KEY"
copy_host_auth = false
model = "gpt-5.4-nano"
sandbox_mode = "workspace-write"
approval_policy = "never"
timeout_seconds = 900
```

`sandbox_mode` is the Codex in-agent sandbox policy. It is not the host-vs-Docker
choice; use `runtime_substrate` for that.

SDK equivalent:

```python
from synth_optimizers import GepaConfig, PolicyConfig, ProposerConfig

GepaConfig(
    policy=PolicyConfig(provider="openai", model="gpt-4.1-nano", api_key_env="OPENAI_API_KEY"),
    proposer=ProposerConfig.local(
        provider="openai",
        auth_mode="api_key",
        api_key_env="OPENAI_API_KEY",
        copy_host_auth=False,
        model="gpt-5.4-nano",
    ),
    # container, taskset, …
)

# Docker proposer (api_key only in v1):
GepaConfig(
    proposer=ProposerConfig.docker_substrate(
        image="ghcr.io/synth-laboratories/codex-gepa-proposer:2026-05-31",
        provider="openai",
        auth_mode="api_key",
        api_key_env="OPENAI_API_KEY",
        model="gpt-5.4-nano",
    ),
    # …
)
```

### OpenRouter proposer

Policy can stay on OpenAI while the proposer uses an OpenRouter model. Set
`provider = "openrouter"` and point `api_key_env` at your OpenRouter key.
GEPA writes a provider-aware Codex config with the OpenRouter base URL.

```bash
export OPENAI_API_KEY="sk-..."          # policy rollouts (container)
export OPENROUTER_API_KEY="sk-or-..."   # proposer
```

```toml
[policy]
provider = "openai"
model = "gpt-4.1-nano"
api_key_env = "OPENAI_API_KEY"

[proposer]
runtime_substrate = "local"
provider = "openrouter"
auth_mode = "api_key"
api_key_env = "OPENROUTER_API_KEY"
copy_host_auth = false
model = "x-ai/grok-4.3"
sandbox_mode = "workspace-write"
approval_policy = "never"
timeout_seconds = 900
```

OpenRouter also works for **policy** rollouts: set `[policy].provider = "openrouter"`
and ensure the container process has `OPENROUTER_API_KEY` in its environment.

### ChatGPT subscription proposer

Subscription models (for example `gpt-5.4-mini`) cannot be driven by a raw Platform
API key through Codex. Use ChatGPT OAuth and point GEPA at your authenticated Codex
home.

1. Install the [Codex CLI](https://github.com/openai/codex) and log in, **or** follow
   [opencode-openai-codex-auth](https://github.com/numman-ali/opencode-openai-codex-auth)
   to build a `~/.codex` OAuth bundle.
2. Confirm `~/.codex/auth.json` exists (`codex auth login`).
3. Configure the proposer — `codex_home` is **required**; GEPA does not silently fall
   back to your host home without it.

```toml
[policy]
provider = "openai"
model = "gpt-4.1-nano"
api_key_env = "OPENAI_API_KEY"

[proposer]
runtime_substrate = "local"
auth_mode = "chatgpt"
codex_home = "~/.codex"
copy_host_auth = true
model = "gpt-5.4-mini"
sandbox_mode = "workspace-write"
approval_policy = "never"
timeout_seconds = 900
```

Allowed proposer models for `auth_mode = "chatgpt"`: `gpt-5.4-mini`, `gpt-5.3-codex`,
`gpt-5.3-codex-spark`, `gpt-5.5`. Do **not** set `api_key_env` in this mode.

Subscription proposer turns are **billable $0** in usage totals; policy rollouts still
accrued against your API key spend normally.

### Gemini and other policy providers

Gemini and other OpenAI-compatible providers are supported on the **policy** side
only today. Configure `[policy].provider`, `base_url`, and the matching key env var
in the container environment. The reflective proposer remains Codex app-server
(OpenAI API key, OpenRouter, or ChatGPT subscription as above).

### Not supported yet: direct DeepSeek via Codex

Direct DeepSeek API keys through Codex app-server are **not** supported in this release.
Codex requires the Responses wire API; DeepSeek rejects `/responses`. Use an
OpenRouter DeepSeek slug for the proposer instead, or wait for a dedicated adapter.
See the [DeepSeek + Codex workaround notes](https://gist.github.com/antenore/c529e055e45559579b08b4961b517f8c).

Preflight errors are intentional: missing `OPENAI_API_KEY` for `auth_mode = "api_key"`,
missing `codex_home` / `auth.json` for `auth_mode = "chatgpt"`, or a disallowed
ChatGPT model id fail before rollouts start.

### Docker proposer substrate

Use Docker when the proposer should run isolated from the host process:

```toml
[proposer]
backend = "codex_app_server"
runtime_substrate = "docker"
execution_mode = "local_process"   # compatibility shim during migration
provider = "openai"
auth_mode = "api_key"
api_key_env = "OPENAI_API_KEY"
model = "gpt-5.4-nano"
sandbox_mode = "workspace-write"
approval_policy = "never"

[proposer.docker]
image = "ghcr.io/synth-laboratories/codex-gepa-proposer:2026-05-31"
workspace_mount_path = "/workspace"
network = "bridge"
extra_env = {}
```

Docker proposer workspaces are staged under
`~/.cache/synth-gepa-docker-workspaces/<run_id>-*/`, mounted into the container,
synced back to the run workspace, then removed. Docker unavailable is a preflight
error; GEPA does not retry on the local substrate.

When `sandbox_mode` is not `danger-full-access`, the Docker substrate grants the
container `SYS_ADMIN` with `seccomp=unconfined` so Codex can run its nested Linux
sandbox (`bubblewrap`) inside the container. This preserves the explicit Codex
sandbox policy instead of silently downgrading it.

`auth_mode = "chatgpt"` with `runtime_substrate = "docker"` is rejected in v1.
Use local substrate for subscription proposers.

Build or pull the pinned image before running docker profiles:

```bash
docker build -t ghcr.io/synth-laboratories/codex-gepa-proposer:2026-05-31 \
  docker/codex-gepa-proposer/
```

More detail for agents: [skills/gepa/SKILL.md](skills/gepa/SKILL.md).
Cross-repo task-container boundary: [containers/skills/containers/SKILL.md](https://github.com/synth-laboratories/containers/blob/main/skills/containers/SKILL.md).

## Engineering style

**Reliability tier:** 2 — Rust GEPA core and platform config are typed at boundaries;
Python SDK mirrors TOML contracts. Ruff + ty on changed Python; `cargo check` on
changed Rust crates before merge.

**Style source:** [SynthStyle](https://github.com/synth-laboratories/backend/blob/main/specifications/tanha/references/synthstyle.md)
(org rules also indexed under `Jstack/.jstack/style/synth_style.md`).

Contributors should prefer:

- One authoritative auth path per `proposer.auth_mode` (no silent `~/.codex` fallback).
- Actionable config errors (missing key, missing `codex_home`, disallowed ChatGPT model).
- Proposer Codex launch wiring in `rust/crates/synth_optimizer_platform/src/agent_runtime/`.
- `runtime_substrate` for host vs docker proposer; `sandbox_mode` for Codex in-agent policy.

Before merge on auth/substrate changes:

```bash
uv run ruff check src/synth_optimizers/gepa.py dev_examples/better_gepa/run_acceptance.py
uv run ty check src/synth_optimizers/gepa.py
cargo check -p synth_optimizer_platform -p synth_gepa -p synth_optimizers_py
cd dev_examples/better_gepa && python run_acceptance.py --profile openai_baseline --mode cost_stop
```

## Links

- [Cookbooks](https://github.com/synth-laboratories/synth-cookbooks-public/tree/main/cookbooks/optimizers/gepa) — runnable GEPA examples
- [synth-containers](https://github.com/synth-laboratories/containers) — the task contract
- [Agent skill](skills/gepa/SKILL.md) — drop into a coding agent to run and adapt GEPA
- [GEPA service OpenAPI](rust/crates/synth_gepa/openapi/gepa-service-v1.yaml)
- [GEPA paper](https://arxiv.org/abs/2507.19457)

## License

Apache-2.0
