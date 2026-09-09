<h1 align="center">synth-optimizers</h1>
<p align="center">A shared optimizer platform for AI applications — starting with open-source GEPA on a language-agnostic task contract.</p>
<p align="center">
<a href="https://pypi.org/project/synth-optimizers/">PyPI</a> ·
<a href="src/synth_optimizers/docs/gepa/containers/20-contract.md">Task contract</a> ·
<a href="https://github.com/synth-laboratories/synth-cookbooks-public/tree/main/cookbooks/optimizers/gepa">Cookbooks</a> ·
<a href="docs/hosted-optimizers.md">Hosted jobs</a>
</p>

`synth-optimizers` provides a shared Rust optimizer core for running search
algorithms against any task exposed through the public optimizer HTTP task
contract.

- **Shared platform core** — reusable Rust machinery for container I/O, workspaces, cache profiles, budgets, telemetry, failure handling, and replayable evidence.
- **Algorithm layer** — [GEPA](https://gepa-ai.github.io/gepa/) is the first public optimizer; future algorithms can plug into the same platform contract.
- **GEPA runs today** — configure GEPA with TOML or `GepaConfig`; it proposes prompt changes, rolls them out, scores them, keeps a Pareto frontier, and emits inspectable run evidence.

## Supported algorithms

| Algorithm | Status | In this repo | Paper & docs |
| --- | --- | --- | --- |
| **GEPA** — reflective prompt evolution | Supported | [`rust/crates/synth_gepa/`](rust/crates/synth_gepa/) (Rust engine + service), [`src/synth_optimizers/gepa.py`](src/synth_optimizers/gepa.py) (Python API), [`skills/gepa/SKILL.md`](skills/gepa/SKILL.md) (agent runbook) | [Paper](https://arxiv.org/abs/2507.19457) · [gepa-ai docs](https://gepa-ai.github.io/gepa/) · bundled HTML via `synth-optimizers gepa console` |
| **GELO** — Go-Explore in prompt space (hosted) | Hosted submit | [`src/synth_optimizers/gelo.py`](src/synth_optimizers/gelo.py), [`skills/gelo/SKILL.md`](skills/gelo/SKILL.md), [`GELO_HOSTED_SDK_CLI_SPEC.md`](GELO_HOSTED_SDK_CLI_SPEC.md) | Bundled HTML via `synth-optimizers gelo console` — [`src/synth_optimizers/docs/gelo/`](src/synth_optimizers/docs/gelo/) |
| **SFT** — supervised fine-tuning | Local + hosted submit | `HostedOptimizerClient.submit_sft()` / `SftService` / `TinkerSftExecutor` | In-process Tinker executor in this repo. Default model `openai/gpt-oss-20b`. |
| **CISPO** — `cispo.slime.v1` | Local + hosted submit | `HostedOptimizerClient.submit_cispo()` / `TinkerCispoExecutor` | True slime CISPO only. Generic importance sampling is not CISPO. |

The shared [`synth_optimizer_platform`](rust/crates/synth_optimizer_platform/)
crate is the substrate for optimizer implementations; GEPA is the first public
local algorithm. GELO remains hosted-only. Standalone SFT and CISPO execute in
this repository against Tinker. Hosted submission is covered in
[`docs/hosted-optimizers.md`](docs/hosted-optimizers.md). Identity rules are in
[`docs/sft-cispo-identity.md`](docs/sft-cispo-identity.md).

### SFT control plane

SFT is served by `synth-optimizers` with an in-process Tinker executor. No
`optimizers-beta` process, URL, or service token is required.

```bash
export TINKER_API_KEY=...
export SYNTH_OPTIMIZERS_SFT_SERVICE_TOKEN=local-qa-token
# Fixture-only local QA without paid Tinker work:
export SYNTH_OPTIMIZERS_SFT_FIXTURE=1
synth-optimizers sft service --db .sft/service.sqlite --bind 127.0.0.1:8878
```

Submit, inspect, and cancel only through the façade:

```bash
synth-optimizers sft validate --config sft.toml
synth-optimizers sft submit --config sft.toml --follow
synth-optimizers sft watch RUN_ID --events
synth-optimizers sft cancel RUN_ID
```

The façade keeps executor-only workspace paths and service credentials private. Its
artifact proxy is available at `/v1/runs/RUN_ID/artifacts/{manifest,events}`.

## Future hosted-algorithm compatibility

MAPO, OHCO, Online Reflexion, and MARL prompt-optimization identifiers are retained
in [`future_algorithms.py`](src/synth_optimizers/future_algorithms.py) so clients can
parse hosted catalogs and historical runs. They are **not supported public optimizer
algorithms**: they carry no local executor, cookbook, or release commitment. New
public algorithms graduate into the table above only after their public API contract,
replay semantics, and end-to-end evidence are ready.

## Install

```bash
pip install synth-optimizers
# or
uv add synth-optimizers
```

This source targets `synth-optimizers==0.2.22` with `synth-containers==0.4.2`.
For an unpublished candidate, build from a checkout as shown below; published
versions are listed on PyPI.

Install [`uv`](https://github.com/astral-sh/uv) for local development and editable installs.

## Local development

Clone the repo and install the local Python/Rust extension in editable mode:

```bash
git clone https://github.com/synth-laboratories/optimizers.git
cd optimizers
uv sync --group dev
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

[taskset]
train_ids = ["train:0", "train:1", "train:2", "train:3"]
heldout_ids = ["test:100", "test:101"]

[gepa.task_pools]
pareto = ["train:0", "train:1", "train:2", "train:3"]
minibatch = ["train:0", "train:1"]
reflection = ["train:0", "train:1", "train:2", "train:3"]
heldout = ["test:100", "test:101"]
```

```python
from synth_optimizers import GepaRun

# Use a complete cookbook config with its task service, policy, and proposer.
# Configure authorized provider credentials before executing a paid run.
result = GepaRun.from_toml("gepa.toml").execute()

print(result.best_candidate)
print("cost: unknown" if result.cost_usd is None else f"cost: ${result.cost_usd:.2f}")
```

The TOML above illustrates task selection, not a standalone task server. Run it
from the GEPA cookbook directory and add the recipe's policy/proposer settings.
The legacy `[dataset]` seed selection is not the current GEPA schema.

CLI:

```bash
synth-optimizers gepa run --config gepa.toml
synth-optimizers gepa service --db service.sqlite
synth-optimizers events compare --left a.jsonl --right b.jsonl
```

Runnable task examples are **not in this repository**. They live in the separate
public repo
[`synth-laboratories/synth-cookbooks-public`](https://github.com/synth-laboratories/synth-cookbooks-public/tree/main/cookbooks/optimizers/gepa)
— Banking77, HotpotQA, MiniGrid, and Crafter. TBLite is optional evaluation
infrastructure. HealthBench is parked because Containers 0.4.2 does not include
its runtime. Config-relative paths resolve against the config file's directory.
Follow the selected cookbook's setup instructions before launching:

```bash
git clone https://github.com/synth-laboratories/synth-cookbooks-public.git
cd synth-cookbooks-public/cookbooks/optimizers/gepa/banking77_container
synth-optimizers gepa run --config gepa.toml
```

The cookbook configs published there still declare the legacy `[dataset]` seed
selection, which `0.2.22` ignores; add `[taskset]` and `[gepa.task_pools]`
blocks like the ones in the quickstart above before one of them will load.

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
- **ChatGPT subscription proposer** — `auth_mode = "chatgpt"` with required `codex_home` (OAuth via [Codex CLI](https://github.com/openai/codex) or [opencode-openai-codex-auth](https://github.com/numman-ali/opencode-openai-codex-auth)); models include `gpt-5.4-mini`, `gpt-5.4`, `gpt-5.3-codex`, `gpt-5.3-codex-spark`, `gpt-5.5`, `gpt-5.6-luna`, `gpt-5.6-sol`, and `gpt-5.6-terra`; proposer usage is $0, policy rollouts still bill normally.
- **Nano-Codex proposer harness** — explicit `[proposer.nano_codex]` opt-in keeps one ChatGPT-authenticated app-server session warm across compatible GEPA generations, caches static task/program context by content digest, records monotonic JSONL events and typed turn receipts, and can replay receipts with zero live model or tool calls. See [`dev_examples/nano_codex_gepa/`](dev_examples/nano_codex_gepa/).
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
- [GEPA task contract](src/synth_optimizers/docs/gepa/containers/20-contract.md) — the public HTTP task contract
- [uv](https://github.com/astral-sh/uv) — Python package and project manager
- [GEPA service OpenAPI](rust/crates/synth_gepa/openapi/gepa-service-v1.yaml)

## License

Apache-2.0
