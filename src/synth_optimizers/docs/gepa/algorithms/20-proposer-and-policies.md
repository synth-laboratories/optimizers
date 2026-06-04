# Proposer & policies

A GEPA run has two model surfaces, and they are deliberately separate:

- **Proposer** — the reflective model (Codex) that reads rollout evidence and **writes new
  candidate prompts**. Runs on the host (or in Docker). Never sees rollout traffic.
- **Policy** — the model that **runs the rollouts** inside the task container. Billed per
  rollout. Rollout requests never carry proposer keys.

## The proposer

The proposer is a Codex app-server. Configure it under `[proposer]` (TOML) or
`ProposerConfig` (SDK). Key fields:

| Field | Meaning |
|-------|---------|
| `backend` | `codex_app_server` (default) or `deepseek_chat`. |
| `runtime_substrate` | `local` (host) or `docker` (`[proposer.docker].image`). |
| `provider` | `openai`, `openrouter`, `deepseek`. |
| `auth_mode` | `auto`, `api_key`, `chatgpt`, or `host`. |
| `model` | e.g. `gpt-5.4-nano`, `gpt-5.4-mini`, `gpt-5.5`. |
| `reasoning_effort` | `none` / `low` / `medium` / `high`. |
| `execution_mode` | `local_process` / `stdio` (JSON-RPC over stdin/stdout) or `websocket` / `ws`. |
| `service_tier` | `default` or `fast` (Codex Fast mode; requires ChatGPT auth). |

### Auth modes

- **`api_key`** — run-local Codex home keyed by an env var (e.g. `OPENAI_API_KEY`); does
  not touch your host `~/.codex` login. Billed normally.
- **`chatgpt`** — ChatGPT subscription via a required `codex_home` (OAuth through the Codex
  CLI). **Proposer usage is $0**; policy rollouts still bill. Required for `service_tier = "fast"`.
- **`host`** — reuse the host's Codex login as-is.
- **`auto`** — pick based on what's configured.

Any `--proposer-*` CLI flag on `gepa run` overrides the matching `[proposer]` field for a
single run (see the [CLI reference](#/cli)).

> Preflight validation fails *before* any rollout if a key is missing, a `codex_home` /
> `auth.json` is absent, or a disallowed ChatGPT model is requested — so misconfigured
> auth never burns rollout budget.

## Policy types

The policy type is how the container executes a rollout for a candidate. It is a property
of the **container**, not something GEPA imposes — but knowing it tells you what the
mutable modules mean.

| Type | Module shape | Used by |
|------|--------------|---------|
| `dag` | A staged program — e.g. `stage1_system`, `stage2_system`. GEPA mutates the stage prompts. | Banking77, HotpotQA, HoVer, HealthBench |
| `react` | A ReAct agent loop driven by a single `react_system_prompt` / `system_prompt`; the policy observes, thinks, acts over many turns. | MiniGrid, Crafter, DungeonGrid, TAU2 Retail |
| `codex` | A Codex app-server sandbox policy (tool use, reasoning) optimized via its `system_prompt`. | FinQA, TBLite-style coding |

The policy's provider/model is set per rollout via the container's `[policy]` config
(`provider`, `model`, `base_url`, `api_key_env`). Providers seen across the catalog include
OpenAI, OpenRouter, Gemini, and DeepSeek. See the [container catalog](#/containers/catalog)
for each task's policy.

## Why the split matters

Because the proposer never touches rollout traffic and the policy never sees proposer
credentials, you can — for example — run a $0 ChatGPT-subscription proposer while policy
rollouts bill against a cheap nano model, and no proposer key can leak into a container
request. The [SDK reference](#/sdk) shows the full TOML for both surfaces.
