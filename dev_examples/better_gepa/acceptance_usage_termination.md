# Acceptance — usage termination across API profiles

Planning source of truth: [`better-ai-auth.txt`](./better-ai-auth.txt) §3.10 (desired UX), §10 (criteria), §11 (merge checklist).

Goal: prove **BudgetEnvelope limits bind** for policy + proposer usage across provider
profiles — not merely that a run completes.

## Prerequisites

External guides (see [better-ai-auth.txt](./better-ai-auth.txt) References):

- **Codex OAuth / ChatGPT proposer:** [opencode-openai-codex-auth](https://github.com/numman-ali/opencode-openai-codex-auth)
- **DeepSeek direct + Codex (gated):** [antenore gist](https://gist.github.com/antenore/c529e055e45559579b08b4961b517f8c)

- Banking77 container: `dev_examples/banking77/banking77_synth_gepa_dev.py --serve`
- Optimizers built with `synth-optimizers` CLI or Python `OptimizerRun`
- API keys present in environment (never commit):

| Profile | Required env |
|---------|----------------|
| `openai_baseline` | `OPENAI_API_KEY` |
| `openai_baseline_docker` | `OPENAI_API_KEY`, Docker/OrbStack, proposer image |
| `openrouter_grok43` | `OPENAI_API_KEY`, `OPENROUTER_API_KEY` |
| `openrouter_grok43_docker` | `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, Docker/OrbStack, proposer image |
| `deepseek_v4_flash` | `OPENAI_API_KEY`, `DEEPSEEK_API_KEY` + `non_western_provider=OK` in TOML |
| `chatgpt_mini` | `OPENAI_API_KEY`, `codex auth login` → `~/.codex` |

## Profiles

Fragment TOMLs under [`profiles/`](./profiles/). Each profile merges:

1. Banking77 container + taskset (from existing `gepa.toml` pattern)
2. Profile-specific `[policy]` / `[proposer]` / `[run]`
3. Shared tight limits from [`profiles/_termination_limits.toml`](./profiles/_termination_limits.toml)

| Profile file | Proposer model | Notes |
|--------------|----------------|-------|
| `banking77_openai_baseline.toml` | `gpt-5.4-nano` (OpenAI API) | Baseline |
| `banking77_openai_baseline_docker.toml` | `gpt-5.4-nano` (OpenAI API) | Docker proposer substrate |
| `banking77_openrouter_grok43.toml` | `x-ai/grok-4.3` (OpenRouter) | First-class OR |
| `banking77_openrouter_grok43_docker.toml` | `x-ai/grok-4.3` (OpenRouter) | Docker proposer + provider routing |
| `banking77_deepseek_v4_flash.toml` | `deepseek-v4-flash` (direct) | Gated |
| `banking77_chatgpt_mini_proposer.toml` | `gpt-5.4-mini` (ChatGPT sub) | Billable proposer $0 |

## Acceptance modes

Run each profile in two modes (harness applies overrides to `_termination_limits.toml`):

### `cost_stop`

- `max_cost_usd = 0.15`
- `max_total_tokens = 2_500_000` (effectively disabled)

**Expect:** terminal `cost_budget_reached` (or `within_budget` with spent ≥ $0.10 once usage_v2 prices rollouts).

### `token_stop`

- `max_cost_usd = 50.0` (effectively disabled)
- `max_total_tokens = 80_000`

**Expect (Phase 3):** terminal `total_token_budget_reached`.

**Interim (today):** ledger admission rejection + graceful finish; assert reserved/spent prompt tokens approach cap without silent LOW-as-zero.

## Assertions (all profiles)

- [ ] `validate_credentials()` passes (or fails without gate keys — deepseek negative case)
- [ ] Run produces ≥1 proposer turn with workspace manifest
- [ ] Terminal status is one of: `cost_budget_reached`, `rollout_budget_reached`, `total_token_budget_reached`, `completed` (if limits too loose — fail the test)
- [ ] `budget_ledger` / result manifest shows spent totals consistent with terminal reason
- [ ] No usage row reported as billable `HIGH` with value `0` when tokens > 0 (post usage_v2)

## Profile-specific

### `openai_baseline`

- [ ] Proposer auth_mode `api_key`; no codex_home required
- [ ] Policy + proposer usage both contribute to cost stop

### Docker proposer profiles

- [ ] `runtime_substrate = "docker"` in final TOML and proposer events
- [ ] Staged workspace is removed after the run
- [ ] `proposal/manifest.json` is synced back into the run workspace
- [ ] Proposer tokens are non-zero, proving Docker usage telemetry parity

### `openrouter_grok43`

- [ ] Proposer provider `openrouter`, model `x-ai/grok-4.3`
- [ ] Proposer turn completes; proposal manifest parseable
- [ ] Usage provenance references OpenRouter / provider API on proposer rows

### `deepseek_v4_flash`

- [ ] Config validation **rejects** run without `[run] non_western_provider = "OK"`
- [ ] Proposer provider `deepseek`, model `deepseek-v4-flash`
- [ ] Proposer billing_class `non_western_api` in usage summary (post usage_v2)
- [ ] Cost stop driven primarily by OpenAI policy spend until DeepSeek pricing wired

### `chatgpt_mini`

- [ ] Proposer billable cost = $0 (`subscription_zero`); api_equivalent > 0
- [ ] Cost stop reflects policy rollouts only
- [ ] Token limits bind on proposer HIGH tokens

## Harness (planned)

```bash
cd optimizers
python dev_examples/better_gepa/run_acceptance.py --profile openai_baseline --mode cost_stop --substrate local
python dev_examples/better_gepa/run_acceptance.py --profile openai_baseline_docker --mode cost_stop
python dev_examples/better_gepa/run_acceptance.py --profile openrouter_grok43 --mode cost_stop
python dev_examples/better_gepa/run_acceptance.py --profile openrouter_grok43_docker --mode cost_stop
```

Use `SYNTH_GEPA_DOCKER_PROPOSER_IMAGE=<image>` to override the pinned Docker image
for local image-build validation. If `docker info` fails, Docker acceptance exits
with an explicit skip message.

## Go-green phases

| Phase | What turns green |
|-------|------------------|
| 1 (now) | Preflight gates; openai baseline runs; deepseek gate reject/accept |
| 2 (usage_v2) | Cost stop on billable lower bound; HIGH rollout tokens from container |
| 3 (stopper) | Token stop terminal reason; stopper snapshot includes token fields |
