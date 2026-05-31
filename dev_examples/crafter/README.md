# Crafter GEPA Container

Crafter-style achievement policy optimization through the public GEPA container
contract.

This dev example is a compact public fixture, not a full private Crafter/Go-EX
service. Each rollout runs a deterministic text simulation with Crafter-like
actions, resources, crafting preconditions, sparse achievement rewards, and
achievement side info.

## Required Env

- `GEMINI_API_KEY` - required for live policy calls.
- Optional: `CRAFTER_POLICY_MODEL` (default `gemini-3.1-flash-lite`),
  `CRAFTER_POLICY_API_KEY_ENV` (default `GEMINI_API_KEY`),
  `CRAFTER_POLICY_BASE_URL` (default Gemini OpenAI-compatible endpoint), and
  `CRAFTER_MAX_TURNS` (default `12`).

## Contract

- `GET /metadata` advertises the GEPA route contract.
- `GET /program` exposes one mutable module: `react_system_prompt`.
- `POST /taskset/tasks` returns deterministic Crafter achievement scenarios.
- `POST /rollout` runs the action policy and returns:
  - `reward_info.outcome_reward`: fraction of target achievements unlocked.
  - `actionable_side_info.achievements`: all unlocked achievements.
  - `actionable_side_info.missing_achievements`: remaining target achievements.
  - `trace.event_history`: per-turn action and achievement events.

The top-level `actionable_side_info` field is intentional: the GEPA platform
projects it into sensor and reflective frames for proposer use.

## Run With SDK

From the repository root:

```bash
bash dev_examples/crafter/run_fresh_gepa.sh
```

`run_gepa_sdk.py` serves the container through `synth_containers.Container`,
builds `GepaConfig` directly, runs async rollouts, and optimizes two objectives:
achievement unlock rate and turn count. It also starts from
`ProposerPromptConfig.from_defaults()` and appends Crafter-specific proposer
guidance for achievement side information.
