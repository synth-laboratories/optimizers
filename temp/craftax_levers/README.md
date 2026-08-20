# Craftax lever containers (v0)

Split **env HTTP** + **policy service** + **GEPA orchestrator**. Toy Craftax-shaped
grid (collect adjacent wood, avoid lava). No JAX.

ReAct requires `OPENAI_API_KEY`. There is no local stub: `run_episode` calls the
model every tick and steps `parse_action(llm_text)` only.

The ReAct search object is an **entire script** (`react_loop.py`) that must define
`run_episode(env, prompt, seed=0, max_steps=16)`. `env` is an HTTP client to the
env service (`reset` / `step`). Applying a new script writes the file and
`POST /restart_policy` — the policy process is killed and respawned; the env
process stays up.

`SPEEDRUNNER_HARNESS` is a SpeedRunner-style actor ([arXiv:2608.11338](https://arxiv.org/abs/2608.11338)):
the LLM selects a public skill; the skill is a program that expands to primitive
`env.step`s without further model calls. That is a `harness_module` rewrite, not
a prompt overlay.

```
optimizer  -- /program /candidates /rollout -->  orchestrator
                                       |
                                       +--> supervisor POST /restart_policy
                                       +--> policy service  (subprocess, loads react_loop.py)
                                       |      run_episode(env, prompt) over HTTP
                                       +--> env service     (subprocess, never restarted)
```

## Run

```bash
cd temp/craftax_levers
uv sync --extra dev

uv run python run_code_policy.py   # GEPA http://127.0.0.1:19100
uv run python run_react.py         # GEPA http://127.0.0.1:19200
uv run python run_gepa.py          # GEPA search on code + ReAct (loads keys from backend/.env.local)
uv run pytest -q
uv run python verify.py
```
