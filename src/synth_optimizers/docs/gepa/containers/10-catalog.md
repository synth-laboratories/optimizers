# Container catalog

Every container shipped under `cookbooks/optimizers/gepa/`. Each is a self-contained task:
launch it, point a `gepa.toml` at its URL, and run. Ports below are the defaults each
`synth_service_app.py` binds.

## At a glance

| Container | Task | Policy | Mutable modules | Scoring |
|-----------|------|--------|-----------------|---------|
| `banking77_container` | Banking77 intent classification (77 labels) | dag | `stage2_system` | Exact-match accuracy |
| `banking77_container_rust` | Banking77, in Rust (Axum) | dag | `stage2_system` | Exact-match accuracy |
| `banking77_container_ts` | Banking77, in TypeScript (Hono) | dag | `stage2_system` | Exact-match accuracy |
| `hotpotqa_container` | HotpotQA multi-hop QA over distractors | dag | `stage1_system` | Token F1 vs gold |
| `hover_container` | HoVer claim verification (supports/refutes) | dag | `stage1_system`, `stage1_user` | Binary accuracy |
| `healthbench_container` | HealthBench professional medical QA | dag | `stage1_system` | LLM rubric (fraction passed) |
| `harvey_lab_container` | Harvey LAB tax-law legal agent | dag | `system_prompt` | LLM rubric (fraction passed) |
| `finqa_container` | FinQA financial QA over 10-K tables | codex | `system_prompt` | FinQA numeric verifier |
| `tblite_container` | Terminal-Bench-Lite Python coding | codex | `starting_prompt` | Real pytest (1.0/0.0) |
| `minigrid_container` | MiniGrid Gymnasium (DoorKey etc.) | react | `system_prompt` | Episode reward |
| `crafter_container` | Craftax survival agent | react | `react_system_prompt` | Episode reward |
| `dungeongrid_container` | DungeonGrid multi-hero dungeon | react | `react_system_prompt` | Episode reward (+achievements) |
| `tau2_retail_container` | TAU2 retail customer-service agent | react | `domain_policy` | TAU2 native evaluator |

Policy types are explained in [Proposer & policies](#/algorithms/proposer-and-policies).

## Classification & QA (dag policies)

- **Banking77** (`banking77_container`, port 8765) — classify a customer query into one of
  77 intents; dataset `PolyAI/banking77`. The canonical "hello world" of GEPA: one mutable
  system prompt, exact-match reward, fast rollouts. The **Rust** (port 8810) and
  **TypeScript** (port 8810) ports run an embedded 6-row fixture to prove the contract is
  language-agnostic.
- **HotpotQA** (`hotpotqa_container`, port 8772) — multi-hop QA over distractor passages;
  reward is token F1 against gold spans.
- **HoVer** (`hover_container`) — evidence-based claim verification; optimizes both a system
  and a user template (`stage1_system`, `stage1_user`).
- **HealthBench** (`healthbench_container`, port 8814) — physician answers scored by an LLM
  rubric judge against per-row criteria; reward = fraction of criteria passed.
- **Harvey LAB** (`harvey_lab_container`, port 8771) — legal associate on tax matters,
  rubric-judged; run `prepare_dataset.py` first.

## Tool-using agents (codex policies)

- **FinQA** (`finqa_container`, port 8106) — a Codex app-server analyst inspects SEC 10-K
  tables; scored by the real FinQA numeric verifier. Requires a Codex ChatGPT home.
- **TBLite** (`tblite_container`, port 8770) — Terminal-Bench-Lite coding tasks; the agent
  writes `solution.py` and is verified by **real pytest** in an isolated subprocess
  (30s timeout). Binary reward: all tests pass or nothing. 51 tasks (35 train / 16 heldout).

## Environments (react policies)

These run full multi-turn episodes per rollout — real model cost per turn.

- **MiniGrid** (`minigrid_container`, port 8769) — Gymnasium `MiniGrid-DoorKey-5x5-v0` and
  variants; reward only on goal success, discounted by steps.
- **Crafter** (`crafter_container`, port 8768) — a real Craftax (JAX) survival env; the
  ReAct prompt is tuned to prioritize wood → table → tools → stone/coal/iron and avoid lava.
- **DungeonGrid** (`dungeongrid_container`, port 8773) — turn-based dungeon crawler with
  multi-hero party control; reward includes achievement bonuses.
- **TAU2 Retail** (`tau2_retail_container`, port 8774) — `tau2-bench` retail with a user
  simulator and a tool suite; GEPA optimizes the `domain_policy`. 20 train / 94 heldout.

## Launching a container

Each Python container is launched the same way (see its `README.md` for exact flags):

```bash
cd cookbooks/optimizers/gepa
uv run python banking77_container/synth_service_app.py --host 127.0.0.1 --port 8765
```

Then in `gepa.toml`:

```toml
[container]
url = "http://127.0.0.1:8765"
command = ["uv", "run", "python", "banking77_container/synth_service_app.py", "--port", "8765"]
```

`command` lets `gepa run` start the container for you. To author your own task, see the
[contract](#/containers/contract).
