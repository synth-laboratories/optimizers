# synth-optimizers skills

Portable agent skills for working with the public optimizers in this repo.

- `gepa/` — run, configure, debug, and adapt the public Rust GEPA optimizer
  (proposer auth, `runtime_substrate` local/docker, task containers, budgets).
  GEPA is **local and hosted**.
- `gelo/` — configure and debug **hosted-only** GELO (Go-Explore in prompt space):
  per-LLM-call checkpoints, async container contracts, achievement ladders, resume
  semantics, three-layer storage (env container / optimizer evidence / hosted durable),
  and `HostedOptimizerClient` submit/watch. Local execute is not part of the
  public product surface.

Each skill is a folder with a required `SKILL.md` (Agent Skills format). Runnable
cookbooks that exercise these optimizers live in
[`synth-cookbooks-public`](https://github.com/synth-laboratories/synth-cookbooks-public)
for public GEPA examples and the hosted GELO launch quickstart.
