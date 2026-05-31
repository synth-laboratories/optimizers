# Agent Runtime

Tier: 2

This subtree owns generic agent execution plumbing for optimizer algorithms:
credential launch resolution, run-local Codex homes, local/Docker proposer
substrates, Codex app-server JSON-RPC, and Codex usage normalization.

Algorithm crates own their domain workspaces, prompts, and result parsing. They
call this platform boundary to run a turn and inspect normalized transport
messages. The platform crate must not import `synth_gepa`.

## Modules

- `substrate`: the `ExecutionSubstrate` typology (`local`, `docker`).
- `session`: one Codex turn boundary shared by substrates.
- `local`: host subprocess substrate.
- `docker`: staged-workspace Docker substrate; no host fallback.
- `supervisor`: operator-visible process/container cleanup receipts.
- `codex_home`: prepares API-key and ChatGPT-authenticated Codex homes.
- `app_server`: starts the selected app-server process, sends JSON-RPC requests,
  waits for turn lifecycle events, and drains stderr for actionable failures.
- `usage`: normalizes Codex turn usage messages into flat compatibility usage.
