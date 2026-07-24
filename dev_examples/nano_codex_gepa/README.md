# Nano-Codex for public GEPA

Nano-Codex is an opt-in proposer execution harness for public Rust GEPA. It
keeps the Codex app-server and GEPA proposer thread alive across compatible
generations instead of paying process, authentication, and thread setup costs
for every proposal.

The normal GEPA workspace contract and manifest importer remain authoritative.
Nano-Codex only changes how the proposer turn is executed:

- the session key binds the GEPA run, proposer role, model/config, task info,
  prompt program, target modules, and proposer prompt;
- changing any static input creates a new session instead of reusing stale
  context;
- every turn has a typed run/role/round/treatment/parent/workspace identity;
- live turns emit monotonic JSONL events and a typed receipt with cold-start,
  first-token, tool, manifest-validation, and total latency;
- replay reads a recorded turn receipt and makes zero live model or tool calls;
- timeout interrupts the active turn and invalidates the session;
- tools outside `search`, `read`, `apply_patch`, and `exec` fail the turn;
- there is no one-shot or API-key fallback when Nano-Codex is enabled.

Nano-Codex requires the local proposer substrate and the existing ChatGPT
Codex authentication bundle. It does not accept a raw model API key as an
alternate path.

Merge [`nano_codex_fragment.toml`](nano_codex_fragment.toml) into a complete
GEPA profile, then run GEPA normally:

```text
synth-optimizers gepa run --config gepa.toml
```

Receipts default to each proposer workspace's `.nano_codex/` directory. Set
`record_dir` to collect all generations under one run-level directory.

## Replay

Use the same GEPA inputs and request identities, point `replay_dir` at the
original record root, and use a distinct `record_dir` for replay receipts:

```toml
[proposer.nano_codex]
enabled = true
mode = "replay"
record_dir = ".out/nano-replay"
replay_dir = ".out/nano-live"
max_turns_per_session = 16
allowed_tools = ["search", "read", "apply_patch", "exec"]
```

Replay validates both the typed request identity and static-context digest. A
changed task, program, target module, proposer prompt, model, or proposer
configuration fails loudly instead of replaying incompatible output.

## Captured proof

`evidence/banking77_live_replay_v6.json` records a two-generation live run and
its full replay. The second live generation reused both the persistent
nano-Codex session and static-context cache. Replay restored each generation's
`proposal/manifest.json`, made zero live model or tool calls, and produced the
same accepted best candidate and rewards as the live run.

## Compatibility boundary

The feature is disabled by default. Existing GEPA profiles, candidate
manifests, workspace layouts, proposal parsing, and candidate admission are
unchanged when it is off. Reflective staleness review currently rejects a
Nano-Codex configuration rather than falling back to the one-shot proposer.
