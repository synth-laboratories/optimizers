# Migration: Tinker SFT and CISPO into public `optimizers`

Public `optimizers` is the authoritative, self-contained home for standalone Tinker
SFT and true CISPO (`cispo.slime.v1`). `optimizers-beta` is a historical/reference
implementation only and is not required at runtime.

Source inventory is relative to:

- public `optimizers` at `eaead59b03118fb3eaede1467e3ead9ab015b12b` before this landing
- `optimizers-beta` `933af0e9e437d9fcd3b51d49e041ec91a66bc9e0`
- `optimizers-beta` training crate snapshot `d0b8577040cad9a52b45125eee4a3094b40c3185`
- slime upstream `41014d1f29e201137fdffce737bb8bac65bc5219`

## Provenance

| Original beta module | Public module | Mode | Behavioral differences |
| --- | --- | --- | --- |
| `crates/synth_training/src/algorithms/cispo_slime/mod.rs` | `src/synth_optimizers/cispo.py` | Adapted | Same clipping, stop-gradient, and unbiased group std. Python port of the pinned slime fixture. |
| `crates/synth_training/src/provider.rs` | `src/synth_optimizers/providers/protocols.py` | Adapted | Independent capability names instead of one “Tinker available” flag. |
| `crates/synth_training/src/providers/tinker/mod.rs` | `src/synth_optimizers/providers/tinker/` | Adapted | Shared adapter for SFT and CISPO. Credentials from `TINKER_API_KEY` / `TINKER_BASE_URL` only. No Keychain. |
| `crates/synth_go_ex/src/plugins/sft.rs` | `src/synth_optimizers/sft_executor.py` | Adapted | Standalone `algorithm_id="sft"`. Does not define or reuse `goex.sft.v1`. |
| `src/sft_standalone.rs` | *(not copied)* | Replaced | Beta smoke artifact generator is not the public executor. |
| Hosted `BetaSftExecutorClient` | `TinkerSftExecutor` | Replaced | In-process Tinker execution. No beta URL, token, or HTTP executor. |
| nanoclassify `train_tinker_banking77.py` | `recipes/banking77.py` + CISPO loop | Adapted | Banking77 recipes keep frozen held-out identity. CISPO uses slime clip bounds, not a generic Tinker IS run. Live chat tokens use Prime Intellect `renderers` (`gpt-oss` Harmony), not a second `apply_chat_template` pass. |

Copyright and license headers are preserved on adapted CISPO math (`Apache-2.0`).

## Identity rules

| Surface | `algorithm_id` | Implementation |
| --- | --- | --- |
| Standalone SFT | `sft` | `sft.tinker.v1` |
| Standalone CISPO | `cispo` | `slime-reference` / `cispo.slime.v1` |
| GoEx SFT lane | `go-ex` | Plugin `goex.sft.v1` may later consume the public executor; it is not the public SFT contract |
| Generic importance sampling | *(not CISPO)* | Preflight returns `unsupported` |

CISPO starts only when every capability is present:

- `sft.train`
- `checkpoint.sample`
- `rollout.grouped`
- `trajectory.logprobs`
- `training.importance_weights`
- `cispo.slime.v1` (validated after a real canary)

`cispo.slime.v1` is validated after a paid canary. The first live receipt is
`docs/receipts/tinker-gpt-oss-20b-banking77-canary-cispo/cispo.slime.v1.receipt.json`
(`validated=true`, `paid_update=true`, `openai/gpt-oss-20b`,
`renderers.gpt-oss.low.v1`). Point a live client at it with
`TINKER_CISPO_VALIDATION_RECEIPT`. Fixture tests may still mark it validated
explicitly. `allow_unvalidated_canary` remains for a first paid run only.

Bounded live canary (requires `TINKER_API_KEY` or `--env-file`):

```bash
uv run python scripts/run_tinker_banking77_canary.py \
  --env-file /path/to/.env \
  --output-dir docs/receipts/tinker-gpt-oss-20b-banking77-canary
```

Reuse an existing SFT checkpoint instead of paying for another SFT step with
`--sft-events path/to/sft.events.json`. Tinker sampler names are sanitized
(no colons) before `save_weights_for_sampler`.

## Cutover

1. Shared Tinker adapter is the only provider client.
2. Public SFT service runs `TinkerSftExecutor` in-process.
3. `BetaSftExecutorClient` and beta runtime configuration are deleted.
4. CISPO closed loop is wired behind the same job store and event journal.
5. Banking77 recipes are first-class and digest-addressed.
6. Workshop consumes `read_models` summaries and paginated collections.

## Workshop and release

Read models live in `src/synth_optimizers/read_models.py`. Release-owned visual
and CUA journeys belong in `workshop-release`, not the Workshop application
repository.

Live Desktop SFT/CISPO (click recipe → `optimizer.sft.live.v1` /
`optimizer.cispo.live.v1` collections) is scoped in
`workshop-readmodel-cua/docs/HANDOFF_SFT_CISPO_LIVE_GEPA_PARITY.md`. Public
services already emit GEPA-shaped event pages; remaining work is operator
process wiring, a CISPO CLI module (not `cli.py`), and Desktop sending a real
`cispo.request.v1` instead of a container-bind stub.
