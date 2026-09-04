# Container-first RL: handoff

Written 2026-09-03 and completed later that day. Nothing is pushed. The
original snapshot remains below the completion record so failed attempts and
the reasons for the final design are not erased.

## Completion record

The runnable milestone is complete across the five real images:

- Free socket matrix: Banking77, HealthBench2, Craftax, Harbor-TBLite, and
  DungeonGrid all reached one update. DungeonGrid published both trainable
  parameter groups, proving the multi-policy route and roster binding.
- Paid floor: Banking77, HealthBench2, Craftax, and Harbor-TBLite each ran
  exactly three groups against Tinker with renderer agreement proven by the
  real gpt-oss canary (`cad55fad220833fbcddf19ad833e1f53`). HealthBench2
  reached one update over 12 examples / 12,038 training tokens and published
  `pg-answer@1`. The other three correctly skipped three zero-advantage groups
  apiece; they sampled real policies but did not fabricate a training update.
- Paired evaluation: `b77-paired-free-01` compared immutable baseline and
  trained checkpoints on the same task/seed and wrote a durable receipt. Both
  arms scored 0.0, so the recorded result is a tie rather than missing data.

Durable evidence is under `/tmp/synth-container-first-e2e`, notably
`receipts_healthbench2_paid_02`, `receipts_banking77_paid`,
`receipts_craftax_paid_02`, `receipts_tblite_paid_05`, and
`evaluation-b77-paired-free-01`. Tinker's adapter did not return a monetary
cost (the receipt marks `cost_missing`), so the run cannot honestly state an
actual dollar total; execution remained inside the declared $20 aggregate cap.

Defects fixed during completion:

- origin-backed probe policies are dispatchable but forcibly non-trainable;
- multi-agent seats get distinct routes even when they share weights;
- multi-policy pins no longer claim one component's revision as the whole set;
- image evidence uses authoritative gateway prompt ids and gateway-declared
  branch/compaction provenance;
- the renderer canary now compares actual provider tokens;
- paired evaluation finalizes `awaiting_score`, verifies artifacts in the live
  plane, closes that plane, and writes the receipt;
- the harness is durable, returns the real CLI exit code, and keeps artifacts
  outside the repository;
- paid Tinker assembly warms a named session before renderer verification;
  TBLite additionally uses a declared 32K prompt-compaction budget and a
  three-step conformance horizon.

The TBLite decision is to exercise its external bound-policy route. That route
preserves raw Tinker text/tokens and does not call MiniSwe's direct
`_complete_tinker_sampler` rewrite. The direct harness rewrite remains
incompatible with strict-prefix training evidence and is not represented as
tested.

Two boundaries remain intentionally unresolved rather than faked:

- DungeonGrid's Rust wire still has no party-message verb, so authored party
  communication cannot be demonstrated until the upstream engine supports it.
- `wire_apis/default_wire_api` and typed `PinnedIdentity` require a versioned
  contract migration. The v1 scalar/string fields remain compatible for the
  proven no-opponent matrix; adding an opponent or a dual-wire image should be
  gated on that migration.

Final verification after the throughput follow-on: optimizers 1,039 passed;
container worktree 801 passed and 8 skipped, with four pre-existing C2-01
failures plus one load-test flake that passed alone on rerun. Its directly
changed contract surface passed 53 tests. Banking77 passed 60 tests with its
known metadata assertion still failing; its directly changed CISPO suite passed
29. The earlier image results remain HealthBench2 42, Craftax 38,
Harbor-TBLite 42, and DungeonGrid 34. TBLite's missing-corpus capacity test
also remains outside the CISPO scope.

## Banking77 throughput experiment

The follow-on paid experiment made Banking77 attempt execution genuinely
asynchronous and exercised groups of eight over a socket. The strongest run,
`b77_throughput_uplift_04`, completed 272 attempts across 34 sampled groups in
a 283.11-second terminal-completion span: **0.957 completed attempts/second**.
It produced three training updates from 12 trained groups (22 zero-advantage
groups were skipped) with no stale-policy discards. A rendezvous sampler test
independently proves that eight submitted attempts overlap rather than merely
being queued in an eight-wide group.

This demonstrates high-throughput execution, not quality uplift. Paired
heldout results were:

- run 04, 32 examples: 0.6875 baseline, 0.6875 trained (32 ties);
- run 05, 32 examples: 0.7500 baseline, 0.71875 trained (one loss, 31 ties);
- run 06, one targeted example: 0.0 baseline, 0.0 trained (tie).

Further paid tuning was stopped because the evidence did not support an
accuracy-uplift claim. Run 05 completed ten updates and run 06 five updates;
neither reversed that conclusion.

The cumulative receipted lower bound, including the earlier conformance runs,
is **653,384 token-events**: 351,295 prompt, 51,178 generated, and 250,911
training tokens. It covers 694 successful paid attempts and 778 sampling
calls. Evaluation-token usage and failed/retried calls are not available in
the receipts, so they are deliberately excluded. At the published uncached
gpt-oss-20b rates, the counted portion estimates to $0.186; Tinker's billing
feed had not yet ingested the relevant hours, so that is not an actual charge.
The experiment remained well below its declared $10 maximum.

Artifacts are under `/tmp/synth-container-first-e2e`, specifically
`receipts_banking77_throughput_paid_{04,05,06}` and
`receipts_banking77_throughput_eval_{04,05,06}`. The reproducible bounded
configuration is `docs/e2e/configs/run_b77_throughput_paid.toml`.

The later scale run is recorded separately in
`docs/receipts/banking77-scale20-uplift-20260903.md` and its machine-readable
companion. It completed 448 rollouts at 29.90/minute, a measured 1.80x overlap
uplift over serializing the same call durations. Fourteen optimizer updates did
not improve quality: heldout moved 81.82% to 80.52%, and the fixed-alpha train
EMA moved 81.25% to 79.51%.

## Original snapshot

The design document is
`docs/receipts/tblite-cispo-orbstack30-sync-async-benchmark-20260902.md`
(2,498 lines, committed). It is the specification; this
file is the state of the work against it.

## What exists

An RL plane that talks to a container through a declared contract, and six
containers that speak it. CISPO is a preset of that plane, not its shape.

| Where | Branch | Commits | State |
|---|---|---|---|
| `~/GitHub/optimizers` | `feat/container-first-rl` | 27 from `96d7bba` | committed, **not pushed**, no upstream |
| `~/GitHub/wt-containers-cispo-conformance` (worktree of `containers`) | `feat/cispo-container-conformance` | 4 from `a5743ef` | committed, **not pushed**, no upstream |
| `~/GitHub/evals` | `agent/workshop-evals-v04` | — | **uncommitted**, see below |
| `~/GitHub/containers` | `fix/workshop-proxy-bearer` | — | **not ours**; another agent's uncommitted work. Do not touch. |

Tests: optimizers **794**; container worktree **305**; images banking77 59,
healthbench2 42, craftax 38, harbor-tblite 43, dungeongrid 34.

## What is proven, and by what

One paid run completed, against the reference counter container:

```
run paid_gate_counter_03 target_train_updates_reached: updates=1 sampled_groups=2
  pg-0: ckpt_516d3a45… ref=tinker://a20f4edd-…/sampler_weights/…
```

Real Tinker, real `openai/gpt-oss-20b`, one skipped zero-advantage group, one
trained group, 160 training tokens, `pg-0@0 → pg-0@1` both published with
separate sampler-weight digests. That satisfies acceptance gates 4, 7 and 8 on
one container.

Four real images then completed the same run over a socket on the **free**
plane (`e2e_plane:unpaid`): banking77, healthbench2, craftax, harbor-tblite —
each with a probe at zero cost, a trained group, a published revision, and a
31-file receipt directory.

Nothing else is proven. In particular: no paid run against any real image, no
paired evaluation, no MARL run, no run at the operational floor (3 groups/step).

## What remains, in order

1. **Re-run dungeongrid over a socket.** It was blocked by a roster-binding bug
   in the worktree adapter, fixed in `c08c1d9` and never re-run. This is the
   only thing standing between here and a demonstrated multi-policy path.
2. **A paid run on banking77.** Cheapest real image: one sampler call per
   attempt, no nested Docker. This is the first acceptance gate that measures
   the milestone policy on a real task.
3. **Paid runs on healthbench2 and craftax.** Craftax is the first multi-turn
   paid run, so it is the first real exercise of turn bridging.
4. **Decide the TBLite conflict** (below), then run it.
5. **Paired evaluation.** `rl/evaluation.py` and `rl evaluate` exist and are
   tested; no run has used them.
6. **Fix the renderer identity hole** (below). Do this before anyone reasons
   from a receipt.

## Decisions waiting for a person

**TBLite cannot serve the milestone policy.** `MiniSweAgent._complete_tinker_sampler`
(`containers` worktree, `policies/mini_swe.py:424`) rewrites the assistant turn
it stores whenever the model starts with `openai/gpt-oss-`. The next prompt
then does not extend the last sequence, so the evidence is refused. Three
options, in the order I would take them: stop the rewrite for this policy;
fork a branch per rewrite with an honest rule name and accept one sealed
segment per turn; or run the TBLite gate on another policy and say in the
receipt that the milestone policy was not measured. Reasoning is in the design
note under "The milestone policy and the TBLite harness disagree".

**DungeonGrid cannot declare a party channel.** The Rust engine's HTTP wire has
no message verb (`action_from_string` has no `message` branch,
`legal_action_strings` never offers one), so no policy on that wire can author
a party message. The container declares no channel rather than one that would
always be empty, which our own dropped-channel rule makes an evidence failure.
The carrying half is written and tested. Fix is upstream in the engine; until
then the party-communication half of the DungeonGrid evidence-matrix row is not
demonstrable.

**Two declarations are too narrow.** `PolicyFacts.wire_api` is scalar, so an
image serving both wires can advertise one; it should be
`wire_apis: tuple[str, ...]` plus `default_wire_api`, with membership rather
than equality at binding. `pinned_identity` is an untyped string, so a
non-trainable instance cannot say whether it is a frozen checkpoint, an
external model, or a scripted baseline; it should be
`PinnedIdentity(kind, identity, revision)`. Neither blocks the MARL gate as
fixtured. The second is reached the moment an opponent appears.

## Known defects

**The renderer identity in a receipt can be wrong.** Startup asserts the bound
renderer profile equals the container's declared profile — but the bound
profile *is* the declared one, so the assertion cannot fail. The paid run's
`renderer.json` names `synth_containers.cispo.whitespace.v1` while the tokens
came from Tinker's gpt-oss tokenizer. `agreement_proven: false` is on the
receipt and is doing its job; the fix is for containers to declare
`canary_digest` (see `CANARY_MESSAGES` and `RendererProfile.assert_renders_like`)
so the plane can compare real tokens. Until then, treat renderer identity in
any receipt as unverified.

**The reference container's evidence is self-consistent by construction.**
`cispo_target._run_episode` renders its own prompt token ids with a local
whitespace hash while stamping `engine_meta` provenance, ignoring the
authoritative ids the gateway returns in `synth_capture.prompt_token_ids`. It
also hardcodes `finish_reason="stop_token"` and drops the capture's
`branch_id`/`parent_branch_id`/`compaction`. In the paid run the gateway sealed
a fork at token 109 and the container's evidence records the two turns as one
unbroken sequence. This is the reference container only; the five real images
capture the sampler's ids.

**A container must be restarted between runs of the same `run_id`.**
Idempotency keys are stable per run id, so a repeat replays a terminal attempt
and `renew` 500s. Use a fresh `run_id` per run.

**A relative `[artifacts]` path writes into the caller's cwd.** One run left a
`checkpoints.sqlite3` in the repo root. Use absolute paths.

## How to run it

Free socket run, any image (serve scripts and configs are in `docs/e2e/`):

```
# 1. serve the container (from the containers worktree, or the image's dir)
uv run --with pytest --with uvicorn python docs/e2e/serve_banking77.py 8231

# 2. drive it (from ~/GitHub/optimizers)
PYTHONPATH=.../e2e uv run synth-optimizers rl run \
  --config docs/e2e/configs/run_banking77.toml \
  --receipts /abs/path/receipts --plane e2e_plane:unpaid
```

Paid: `--plane paid_plane:paid`. That plane reads `TINKER_API_KEY` from the
project-local file named by `$SYNTH_TINKER_ENV_FILE`. The ordinary paid config
is bounded to one update, group size 2, and at most three sampled groups;
**widening it costs money**.

Read the container's traceback from its server log, not the 500 the client
reports: `http_adapter._cispo_call` types only one error, so every other
refusal reaches the executor as a bare 500. Giving it typed errors would save
the next person several round trips.

## What to commit

In `~/GitHub/optimizers`, the design note is still untracked:
`docs/receipts/tblite-cispo-orbstack30-sync-async-benchmark-20260902.md`.
It is the specification for all of this and should be committed.

In `~/GitHub/evals`, the image work is uncommitted and lives among 200+ other
uncommitted files that are **not ours**. Stage only these:

```
containers/images/banking77/{README.md,CISPO_RESULT.md}
containers/images/banking77/banking77_classify/{cispo.py,stack.py,targets.py,__init__.py,runtime.py}
containers/images/banking77/tests/test_banking77_cispo_contract.py
containers/images/healthbench2/{README.md,image.toml}
containers/images/healthbench2/healthbench_chat/{cispo.py,routes.py,runtime.py,targets.py}
containers/images/healthbench2/tests/{test_healthbench_cispo_contract.py,test_healthbench_platform.py}
containers/images/craftax-gamebench-rust/README.md
containers/images/craftax-gamebench-rust/craftax_gold/{cispo.py,gepa.py,stack.py,targets.py,__init__.py}
containers/images/craftax-gamebench-rust/tests/{test_craftax_cispo_contract.py,test_craftax_gold_environment.py}
containers/images/harbor-tblite/README.md
containers/images/harbor-tblite/harbor_tblite/{cispo.py,stack.py,targets.py,__init__.py,__main__.py}
containers/images/harbor-tblite/tests/
containers/images/dungeongrid-gold/
```

## Pushing

Three branches, none with an upstream. Push order matters only in that the
images import the container contract:

1. `containers` — the worktree branch `feat/cispo-container-conformance`
   (`cd ~/GitHub/wt-containers-cispo-conformance && git push -u origin HEAD`).
   The main `containers` checkout has another agent's uncommitted work on a
   different branch; leave it alone.
2. `optimizers` — `feat/container-first-rl`.
3. `evals` — commit the list above onto `agent/workshop-evals-v04`, which is
   already 6 commits ahead of its upstream with work that is not ours. If that
   is awkward, branch off it first.

The images resolve `synth_containers.cispo_*` from a checkout, not the pinned
wheel (`install_cispo_import_path()` reads `SYNTH_CONTAINERS_CISPO_SRC`, else a
sibling worktree). Once the containers branch is released, that bootstrap can
be simplified, and `/info` reports which path it took.

## Two pre-existing failures, not ours

`banking77/tests/test_banking77_platform.py::test_metadata_is_content_not_a_fold`
fails on clean HEAD. `harbor-tblite`'s capacity test needs a `/var/lib/tblite`
corpus that does not exist on this machine.
