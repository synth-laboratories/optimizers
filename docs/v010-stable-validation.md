# Stable 0.2.22 validation — 2026-09-09

Candidate worktree: `wt-optimizers-v010-stable`, branch
`codex/v010-stable-packages`, based on `dcf7e8f`.

## Verified

- Python suite: **1,334 passed, 12 skipped**, 257.05 seconds.
- Rust workspace suite excluding the PyO3 test harness: **137 passed**.
- Source Ruff, production metadata checks, and `git diff --check`: passed.
- Stable macOS arm64 ABI3 wheel and source distribution built successfully.
- Fresh environment installation of the wheel and stable Containers succeeds;
  the native extension imports, the CLI starts, and TBLite is absent.
- Root production lock and separate TBLite eval lock both resolve. Required
  vendored Containers wheels are now tracked, avoiding clean-checkout failures.

## Changes and scope

Rust and Python versions are 0.2.22; the production Containers pin is 0.4.2.
Only the five local crate versions changed in Cargo.lock; third-party versions
were preserved. Workshop's separate embedded 0.2.21 source/lock is unchanged.

The live-tail regression now asserts the authoritative lifecycle event. A
separate 1,002-event regression found and fixes terminal replay truncation after
two pages. The mocked renderer-profile test now mocks package metadata too,
retaining its version assertion without requiring an optional renderer install.

TBLite research fixtures moved under `evals/tblite/tests`; their two missing
research scripts remain an explicit eval-only blocker. Self-contained async
runner tests remain in the production unit suite. No paid experiment was run.

Publication CI now runs Python tests, builds macOS arm64 and Linux x86_64 wheels,
checks production metadata/native extension presence, and fresh-installs each.
The Linux jobs have not yet executed remotely; local macOS evidence is not a
substitute. The four-file formatting/size-ratchet conflict is resolved by
extracting intact test modules: formatting and all four ratchet checks pass
without raising any ceiling or removing assertions. All 137 Rust tests passed
again after extraction, and Clippy passed with `-D warnings`.

## Built candidate hashes (not public artifacts)

- macOS wheel: `022beaa1bc189947bbc4fc807f892872293898609c8f30b21981cb6c814aae3b`
- source archive: `ed63bdbf004542e9f22442f9eb9d150c729cd413aba9ca63d51f8f4cc083ae10`
- vendored stable Containers wheel from candidate `30287d0`:
  `98bb5184ec2661f2a02974f65f373b88e9b28b5d2541dbf644c0d0b552e40e9f`

These locally built bytes preceded this documentation commit. Final publication
must rebuild from the reviewed source and record its own hashes/provenance.
Merge/release permissions, protected CI, immutable tags, Containers publication
first, and a public-index installation remain open. Nothing is published yet.
