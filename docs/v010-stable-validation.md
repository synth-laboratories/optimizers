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
- vendored stable Containers wheel, as tracked in this worktree:
  `02943f7e00e281b04b07b8f5d5f3084ab52a170d76efe400502b8e72b9c3a81b`. The hash
  previously recorded here,
  `98bb5184ec2661f2a02974f65f373b88e9b28b5d2541dbf644c0d0b552e40e9f`, matches no
  blob under `vendor/synth-containers/` and is stale.
- Containers `0.4.2` **as published on PyPI** on 2026-09-09 is a different blob:
  wheel `61773e43bfce893437b0b27bcaf03233c8e301e114374d7ea5a159a95ceb2b67`,
  sdist `2aa064d62debc47647e6a05372475158c3a1bb9da7c8d8fdbdee3f25afb5295a`.

  Reconciled -- three distinct blobs all legitimately carry version `0.4.2`, and
  hash inequality between them is expected rather than a provenance failure:

  1. The tag-triggered publish run rebuilt the distributions from tag `v0.4.2` in
     its own `Build 0.4.2` job instead of reusing the candidate CI artifacts, so
     the published wheel (`61773e43...`) differs from the candidate CI wheel
     (`7a79b345...`) only in non-reproducible archive metadata. Wheel size is
     identical at 912483 bytes; the sdist differs by 72 bytes.
  2. The vendored development copy (`02943f7e...`) was built at Containers
     `20c4f1a`, the candidate's immediate parent.

  Content equivalence is the correct check, and it was verified directly against
  the downloaded published wheel: all **224** Python modules are byte-identical
  to `src/` at the Containers candidate
  `8cde9e3c6f5daa2fef9fe5fe822263e2c7c34b94`, with zero mismatches and no extra
  modules, and the published `METADATA` long description is byte-identical to the
  tagged `README.md`. The vendored copy's 224 modules are likewise byte-identical
  to that same candidate, because candidate commit `8cde9e3c` changed only
  `README.md` and `tests/test_readme_smoke.py` and touched no `src/` file -- so
  the vendored wheel differs from the published one in packaged documentation
  bytes only, never in code.

  Practical consequence: hash equality against the candidate CI artifact is not a
  valid release check for a tag-built publish. Verify the published wheel's code
  against the tagged source instead.

These locally built bytes preceded this documentation commit. Final publication
must rebuild from the reviewed source and record its own hashes/provenance.
Merge/release permissions, protected CI, immutable tags, and a public-index
installation of `synth-optimizers` remain open. Containers `0.4.2` is published
on PyPI as of 2026-09-09 and is no longer a prerequisite; `synth-optimizers`
`0.2.22` is still unpublished (PyPI serves `0.2.16`).
