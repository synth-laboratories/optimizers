# Eval → HealthBench local witness

Local end-to-end acceptance passed for `SYN-4925` on October 21, 2026.

- Run: `healthbench-eval-accepted-20261021`
- Cohort: `sha256:c3e4ca1201452777b7cbf7e1ca7aba95a6cdc11743857f980846fb396483a5f7`
- Image: `sha256:047c5d59e67e745aa9c7d226999ad423cd67c93e7bcd1b20f2447160ef48d9a8`
- Two valid trials; two answers and all 29 rubric judgments retained.
- Provider-reported accepted-run charges: **$0.0650544**.
- **85 provider-free checks pass.** Two fresh installed readers verify all adopted artifact hashes. A terminal reconnect leaves all 28 original run files unchanged.
- Exact owned containers are independently absent; both HTTP services stopped.
- Image, startup/input bundle and a development installation snapshot are exported with hashes in `cohort.json`.

`acceptance.json` contains the checked `ship-evidence` block. Raw TOML, plan/build/image receipts, runtime agreement, IDs, committed events, results, provider usage and cleanup are retained beside this report. Re-derive exports and readback with `python finalize_witness.py`; it performs no provider compute.

The first execution smoke passed. A later charge-receipt attempt refused a truncated answer. Both remain visible; the accepted protocol freezes a 1,536-token answer ceiling plus a 400-word instruction before execution. No result is described as optimization uplift. The grader is the custom `healthbench.local_rubric.v1`, not the reference protocol. The final row is unobserved and no proposer runs.

The first smoke's provider charge is unfinalized because the old runtime discarded generation IDs; its retained token-priced usage is $0.0832028. The refused-attempt charges are provider-reported $0.047774. Accepted-run charges are finalized from original generation metadata or, where metadata returns 404, the original completion's explicit provider-reported cost; the exact authority is retained per call. Combined known/reported plus first-smoke token-priced usage is $0.1960312, within the original $2 cap.

This is a local Linux ARM64 portable release and macOS ARM64 development installation witness. The installed snapshot reuses an existing native extension, which Eval does not invoke; it is not a newly published optimizer wheel. Hosted Rhodes/Tunnel acceptance and the other four matrix witnesses remain separate. Pressure repo invariants pass; scoped host warnings concern shared OrbStack/memory usage, and no foreign cleanup was attempted.
