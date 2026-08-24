# Synth portable contract spine v1

This directory is the v0.8 wire authority for `synth.candidate.v1`,
`synth.task-contract.v1`, lineage edges, and evidence receipts. Candidate IDs
remain useful human handles; the top-level `digest` is the immutable join key.
Existing optimizer candidate payloads map into this envelope instead of gaining
a second identity.

## Canonical bytes

`digest = "sha256:" + hex(sha256("synth.canonical-json.v1\\0" || canonical_json))`.

- JSON objects sort keys by unsigned UTF-8 bytes; arrays retain order.
- JSON is UTF-8 with no whitespace. Unicode is preserved exactly: NFC and NFD
  are intentionally different content.
- Values are null, booleans, strings, arrays, objects, or integers in
  `[-9007199254740991, 9007199254740991]`. Floats are forbidden, including
  mathematically integral floats. Measurements belong in evidence bodies or
  must use a declared integer unit.
- Strings use JSON escapes for quotes, backslash, and control characters; they
  are not ASCII-escaped.
- A self-digest is calculated with only the root `digest` member omitted.
  Nested digest fields remain included.
- Digests are lowercase `sha256:` plus exactly 64 hexadecimal characters.

Stable errors include `schema_unsupported`, `field_missing`, `field_unknown`,
`field_type`, `digest_malformed`, `digest_mismatch`,
`canonical_float_forbidden`, `canonical_non_finite`,
`canonical_integer_range`, `lineage_edge_type`, `lineage_endpoint_kind`,
`evaluation_authority_missing`, and `idempotency_conflict`.

## Ownership and extensions

- The schemas and fixtures here are canonical. Python owns the producer adapter;
  Rust and TypeScript are parity consumers of the same corpus.
- Optimizers emits `synth.workshop-experiment-fact.v1` events only. Workshop
  CoreRuntime eventually validates these and remains the single SQLite writer.
- Containers vendors a release-stamped copy of the two identity fixtures and
  validates the exact digests at register-then-run admission.
- NanoHorizon and Shoal may add new candidate `kind` strings, evidence media
  types, receipts, and separately versioned extension records. They must not
  change or reinterpret v1 identity bytes.
- Lineage is explicit and append-only. Names, folders, ports, and timestamps are
  never lineage authority.

Run parity with:

```sh
uv run pytest tests/test_portable_contracts.py
cargo test -p synth_optimizer_platform --test portable_contract_parity
node --experimental-strip-types contracts/typescript/parity_test.ts
```
