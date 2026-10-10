# Python Wrapper

This directory is intentionally thin.

Python exists to provide a stable import shape and CLI entry point for users who
install the package with `pip` or `uv`. Optimizer behavior lives in Rust.

## Public Imports

```python
from synth_optimizers import CacheMissError, GepaRun, GepaRunResult
```

The standing HTTP service owns workspace submission, claiming and lifecycle
control. Its native entry point is public:

```python
from synth_optimizers import gepa_serve
```

Direct `workspace_submit_run_request`, `workspace_status` and recovery functions
are not exported by the current Python extension. Use the standing service's
routes for those operations.

## Files

Planned files:

- `__init__.py`: exports version, `GepaRun`, `GepaRunResult`, workspace
  service entry point, and exceptions.
- `cli.py`: small argparse or click-free CLI that calls the PyO3 module.
- `py.typed`: marks the wrapper as typed once annotations are complete.

## Errors

All Rust optimizer failures are exported as `SynthOptimizerError` subclasses.
Each class has a stable `error_code` class attribute. The most important public
ones are:

- `CacheMissError`: readonly replay hit an uncached boundary.
- `CancelledError`: optimizer execution observed a cancellation signal.
- `ContainerContractError`: the container boundary failed or did not advertise
  the GEPA contract.
- `ConfigError`: TOML/config validation failed.
- `ProposerError`: the configured proposer backend failed.

## Non-Goals

- No Python GEPA implementation.
- No private optimizer compatibility path.
- No arbitrary runtime imports from user code.
- No hidden local service startup outside explicit `gepa_serve` or the
  config-defined container process.

## Application custody IPC

`tunnel-custody` accepts one bounded private JSON stdin request for Workshop's
shared-connector cloud submission or exit reconciliation. It reuses the SDK's
canonical handoff and fsynced private receipts. Its HTTP adapter refuses redirects
and oversized authority replies; it never retries an uncertain effect. Submission
is currently limited to GEPA/go-ex until SFT/CISPO parent/child mapping is qualified.
Closes preserve accepted executor custody and shared connector/route lifetime.
The actual installed ABI, stdio refusal, HTTP bounds and canonical API tests are
retained in `records/tunnel-v2-20261009/` and Synth Tunnel's corresponding E2E
receipts. This feature has not been published as a runtime distribution.
