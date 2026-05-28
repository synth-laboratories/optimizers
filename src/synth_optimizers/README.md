# Python Wrapper

This directory is intentionally thin.

Python exists to provide a stable import shape and CLI entry point for users who
install the package with `pip` or `uv`. Optimizer behavior lives in Rust.

## Public Imports

```python
from synth_optimizers import CacheMissError, GepaRun, GepaRunResult
```

Workspace service helpers are also re-exported from Rust:

```python
from synth_optimizers import workspace_status, workspace_submit_run_request
```

The GEPA service helpers are exported for callers that want the Rust standing
service from Python:

```python
from synth_optimizers import gepa_serve, gepa_service_run_next
```

Queue recovery is public for service supervisors:

```python
from synth_optimizers import workspace_recover_expired_run_requests
```

## Files

Planned files:

- `__init__.py`: exports version, `GepaRun`, `GepaRunResult`, workspace
  helpers, and exceptions.
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
