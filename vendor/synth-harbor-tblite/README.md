# Harbor TBLite installed test dependency

This wheel packages the unchanged Harbor TBLite adapter from Evals commit
`b95df8d4d` (`containers/images/harbor-tblite`). It includes the task pins and
requires the coordinated Containers candidate `0.4.2.dev20260903`.

Build with `uv build --wheel containers/images/harbor-tblite` in Evals.
`uv sync --group dev` installs it without sibling checkout paths or PYTHONPATH.
The wheel SHA256 is pinned in `uv.lock`. This is a test dependency; it does not
install Harbor task environments or authorize provider-backed runs.
