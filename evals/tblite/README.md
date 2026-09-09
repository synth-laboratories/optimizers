# TBLite evaluation environment

This independent uv project preserves TBLite's exact development Containers
dependency. It is not part of the production project's dependency groups or
release gates. From this directory, use `uv sync --locked` to provision its
evaluation dependencies. Its lockfile must not be merged into the root lock.

This environment is for the vendored TBLite evaluation tooling, not evidence of
a production Optimizers installation. Root Optimizers builds and tests use the
root environment with stable Containers. Tests combining the current optimizer
with this older TBLite runtime need a separately validated compatibility setup;
do not override production pins to accommodate the evaluation dependency.

No TBLite publication or paid evaluation is required to release Optimizers.

The native Vim and continuation-recipe regression tests live under `tests/`
here, not the production test root. They currently require two absent research
sources: `docs/e2e/tblite_native_vim_grader.py` and
`docs/e2e/screen_tblite_preparation_repairs.py`. The eval lane remains blocked
until those reviewed sources are supplied; no passing eval claim is made.
The self-contained async-runner unit tests remain in the root test suite.
