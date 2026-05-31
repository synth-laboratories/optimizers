<h1 align="center">synth-optimizers</h1>
<p align="center">Prompt-program optimizers for production agents — open-source GEPA on a language-agnostic task contract.</p>
<p align="center">
<a href="https://pypi.org/project/synth-optimizers/">PyPI</a> ·
<a href="https://github.com/synth-laboratories/containers">Containers</a> ·
<a href="https://github.com/synth-laboratories/synth-cookbooks-public/tree/main/cookbooks/optimizers/gepa">Cookbooks</a>
</p>

`synth-optimizers` runs [GEPA](https://arxiv.org/abs/2507.19457) — reflective prompt
evolution — against any task exposed through the
[`synth-containers`](https://github.com/synth-laboratories/containers) HTTP contract.
Point it at a container and a TOML config; it proposes prompt changes, rolls them out,
scores them, keeps a Pareto frontier, and returns a deployable candidate with replayable
evidence.

- **Container boundary** — the optimizer only speaks HTTP. It never imports your task code or sees your model credentials.
- **Inspectable runs** — every candidate carries its prompt diff, per-seed scores, rollout traces, cache profile, and usage.
- **Rust core, thin Python** — the GEPA state machine is Rust (PyO3); the Python surface is just `GepaRun` and a CLI.

## Install

```bash
pip install synth-optimizers
# or
uv add synth-optimizers
```

## Local Better SDK Dev

This branch pair expects:

- `containers`: `better-sdk`, package `synth-containers==0.2.0.dev20260531`
- `optimizers`: `better-sdk`, package `synth-optimizers==0.2.0.dev20260531`

Install both editable checkouts with `uv`:

```bash
cd /Users/joshpurtell/Documents/GitHub/optimizers
uv sync --group dev
uv pip install -e /Users/joshpurtell/Documents/GitHub/containers
uv pip install -e /Users/joshpurtell/Documents/GitHub/optimizers
```

Verify the installed paths and versions:

```bash
uv run --project /Users/joshpurtell/Documents/GitHub/optimizers python -c "import importlib.metadata as m, synth_containers, synth_optimizers; print(synth_containers.__file__); print(synth_optimizers.__file__); print(m.version('synth-containers')); print(synth_optimizers.__version__)"
```

The SDK validation set lives in local `dev_examples/` (gitignored). After editable install:

```bash
cd /Users/joshpurtell/Documents/GitHub/optimizers
bash dev_examples/banking77/run_fresh_gepa.sh
bash dev_examples/tblite/run_fresh_gepa.sh
bash dev_examples/crafter/run_fresh_gepa.sh
bash dev_examples/minigrid/run_fresh_gepa.sh
```

## Quickstart

A run is defined entirely by one TOML: which **container** to talk to, which prompt
modules to optimize, and how to score them. The optimizer launches (or connects to)
the container, then only speaks HTTP to it.

```toml
[container]
url = "http://127.0.0.1:8765"
command = ["uv", "run", "python", "banking77_container/synth_service_app.py", "--port", "8765"]

[candidate]
target_modules = ["stage2_system"]

[seed_candidate]
stage2_system = "Classify the query into exactly one Banking77 intent. Return only the label."

[dataset]
train_seeds = [0, 1, 2, 3, 4, 5, 6, 7]
heldout_seeds = [100, 101, 102, 103]
```

GEPA can only optimize against a **live container** — it speaks HTTP, nothing else.
The same `url` you put in `[container]` is the one to confirm is up via `GET /health`
before spending a single rollout:

```python
from synth_containers import Container
from synth_optimizers import GepaConfig, OptimizerRun, TasksetSelection

container = Container("my-task")

# Register task_info, program, dataset, task_rows, and rollout callbacks here.

with container.serve() as handle:
    config = GepaConfig(
        container=handle.connection(),
        taskset=TasksetSelection(train_ids=["train:0", "train:1"], heldout_ids=["test:100"]),
        program=None,
        objectives=None,
        policy=None,
    )
    result = OptimizerRun(config).execute()

print(result.best_candidate)
print(f"cost:       ${result.cost_usd:.2f}")
print(f"frontier:   {result.frontier_path}")
print(f"score plot: {result.score_chart_path}")
print(f"events:     {result.event_feed_path}")
```

`GepaRun.from_toml("gepa.toml").execute()` remains as a compatibility shim, but
new examples should use `OptimizerRun(GepaConfig(...)).execute()` with a
`ContainerConnection` from `synth-containers`.

Or from the CLI:

```bash
synth-optimizers gepa run --config gepa.toml        # one-shot optimization
synth-optimizers gepa service --db service.sqlite   # standing HTTP service
synth-optimizers events compare --left a.jsonl --right b.jsonl
```

A run needs a container URL and a TOML config. The
[GEPA cookbooks](https://github.com/synth-laboratories/synth-cookbooks-public/tree/main/cookbooks/optimizers/gepa)
have runnable examples across task shapes: Banking77, HotpotQA, MiniGrid, TBLite, and Crafter.

## Links

- [Cookbooks](https://github.com/synth-laboratories/synth-cookbooks-public/tree/main/cookbooks/optimizers/gepa) — runnable GEPA examples
- [synth-containers](https://github.com/synth-laboratories/containers) — the task contract
- [Agent skill](skills/gepa/SKILL.md) — drop into a coding agent to run and adapt GEPA
- [GEPA service OpenAPI](rust/crates/synth_gepa/openapi/gepa-service-v1.yaml)
- [GEPA paper](https://arxiv.org/abs/2507.19457)

## License

Apache-2.0
