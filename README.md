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
import urllib.request, json
from synth_optimizers import GepaRun

CONTAINER_URL = "http://127.0.0.1:8765"

with urllib.request.urlopen(f"{CONTAINER_URL}/health", timeout=5) as r:
    assert json.load(r)["status"] == "ok", "container is up but not ready"

result = GepaRun.from_toml("gepa.toml").execute()

print(result.best_candidate["stage2_system"])
print(f"cost:       ${result.cost_usd:.2f}")
print(f"frontier:   {result.frontier_path}")
print(f"score plot: {result.score_chart_path}")
print(f"events:     {result.event_feed_path}")
```

> `from_toml` takes only a path — the container URL lives in the TOML and is the single
> source of truth, so the health check above points at that same `url`. If a connection
> error fires instead of the assertion, no container is running: start it (the
> `[container].command` lets GEPA launch it for you) and re-run.

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
