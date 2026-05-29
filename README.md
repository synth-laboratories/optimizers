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

```python
from synth_optimizers import GepaRun

result = GepaRun.from_toml("gepa.toml").execute()
print(result.best_candidate)
```

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
