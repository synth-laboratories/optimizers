# Pipeline modes

The pipeline mode decides how a generation's **propose → rollout → evaluate** phases are
scheduled. Same search, different throughput/latency trade-offs. Set it via the
`[gepa.pipeline]` TOML section or the `GepaPipeline` factories in the SDK.

| Mode | What it does | Default concurrency |
|------|--------------|---------------------|
| `sync_serial` | Run propose, rollout, evaluate for one candidate, then a generation barrier. Simplest and most deterministic. | candidate=1, rollout=1 |
| `async_pipelined` | Improved throughput via asynchronous rollouts; multiple candidates in flight, full-staleness selection at the barrier. | candidate=4, rollout=8 |
| `flash_evolve` | Overlap the propose / rollout / evaluate lanes; use **staleness review** and **speculative completion** to decide which in-flight work still counts. Highest throughput. | candidate=8, rollout=8 |

## `sync_serial` (default)

```python
from synth_optimizers import GepaPipeline
pipeline = GepaPipeline.sync_serial(rollout_timeout_seconds=600)
```

Propose, roll out, and evaluate one candidate before a generation barrier. No staleness to
reason about — every rollout you score reflects the current candidate. Use it for
reproducibility, small budgets, or debugging.

## `async_pipelined`

```python
pipeline = GepaPipeline.async_pipelined(candidate_concurrency=4, rollout_concurrency=8)
```

Keeps several candidates and rollouts in flight to hide rollout latency, but still
collects results at a generation barrier under the `full` staleness policy. Good middle
ground when rollouts are slow (live model calls, RL episodes) but you want barrier-clean
selection.

## `flash_evolve`

```python
pipeline = GepaPipeline.flash_evolve(
    candidate_concurrency=8,
    rollout_concurrency=8,
    staleness_policy="guarded",
    staleness_delta_max=2,
)
```

Instead of a purely sync loop, Flash Evolve overlaps the propose, rollout, and evaluate
lanes, then uses **staleness review** and **heldout evidence** to decide what survives.
This is the highest-throughput mode and the one wired into `optimizers/rust`. The hard part
it solves: overlapping phases *without corrupting evidence* — the container stays the
source of truth and every surviving candidate is backed by real, attributable rollouts.

### Staleness policy

When phases overlap, a rollout may complete against a candidate that has since been
superseded. The staleness policy decides whether that result still counts:

| Policy | Behavior |
|--------|----------|
| `full` | Strictest — only barrier-fresh results are used (the `*_pipelined`/sync default). |
| `guarded` | Accept results within `staleness_delta_max` generations of the current frontier (Flash Evolve default, `delta_max=2`). |
| `reflective` | Let the reflective layer judge whether stale in-flight work is still usable. |

### Speculative completion

`speculative_alpha` (TOML `[gepa.pipeline.speculative_completion]`) lets Flash Evolve
decide which in-flight rollouts can be completed/used speculatively rather than discarded,
trading a little risk for more usable signal per unit of compute.

### Adaptive stage workers

`adaptive_stage_workers` scales the per-stage worker pool between `min` and `max`
(default 1…128) based on backlog and stale-gap thresholds, so the lane that is the
bottleneck gets more workers automatically.

## TOML equivalent

```toml
[gepa.pipeline]
mode = "flash_evolve"
staleness_policy = "guarded"
delta_max = 2
max_in_flight_candidates = 8

[gepa.pipeline.workers]
min = 1
max = 128
backlog_threshold = 2
stale_gap_threshold = 2

[gepa.pipeline.speculative_completion]
enabled = true
alpha = 0.5
```

All three modes share the same rollout transport (`sync` or `async`) and timeout; the mode
only governs scheduling and staleness, not correctness of the final reported result —
heldout evidence always gates what is accepted.
