# Pipeline modes

The pipeline mode decides how a generation's **propose → rollout → evaluate** phases are
scheduled. Same search, different throughput/latency trade-offs. Set it via the
`[gepa.pipeline]` TOML section or the `GepaPipeline` factories in the SDK.

| Mode | What it does | Default concurrency |
|------|--------------|---------------------|
| `sync_serial` | Run propose, rollout, evaluate for one candidate, then a generation barrier. Simplest and most deterministic. | candidate=1, rollout=1 |
| `async_pipelined` | Improved throughput via asynchronous rollouts; multiple candidates in flight, full-staleness selection at the barrier. | candidate=4, rollout=8 |
| `flash_evolve` | Drop the generation barrier and run the propose / rollout / evaluate lanes on background workers, using **staleness review** and **speculative completion** to decide which in-flight work still counts. | candidate=8, rollout=8 |

Whether `flash_evolve` is actually faster than `sync_serial` on *your* workload is a
measurement, not a property of the mode — see [Measuring overlap](#measuring-overlap).

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
The hard part it solves: overlapping phases *without corrupting evidence* — the container
stays the source of truth and every surviving candidate is backed by real, attributable
rollouts.

Overlap needs two independent things, and for a long time this mode only had one:

1. **No generation barrier**, so generation `n+1`'s proposer can be *admitted* while
   generation `n`'s full-train rollouts are still outstanding. `flash_evolve` has always
   had this.
2. **Background lane execution**, so admitted lanes actually run at the same time. Until
   this landed, the driver executed every leased lane job inline on its own tick: two
   lanes could hold leases simultaneously but never occupy wall clock simultaneously.

On the 2026-06-02 Banking77 matrix (`20260602052705`), with only (1), `flash_evolve` ran
**741s vs 509s** for `sync_serial` — a 0.687x *regression* — at identical heldout quality
(0.750 both), with measured propose/rollout overlap of ~0.33s. `pipeline.background_execution`
supplies (2); it defaults to `true` for `flash_evolve` and `false` for every other mode.

```toml
[gepa.pipeline]
mode = "flash_evolve"
background_execution = true   # default for this mode; set false to reproduce the old behaviour
background_workers = 10       # default: workers.propose + workers.rollout + workers.evaluate
```

Each background worker opens its own workspace and request-cache handle on the shared
sqlite files (both WAL with a busy timeout). The driver tick stays single-threaded and
still does every state-machine decision — folding outcomes, staleness, budgets, adaptive
concurrency. Workers only execute the job and report back.

### Measuring overlap

`cursor.pipeline_state.lane_overlap` — also on `gepa.run.finished` and in the run
metadata — reports the numbers that decide whether the mode is earning its name:

| Field | Meaning |
|-------|---------|
| `overlap_seconds` | Wall time with **at least one propose lane job and one rollout lane job executing**. Not lease overlap. |
| `overlap_ratio` | `overlap_seconds / lane_busy_seconds`. |
| `propose_busy_seconds` / `rollout_busy_seconds` / `evaluate_busy_seconds` | Per-lane busy wall time. |
| `max_concurrent_lane_jobs` | Peak simultaneously-executing lane jobs. |
| `mean_stale_gap` | Mean pool-version gap of items the staleness policy acted on. |

A `flash_evolve` run whose `overlap_seconds` is a fraction of a second is not overlapping,
whatever the mode flag says. Judge the mode on `overlap_seconds` plus wall clock at matched
heldout — never on the flag alone, and never on a 1-generation config, where there is no
generation `n+1` proposer to overlap with anything.

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
background_execution = true
background_workers = 10

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
