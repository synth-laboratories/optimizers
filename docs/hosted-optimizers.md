# Hosted Optimizers

Hosted optimizer jobs run through the Synth API with `SYNTH_API_KEY`. The public
client and CLI submit to `/api/v1/optimizers/runs`; the backend owns auth,
billing, durable run status, events, and artifacts. No public workflow requires
a separate optimizer service token.

By default, `synth-optimizers` also sends a best-effort usage-registration event
when an optimizer run is kicked off. The event is intentionally content-free:
it sends only the optimizer algorithm, client surface, package name, and package
version. It does not send prompts, config, code, artifacts, repo paths, run ids,
or user content; the backend derives source IP from the request context. Disable
it with `SYNTH_OPTIMIZERS_DISABLE_USAGE_REGISTRATION=1`,
`HostedOptimizerClient(register_usage=False)`, `[usage_registration] enabled =
false` for local GEPA TOML, or the CLI `--disable-usage-registration` flag.

## GEPA

GEPA supports both local execution and hosted jobs.

```bash
export SYNTH_API_KEY="..."

synth-optimizers gepa submit \
  --config gepa.toml \
  --tunnel-url http://127.0.0.1:8765 \
  --tunnel-provider synth_tunnel \
  --follow

synth-optimizers gepa watch gepa_... --events
```

Use `--tunnel-provider synth_tunnel`, `cloudflared`, or `ngrok` to choose the
tunnel provider for a local task container. `--tunnel-url` and
`--container-pool POOL_ID` are mutually exclusive.
RunPod can host the task container for proofs; start the selected tunnel
connector in the RunPod pod so `127.0.0.1` resolves to that task service.

Python:

```python
from synth_optimizers import GepaConfig
from synth_optimizers.hosted import HostedOptimizerClient

client = HostedOptimizerClient()
response = client.submit(GepaConfig.from_toml("gepa.toml"))
record = client.wait_for_run(response.run_id, timeout_seconds=3600)
events = list(client.event_backfill(response.run_id, limit=3))
candidates = client.get_state_slice(response.run_id, "candidates")
frontier_events = list(client.algorithm_events(response.run_id, limit=10))
```

## GELO

GELO is hosted-only in the public package. There is no public `gelo run`; local
GELO execution is not part of the public product surface.

```bash
export SYNTH_API_KEY="..."

synth-optimizers gelo startup

synth-optimizers gelo materialize \
  --preset crafter_smoke \
  --container-url http://127.0.0.1:8943 \
  -o .out/crafter_goex.json

synth-optimizers gelo submit \
  --config .out/crafter_goex.json \
  --tunnel-url http://127.0.0.1:8943 \
  --tunnel-provider synth_tunnel \
  --follow

synth-optimizers gelo watch goex_... --slice board --goex-events
```

Python:

```python
from synth_optimizers.gelo import GeloPreset
from synth_optimizers.hosted import HostedOptimizerClient

client = HostedOptimizerClient()

with client.open_tunnel("http://127.0.0.1:8943", provider="synth_tunnel") as tunnel:
    config = GeloPreset.crafter_smoke().to_config(container_tunnel=tunnel)
    response = client.submit(config)
    record = client.wait_for_run(response.run_id, timeout_seconds=3600)

board = client.get_state_slice(response.run_id, "board")
events = list(client.event_backfill(response.run_id, limit=3))
```

GELO submit defaults to launch-promo billing. Use `billing_mode="paid"` in
`client.submit(...)` / `client.submit_gelo(...)`, or `--billing-mode paid` in the
CLI, for normal paid hosted GELO without launch-promo model and concurrency gates.

## Online Reflexion

Online Reflexion is hosted-only in the public package. The public client submits
the `online-reflexion` run kind and reads the receipt/audit APIs used for
release evidence. The optimizer itself must use the backend-enforced row cursor
contract; no public workflow starts a Reflexion side service.

```bash
export SYNTH_API_KEY="..."

synth-optimizers reflexion startup

synth-optimizers reflexion submit \
  --config dev_examples/online_reflexion/native_smoke.json \
  --follow

synth-optimizers reflexion receipt orx_... --json
synth-optimizers reflexion audit --run-id orx_... --json
synth-optimizers reflexion evidence-packet \
  --run-id orx_... \
  --evidence-notes-file dev_examples/online_reflexion/evidence_notes_template.json \
  --out artifacts/online_reflexion_evidence_packet.json \
  --json

synth-optimizers reflexion validate-evidence-notes \
  --evidence-notes-file dev_examples/online_reflexion/evidence_notes_template.json
```

Python:

```python
from synth_optimizers.hosted import HostedOptimizerClient

client = HostedOptimizerClient()
response = client.submit_online_reflexion(config)
record = client.wait_for_run(response.run_id, timeout_seconds=3600)
receipt = client.online_reflexion_receipt(record.run_id)
audit = client.online_reflexion_receipt_audit(record.run_id)
packet = client.online_reflexion_evidence_packet(run_ids=[record.run_id])
```

`reflexion startup` returns the backend-advertised Online Reflexion release
evidence metadata, including required eval lanes, the `release_blog_growth`
gate key, release-gate checks, and the standard artifact bundle. Treat it as a
pre-submit/pre-publish schema check: a backend that does not advertise
`online-reflexion` or the release evidence metadata is not launch evidence.
Text output includes the release-evidence schema, release gate, lane/check
counts, standard-artifact count, owner-approval requirement, and EffortBench
Chinese-wall status; `--json` returns the full metadata object.
For release automation, run `reflexion startup --require-online-reflexion
--require-online-reflexion-release-metadata`; it exits non-zero when the
target backend is missing the hosted run kind or current release schema.

The evidence packet composes receipt audits with explicit evidence for the
Craftax rotated 121-125 repeats, ALFWorld 6/6 x3, EBR first scale compare,
Harvey LAB pilot, hosted staging smoke, and the release/blog/growth readiness
gate. It reports `not_ready`, `ready_for_owner_review`, or `ready`; public copy
is not approved unless all evidence gates are complete and the human
blog/release owner approval is supplied.

The evidence notes file must be structured by lane. A lane does not complete
from a bare `true` or generic `status=pass`; it completes only when the lane
object carries the specific proof required for that claim:

- `craftax_rotated_121_125`: `heldout_window="121-125"`, repeat 2+3 run or
  artifact ids, `ci_excludes_zero=true`, per-inject harm at or below 15%, and
  `zero_invalid_injections=true`.
- `alfworld_6x6_x3`: `matched_tasks>=6`, three clean repeat run or artifact ids,
  no truncated runs in the verdict, and an explicit verdict.
- `ebr_first_scale_compare`: scale-compare proof, a run or artifact id, and a
  verdict.
- `harvey_lab_pilot`: Tax split, 25 train / 9 heldout, criteria mapped to typed
  failure signals, and a run or artifact id.
- `hosted_staging_smoke`: staging terminal-success run id, strict receipt audit
  pass, standard artifact bundle, and a policy-never-blocks receipt.
- `release_blog_growth`: docs/runbooks ready, SDK/CLI/Stack operator paths
  ready, changelog or release notes ready, blog claims mapped to receipt-backed
  evidence, launch/growth plan ready, EffortBench Chinese-wall review complete,
  and at least one referenced release/blog/growth artifact path.

Use `dev_examples/online_reflexion/release_blog_growth_plan.md` as the
release/blog/growth artifact path. It is a fill-in evidence plan, not proof by
itself; the `release_blog_growth` booleans should remain false until the file
contains concrete paths, SHAs, run ids, owners, claim mappings, growth plan, and
EffortBench Chinese-wall review.

The packet returns per-lane `validation.checks`, `release_gate.checks`, and
their `missing_requirements`, which are the operator checklist for turning
attached evidence into complete evidence.
Use `reflexion validate-evidence-notes` locally while filling the evidence file;
it does not contact the backend and exits non-zero until every lane plus
`release_blog_growth` is complete. The backend `evidence-packet` command is
still required for receipt audit and public-copy readiness. Use
`reflexion evidence-packet --out <path>` to materialize the packet JSON that
release, blog, and growth review cite.

The receipt audit is the public release gate. It requires the standard artifact
bundle (`events`, `exposures`, `lever_effects`, `summary`), row-backed exposure
and lever-effect causality, matching summary row counts, `policy_never_blocks`,
missed-window receipts, `lever_scope=internal_gated_surfaces_only`, and no
production SMR edit claims.

## Streaming and state

Hosted GEPA, GELO, MAPO, and Online Reflexion share the generic optimizer
observability routes:

- `events()` / `event_backfill()` for lifecycle events.
- `algorithm_event_stream()` / `algorithm_events()` for normalized algorithm events.
- `get_state()` for the latest cursor.
- `get_state_slice("candidates")` and `get_state_slice("frontier")` for live search state.

GELO also exposes compatibility aliases `goex_events()` and `goex_event_stream()`.
Prefer the generic `algorithm_*` methods for new SDK and CLI integrations.

### Extension fields

The public hosted GELO path accepts the documented base Go-Explore prompt-space
configuration. Experimental extension fields are not a launch surface; clients
should omit them unless a Synth operator provides a dated compatibility note for
the target hosted API.

## Launch Evidence

Before promoting a hosted optimizer release, capture evidence for the exact
target API and commits being promoted:

- `synth-optimizers gelo startup` shows `go-ex` available and billing feature ids configured.
- `synth-optimizers reflexion startup --require-online-reflexion
  --require-online-reflexion-release-metadata` exits 0, shows
  `online-reflexion` available and billing feature ids configured, and prints
  `online_reflexion_release_evidence schema=online_reflexion_release_evidence.v1`
  with `release_gate=release_blog_growth`.
- GEPA active cancel reaches terminal cancellation through the backend-hosted path.
- GEPA, GELO, MAPO, and Online Reflexion public run reads survive optimizer
  service restart/reconcile.
- Public lifecycle event evidence uses bounded `event_backfill()` reads; streaming
  `events()` tails are for monitoring, not finite launch proof.
- GELO `gelo submit --preset crafter_smoke --tunnel-url ... --follow` reaches terminal success.
- Online Reflexion `reflexion submit --config ... --follow` reaches terminal
  success on staging and `reflexion audit --run-id ...` passes.
- `reflexion evidence-packet ...` returns `ready_for_owner_review` or `ready`
  only after all required eval-lane evidence is attached.
- `gelo watch RUN_ID --slice board` returns public board rows without internal paths.
- Artifacts needed by the public client, such as `checkpoint_frontier`, are readable.
- Billing evidence records optimizer spend for GEPA, GELO, MAPO, and Online
  Reflexion, or a launch waiver is recorded.

The hosted launch checklist under `specifications/daily/<date>/` is authoritative
for final promotion evidence.
