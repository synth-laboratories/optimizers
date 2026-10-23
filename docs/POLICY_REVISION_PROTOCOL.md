# Policy revision consumer migration

Review candidate, 2026-10-23. Producer specification:
`containers/docs/specs/policy-revision-v1.md`, source
`d5e2b48b93213a95d3c5fde9fbca85e7237f028c`.

The shared `ContainerClient` observes `policy_revision` during the existing
metadata preflight. For `prompt_program` / `next_rollout_update.v1`, it publishes
prompt fields and inference configuration, checks the preparation's actual
binding, caches the prepared revision, and sends only `policy_revision_id` on the
rollout wire. It checks the response's admitted ID. A publication/readiness
failure never retries as an inline candidate. Targets lacking this adapter keep
their versioned GEPA v2 wire during the compatibility window.

Run metadata/contract preflight before dispatch. Discovery is not authorization,
budget admission or a weight-loading receipt. Recreate the client/preflight when
the target process changes; process-local preparation caches do not survive a
target restart. A stale revision refuses; retain and republish the manifest rather
than silently starting with the target's default.

`contracts/policy-revision-v1/` vendors the producer schema and canonical byte
fixtures. The Python dependency and uv source use the exact retained
`synth_containers-0.4.6.dev2026102302` wheel; its SHA-256/source and all 314 packaged
Python file matches are in `vendor/synth-containers/policy-revision-20261023/provenance.json`.
This is an unpublished source candidate, not a PyPI release.

The loopback `policy_revision_probe` example is a provider-free conformance
client. Against the fixture Banking77 producer it verifies A/B/A binding and
readback; the runner also captures the HTTP request bodies and actual prompt
arguments. It is not a provider performance or scientific-score test.

MIPRO is present in the owner's dirty primary checkout, not this maintained
`origin/dev` snapshot. Its source must be selected independently. It should use
the same transport with typed `module_demos`; a task adapter must enforce
training-pool membership, disjoint selection/heldout roles, and winner freeze.
Neither the generic prompt adapter nor Banking77's current GEPA-only adapter
claims demo support. Do not encode demo ID arrays into prompt-field strings to
bypass this refusal.

MAPO's prompt/protocol stage rules remain in the optimizer. A policy-set update
needs a complete roster, all-component readiness, atomic visibility and the
combined checkpoint-fork profile. Those profiles are specified but are not
advertised by the initial next-rollout implementation.
