# Shared connector custody

`borrow_shared_synth_tunnel(..., custody_directory=...)` owns a distinct lease,
not the connector service or its routes. By default private, fsynced receipts
live under `.synth/tunnel-leases` in the caller's current directory. Use persistent
local storage for callers that must recover after process/container restart.
Receipts contain grant route secrets; do not publish them or put them in logs.
The directory is bounded and refuses additional grants when full. Retention and
archival need a separate policy; unknown receipts cannot be silently evicted.

The client retains exact identities before grant, offer, submission and revoke.
`recover_shared_synth_tunnel(receipt_path)` resolves grant/offer custody through
canonical reads without repeating those mutations. Missing authority or missing
receipts leave the operation in doubt; a timeout is not a refusal. Local receipt
revision CAS fences stale recovered handles. POSIX flock/fsync provide the local
adapter; a Windows or network-filesystem adapter is not qualified.

Hosted submission prepares a job-bound offer using `run_id`, then
`idempotency_key`, or a generated retained job ID. Job IDs accept alphanumerics,
underscores and hyphens, up to 128 bytes. The hosted descriptor includes the
exact offer/job binding. A lost submit reply raises `TunnelCustodyInDoubt` and
will not automatically POST again; `submission_status()` performs a read.
A saved acknowledged response can be returned for the same input digest.

Offering or submitting does not transfer custody. The executor must fsync its
queued job, accept the exact offer and obtain live authority before origin work.
Closing a handle with a matching accepted handoff detaches local observation
without revoking the executor's lease. Before acceptance close uses the caller's
original lease revision, so a race cannot revoke a newer executor owner. A lost
revoke reply is reconciled by command/lease lookup, never a fresh revoke.

Real API/Postgres reply-loss and disposal proof is retained in Synth Tunnel's
`records/connector-20261008/public-custody-current/backend-api.json`. That proof
uses fixture-enrolled API keys and does not yet qualify deployed executor work.
