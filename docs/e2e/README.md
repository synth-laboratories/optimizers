# End-to-end harness

What drove the runs in `../HANDOFF_CONTAINER_FIRST_RL_2026-09-03.md`. These are
not tests — they stand a real container up in one process and drive it with the
shipped CLI from another, which is the only way the two halves are ever on
opposite ends of a socket.

- `serve_*.py` — one per container: builds the image the way its own tests do,
  but with an HTTP sampler that posts to whatever origin the executor bound.
- `e2e_plane.py` — `unpaid`: the real client, gateway, binder and catalog,
  with a stubbed provider. Free.
- `paid_plane.py` — `paid`: nothing stubbed. **Spends money.** Reads the
  credential from `$SYNTH_TINKER_ENV_FILE`, or the path baked into it.
- `configs/` — a `cispo.container.v1` document per container. Each is bounded
  to one update, group size 2, at most three sampled groups. Widening any of
  those on the paid plane costs real money.

Two things that will bite:

- Use a fresh `run_id` per run. Idempotency keys are stable per run id, so a
  repeat against a live container replays a terminal attempt.
- Read the container's own log for the reason behind a 500. The adapter types
  only one error, so everything else arrives as a bare Internal Server Error.
