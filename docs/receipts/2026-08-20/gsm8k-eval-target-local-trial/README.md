# GSM8K eval target — local trial receipt (2026-08-20)

One `eval.target.v1` trial of the **locally built** `gsm8k-eval-target` image
(`docker/gsm8k-eval-target/build.sh`, containers `9916cd7479c1029d7dd9091db43f52af1cdf484e`,
local image id `sha256:a454cd9e80c775703f1cbcd4ead294ffe53e9420c69cb244f9304e648ad4935e`)
against a real v0.7 `synth-mlx-rl` on the host (`Qwen/Qwen3.5-0.8B`, offline) via
`http://host.docker.internal:8791/v1/chat/completions`, run with the executor's
exact `docker run` flags (bridge network, read-only `/input`, bearer via env).

Mechanism receipt, not a score: seed 101 of `gsm8k-test`, base model, greedy.
The policy answered in the marked format (`parse_mode: exact`) and was wrong
(`benchmark_status: failed`, `accuracy: 0.0`); the served snapshot matched the
pinned `policy_snapshot_id` (`snapshot_pinned: true`); `proxy_request_ids` is
recorded in `evidence` and the trace.

Files: `trial.json` (input), `result.json`, `trace.jsonl`, `events.jsonl`,
`dataset.json` (the pin the container verified at start: `openai/gsm8k` @
`740312add88f781978c0658806c59bc2815b9866`, shuffle seed `20260820`).
