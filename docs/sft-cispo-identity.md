# SFT, GoEx SFT, importance sampling, and CISPO

These names are not interchangeable.

## Standalone SFT

- `algorithm_id="sft"`
- Implementation `sft.tinker.v1`
- Public API: `submit_sft`, `SftService`, `TinkerSftExecutor`
- Supervised next-token training on a fingerprintable dataset
- Default model `openai/gpt-oss-20b` on Tinker

The sqlite journal is the record. `GET /v1/runs/{id}/optimizer-events` backfills the same event objects as JSON; `GET /v1/runs/{id}/optimizer-events/stream` (or `?stream=1`) mirrors them over SSE. Submit returns a run id immediately so a client can tail while training runs; dropping the reader does not stop the job.

## GoEx SFT lane

- Parent `algorithm_id="go-ex"`
- Plugin identity `goex.sft.v1`
- A GELO/Go-Explore theme lane that may later call the public SFT executor
- It does not define the public SFT contract and must not appear as standalone SFT

## Generic importance sampling

- A Tinker `importance_sampling` or group-relative IS run
- Valid RLVR mechanism, **not** CISPO
- Must not set `algorithm_id="cispo"` or `implementation_version="cispo.slime.v1"`

## True CISPO

- `algorithm_id="cispo"` only with `implementation="slime-reference"` and
  `implementation_version="cispo.slime.v1"`
- Group-relative advantages, behavior and current log-probabilities, slime
  clipping with stop-gradient on the ratio
- Preflight returns `unsupported` rather than silently downgrading
- Public HTTP: `CispoService` (`POST /v1/runs`, `GET /v1/runs/{id}/optimizer-events/stream`).
  The sqlite journal is the record; SSE is a mirror. Submit returns a run id
  immediately so a client can tail events. Disconnecting SSE does not fail the
  job. This service rejects `algorithm_id="sft"`.

## Chat rendering

Live Tinker SFT/CISPO on `openai/gpt-oss-20b` uses Prime Intellect's
`renderers` package (`GptOssRendererConfig`, Harmony). That is the token-in
path: `render_ids` / `build_training_sample` / `parse_response`, not a second
`apply_chat_template` pass over sampled text.

Banking77 pins `renderer_version="renderers.gpt-oss.low.v1"` (low reasoning
effort). Fixture tests keep a stand-in tokenizer and do not import `renderers`.
