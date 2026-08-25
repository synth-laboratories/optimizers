# Craftax eval.target.v1 target.
#
# The GameBench craftax-singleplayer task and its native Rust REPL are baked in,
# so a trial needs no host substrate, registry, or network. build.sh stages the
# task from a pinned GameBench checkout; nothing is vendored into this repo.
FROM rust:1.97-slim AS rust-builder

COPY gamebench /opt/gamebench
WORKDIR /opt/gamebench/tasks/craftax-singleplayer/gold_rust
RUN cargo build --release --locked --bin craftax_repl

FROM python:3.11-slim

ARG GAMEBENCH_SOURCE_COMMIT=unknown
ARG GAMEBENCH_SOURCE_SUBPATH=tasks/craftax-singleplayer
LABEL ai.synth.eval.target.contract="eval.target.v1"
LABEL ai.synth.eval.target.source-repo="gamebench"
LABEL ai.synth.eval.target.source-commit="${GAMEBENCH_SOURCE_COMMIT}"
LABEL ai.synth.eval.target.source-subpath="${GAMEBENCH_SOURCE_SUBPATH}"

RUN apt-get update && apt-get install -y --no-install-recommends bubblewrap ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 target
COPY gamebench /opt/gamebench
COPY --from=rust-builder \
    /opt/gamebench/tasks/craftax-singleplayer/gold_rust/target/release/craftax_repl \
    /opt/gamebench/tasks/craftax-singleplayer/gold_rust/target/release/craftax_repl
COPY target.py /app/target.py
COPY shared /opt/eval
# The trusted harness verifies and atomically materializes the fixture binary
# at runtime. Its target/release directory must therefore be writable by the
# dedicated non-root evaluator user; candidate code remains isolated elsewhere.
RUN chown -R target:target \
    /opt/gamebench/tasks/craftax-singleplayer/gold_rust/target
# The outer eval container is the security boundary. GameBench records this as
# container_standalone and avoids attempting a nested bubblewrap namespace.
ENV GAMEBENCH_STANDALONE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

USER target
WORKDIR /opt/gamebench/tasks/craftax-singleplayer
ENTRYPOINT ["python", "/app/target.py"]
