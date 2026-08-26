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
    /tmp/craftax_repl
# This image has just built the REPL from the exact source tree copied above.
# Put that binary where GameBench's non-fixture host path expects it and make
# the local eval image prefer it over the separately published AArch64 fixture.
# The published fixture remains the default everywhere outside this image.
RUN set -eux; \
    case "$(uname -m)" in \
      aarch64|arm64) host_arch=aarch64 ;; \
      x86_64|amd64) host_arch=x86_64 ;; \
      *) echo "unsupported Craftax image architecture: $(uname -m)" >&2; exit 1 ;; \
    esac; \
    host_dir="/opt/gamebench/tasks/craftax-singleplayer/gold_rust/target/gamebench-host/linux-${host_arch}/release"; \
    mkdir -p "$host_dir"; \
    install -m 0755 /tmp/craftax_repl "$host_dir/craftax_repl"; \
    rm /tmp/craftax_repl
RUN python - <<'PY'
from pathlib import Path

path = Path("/opt/gamebench/tasks/craftax-singleplayer/containers/codepolicy/rust_repl_session.py")
source = path.read_text(encoding="utf-8")
needle = '''def _host_uses_linux_aarch64_fixture() -> bool:\n    return _host_platform_identity() == ("linux", "aarch64")\n'''
replacement = '''def _host_uses_linux_aarch64_fixture() -> bool:\n    if os.environ.get("GAMEBENCH_USE_BAKED_REPL") == "1":\n        return False\n    return _host_platform_identity() == ("linux", "aarch64")\n'''
if source.count(needle) != 1:
    raise SystemExit("unexpected GameBench Rust REPL fixture selector")
path.write_text(source.replace(needle, replacement), encoding="utf-8")

sweep = Path("/opt/gamebench/tasks/craftax-singleplayer/scripts/run_policy_sweep.py")
source = sweep.read_text(encoding="utf-8")
needle = '''    return trusted_declarative or (\n'''
replacement = '''    trusted_outer_container = (\n        os.environ.get("EVAL_TRUSTED_OUTER_CONTAINER") == "report_only"\n        and isinstance(receipt, Mapping)\n        and receipt.get("contract") == "process_observation_action.v1"\n        and receipt.get("sandbox") == "process_fallback"\n        and receipt.get("suite_visible") is False\n        and receipt.get("output_visible") is False\n    )\n    return trusted_declarative or trusted_outer_container or (\n'''
if source.count(needle) != 1:
    raise SystemExit("unexpected GameBench isolation receipt validator")
sweep.write_text(source.replace(needle, replacement), encoding="utf-8")
PY
COPY target.py /app/target.py
COPY shared /opt/eval
# The fixture-bound REPL resolves task defaults through the canonical
# GameBench container workspace prefix.
RUN ln -s /opt/gamebench /workspace
# The trusted harness verifies and atomically materializes the fixture binary
# at runtime. Its target/release directory must therefore be writable by the
# dedicated non-root evaluator user; candidate code remains isolated elsewhere.
RUN chown -R target:target \
    /opt/gamebench/tasks/craftax-singleplayer/gold_rust/target
# The outer eval container is the security boundary. GameBench records this as
# container_standalone and avoids attempting a nested bubblewrap namespace.
ENV GAMEBENCH_STANDALONE=1 \
    GAMEBENCH_USE_BAKED_REPL=1 \
    EVAL_TRUSTED_OUTER_CONTAINER=report_only \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER target
WORKDIR /opt/gamebench/tasks/craftax-singleplayer
ENTRYPOINT ["python", "/app/target.py"]
