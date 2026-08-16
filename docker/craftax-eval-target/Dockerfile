# Craftax eval.target.v1 target.
#
# The GameBench craftax-singleplayer task (symbolic `gold_python` engine, world
# defaults, policy sweep) is baked in, so a trial needs no host substrate and no
# network. build.sh stages the task from a pinned gamebench checkout; nothing is
# vendored into this repo.
FROM python:3.11-slim

ARG GAMEBENCH_SOURCE_COMMIT=unknown
ARG GAMEBENCH_SOURCE_SUBPATH=tasks/craftax-singleplayer
LABEL ai.synth.eval.target.contract="eval.target.v1"
LABEL ai.synth.eval.target.source-repo="gamebench"
LABEL ai.synth.eval.target.source-commit="${GAMEBENCH_SOURCE_COMMIT}"
LABEL ai.synth.eval.target.source-subpath="${GAMEBENCH_SOURCE_SUBPATH}"

RUN useradd --create-home --uid 10001 target
COPY gamebench /opt/gamebench
COPY target.py /app/target.py
COPY shared /opt/eval
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

USER target
WORKDIR /opt/gamebench/tasks/craftax-singleplayer
ENTRYPOINT ["python", "/app/target.py"]
