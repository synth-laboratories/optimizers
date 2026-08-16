# GameBench/Harbor eval.target.v1 target.
#
# Harbor's workspace layout and verifier surface, wrapped in the generic target
# contract. build.sh stages the GameBench task tree from a pinned checkout.
FROM python:3.11-slim

ARG GAMEBENCH_SOURCE_COMMIT=unknown
LABEL ai.synth.eval.target.contract="eval.target.v1"
LABEL ai.synth.eval.target.source-repo="gamebench"
LABEL ai.synth.eval.target.source-commit="${GAMEBENCH_SOURCE_COMMIT}"
LABEL ai.synth.eval.target.harbor.workspace-root="/workspace"

RUN useradd --create-home --uid 10001 target
COPY gamebench /workspace/gamebench
COPY verify_single_candidate.py /opt/harbor/verify_single_candidate.py
COPY target.py /app/target.py
COPY shared /opt/eval
RUN mkdir -p /workspace/candidates /logs/verifier \
    && chown -R target:target /workspace/candidates /logs

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    GAMEBENCH_TASK=craftax-singleplayer \
    CANDIDATE_SUBDIR=craftax \
    GAMEBENCH_WORKSPACE_ROOT=/workspace \
    GAMEBENCH_ROOT=/workspace/gamebench \
    HARBOR_LOG_DIR=/logs/verifier

USER target
WORKDIR /workspace
ENTRYPOINT ["python", "/app/target.py"]
