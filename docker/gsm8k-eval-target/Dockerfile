# GSM8K eval.target.v1 target for `mlx-lora.v1` candidates.
#
# The pinned dataset rows (openai/gsm8k @ GSM8K_REVISION, verified against the
# split digests recorded in containers' gsm8k_world.py) are baked in, and the
# world module itself is vendored from a pinned containers commit, so a trial
# needs no dataset download and scores with exactly the parser the containers
# target uses. The policy is NOT in the image: it is a snapshot registered with
# the host's synth-mlx-rl service, reached over the recipe-owned route.
FROM python:3.11-slim

ARG GSM8K_REVISION=unknown
ARG CONTAINERS_SOURCE_COMMIT=unknown
LABEL ai.synth.eval.target.contract="eval.target.v1"
LABEL ai.synth.eval.target.task="gsm8k"
LABEL ai.synth.eval.target.dataset="openai/gsm8k"
LABEL ai.synth.eval.target.dataset-revision="${GSM8K_REVISION}"
LABEL ai.synth.eval.target.source-repo="containers"
LABEL ai.synth.eval.target.source-commit="${CONTAINERS_SOURCE_COMMIT}"
LABEL ai.synth.eval.target.source-subpath="src/synth_containers/platform/gsm8k_world.py"

RUN useradd --create-home --uid 10001 target
COPY gsm8k_world.py /opt/eval/gsm8k_world.py
COPY gsm8k /opt/gsm8k
COPY target.py /app/target.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

USER target
WORKDIR /app
ENTRYPOINT ["python", "/app/target.py"]
