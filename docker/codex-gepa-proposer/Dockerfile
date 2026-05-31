FROM node:20.20.0-bookworm-slim

ARG CODEX_CLI_PACKAGE=@openai/codex@0.135.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bubblewrap \
        ca-certificates \
        git \
        jq \
        python3 \
        python3-pip \
        ripgrep \
    && npm install -g "${CODEX_CLI_PACKAGE}" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
