# Deterministic eval.target.v1 fixture. No network, no credentials, no state
# beyond /output: the target reads the mounted candidate and the trial manifest
# and writes exactly the contract's files.
FROM python:3.11-slim

RUN useradd --create-home --uid 10001 target
WORKDIR /app
COPY target.py /app/target.py

USER target
ENTRYPOINT ["python", "/app/target.py"]
