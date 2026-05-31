#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"

load_api_key_if_needed() {
  local key_name="$1"
  if [[ -n "${!key_name:-}" ]]; then
    return
  fi
  local env_file=""
  local env_value=""
  for env_file in \
    "$SCRIPT_DIR/.env" \
    "$REPO_ROOT/../synth-ai/.env" \
    "$REPO_ROOT/../synth-dev/.env.shared" \
    "$REPO_ROOT/../backend/.env.local"
  do
    if [[ -f "$env_file" ]]; then
      env_value="$(KEY="$key_name" awk -F= '$1 == ENVIRON["KEY"] { sub(/^[^=]*=/, ""); print; exit }' "$env_file")"
      env_value="${env_value%\"}"
      env_value="${env_value#\"}"
      if [[ -n "$env_value" ]]; then
        export "$key_name=$env_value"
        echo "GEPA loaded $key_name from $env_file"
        return
      fi
    fi
  done
}

POLICY_API_KEY_ENV="${GEPA_POLICY_API_KEY_ENV:-GEMINI_API_KEY}"
PROPOSER_API_KEY_ENV="${GEPA_PROPOSER_API_KEY_ENV:-OPENAI_API_KEY}"
load_api_key_if_needed "$POLICY_API_KEY_ENV"
load_api_key_if_needed "$PROPOSER_API_KEY_ENV"
if [[ -z "${!POLICY_API_KEY_ENV:-}" ]]; then
  echo "error: $POLICY_API_KEY_ENV is not set; live policy rollouts need an API key" >&2
  exit 1
fi
if [[ -z "${!PROPOSER_API_KEY_ENV:-}" ]]; then
  echo "error: $PROPOSER_API_KEY_ENV is not set; Codex proposer auth needs an API key" >&2
  exit 1
fi

cd "$SCRIPT_DIR"
exec uv run --project "$REPO_ROOT" python "$SCRIPT_DIR/run_gepa_sdk.py" "$@"
