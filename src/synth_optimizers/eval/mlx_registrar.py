"""Host-side `PolicySnapshotRegistrar` for `mlx-lora.v1` candidates.

The trial container cannot load an adapter: the inference service runs on the
host. This module posts the staged candidate directory to `synth-mlx-rl` and
returns the immutable snapshot id the trial pins. Re-registering the same
artifact digest returns the same id.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .home import EvalHome
from .models import EvalContractError, LOCAL_HTTP_HOSTS, MLX_LORA_POLICY_KIND
from .recipes import EvalRecipe
from .runner import PolicySnapshotRegistrar, WorkerManifest

JsonPoster = Callable[[str, dict[str, Any], dict[str, str], float], dict[str, Any]]

DEFAULT_MLX_URL = "http://127.0.0.1:8787"


def snapshot_id_for_digest(artifact_digest: str) -> str:
    hex_part = artifact_digest.split(":", 1)[-1].strip()
    if not hex_part or not all(character in "0123456789abcdefABCDEF" for character in hex_part):
        raise EvalContractError(f"cannot derive a snapshot id from digest {artifact_digest!r}")
    return f"snap_{hex_part.lower()}"


def _post_json(
    url: str, payload: dict[str, Any], headers: dict[str, str], timeout: float
) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EvalContractError(
            f"synth-mlx-rl refused policy registration ({exc.code}): {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise EvalContractError(
            f"synth-mlx-rl is unreachable at {url}: {exc.reason}"
        ) from exc
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise EvalContractError(f"synth-mlx-rl returned a non-JSON body: {raw[:200]}") from exc
    if not isinstance(decoded, dict):
        raise EvalContractError("synth-mlx-rl policy registration returned a non-object")
    return decoded


class MlxHttpPolicySnapshotRegistrar:
    """Turns a staged candidate directory into an immutable MLX snapshot id."""

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeout: float = 30.0,
        poster: JsonPoster | None = None,
    ) -> None:
        parsed = urlparse(base_url)
        host = (parsed.hostname or "").lower()
        if parsed.scheme != "http" or host not in LOCAL_HTTP_HOSTS:
            raise EvalContractError(
                f"mlx inference URL must be a local http origin, got {base_url!r}"
            )
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._poster = poster or _post_json
        self._ids: dict[str, str] = {}

    def register(
        self,
        *,
        candidate_id: str,
        artifact_digest: str,
        policy_dir: Path,
    ) -> str:
        previous = self._ids.get(artifact_digest)
        if previous:
            return previous
        snapshot_id = snapshot_id_for_digest(artifact_digest)
        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        body = self._poster(
            f"{self.base_url}/v1/synth/policies/register",
            {
                "policy_dir": str(policy_dir),
                "snapshot_id": snapshot_id,
                "artifact_digest": artifact_digest,
                "candidate_id": candidate_id,
            },
            headers,
            self.timeout,
        )
        returned = body.get("policy_snapshot_id")
        if not isinstance(returned, str) or not returned.strip():
            raise EvalContractError(
                "synth-mlx-rl policy registration returned no policy_snapshot_id"
            )
        returned = returned.strip()
        self._ids[artifact_digest] = returned
        return returned


def registrar_for_manifest(
    manifest_path: Path,
) -> PolicySnapshotRegistrar | None:
    """Build a registrar when the recipe's candidates cannot load themselves."""

    manifest = WorkerManifest.load(manifest_path)
    home = EvalHome.open(manifest.home, create=False)
    recipe: EvalRecipe = home.recipe(manifest.recipe_id)
    if MLX_LORA_POLICY_KIND not in recipe.target.policy_kinds:
        return None
    url = manifest.mlx_inference_url or home.config.mlx_inference_url
    token = None
    if "SYNTH_MLX_RL_TOKEN" in recipe.secrets:
        try:
            token = home.resolve_secret(
                "SYNTH_MLX_RL_TOKEN", declared=recipe.secrets
            )
        except EvalContractError:
            token = None
    return MlxHttpPolicySnapshotRegistrar(url, token=token)
