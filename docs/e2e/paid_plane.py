"""The real provider, assembled the way a production run assembles it.

Nothing is stubbed here. The only difference from `rl run` with no --plane is
that the credential is read from a named file rather than the ambient
environment, because it does not live in this shell.
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any
from urllib.parse import urlsplit

from synth_optimizers.contracts.rl_records import digest
from synth_optimizers.providers.protocols import ProviderCheckpoint
from synth_optimizers.rl.binder import TRAINING_STATE_KIND, revision_number_of
from synth_optimizers.rl.gateway import PromptBudget
from synth_optimizers.rl.plane import ProviderArtifactProbe, build_plane, build_provider, open_catalog
from synth_optimizers.rl.resolver import EvaluationResolver, MappingArtifactProbe, ResolutionScope

ARTIFACT_DIGESTS_ENV = "SYNTH_E2E_ARTIFACT_DIGESTS"

#: Where the Tinker credential lives when it is not already in the environment.
#: Override with SYNTH_TINKER_ENV_FILE; the default is where it happened to be
#: on the machine this was written on, which is not a promise about yours.
KEY_FILE = os.environ.get(
    "SYNTH_TINKER_ENV_FILE", "/Users/joshuapurtell/GitHub/frontend/.env.local"
)


def _load_credential() -> None:
    if os.environ.get("TINKER_API_KEY"):
        return
    for line in pathlib.Path(KEY_FILE).read_text().splitlines():
        if line.startswith("TINKER_API_KEY="):
            os.environ["TINKER_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
            return
    raise SystemExit(f"no TINKER_API_KEY in {KEY_FILE}")


def _canonical_tinker_identity(reference: Any, artifact_digest: Any) -> tuple[str, str]:
    if not isinstance(reference, str) or not reference.strip():
        raise RuntimeError("artifact reference must be a non-empty tinker:// URI")
    stripped_reference = reference.strip()
    parsed = urlsplit(stripped_reference)
    if parsed.scheme != "tinker" or not parsed.netloc or parsed.query or parsed.fragment:
        raise RuntimeError(f"artifact reference {reference!r} must be a tinker:// URI")
    canonical_reference = "tinker://" + stripped_reference.split("://", 1)[1]
    if not isinstance(artifact_digest, str):
        raise RuntimeError(
            f"artifact digest for {canonical_reference!r} must be a sha256 string"
        )
    canonical_digest = artifact_digest.strip().lower()
    valid_digest = (
        len(canonical_digest) == 71
        and canonical_digest.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in canonical_digest[7:])
    )
    if not valid_digest:
        raise RuntimeError(
            f"artifact digest for {canonical_reference!r} must be a sha256 string"
        )
    return canonical_reference, canonical_digest


def _resume_artifact_probe(provider: Any) -> Any:
    path = os.environ.get(ARTIFACT_DIGESTS_ENV, "").strip()
    if not path:
        return ProviderArtifactProbe(provider)
    source = pathlib.Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot load {ARTIFACT_DIGESTS_ENV}={source}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{ARTIFACT_DIGESTS_ENV} must contain a JSON object")
    digests: dict[str, str] = {}
    for reference, artifact_digest in payload.items():
        try:
            canonical_reference, canonical_digest = _canonical_tinker_identity(
                reference, artifact_digest
            )
        except RuntimeError as error:
            raise RuntimeError(f"{ARTIFACT_DIGESTS_ENV}: {error}") from error
        if canonical_reference in digests:
            raise RuntimeError(
                f"{ARTIFACT_DIGESTS_ENV} contains duplicate reference {canonical_reference!r}"
            )
        digests[canonical_reference] = canonical_digest
    return MappingArtifactProbe(digests=digests)


def _restore_parent(
    config: Any, provider: Any, parameter_group_id: str, artifact_probe: Any
) -> dict[str, str] | None:
    """Resolve every local identity constraint before asking Tinker to restore."""

    selector = config.model.resume_from_checkpoint
    if selector is None:
        return None
    catalog = open_catalog(config)
    try:
        resolution = EvaluationResolver(catalog, probe=artifact_probe).resolve_training_state(
            selector,
            scope=ResolutionScope(parameter_group_id=parameter_group_id),
        )
        policy = resolution.policy_for_group(parameter_group_id)
    finally:
        catalog.close()
    if policy.base_model != config.model.id:
        raise RuntimeError(
            f"resume checkpoint base model {policy.base_model!r} does not match "
            f"configured model {config.model.id!r}"
        )
    artifact = policy.artifact
    reference, artifact_digest = _canonical_tinker_identity(artifact.ref, artifact.digest)
    checkpoint = ProviderCheckpoint(
        checkpoint_id=policy.checkpoint_id,
        provider_reference=reference,
        step=revision_number_of(policy.policy_revision_id),
        digest=artifact_digest,
        kind=TRAINING_STATE_KIND,
        resume_token=reference,
        model_id=policy.base_model,
    )
    provider.restore_session(
        checkpoint,
        request_id="restore-" + digest(
            {
                "run_id": config.run_id,
                "parameter_group_id": parameter_group_id,
                "checkpoint_id": policy.checkpoint_id,
            },
            length=32,
        ),
    )
    return {"ref": reference, "digest": artifact_digest}


def paid(config: Any = None, **kwargs: Any) -> Any:
    if config is None:
        raise SystemExit("this plane needs --config: it is assembled from one")
    _load_credential()
    provider = build_provider(config)
    # Tinker's authoritative renderer is attached to a training session. Prime
    # it via the same deterministic create/restore request the binder will
    # idempotently reuse, so compatibility is checked against the actual model.
    parameter_group_id = os.environ.get("SYNTH_E2E_PARAMETER_GROUP", "pg-0")
    resume_selector = config.model.resume_from_checkpoint
    if resume_selector is not None:
        resume_probe = kwargs.get("artifact_probe")
        if resume_probe is None:
            resume_probe = _resume_artifact_probe(provider)
        kwargs["artifact_probe"] = resume_probe
        resume_identity = _restore_parent(
            config, provider, parameter_group_id, resume_probe
        )
        if resume_identity is not None:
            setattr(provider, "_resume_artifact_identity", resume_identity)
    else:
        request_id = "session-" + digest(
            {
                "run_id": config.run_id,
                "parameter_group_id": parameter_group_id,
                "base_model": config.model.id,
                "rank": config.model.rank,
                "seed": 0,
            },
            length=32,
        )
        provider.create_session(
            config.model.id, rank=config.model.rank, seed=0, request_id=request_id
        )
    prompt_limit = os.environ.get("SYNTH_E2E_MAX_CONTEXT_TOKENS")
    if prompt_limit:
        kwargs["prompt_budget"] = PromptBudget(
            max_prompt_tokens=int(prompt_limit), policy="compact"
        )
    return build_plane(config, provider=provider, **kwargs)
