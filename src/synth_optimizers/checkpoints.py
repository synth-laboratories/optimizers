"""Typed contracts for the searchable saved-LoRA checkpoint library."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class SavedLoraContractError(ValueError):
    """Raised when the backend returns an invalid checkpoint contract."""


def _required_text(payload: Mapping[str, Any], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise SavedLoraContractError(f"saved LoRA checkpoint missing {name}")
    return value.strip()


def _optional_text(payload: Mapping[str, Any], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise SavedLoraContractError(f"saved LoRA checkpoint {name} must be text")
    return value


@dataclass(frozen=True, slots=True)
class SavedLoraStorage:
    backend: str
    bucket: str
    key: str
    version: str | None = None
    etag: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    content_type: str = "application/x-tar"

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SavedLoraStorage":
        size = payload.get("size_bytes")
        if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size < 0):
            raise SavedLoraContractError("saved LoRA storage size_bytes must be non-negative")
        return cls(
            backend=_required_text(payload, "backend"),
            bucket=_required_text(payload, "bucket"),
            key=_required_text(payload, "key"),
            version=_optional_text(payload, "version"),
            etag=_optional_text(payload, "etag"),
            sha256=_optional_text(payload, "sha256"),
            size_bytes=size,
            content_type=_required_text(payload, "content_type"),
        )


@dataclass(frozen=True, slots=True)
class SavedLoraLineage:
    """Canonical optimizer run and provider checkpoint that produced an archive."""

    optimizer_algorithm: str | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    source_checkpoint_id: str | None = None
    provider_checkpoint_reference: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SavedLoraLineage":
        return cls(
            optimizer_algorithm=_optional_text(payload, "optimizer_algorithm"),
            run_id=_optional_text(payload, "run_id"),
            attempt_id=_optional_text(payload, "attempt_id"),
            source_checkpoint_id=_optional_text(payload, "source_checkpoint_id"),
            provider_checkpoint_reference=_optional_text(
                payload, "provider_checkpoint_reference"
            ),
        )


@dataclass(frozen=True, slots=True)
class SavedLoraCheckpoint:
    checkpoint_id: str
    org_id: str
    owner_user_id: str | None
    visibility: str
    name: str
    description: str
    provider: str
    checkpoint_kind: str
    base_model: str
    status: str
    storage: SavedLoraStorage
    tags: tuple[str, ...] = ()
    provider_checkpoint_reference: str | None = None
    run_id: str | None = None
    attempt_id: str | None = None
    source_checkpoint_id: str | None = None
    optimizer_algorithm: str | None = None
    lineage: SavedLoraLineage = field(default_factory=SavedLoraLineage)
    lora_rank: int | None = None
    step: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None
    archived_at: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SavedLoraCheckpoint":
        storage = payload.get("storage")
        tags = payload.get("tags", [])
        metadata = payload.get("metadata", {})
        lineage = payload.get("lineage", {})
        if not isinstance(storage, Mapping):
            raise SavedLoraContractError("saved LoRA checkpoint missing storage object")
        if not isinstance(tags, Sequence) or isinstance(tags, str | bytes):
            raise SavedLoraContractError("saved LoRA checkpoint tags must be a list")
        if not isinstance(metadata, Mapping):
            raise SavedLoraContractError("saved LoRA checkpoint metadata must be an object")
        if not isinstance(lineage, Mapping):
            raise SavedLoraContractError("saved LoRA checkpoint lineage must be an object")
        return cls(
            checkpoint_id=_required_text(payload, "checkpoint_id"),
            org_id=_required_text(payload, "org_id"),
            owner_user_id=_optional_text(payload, "owner_user_id"),
            visibility=_required_text(payload, "visibility"),
            name=_required_text(payload, "name"),
            description=_optional_text(payload, "description") or "",
            provider=_required_text(payload, "provider"),
            checkpoint_kind=_required_text(payload, "checkpoint_kind"),
            provider_checkpoint_reference=_optional_text(payload, "provider_checkpoint_reference"),
            run_id=_optional_text(payload, "run_id"),
            attempt_id=_optional_text(payload, "attempt_id"),
            source_checkpoint_id=_optional_text(payload, "source_checkpoint_id"),
            optimizer_algorithm=_optional_text(payload, "optimizer_algorithm"),
            lineage=SavedLoraLineage.from_payload(lineage or payload),
            base_model=_required_text(payload, "base_model"),
            lora_rank=payload.get("lora_rank"),
            step=payload.get("step"),
            status=_required_text(payload, "status"),
            storage=SavedLoraStorage.from_payload(storage),
            tags=tuple(str(tag) for tag in tags),
            metadata=dict(metadata),
            created_at=_optional_text(payload, "created_at"),
            updated_at=_optional_text(payload, "updated_at"),
            archived_at=_optional_text(payload, "archived_at"),
            raw=dict(payload),
        )


@dataclass(frozen=True, slots=True)
class SavedLoraCheckpointPage:
    items: tuple[SavedLoraCheckpoint, ...]
    total: int
    limit: int
    offset: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SavedLoraCheckpointPage":
        items = payload.get("items")
        if not isinstance(items, Sequence) or isinstance(items, str | bytes):
            raise SavedLoraContractError("saved LoRA checkpoint page missing items list")
        if any(not isinstance(item, Mapping) for item in items):
            raise SavedLoraContractError("saved LoRA checkpoint page item is not an object")
        return cls(
            items=tuple(SavedLoraCheckpoint.from_payload(item) for item in items),
            total=int(payload.get("total", 0)),
            limit=int(payload.get("limit", 0)),
            offset=int(payload.get("offset", 0)),
        )


@dataclass(frozen=True, slots=True)
class SavedLoraRunPage:
    """Saved LoRAs for one run, with exact kind counts and run identity."""

    run_id: str
    attempt_id: str | None
    optimizer_algorithm: str
    run_status: str
    items: tuple[SavedLoraCheckpoint, ...]
    counts: Mapping[str, int]
    total: int
    limit: int
    offset: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SavedLoraRunPage":
        run = payload.get("run")
        items = payload.get("items")
        counts = payload.get("counts", {})
        if not isinstance(run, Mapping):
            raise SavedLoraContractError("saved LoRA run page missing run object")
        if not isinstance(items, Sequence) or isinstance(items, str | bytes):
            raise SavedLoraContractError("saved LoRA run page missing items list")
        if not isinstance(counts, Mapping):
            raise SavedLoraContractError("saved LoRA run page counts must be an object")
        return cls(
            run_id=_required_text(run, "run_id"),
            attempt_id=_optional_text(run, "attempt_id"),
            optimizer_algorithm=_required_text(run, "optimizer_algorithm"),
            run_status=_required_text(run, "status"),
            items=tuple(SavedLoraCheckpoint.from_payload(item) for item in items),
            counts={str(key): int(value) for key, value in counts.items()},
            total=int(payload.get("total", 0)),
            limit=int(payload.get("limit", 0)),
            offset=int(payload.get("offset", 0)),
        )


@dataclass(frozen=True, slots=True)
class OptimizerRunArtifact:
    artifact_id: str
    run_id: str
    artifact_name: str
    content_type: str | None
    size_bytes: int
    sha256: str | None
    storage_backend: str
    uri: str
    download_path: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OptimizerRunArtifact":
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise SavedLoraContractError("optimizer artifact metadata must be an object")
        return cls(
            artifact_id=_required_text(payload, "artifact_id"),
            run_id=_required_text(payload, "run_id"),
            artifact_name=_required_text(payload, "artifact_name"),
            content_type=_optional_text(payload, "content_type"),
            size_bytes=int(payload.get("size_bytes", 0)),
            sha256=_optional_text(payload, "sha256"),
            storage_backend=_required_text(payload, "storage_backend"),
            uri=_required_text(payload, "uri"),
            download_path=_required_text(payload, "download_path"),
            metadata=dict(metadata),
        )


@dataclass(frozen=True, slots=True)
class OptimizerRunOutputs:
    """All automatically persisted outputs for one optimizer run."""

    run_id: str
    attempt_id: str | None
    optimizer_algorithm: str
    run_status: str
    result: Mapping[str, Any] | None
    artifacts: tuple[OptimizerRunArtifact, ...]
    model_checkpoints: tuple[SavedLoraCheckpoint, ...]
    counts: Mapping[str, int]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OptimizerRunOutputs":
        run = payload.get("run")
        result = payload.get("result")
        artifacts = payload.get("artifacts")
        checkpoints = payload.get("model_checkpoints")
        counts = payload.get("counts")
        if not isinstance(run, Mapping):
            raise SavedLoraContractError("optimizer outputs missing run object")
        if result is not None and not isinstance(result, Mapping):
            raise SavedLoraContractError("optimizer result must be an object or null")
        if not isinstance(artifacts, Sequence) or isinstance(artifacts, str | bytes):
            raise SavedLoraContractError("optimizer outputs missing artifacts list")
        if not isinstance(checkpoints, Sequence) or isinstance(checkpoints, str | bytes):
            raise SavedLoraContractError("optimizer outputs missing model checkpoints list")
        if not isinstance(counts, Mapping):
            raise SavedLoraContractError("optimizer output counts must be an object")
        return cls(
            run_id=_required_text(run, "run_id"),
            attempt_id=_optional_text(run, "attempt_id"),
            optimizer_algorithm=_required_text(run, "optimizer_algorithm"),
            run_status=_required_text(run, "status"),
            result=dict(result) if result is not None else None,
            artifacts=tuple(OptimizerRunArtifact.from_payload(item) for item in artifacts),
            model_checkpoints=tuple(
                SavedLoraCheckpoint.from_payload(item) for item in checkpoints
            ),
            counts={str(key): int(value) for key, value in counts.items()},
        )


@dataclass(frozen=True, slots=True)
class SavedLoraUploadIntent:
    checkpoint: SavedLoraCheckpoint
    method: str
    url: str
    expires_in: int
    content_type: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SavedLoraUploadIntent":
        upload = payload.get("upload")
        if not isinstance(upload, Mapping):
            raise SavedLoraContractError("saved LoRA upload response missing upload object")
        return cls(
            checkpoint=SavedLoraCheckpoint.from_payload(payload),
            method=_required_text(upload, "method"),
            url=_required_text(upload, "url"),
            expires_in=int(upload.get("expires_in", 0)),
            content_type=_required_text(upload, "content_type"),
        )


@dataclass(frozen=True, slots=True)
class SavedLoraDownload:
    checkpoint_id: str
    url: str
    expires_in: int
    content_type: str
    size_bytes: int | None
    sha256: str | None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SavedLoraDownload":
        return cls(
            checkpoint_id=_required_text(payload, "checkpoint_id"),
            url=_required_text(payload, "url"),
            expires_in=int(payload.get("expires_in", 0)),
            content_type=_required_text(payload, "content_type"),
            size_bytes=payload.get("size_bytes"),
            sha256=_optional_text(payload, "sha256"),
        )


__all__ = [
    "SavedLoraCheckpoint",
    "SavedLoraCheckpointPage",
    "SavedLoraContractError",
    "SavedLoraDownload",
    "OptimizerRunArtifact",
    "OptimizerRunOutputs",
    "SavedLoraLineage",
    "SavedLoraRunPage",
    "SavedLoraStorage",
    "SavedLoraUploadIntent",
]
