"""Staging policy source into an immutable, content-addressed candidate set.

Staging happens before a run starts and is the reason the start API can accept
a `candidate_set_id` instead of paths or inline code. Once staged, an artifact
is read-only and addressed by the digest of its contents, so the evidence a run
produces can always be traced back to exactly the bytes that were evaluated.

Workshop performs the same staging from an attached workspace; this module is
the reference implementation and the path used by tests and operators.
"""

from __future__ import annotations

import shutil
import stat
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .home import EvalHome
from .models import (
    MLX_LORA_POLICY_KIND,
    CandidateSet,
    EvalContractError,
    PolicyCandidate,
    digest_of_tree,
    read_mlx_lora_policy,
    write_json,
)


@dataclass(frozen=True, slots=True)
class CandidateSource:
    label: str
    path: Path
    entrypoint: str
    kind: str = "python-code.v1"
    is_baseline: bool = False


def stage_candidate_set(
    home: EvalHome,
    sources: Sequence[CandidateSource],
    *,
    set_id: str | None = None,
) -> CandidateSet:
    if not sources:
        raise EvalContractError("a candidate set needs at least one candidate")
    labels = [source.label for source in sources]
    if len(set(labels)) != len(labels):
        raise EvalContractError("candidate labels must be unique inside a set")
    baselines = [source for source in sources if source.is_baseline]
    if len(baselines) > 1:
        raise EvalContractError("a candidate set may designate at most one baseline")

    set_id = set_id or f"policy_set_{uuid.uuid4().hex[:12]}"
    root = home.candidates_dir / set_id
    if root.exists():
        raise EvalContractError(f"candidate set {set_id} already exists and is immutable")
    artifacts = root / "artifacts"
    artifacts.mkdir(parents=True)

    candidates: list[PolicyCandidate] = []
    baseline_id: str | None = None
    for source in sources:
        origin = source.path.expanduser()
        if not origin.exists():
            raise EvalContractError(f"policy source does not exist: {origin}")
        metadata: dict[str, Any] = {"source": {"kind": "workspace", "name": origin.name}}
        if source.kind == MLX_LORA_POLICY_KIND:
            # Refused here rather than at run time: an adapter set whose bytes
            # are wrong is not evidence that arrived late, it is a candidate
            # set that should never have been sealed.
            metadata["mlx_lora"] = read_mlx_lora_policy(origin).to_json()
        staged = artifacts / f"pending_{uuid.uuid4().hex[:12]}"
        if origin.is_dir():
            shutil.copytree(origin, staged)
        else:
            staged.mkdir()
            shutil.copy2(origin, staged / origin.name)
        digest = digest_of_tree(staged)
        final = artifacts / digest.split(":", 1)[1]
        if final.exists():
            shutil.rmtree(staged)  # identical bytes already staged
        else:
            staged.rename(final)
            _freeze(final)
        candidate = PolicyCandidate(
            id=f"policy_{uuid.uuid4().hex[:12]}",
            label=source.label,
            kind=source.kind,
            artifact_uri=f"local-artifact://sha256/{digest.split(':', 1)[1]}",
            artifact_digest=digest,
            entrypoint=source.entrypoint,
            metadata=metadata,
        )
        candidates.append(candidate)
        if source.is_baseline:
            baseline_id = candidate.id

    candidate_set = CandidateSet(
        id=set_id,
        candidates=tuple(candidates),
        baseline_id=baseline_id,
        created_at=datetime.now(UTC).isoformat(),
        root=root,
    )
    write_json(root / "candidate_set.json", candidate_set.to_json())
    return candidate_set


def _freeze(root: Path) -> None:
    """Make a staged artifact read-only so later runs cannot be re-based."""

    for path in sorted(root.rglob("*"), reverse=True):
        mode = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
        if path.is_dir():
            mode |= stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
        path.chmod(mode)
    root.chmod(stat.S_IRUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IXGRP)
