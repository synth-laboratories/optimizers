from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "paid_plane", Path(__file__).parents[1] / "docs/e2e/paid_plane.py"
)
assert _SPEC is not None and _SPEC.loader is not None
paid_plane = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(paid_plane)


class _Catalog:
    closed = False

    def close(self) -> None:
        self.closed = True


class _Resolution:
    def policy_for_group(self, parameter_group_id: str) -> SimpleNamespace:
        assert parameter_group_id == "pg-0"
        return SimpleNamespace(base_model="vendor/wrong-model")


class _Resolver:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def resolve_training_state(self, *_args: object, **_kwargs: object) -> _Resolution:
        return _Resolution()


def test_paid_plane_refuses_resume_model_mismatch_before_provider_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _Catalog()
    provider = SimpleNamespace(restore_session=lambda *_args, **_kwargs: pytest.fail("restored"))
    config = SimpleNamespace(
        run_id="stage-2",
        model=SimpleNamespace(
            id="vendor/right-model", resume_from_checkpoint="ckpt_immutable"
        ),
    )
    monkeypatch.setattr(paid_plane, "open_catalog", lambda _config: catalog)
    monkeypatch.setattr(paid_plane, "EvaluationResolver", _Resolver)
    monkeypatch.setattr(paid_plane, "ProviderArtifactProbe", lambda _provider: object())

    with pytest.raises(RuntimeError, match="does not match configured model"):
        paid_plane._restore_parent(config, provider, "pg-0", object())

    assert catalog.closed is True


def test_restore_parent_canonicalizes_identity_and_restores_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = _Catalog()
    calls: list[object] = []
    provider = SimpleNamespace(
        restore_session=lambda checkpoint, **_kwargs: calls.append(checkpoint)
    )
    config = SimpleNamespace(
        run_id="stage-2",
        model=SimpleNamespace(id="vendor/model", resume_from_checkpoint="ckpt_parent"),
    )
    artifact_digest = "sha256:" + "AB" * 32
    policy = SimpleNamespace(
        base_model="vendor/model",
        checkpoint_id="ckpt_parent",
        policy_revision_id="pg-0@8",
        artifact=SimpleNamespace(ref="  tinker://state  ", digest=f" {artifact_digest} "),
    )

    class Resolver:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def resolve_training_state(self, *_args: object, **_kwargs: object) -> object:
            return SimpleNamespace(policy_for_group=lambda _group: policy)

    monkeypatch.setattr(paid_plane, "open_catalog", lambda _config: catalog)
    monkeypatch.setattr(paid_plane, "EvaluationResolver", Resolver)

    identity = paid_plane._restore_parent(config, provider, "pg-0", object())

    assert identity == {"ref": "tinker://state", "digest": artifact_digest.lower()}
    assert len(calls) == 1
    assert calls[0].provider_reference == "tinker://state"
    assert calls[0].digest == artifact_digest.lower()
    assert calls[0].step == 8
    assert catalog.closed is True


def test_resume_artifact_probe_uses_independent_digest_map(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reference = "tinker://state"
    artifact_digest = "sha256:" + "ab" * 32
    path = tmp_path / "digests.json"
    path.write_text(json.dumps({reference: artifact_digest}), encoding="utf-8")
    monkeypatch.setenv(paid_plane.ARTIFACT_DIGESTS_ENV, str(path))

    probe = paid_plane._resume_artifact_probe(SimpleNamespace(artifacts={}))

    assert probe.exists(reference) is True
    assert probe.digest_of(reference) == artifact_digest
    assert probe.exists("tinker://missing") is False
    with pytest.raises(Exception, match="does not exist"):
        probe.digest_of("tinker://missing")


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"tinker://state": 3},
        {"tinker://state": "sha256:short"},
        {"": "sha256:" + "ab" * 32},
        {"provider://state": "sha256:" + "ab" * 32},
    ],
)
def test_resume_artifact_probe_rejects_malformed_maps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, payload: object
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv(paid_plane.ARTIFACT_DIGESTS_ENV, str(path))

    with pytest.raises(RuntimeError):
        paid_plane._resume_artifact_probe(SimpleNamespace())


def test_paid_plane_shares_one_resume_probe_with_prewarm_and_binder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = paid_plane.MappingArtifactProbe(
        digests={
            "tinker://state": "sha256:" + "ab" * 32,
            "tinker://unrelated": "sha256:" + "cd" * 32,
        }
    )
    provider = SimpleNamespace()
    config = SimpleNamespace(
        run_id="stage-2",
        model=SimpleNamespace(
            id="vendor/model",
            resume_from_checkpoint="ckpt_parent",
            rank=8,
        ),
    )
    seen: dict[str, object] = {}
    monkeypatch.setattr(paid_plane, "_load_credential", lambda: None)
    monkeypatch.setattr(paid_plane, "build_provider", lambda _config: provider)
    monkeypatch.setattr(
        paid_plane,
        "_restore_parent",
        lambda _config, _provider, _group, artifact_probe: (
            seen.update(prewarm_probe=artifact_probe)
            or {"ref": "tinker://state", "digest": "sha256:" + "ab" * 32}
        ),
    )
    monkeypatch.setattr(
        paid_plane,
        "build_plane",
        lambda _config, **kwargs: seen.update(binder_probe=kwargs["artifact_probe"])
        or "plane",
    )

    result = paid_plane.paid(config, artifact_probe=probe)

    assert result == "plane"
    assert seen == {"prewarm_probe": probe, "binder_probe": probe}
    assert provider._resume_artifact_identity == {
        "ref": "tinker://state",
        "digest": "sha256:" + "ab" * 32,
    }
