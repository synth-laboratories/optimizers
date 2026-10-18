from types import SimpleNamespace

from synth_optimizers.eval.commands import _runtime_recipe_readiness
from synth_optimizers.eval.executor import ContainerRuntimeError


def recipe(*, digest: str | None = "sha256:" + "ab" * 32):
    return SimpleNamespace(
        id="eval.craftax.code-policy.smoke.v1",
        image="ghcr.io/synth-laboratories/workshop-craftax-eval-target",
        image_digest=digest,
        unavailable_reason=None if digest else "target image is not published and pinned yet",
    )


class Executor:
    def __init__(self, error: str | None = None) -> None:
        self.error = error

    def resolve_reference(self, image: str, digest: str) -> str:
        if self.error:
            raise ContainerRuntimeError(self.error)
        return f"{image}@{digest}"


def test_doctor_blocks_digest_pinned_but_missing_image_before_run_creation():
    result = _runtime_recipe_readiness([recipe()], Executor("image is not present locally"))
    assert result == [
        {
            "id": "eval.craftax.code-policy.smoke.v1",
            "available": False,
            "reason": "image is not present locally",
            "image": "ghcr.io/synth-laboratories/workshop-craftax-eval-target",
            "imageDigest": "sha256:" + "ab" * 32,
            "resolvedReference": None,
        }
    ]


def test_doctor_advertises_only_the_exact_resolved_digest():
    result = _runtime_recipe_readiness([recipe()], Executor())
    assert result[0]["available"] is True
    assert result[0]["resolvedReference"].endswith("@sha256:" + "ab" * 32)


def test_local_image_id_precedes_identical_repository_digest(monkeypatch):
    from synth_optimizers.eval.executor import OciTrialExecutor

    monkeypatch.setattr("synth_optimizers.eval.executor.shutil.which", lambda _: "/usr/bin/docker")
    executor = OciTrialExecutor()
    digest = "sha256:" + "ba" * 32
    monkeypatch.setattr(executor, "image_digests", lambda _: (digest, (digest,)))
    assert executor.resolve_reference("craftax-eval-target", digest) == digest


def test_cancel_removes_created_container_and_retries_after_client_exit(monkeypatch):
    import subprocess

    from synth_optimizers.eval.executor import OciTrialExecutor

    monkeypatch.setattr("synth_optimizers.eval.executor.shutil.which", lambda _: "/usr/bin/docker")
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if len(calls) == 1:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("synth_optimizers.eval.executor.subprocess.run", run)
    waited = []
    process = SimpleNamespace(wait=lambda **kwargs: waited.append(kwargs))
    OciTrialExecutor()._kill("owned-created-container", process)
    assert waited == [{"timeout": 10}]
    assert len(calls) == 2
    assert all(call[0] == ["/usr/bin/docker", "rm", "--force", "owned-created-container"] for call in calls)
    assert all(call[1]["timeout"] == 15 for call in calls)
