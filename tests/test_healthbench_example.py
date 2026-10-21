"""The installed example must never turn observer reconnect into new compute."""

import json
from types import SimpleNamespace

import pytest

from synth_optimizers.eval import healthbench


def test_terminal_run_reconnect_only_reads(tmp_path, monkeypatch, capsys):
    root = tmp_path / "runs/original"
    root.mkdir(parents=True)
    (root / "result_manifest.json").write_text(
        json.dumps({"run_id": "original", "artifacts": [], "trials": [{"valid": True}]})
    )
    monkeypatch.setattr(healthbench, "_run", lambda args: pytest.fail("duplicate compute"))
    assert healthbench.run(SimpleNamespace(home=tmp_path, run_id="original")) == 0
    assert json.loads(capsys.readouterr().out)["run_id"] == "original"


def test_read_refuses_changed_evidence(tmp_path):
    root = tmp_path / "runs/original"
    root.mkdir(parents=True)
    artifact = root / "answer.json"
    artifact.write_text("changed")
    (root / "result_manifest.json").write_text(
        json.dumps({"artifacts": [{"path": str(artifact), "digest": "sha256:bad"}]})
    )
    with pytest.raises(ValueError, match="artifact_digest"):
        healthbench.read(SimpleNamespace(home=tmp_path, run_id="original"))


def test_watch_caps_ten_rollouts_and_fixed_status(tmp_path, capsys):
    root = tmp_path / "runs/original"
    root.mkdir(parents=True)
    (root / "events.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "event": "eval.trial.event",
                    "trial_id": str(i),
                    "container_event": {"kind": "rubric.grade"},
                }
            )
            + "\n"
            for i in range(14)
        )
    )
    healthbench.watch(SimpleNamespace(home=tmp_path, run_id="original", once=True))
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 11
    assert all("rubric.grade" in line for line in lines[:10])
    assert lines[0].startswith("4 ")
    assert lines[-1] == "Status: eval.trial.event"


def test_package_descriptor_rejects_unknowns_without_build(tmp_path):
    from synth_containers.package_build import plan_container_package
    from synth_containers.package_files import ContainerPackageError

    path = tmp_path / "container-package.toml"
    path.write_text('schema_version = "synth.container-package.v1"\nunknown = true\n')
    with pytest.raises(ContainerPackageError, match="manifest_invalid"):
        plan_container_package(path)


def test_package_output_refuses_unpinned_registry_identity():
    from pydantic import ValidationError
    from synth_containers.package_models import ImageArtifactIdentity

    with pytest.raises(ValidationError):
        ImageArtifactIdentity(
            kind="oci_image",
            digest_kind="oci_manifest",
            reference="mutable:latest",
            digest="sha256:" + "a" * 64,
            platform="linux/arm64",
        )


def test_tty_uses_fixed_right_status_area(tmp_path, monkeypatch):
    import io
    import os

    root = tmp_path / "runs/original"
    root.mkdir(parents=True)
    (root / "events.jsonl").write_text(
        "".join(
            json.dumps({"event": "eval.trial.event", "trial_id": str(i)}) + "\n" for i in range(14)
        )
    )

    class Terminal(io.StringIO):
        def isatty(self):
            return True

    terminal = Terminal()
    monkeypatch.setattr(healthbench.sys, "stdout", terminal)
    monkeypatch.setattr(
        healthbench.shutil, "get_terminal_size", lambda: os.terminal_size((100, 24))
    )
    healthbench.watch(SimpleNamespace(home=tmp_path, run_id="original", once=True))
    output = terminal.getvalue()
    assert len(output.splitlines()) == 10
    assert "Hidden: 4" in output and "Status" in output


def test_active_run_reconnect_observes_instead_of_dispatch(tmp_path, monkeypatch):
    import fcntl

    called = []
    monkeypatch.setattr(healthbench, "_run", lambda args: pytest.fail("duplicate compute"))
    monkeypatch.setattr(healthbench, "watch", lambda args: called.append(args.run_id) or 0)
    with (tmp_path / "healthbench-controller.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert healthbench.run(SimpleNamespace(home=tmp_path, run_id="original")) == 0
    assert called == ["original"]


def test_invalid_trial_is_not_successful_run(tmp_path):
    root = tmp_path / "runs/original"
    root.mkdir(parents=True)
    (root / "result_manifest.json").write_text(json.dumps({"trials": [{"valid": False}]}))
    assert healthbench.result_exit_code(SimpleNamespace(home=tmp_path, run_id="original")) == 1


def test_charge_reconciliation_keeps_completion_authority_when_metadata_missing(
    tmp_path, monkeypatch
):
    import httpx

    output = tmp_path / "runs/original/trials/one/output"
    output.mkdir(parents=True)
    records = [
        {"state": "reserved"},
        {
            "state": "completed",
            "call_id": "call-0",
            "role": "answer",
            "provider_receipt": {
                "provider": "openrouter",
                "id": "gen-original",
                "usage": {"cost": 0.01},
            },
        },
    ]
    (output / "provider.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(404))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    receipt = tmp_path / "charges.json"
    assert (
        healthbench.reconcile(SimpleNamespace(home=tmp_path, run_id="original", output=receipt))
        == 0
    )
    value = json.loads(receipt.read_text())
    assert value["total_cost_usd"] == 0.01 and value["charge_final"] is True
    assert value["generations"][0]["charge_source"] == "completion_usage"
