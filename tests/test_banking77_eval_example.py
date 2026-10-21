"""Freeze actual Banking77 source coordinates and reuse qualified Eval custody."""

import csv
import hashlib
import json
from types import SimpleNamespace

import pytest

from synth_optimizers.eval import banking77, healthbench


@pytest.fixture
def inputs(tmp_path):
    images = tmp_path / "evals/containers/images"
    task = images / "banking77/banking77_classify"
    task.mkdir(parents=True)
    (task / "world.py").write_text(
        'CLASSIFY_SYSTEM = "Classify"\nBANKING77_LABELS = ("card_arrival",)\n'
    )
    (task / "runtime.py").write_text("# frozen scoring fixture\n")
    (images / "banking77/eval_adapter.py").write_text("# trusted adapter fixture\n")
    common = images / "healthbench2"
    common.mkdir()
    for name in ("eval_native.py", "eval_adapter.py", "build_eval.py"):
        (common / name).write_text("# immutable shared fixture\n")
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / "synth_containers-0.0-py3-none-any.whl").write_bytes(b"descriptor fixture only")
    dataset = tmp_path / "source"
    dataset.mkdir()
    provenance = {"commit": "a" * 40, "files": {}}
    for split in ("train", "heldout"):
        path = dataset / f"banking77-{split}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("text", "category"))
            writer.writeheader()
            writer.writerows(
                {"text": f"{split}-{i}", "category": "card_arrival"} for i in range(81)
            )
        provenance["files"][split] = {
            "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        }
    (dataset / "source-provenance.json").write_text(json.dumps(provenance))
    prices = tmp_path / "prices.json"
    prices.write_text(
        json.dumps(
            {
                "data": [
                    {
                        "id": "openai/gpt-4.1-mini",
                        "pricing": {"prompt": "0.0000004", "completion": "0.0000016"},
                    }
                ]
            }
        )
    )
    return SimpleNamespace(
        evals_root=tmp_path / "evals",
        wheels=wheels,
        dataset=dataset,
        prices=prices,
        output=tmp_path / "release",
        base_image="python@sha256:" + "b" * 64,
        platform="linux/arm64",
    )


def test_portable_freeze_preserves_real_source_coordinates(inputs):
    from synth_containers.package_build import plan_container_package

    banking77.freeze(inputs)
    protocol = json.loads((inputs.output / "protocol.json").read_text())
    assert {k: v[0]["seed"] for k, v in protocol["panels"].items()} == {
        "baseline": 0,
        "selection": 40,
        "final": 80,
    }
    assert len({p[0]["row_digest"] for p in protocol["panels"].values()}) == 3
    assert protocol["judge"] == {"kind": "deterministic", "calls": 0, "cost_usd": 0}
    assert protocol["proposer"] == {"enabled": False, "calls": 0}
    assert (
        plan_container_package(inputs.output / "container-package.toml").manifest.package_id
        == banking77.PACKAGE
    )
    assert (inputs.output / "eval_budget.py").read_bytes() == (
        inputs.evals_root / "containers/images/healthbench2/eval_adapter.py"
    ).read_bytes()


def test_source_provenance_drift_refuses_before_compute(inputs):
    with (inputs.dataset / "banking77-heldout.csv").open("a") as handle:
        handle.write("drift,card_arrival\n")
    with pytest.raises(ValueError, match="source_provenance_digest_mismatch"):
        banking77.freeze(inputs)


def test_banking77_run_uses_common_controller_and_trusted_profile(monkeypatch):
    observed = []
    monkeypatch.setattr(healthbench, "run", lambda args, **kwargs: observed.append(kwargs) or 0)
    assert banking77.run(SimpleNamespace()) == 0
    assert observed == [{"profile": banking77.PROFILE}]


def test_banking77_reconnect_only_reads_terminal_original(tmp_path, monkeypatch):
    root = tmp_path / "runs/original"
    root.mkdir(parents=True)
    (root / "result_manifest.json").write_text(
        json.dumps({"run_id": "original", "artifacts": [], "trials": [{"valid": True}]})
    )
    monkeypatch.setattr(
        healthbench, "_run", lambda *a, **kw: pytest.fail("duplicate provider compute")
    )
    assert banking77.run(SimpleNamespace(home=tmp_path, run_id="original")) == 0
