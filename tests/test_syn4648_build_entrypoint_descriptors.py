"""SYN-4648: build-entrypoint declarations and artifact descriptors for advertised packages.

The maintained ``synth.container-package.v1`` declarations are produced by the
installed Eval release freezers (``synth_optimizers.eval.healthbench.freeze`` and
``synth_optimizers.eval.banking77.freeze``) and validated by Containers'
``plan_container_package`` / ``validate_container_build_output``. This test runs
both REAL producers against small, checked-in stand-in sources (no evals checkout,
no provider, no image build) and requires:

* the exact ``container-package.toml`` bytes and frozen build-plan JSON, pinned by
  sha256 in ``MANIFEST.json``;
* a complete-input cache identity (manifest, every file digest/mode, directories,
  platform, dependency digests) bound into ``input_digest``;
* a versioned ``synth.container-artifact.v1`` descriptor for a frozen ``resolve``
  output, which records ``artifact_verification=not_checked`` and
  ``runtime_validation=not_run`` -- build success is not runtime compatibility;
* typed refusal codes for missing/invalid entrypoints and unsupported formats/profiles.

Regenerate deliberately:
    PYTHONPATH=src python tests/test_syn4648_build_entrypoint_descriptors.py --regenerate
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from synth_containers.package_build import (
    plan_container_package,
    validate_container_build_output,
)
from synth_containers.package_compatibility import (
    PackageRuntimeSupport,
    check_package_compatibility,
)
from synth_containers.package_files import ContainerPackageError

from synth_optimizers.eval import banking77, healthbench

FIXTURES = Path(__file__).parent / "fixtures" / "syn4648_build_descriptors"
MANIFEST = FIXTURES / "MANIFEST.json"
BASE_IMAGE = "python:3.12-slim@sha256:" + "1" * 64
PLATFORM = "linux/arm64"
WHEEL_NAME = "synth_containers-0.4.6-py3-none-any.whl"
WHEEL_BYTES = b"syn4648 stand-in wheel bytes\n"
PRICES = {
    "data": [
        {"id": model, "pricing": {"prompt": "0.0000004", "completion": "0.0000016"}}
        for model in ("openai/gpt-4.1-mini", "openai/gpt-4.1")
    ]
}
ENTRYPOINT = "#!/usr/bin/env python3\n# stand-in for evals healthbench2/build_eval.py\n"


def _common_sources(root: Path) -> SimpleNamespace:
    images = root / "evals/containers/images"
    hb = images / "healthbench2"
    (hb / "healthbench_chat").mkdir(parents=True)
    (hb / "healthbench_chat/runtime.py").write_text("# stand-in grader runtime\n")
    (hb / "eval_adapter.py").write_text("# stand-in adapter\n")
    (hb / "eval_native.py").write_text("# stand-in native consumer\n")
    (hb / "build_eval.py").write_text(ENTRYPOINT)
    wheels = root / "wheelhouse"
    wheels.mkdir()
    (wheels / WHEEL_NAME).write_bytes(WHEEL_BYTES)
    prices = root / "provider-models.json"
    prices.write_text(json.dumps(PRICES, sort_keys=True) + "\n")
    return SimpleNamespace(
        evals_root=root / "evals", wheels=wheels, prices=prices,
        base_image=BASE_IMAGE, platform=PLATFORM,
    )


def _stage_healthbench(root: Path) -> Path:
    args = _common_sources(root)
    dataset = root / "healthbench.jsonl"
    dataset.write_text("".join(
        json.dumps({"prompt_id": f"p{i}", "prompt": [{"role": "user", "content": f"q{i}"}],
                    "rubrics": [{"criterion": f"c{i}", "points": 1}]}, sort_keys=True) + "\n"
        for i in range(3)
    ))
    args.dataset, args.output = dataset, root / "bundle"
    healthbench.freeze(args)
    return args.output / "container-package.toml"


def _stage_banking77(root: Path) -> Path:
    args = _common_sources(root)
    classify = args.evals_root / "containers/images/banking77/banking77_classify"
    classify.mkdir(parents=True)
    (classify / "runtime.py").write_text("# stand-in scorer runtime\n")
    (classify / "world.py").write_text(
        'CLASSIFY_SYSTEM = "Classify into one Banking77 label."\n'
        'BANKING77_LABELS = ["card_arrival", "lost_or_stolen_card"]\n'
    )
    (classify.parent / "eval_adapter.py").write_text("# stand-in banking77 adapter\n")
    dataset = root / "banking77-dataset"
    dataset.mkdir()
    files = {}
    for split, count in (("train", 2), ("heldout", 81)):
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["text", "category"])
        for i in range(count):
            writer.writerow([f"{split} query {i}", "card_arrival"])
        data = buffer.getvalue().encode()
        (dataset / f"banking77-{split}.csv").write_bytes(data)
        files[split] = {"digest": "sha256:" + hashlib.sha256(data).hexdigest()}
    (dataset / "source-provenance.json").write_text(
        json.dumps({"commit": "a" * 40, "files": files}, sort_keys=True) + "\n"
    )
    args.dataset, args.output = dataset, root / "bundle"
    banking77.freeze(args)
    return args.output / "container-package.toml"


STAGERS: dict[str, Callable[[Path], Path]] = {
    "healthbench2": _stage_healthbench,
    "banking77": _stage_banking77,
}


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _frozen_output(input_digest: str) -> dict[str, Any]:
    digest = "sha256:" + "2" * 64
    return {
        "schema_version": "synth.container-build-output.v1",
        "action": "resolve",
        "input_digest": input_digest,
        "artifact": {
            "kind": "oci_image", "digest_kind": "oci_manifest",
            "reference": f"ghcr.io/synth-laboratories/fixture@{digest}",
            "digest": digest, "platform": PLATFORM,
        },
    }


def produce(package: str, root: Path) -> dict[str, bytes]:
    manifest_path = STAGERS[package](root)
    plan = plan_container_package(manifest_path)
    descriptor = validate_container_build_output(plan, _frozen_output(plan.input_digest))
    return {
        "container-package.toml": manifest_path.read_bytes(),
        "build-plan.json": _canonical(plan.model_dump(mode="json")),
        "artifact-descriptor.json": _canonical(descriptor.model_dump(mode="json")),
    }


@pytest.mark.parametrize("package", sorted(STAGERS))
def test_real_freezer_and_planner_reproduce_frozen_descriptor_bytes(
    package: str, tmp_path: Path
) -> None:
    manifest = json.loads(MANIFEST.read_text())["packages"][package]
    for name, data in produce(package, tmp_path).items():
        frozen = (FIXTURES / package / name).read_bytes()
        assert _sha256(frozen) == manifest[name], f"{package}/{name}: manifest pin mismatch"
        assert data == frozen, f"{package}/{name}: producer drifted from frozen bytes"


@pytest.mark.parametrize("package", sorted(STAGERS))
def test_descriptor_carries_complete_input_cache_identity(package: str) -> None:
    descriptor = json.loads((FIXTURES / package / "artifact-descriptor.json").read_text())
    assert descriptor["schema_version"] == "synth.container-artifact.v1"
    assert descriptor["artifact_verification"] == "not_checked"
    assert descriptor["runtime_validation"] == "not_run"
    plan = descriptor["plan"]
    assert plan["schema_version"] == "synth.container-build-plan.v1"
    build = plan["manifest"]["build"]
    assert build["entrypoint"] == "build_eval.py" and build["platform"] == PLATFORM
    paths = {entry["path"]: entry for entry in plan["files"]}
    assert paths["build_eval.py"]["executable"] is True
    assert "container-package.toml" in paths
    for declared in build["inputs"]:
        assert declared in paths or declared in plan["directories"] or any(
            path.startswith(declared + "/") for path in paths
        ), f"{package}: declared input {declared} absent from cache identity"
    assert all(entry["digest"].startswith("sha256:") for entry in plan["files"])
    dependency = plan["manifest"]["dependencies"][0]
    assert dependency["digest"] == _sha256(WHEEL_BYTES)
    assert plan["manifest"]["runtime"]["readiness"]["kind"] == "adapter"
    assert {c["version"] for c in plan["manifest"]["contracts"]} >= {"eval.container-result.v1"}
    assert descriptor["output"]["input_digest"] == plan["input_digest"]


def _staged(tmp_path: Path) -> Path:
    return _stage_healthbench(tmp_path)


def _code(call: Callable[[], Any]) -> str:
    with pytest.raises(ContainerPackageError) as caught:
        call()
    return str(caught.value)


def test_missing_build_entrypoint_is_refused_with_typed_code(tmp_path: Path) -> None:
    manifest = _staged(tmp_path)
    (manifest.parent / "build_eval.py").unlink()
    assert _code(lambda: plan_container_package(manifest)) == FROZEN_CODES["missing_entrypoint"]


def test_non_executable_build_entrypoint_is_refused_with_typed_code(tmp_path: Path) -> None:
    manifest = _staged(tmp_path)
    (manifest.parent / "build_eval.py").chmod(0o644)
    assert _code(lambda: plan_container_package(manifest)) == FROZEN_CODES["invalid_entrypoint"]


def test_escaping_build_entrypoint_is_refused_with_typed_code(tmp_path: Path) -> None:
    manifest = _staged(tmp_path)
    manifest.write_text(manifest.read_text().replace(
        'entrypoint = "build_eval.py"', 'entrypoint = "../build_eval.py"'))
    assert _code(lambda: plan_container_package(manifest)) == FROZEN_CODES["escaping_entrypoint"]


def test_unknown_manifest_field_is_refused_with_typed_code(tmp_path: Path) -> None:
    manifest = _staged(tmp_path)
    manifest.write_text(manifest.read_text().replace(
        'package_id =', 'surprise = 1\npackage_id ='))
    assert _code(lambda: plan_container_package(manifest)) == FROZEN_CODES["unknown_field"]


def test_unsupported_manifest_version_is_refused_with_typed_code(tmp_path: Path) -> None:
    manifest = _staged(tmp_path)
    manifest.write_text(manifest.read_text().replace(
        "synth.container-package.v1", "synth.container-package.v2"))
    assert _code(lambda: plan_container_package(manifest)) == FROZEN_CODES["unsupported_version"]


def test_unsupported_manifest_format_is_refused_with_typed_code(tmp_path: Path) -> None:
    manifest = _staged(tmp_path)
    yaml = manifest.with_suffix(".yaml")
    yaml.write_bytes(manifest.read_bytes())
    assert _code(lambda: plan_container_package(yaml)) == FROZEN_CODES["unsupported_format"]


def test_build_output_for_other_inputs_is_refused_with_typed_code(tmp_path: Path) -> None:
    plan = plan_container_package(_staged(tmp_path))
    other = _frozen_output("sha256:" + "3" * 64)
    assert _code(lambda: validate_container_build_output(plan, other)) == FROZEN_CODES[
        "output_input_mismatch"
    ]


def test_unsupported_runtime_profile_is_reported_with_typed_code(tmp_path: Path) -> None:
    plan = plan_container_package(_staged(tmp_path))
    descriptor = validate_container_build_output(plan, _frozen_output(plan.input_digest))
    support = PackageRuntimeSupport.model_validate({
        "schema_version": "synth.package-runtime-support.v1",
        "runtime_id": "syn4648-http-only",
        "platforms": [PLATFORM],
        "runtime_profiles": ["http"],
        "readiness_kinds": ["adapter"],
        "readiness_adapters": ["healthbench-http-eval-v1"],
        "contracts": [
            {"name": "eval", "version": "eval.container-result.v1"},
            {"name": "gepa", "version": "synth_optimizers.gepa.v2"},
            {"name": "containers", "version": "container_contract.v1"},
        ],
    })
    report = check_package_compatibility(descriptor, support)
    assert not report.declaration_supported
    assert [issue.code for issue in report.issues] == ["runtime_profile_unsupported"]
    assert report.runtime_qualification == "not_run" and report.readiness == "not_observed"


FROZEN_CODES = {
    "missing_entrypoint": "container_package_input_unavailable",
    "invalid_entrypoint": "container_package_manifest_invalid",
    "escaping_entrypoint": "container_package_manifest_invalid",
    "unknown_field": "container_package_manifest_invalid",
    "unsupported_version": "container_package_manifest_invalid",
    "unsupported_format": "container_package_format_unsupported",
    "output_input_mismatch": "container_package_build_output_invalid",
}


def _regenerate() -> None:
    pins: dict[str, dict[str, str]] = {}
    for package in sorted(STAGERS):
        with tempfile.TemporaryDirectory() as scratch:
            outputs = produce(package, Path(scratch))
        (FIXTURES / package).mkdir(parents=True, exist_ok=True)
        pins[package] = {}
        for name, data in outputs.items():
            (FIXTURES / package / name).write_bytes(data)
            pins[package][name] = _sha256(data)
    MANIFEST.write_bytes(_canonical(
        {"schema_version": "syn4648.build-descriptor-fixtures.v1", "packages": pins}
    ))


if __name__ == "__main__":
    if sys.argv[1:] != ["--regenerate"]:
        raise SystemExit("usage: test_syn4648_build_entrypoint_descriptors.py --regenerate")
    _regenerate()
