import importlib.util
from pathlib import Path
import tomllib
from zipfile import ZipFile

import pytest

spec = importlib.util.spec_from_file_location(
    "production_wheel", Path(__file__).parents[1] / "scripts/check-production-wheel.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("tblite,native,containers,valid", [
    (False, True, "0.4.5.dev20261009", True),
    (True, True, "0.4.5.dev20261009", False),
    (False, False, "0.4.5.dev20261009", False),
    (False, True, "0.4.3", False),
])
def test_production_wheel_contract(tmp_path, monkeypatch, tblite, native, containers, valid):
    monkeypatch.chdir(Path(__file__).parents[1])
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    wheel = tmp_path / "test.whl"
    with ZipFile(wheel, "w") as archive:
        metadata = f"Name: synth-optimizers\nVersion: {project['version']}\nRequires-Dist: synth-containers=={containers}\n"
        if tblite:
            metadata += "Requires-Dist: synth-harbor-tblite==0.1.1\n"
        archive.writestr(f"synth_optimizers-{project['version']}.dist-info/METADATA", metadata)
        if native:
            archive.writestr("synth_optimizers/_synth_optimizers.abi3.so", b"fixture")
    if valid:
        module.check(wheel)
    else:
        with pytest.raises(ValueError):
            module.check(wheel)


def test_stable_optimizer_refuses_development_containers(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "synth-optimizers"\nversion = "0.2.23"\n'
        'dependencies = ["synth-containers==0.4.5.dev20261009"]\n'
    )
    monkeypatch.chdir(tmp_path)
    wheel = tmp_path / "stable.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("synth_optimizers-0.2.23.dist-info/METADATA",
                         "Name: synth-optimizers\nVersion: 0.2.23\n"
                         "Requires-Dist: synth-containers==0.4.5.dev20261009\n")
        archive.writestr("synth_optimizers/_synth_optimizers.abi3.so", b"fixture")
    with pytest.raises(ValueError, match="stable optimizer wheel"):
        module.check(wheel)
