import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

spec = importlib.util.spec_from_file_location(
    "production_wheel", Path(__file__).parents[1] / "scripts/check-production-wheel.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("tblite,native,containers,valid", [
    (False, True, "0.4.3", True),
    (True, True, "0.4.3", False),
    (False, False, "0.4.3", False),
    (False, True, "0.4.2", False),
])
def test_production_wheel_contract(tmp_path, monkeypatch, tblite, native, containers, valid):
    monkeypatch.chdir(Path(__file__).parents[1])
    wheel = tmp_path / "test.whl"
    with ZipFile(wheel, "w") as archive:
        metadata = f"Name: synth-optimizers\nVersion: 0.2.22\nRequires-Dist: synth-containers=={containers}\n"
        if tblite:
            metadata += "Requires-Dist: synth-harbor-tblite==0.1.1\n"
        archive.writestr("synth_optimizers-0.2.22.dist-info/METADATA", metadata)
        if native:
            archive.writestr("synth_optimizers/_synth_optimizers.abi3.so", b"fixture")
    if valid:
        module.check(wheel)
    else:
        with pytest.raises(ValueError):
            module.check(wheel)
