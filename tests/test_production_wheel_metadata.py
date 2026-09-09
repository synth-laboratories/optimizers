import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

spec = importlib.util.spec_from_file_location(
    "production_wheel", Path(__file__).parents[1] / "scripts/check-production-wheel.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("tblite,native,valid", [(False, True, True), (True, True, False), (False, False, False)])
def test_production_wheel_contract(tmp_path, monkeypatch, tblite, native, valid):
    monkeypatch.chdir(Path(__file__).parents[1])
    wheel = tmp_path / "test.whl"
    with ZipFile(wheel, "w") as archive:
        metadata = "Name: synth-optimizers\nVersion: 0.2.22\nRequires-Dist: synth-containers==0.4.2\n"
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
