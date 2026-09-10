"""Check release wheel metadata without importing or installing the package."""

from email.parser import BytesParser
from pathlib import Path
import sys
import tomllib
from zipfile import ZipFile


def check(wheel: Path) -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text())["project"]
    with ZipFile(wheel) as archive:
        metadata_names = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise ValueError("wheel must contain exactly one package metadata record")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        if metadata["Name"] != project["name"] or metadata["Version"] != project["version"]:
            raise ValueError("wheel identity differs from the release source")
        dependencies = metadata.get_all("Requires-Dist", [])
        if any("tblite" in dependency.lower() for dependency in dependencies):
            raise ValueError("TBLite must never be a production wheel dependency")
        if not any(dependency.replace(" ", "") == "synth-containers==0.4.3" for dependency in dependencies):
            raise ValueError("production wheel must pin stable Containers 0.4.3")
        if not any("_synth_optimizers" in name and name.endswith((".so", ".pyd")) for name in archive.namelist()):
            raise ValueError("wheel is missing its native optimizer extension")
    print(f"Verified production metadata and native extension: {wheel.name}")


if __name__ == "__main__":
    wheels = [Path(value) for value in sys.argv[1:]]
    if not wheels:
        raise SystemExit("usage: check-production-wheel.py WHEEL [WHEEL ...]")
    for wheel in wheels:
        check(wheel)
