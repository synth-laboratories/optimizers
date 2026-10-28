"""SYN-4648: authored TOML -> exact native request bytes, per maintained config family.

Each family under ``tests/fixtures/syn4648_cli_toml/<family>/`` checks in the
authored TOML and the frozen native request bytes the REAL parser/resolver must
reproduce. ``MANIFEST.json`` pins the sha256 of both files so a fixture cannot be
silently rewritten alongside a producer change.

Native request per family (the object handed to the next owner):

* ``gepa``     -- ``GepaConfig.from_toml(path).to_toml_dict()``; written as the
  ``*.sdk.toml`` the Rust ``GepaRun`` consumes.
* ``gelo``     -- ``GeloMaterializer.from_paths(path).materialize()``; the go-ex
  config JSON submitted to the native/hosted executor.
* ``rl_cispo`` -- ``rl.config.load(path).redacted_payload()``; the resolved
  ``cispo.container.v1`` effective configuration retained in run receipts.
* ``sft``      -- ``sft._executor_config(SftConfig.from_toml(text))``; the
  executor request.

Absolute paths derived from the declaring document are written as
``${CONFIG_DIR}`` so the bytes are checkout-independent.

Regenerate deliberately (never in a test run):
    PYTHONPATH=src python tests/test_syn4648_cli_toml_native_fixtures.py --regenerate
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from synth_optimizers.gelo import GeloMaterializeError, GeloMaterializer
from synth_optimizers.gepa import GepaConfig
from synth_optimizers.rl import config as rl_config
from synth_optimizers.sft import SftConfig, SftServiceError, _executor_config

FIXTURES = Path(__file__).parent / "fixtures" / "syn4648_cli_toml"
MANIFEST = FIXTURES / "MANIFEST.json"
CONFIG_DIR_TOKEN = "${CONFIG_DIR}"


def _gepa(path: Path) -> dict[str, Any]:
    return GepaConfig.from_toml(path).to_toml_dict()


def _gelo(path: Path) -> dict[str, Any]:
    return GeloMaterializer.from_paths(path).materialize()


def _rl_cispo(path: Path) -> dict[str, Any]:
    return rl_config.load(path).redacted_payload()


def _sft(path: Path) -> dict[str, Any]:
    return _executor_config(SftConfig.from_toml(path.read_text(encoding="utf-8")))


PRODUCERS: dict[str, tuple[str, Callable[[Path], dict[str, Any]]]] = {
    "gepa": ("synth_optimizers.gepa.GepaConfig.from_toml().to_toml_dict()", _gepa),
    "gelo": ("synth_optimizers.gelo.GeloMaterializer.from_paths().materialize()", _gelo),
    "rl_cispo": ("synth_optimizers.rl.config.load().redacted_payload()", _rl_cispo),
    "sft": ("synth_optimizers.sft._executor_config(SftConfig.from_toml())", _sft),
}


def native_bytes(family: str, authored: Path) -> bytes:
    payload = PRODUCERS[family][1](authored)
    text = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
    text = text.replace(str(authored.parent.resolve()), CONFIG_DIR_TOKEN)
    text = text.replace(str(authored.parent), CONFIG_DIR_TOKEN)
    return (text + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("family", sorted(PRODUCERS))
def test_manifest_pins_checked_in_fixture_bytes(family: str) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = manifest["families"][family]
    assert entry["producer"] == PRODUCERS[family][0]
    assert _sha256((FIXTURES / family / "authored.toml").read_bytes()) == entry["authored_sha256"]
    assert _sha256((FIXTURES / family / "native.json").read_bytes()) == entry["native_sha256"]


@pytest.mark.parametrize("family", sorted(PRODUCERS))
def test_real_resolver_reproduces_frozen_native_bytes(family: str) -> None:
    expected = (FIXTURES / family / "native.json").read_bytes()
    actual = native_bytes(family, FIXTURES / family / "authored.toml")
    assert actual == expected, (
        f"{family}: authored TOML no longer resolves to the frozen native request; "
        f"producer={PRODUCERS[family][0]}"
    )


def test_gepa_paths_resolve_relative_to_declaring_document_not_cwd(tmp_path: Path) -> None:
    relocated = tmp_path / "relocated"
    shutil.copytree(FIXTURES / "gepa", relocated)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    probe = (
        "import json,sys; from pathlib import Path; "
        "from synth_optimizers.gepa import GepaConfig; "
        "print(json.dumps(GepaConfig.from_toml(Path(sys.argv[1])).to_toml_dict()["
        "'proposer'], sort_keys=True))"
    )
    src = Path(__file__).resolve().parents[1] / "src"
    completed = subprocess.run(
        [sys.executable, "-c", probe, str(relocated / "authored.toml")],
        cwd=elsewhere,
        env={"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    proposer = json.loads(completed.stdout)
    assert proposer["prompt"]["best_practices_path"] == str(
        relocated.resolve() / "prompts/best_practices.md"
    )
    assert proposer["codex_home"] == str(relocated.resolve() / "codex-home")
    # Relocation changes nothing but the config directory itself.
    assert native_bytes("gepa", relocated / "authored.toml") == (
        FIXTURES / "gepa" / "native.json"
    ).read_bytes()


def test_frozen_gaps_in_path_resolution_are_explicit() -> None:
    """Paths these producers pass through verbatim (cwd-relative at execution).

    Recorded as frozen behavior, not endorsed: GEPA ``container.cwd`` and RL
    ``[artifacts]`` paths are not resolved against the declaring document.
    """
    gepa = json.loads((FIXTURES / "gepa" / "native.json").read_text())
    assert gepa["container"]["cwd"] == "container"
    rl = json.loads((FIXTURES / "rl_cispo" / "native.json").read_text())
    assert rl["artifacts"]["catalog"] == "artifacts/checkpoints.sqlite3"


# --------------------------------------------------------------------------- #
# Refusals. The Python parsers expose typed refusals only as exception classes
# (and pydantic error ``type`` codes); none carries a stable string code.
# --------------------------------------------------------------------------- #


def _write(tmp_path: Path, family: str, text: str) -> Path:
    target = tmp_path / family / "authored.toml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def _authored(family: str) -> str:
    return (FIXTURES / family / "authored.toml").read_text(encoding="utf-8")


def test_rl_unknown_top_level_section_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "rl", _authored("rl_cispo") + "\n[surprise]\nvalue = 1\n")
    with pytest.raises(rl_config.ConfigError):
        rl_config.load(path)


def test_rl_unknown_field_inside_section_is_refused(tmp_path: Path) -> None:
    text = _authored("rl_cispo").replace("[reward]\n", "[reward]\nunknown_knob = 1\n")
    with pytest.raises(rl_config.ConfigError):
        rl_config.load(_write(tmp_path, "rl", text))


def test_rl_unsupported_schema_version_is_refused(tmp_path: Path) -> None:
    text = _authored("rl_cispo").replace('"cispo.container.v1"', '"cispo.container.v2"')
    with pytest.raises(rl_config.ConfigError):
        rl_config.load(_write(tmp_path, "rl", text))


def test_rl_unsupported_plan_preset_is_refused(tmp_path: Path) -> None:
    text = _authored("rl_cispo").replace('preset = "cispo"', 'preset = "ppo"')
    with pytest.raises(rl_config.ConfigError):
        rl_config.load(_write(tmp_path, "rl", text))


def test_gepa_unknown_field_in_strict_section_is_refused_with_typed_code(tmp_path: Path) -> None:
    text = _authored("gepa").replace("[gepa.task_pools]\n", "[gepa.task_pools]\nsurprise = []\n")
    with pytest.raises(ValidationError) as caught:
        GepaConfig.from_toml(_write(tmp_path, "gepa", text))
    assert [error["type"] for error in caught.value.errors()] == ["extra_forbidden"]


def test_gepa_unknown_top_level_field_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "gepa", _authored("gepa") + "\n[surprise]\nvalue = 1\n")
    with pytest.raises((ValidationError, ValueError)):
        GepaConfig.from_toml(path)


def test_gelo_unsupported_auth_refresh_provider_is_refused(tmp_path: Path) -> None:
    text = _authored("gelo").replace('"synth_tunnel_v2"', '"ngrok"')
    with pytest.raises(GeloMaterializeError):
        GeloMaterializer.from_paths(_write(tmp_path, "gelo", text)).materialize()


def test_gelo_v2_auth_refresh_without_connector_is_refused(tmp_path: Path) -> None:
    text = _authored("gelo").replace('connector_id = "connector-fixture-0001"\n', "")
    with pytest.raises(GeloMaterializeError):
        GeloMaterializer.from_paths(_write(tmp_path, "gelo", text)).materialize()


def test_sft_unsupported_backend_is_refused() -> None:
    text = _authored("sft").replace('backend = "fixture"', 'backend = "modal"')
    with pytest.raises(SftServiceError):
        SftConfig.from_toml(text)


# SYN-5207: ``sft.config.v2`` is the closed authored SFT config. Keys are
# prepended so they sit at the TOML top level (the fixture ends in an
# ``[[examples]]`` table, which would otherwise own them).
SFT_V2 = 'schema_version = "sft.config.v2"\n'


def test_sft_v2_config_is_accepted_and_resolves_like_legacy() -> None:
    legacy = SftConfig.from_toml(_authored("sft")).config_json
    v2 = dict(SftConfig.from_toml(SFT_V2 + _authored("sft")).config_json)
    assert v2.pop("schema_version") == "sft.config.v2"
    assert v2 == legacy


def test_sft_v2_unknown_top_level_field_is_refused() -> None:
    with pytest.raises(SftServiceError, match="unknown top-level fields: surprise_knob"):
        SftConfig.from_toml(SFT_V2 + 'surprise_knob = "x"\n' + _authored("sft"))


def test_sft_v2_refuses_executor_owned_fields() -> None:
    with pytest.raises(SftServiceError, match="resolved_checkpoint_plan"):
        SftConfig.from_toml(SFT_V2 + 'resolved_checkpoint_plan = "x"\n' + _authored("sft"))


def test_sft_unsupported_schema_version_is_refused() -> None:
    with pytest.raises(SftServiceError, match="unsupported SFT config schema_version"):
        SftConfig.from_toml('schema_version = "sft.config.v3"\n' + _authored("sft"))


def test_sft_legacy_config_without_schema_version_stays_open() -> None:
    """Existing hosted submissions keep working: the unversioned form forwards
    unknown top-level keys unchanged (SYN-5207 option (a))."""
    config = SftConfig.from_toml('surprise_knob = "x"\n' + _authored("sft"))
    assert config.config_json["surprise_knob"] == "x"


# --------------------------------------------------------------------------- #
# Python/Rust auth-refresh parity. Rust ``ContainerConfig`` (shared platform
# config.rs) declares ``auth_refresh: Option<ContainerAuthRefreshConfig>``.
# --------------------------------------------------------------------------- #

AUTH_REFRESH_TOML = """
[container.auth_refresh]
provider = "synth_tunnel_v2"
lease_id = "lease-fixture-0001"
connector_id = "connector-fixture-0001"
refresh_interval_seconds = 240
"""


def test_gelo_python_preserves_rust_auth_refresh_fields() -> None:
    native = json.loads((FIXTURES / "gelo" / "native.json").read_text())
    # Field set of Rust ContainerAuthRefreshConfig (provider, lease_id, connector_id,
    # offer_id, job_id, refresh_interval_seconds); unset optionals are omitted.
    assert native["container"]["auth_refresh"] == {
        "provider": "synth_tunnel_v2",
        "lease_id": "lease-fixture-0001",
        "connector_id": "connector-fixture-0001",
        "refresh_interval_seconds": 240,
    }


def test_gepa_python_preserves_rust_auth_refresh_fields(tmp_path: Path) -> None:
    text = _authored("gepa").replace(
        "startup_timeout_seconds = 45\n", "startup_timeout_seconds = 45\n" + AUTH_REFRESH_TOML
    )
    native = GepaConfig.from_toml(_write(tmp_path, "gepa", text)).to_toml_dict()
    assert native["container"].get("auth_refresh") == {
        "provider": "synth_tunnel_v2",
        "lease_id": "lease-fixture-0001",
        "connector_id": "connector-fixture-0001",
        "refresh_interval_seconds": 240,
    }


def _regenerate() -> None:
    families: dict[str, Any] = {}
    for family in sorted(PRODUCERS):
        authored = FIXTURES / family / "authored.toml"
        data = native_bytes(family, authored)
        (FIXTURES / family / "native.json").write_bytes(data)
        families[family] = {
            "producer": PRODUCERS[family][0],
            "authored_sha256": _sha256(authored.read_bytes()),
            "native_sha256": _sha256(data),
        }
    MANIFEST.write_text(
        json.dumps({"schema_version": "syn4648.cli-toml-fixtures.v1", "families": families},
                   sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    if sys.argv[1:] != ["--regenerate"]:
        raise SystemExit("usage: test_syn4648_cli_toml_native_fixtures.py --regenerate")
    _regenerate()


def test_gepa_auth_refresh_refuses_unknown_field(tmp_path: Path) -> None:
    """Rust ``ContainerAuthRefreshConfig`` is deny_unknown_fields; so is Python's."""
    text = _authored("gepa").replace(
        "startup_timeout_seconds = 45\n",
        "startup_timeout_seconds = 45\n" + AUTH_REFRESH_TOML + 'surprise = "x"\n',
    )
    with pytest.raises(ValidationError):
        GepaConfig.from_toml(_write(tmp_path, "gepa", text))


@pytest.mark.parametrize(
    ("needle", "replacement"),
    [
        ('provider = "synth_tunnel_v2"', 'provider = "ngrok"'),
        ('connector_id = "connector-fixture-0001"\n', ""),
    ],
    ids=["unknown-provider", "v2-without-connector"],
)
def test_gepa_auth_refresh_follows_gelo_provider_rule(
    tmp_path: Path, needle: str, replacement: str
) -> None:
    """Same rule as GELO and Rust ContainerConfig::validate_auth_refresh."""
    table = AUTH_REFRESH_TOML.replace(needle, replacement)
    assert table != AUTH_REFRESH_TOML
    text = _authored("gepa").replace(
        "startup_timeout_seconds = 45\n", "startup_timeout_seconds = 45\n" + table
    )
    with pytest.raises(ValidationError):
        GepaConfig.from_toml(_write(tmp_path, "gepa", text))
