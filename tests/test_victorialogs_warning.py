from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_victorialogs():
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "synth_optimizers"
        / "victorialogs.py"
    )
    spec = importlib.util.spec_from_file_location("victorialogs_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_vl_warning_is_telemetry_prefixed(capsys, monkeypatch):
    module = _load_victorialogs()
    monkeypatch.delenv("SYNTH_OPTIMIZERS_REQUIRE_VL", raising=False)
    module._warn_or_raise("VictoriaLogs write URL not configured for synth-optimizers")
    err = capsys.readouterr().err
    assert err.startswith("[telemetry-warning]")
    assert "VictoriaLogs write URL not configured" in err
