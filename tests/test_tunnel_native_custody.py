"""Installed ABI accepts custody DTOs before failing an unpaid validation gate.

Exact queue serialization is covered by Rust platform and actual executor tests.
The supported Python native entry point is GepaRun, not workspace queue helpers.
"""
from uuid import uuid4

import pytest

from synth_optimizers import _synth_optimizers as native


def config_text(extra=""):
    return f'''[run]
run_id = "native_shared_job"
output_dir = "runs"
[container]
url = "http://127.0.0.1:1/fixture"
[container.auth_refresh]
provider = "synth_tunnel_v2"
lease_id = "{uuid4()}"
connector_id = "{uuid4()}"
offer_id = "{uuid4()}"
job_id = "native_shared_job"
refresh_interval_seconds = 30
{extra}
[taskset]
train_ids = []
heldout_ids = ["fixture/heldout"]
'''


def test_native_accepts_handoff_descriptor_before_execution(tmp_path):
    config = tmp_path / "input.toml"
    config.write_text(config_text())
    # Empty train IDs are a deliberate validation gate after DTO deserialization,
    # before output creation, origin traffic, credentials or provider work.
    with pytest.raises(native.ConfigError, match="taskset.train_ids must contain"):
        native.GepaRun.from_toml(str(config)).execute()
    assert not (tmp_path / "runs").exists()


def test_native_unknown_custody_field_refused_before_execution(tmp_path):
    config = tmp_path / "input.toml"
    config.write_text(config_text('invented_authority = "forbidden"'))
    with pytest.raises(native.OptimizerTomlDecodeError, match="unknown field.*invented_authority"):
        native.GepaRun.from_toml(str(config)).execute()
    assert not (tmp_path / "runs").exists()
