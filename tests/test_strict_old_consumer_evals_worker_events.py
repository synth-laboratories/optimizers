"""SYN-4648: the eval worker journal still matches what pinned Evals decodes.

Evals (``core/streaming/source_optimizer.py`` and ``export_trial`` in
``scripts/run_optimizer_eval.py``) tails ``events.jsonl`` from
``synth_optimizers.eval.runner.EventEmitter`` and reads ``eval.trial-result.v1``
records out of ``eval.trial.terminal``. The frozen bytes under
``tests/fixtures/evals_consumer/`` are byte-identical to the copy in the Evals
repository, where the consumer test decodes them.

This test drives the real producer (``EventEmitter`` + ``TrialRecord``) and
requires the pinned key set of every row. A new, renamed or removed field
fails here with the consumer pin named, before Evals sees it.

Runs without a maturin build, like ``test_event_vocabulary.py``: the package
``__init__`` imports the native extension, so modules load from ``src/``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src" / "synth_optimizers"
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "evals_consumer"
CONSUMER = "synth-laboratories/evals@90eba3b660bb1107481b70848ff1fa29843c07d8"


def _load(name: str):
    try:
        return importlib.import_module(f"synth_optimizers.{name}")
    except Exception:
        pass
    alias = "_synth_optimizers_src"
    if alias not in sys.modules:
        package = types.ModuleType(alias)
        package.__path__ = [str(SRC)]
        package.__package__ = alias
        package.__version__ = "0.0.0-source-load"
        sys.modules[alias] = package
        sys.modules.setdefault(f"{alias}._synth_optimizers", types.ModuleType("native"))
    return importlib.import_module(f"{alias}.{name}")


def build_journal(directory: Path) -> Path:
    """One screened trial through the real emitter, as the runner emits it."""

    runner = _load("eval.runner")
    models = _load("eval.models")
    path = directory / "events.jsonl"
    events = runner.EventEmitter(path, run_id="20261028T000000Z-fixture")
    key = models.TrialKey(
        candidate_id="cand_baseline",
        seed=101,
        scenario="craftax-survival-default-48x48",
        stage="screen",
    )
    events.emit(
        "eval.run.planned",
        recipe_id="eval.craftax.code-policy.smoke.v1",
        candidate_set_id="cset_fixture",
        candidates=[{"id": "cand_baseline", "label": "idle", "is_baseline": True}],
        correlation={"evals_run_dir": "/evals/data/results/craftax-optimizer/fixture"},
        manifest_digest="sha256:" + "0" * 64,
        parallelism=2,
        global_capacity=4,
        reclaimed_leases=0,
        planned_trials=1,
    )
    events.emit("eval.trial.queued", trial_id=key.trial_id, **key.to_json())
    events.emit("eval.trial.started", trial_id=key.trial_id, lease_id="lease_fixture", **key.to_json())
    record = models.TrialRecord(
        key=key,
        trial_id=key.trial_id,
        status="evaluated",
        benchmark_status="scored",
        metrics={"reward": 3.0},
        gates={"policy_loaded": True},
        missing_gates=(),
        missing_artifacts=(),
        usage={"llm_calls": 0},
        artifacts=({"kind": "trace", "path": "output/trace.jsonl"},),
        started_at="2026-10-28T00:00:01+00:00",
        finished_at="2026-10-28T00:00:09+00:00",
        exit_code=0,
        error=None,
        evidence_dir="/optimizer/runs/fixture/trials/" + key.trial_id,
    )
    events.emit("eval.trial.terminal", trial_id=key.trial_id, resumed=False, trial=record.to_json())
    events.emit(
        "eval.run.terminal",
        status="completed",
        selection_status="report_only",
        winner_id=None,
        evidence_dir="/optimizer/runs/fixture",
    )
    return path


def _shape(value: Any) -> Any:
    """Recursive key set; volatile values (timestamps) are not compared."""

    if isinstance(value, dict):
        return {key: _shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    return type(value).__name__


def test_frozen_fixture_bytes_match_manifest() -> None:
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    assert manifest["producer"].startswith("synth-laboratories/optimizers@")
    for name, digest in manifest["fixtures"].items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == digest, name


def test_real_producer_reproduces_pinned_worker_event_keys(tmp_path: Path) -> None:
    produced = [json.loads(line) for line in build_journal(tmp_path).read_text().splitlines()]
    frozen = [
        json.loads(line)
        for line in (FIXTURES / "eval_worker_events.v1.jsonl").read_text().splitlines()
    ]
    assert [row["event"] for row in produced] == [row["event"] for row in frozen], CONSUMER
    for current, pinned in zip(produced, frozen, strict=True):
        assert _shape(current) == _shape(pinned), (
            f"eval.worker-event.v1 {current['event']} drifted from the shape decoded by "
            f"{CONSUMER} (core/streaming/source_optimizer.py, scripts/run_optimizer_eval.py "
            "export_trial); roll the consumer first"
        )
