from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class JsonStore:
    """Process-restart durable state for in-flight rollouts and pins."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "rollouts").mkdir(exist_ok=True)
        (self.root / "checkpoints").mkdir(exist_ok=True)
        self._lock = threading.Lock()

    def _rollout_path(self, rollout_id: str) -> Path:
        return self.root / "rollouts" / f"{rollout_id}.json"

    def _checkpoint_path(self, checkpoint_id: str) -> Path:
        return self.root / "checkpoints" / f"{checkpoint_id}.json"

    def put_rollout(self, record: dict[str, Any]) -> None:
        path = self._rollout_path(str(record["rollout_id"]))
        payload = json.dumps(record, indent=2, sort_keys=True)
        with self._lock:
            path.write_text(payload)

    def get_rollout(self, rollout_id: str) -> dict[str, Any] | None:
        path = self._rollout_path(rollout_id)
        with self._lock:
            if not path.exists():
                return None
            return json.loads(path.read_text())

    def list_rollouts(self) -> list[dict[str, Any]]:
        with self._lock:
            paths = sorted((self.root / "rollouts").glob("*.json"))
            records: list[dict[str, Any]] = []
            for path in paths:
                try:
                    records.append(json.loads(path.read_text()))
                except json.JSONDecodeError:
                    continue
            return records

    def put_checkpoint(self, record: dict[str, Any]) -> None:
        path = self._checkpoint_path(str(record["checkpoint_id"]))
        payload = json.dumps(record, indent=2, sort_keys=True)
        with self._lock:
            path.write_text(payload)

    def get_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        path = self._checkpoint_path(checkpoint_id)
        with self._lock:
            if not path.exists():
                return None
            return json.loads(path.read_text())
