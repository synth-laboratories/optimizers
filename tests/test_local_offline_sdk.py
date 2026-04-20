from __future__ import annotations

import json
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prompt_opt.sdk.optimization.policy.v1 import PolicyOptimizationOfflineJob


class _RolloutHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []
    lock = threading.Lock()
    inflight = 0
    max_inflight = 0
    response_delay_seconds = 0.0

    def log_message(self, format: str, *args: Any) -> None:  # pragma: no cover
        del format, args

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/rollout":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        with self.__class__.lock:
            self.__class__.requests.append(payload)
            self.__class__.inflight += 1
            self.__class__.max_inflight = max(self.__class__.max_inflight, self.__class__.inflight)

        candidate = payload.get("policy", {}).get("config", {}).get("candidate", {})
        candidate_content = str(candidate.get("candidate_content", ""))
        if self.__class__.response_delay_seconds > 0:
            time.sleep(self.__class__.response_delay_seconds)
        if "Return exactly one of" in candidate_content or "output schema exactly" in candidate_content:
            reward = 1.0
        elif "deterministic" in candidate_content:
            reward = 0.8
        else:
            reward = 0.1

        response = {
            "metrics": {"outcome_reward": reward},
            "candidate_id": payload.get("policy", {}).get("config", {}).get("candidate_id"),
        }
        encoded = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)
        with self.__class__.lock:
            self.__class__.inflight = max(0, self.__class__.inflight - 1)


class LocalOfflineSdkContainerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _RolloutHandler.requests = []
        cls._server = ThreadingHTTPServer(("127.0.0.1", 0), _RolloutHandler)
        cls._thread = threading.Thread(target=cls._server.serve_forever, daemon=True)
        cls._thread.start()
        cls._base_url = f"http://127.0.0.1:{cls._server.server_address[1]}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls._server.shutdown()
        cls._server.server_close()
        cls._thread.join(timeout=5.0)

    def test_mipro_container_job_uses_candidate_id_retrieval_contract_and_improves(self) -> None:
        job = PolicyOptimizationOfflineJob.create(
            kind="mipro_offline",
            system_name="mipro-contract",
            config={
                "prompt_learning": {
                    "algorithm": "mipro",
                    "execution_mode": "retrieved",
                    "container_url": self._base_url,
                    "task_data": {
                        "validation_examples": [
                            {"seed": 0, "input": "A", "answer": "x"},
                            {"seed": 1, "input": "B", "answer": "y"},
                        ],
                        "train_examples": [
                            {"seed": 0, "input": "A", "answer": "x"},
                            {"seed": 1, "input": "B", "answer": "y"},
                        ],
                    },
                    "mipro": {
                        "initial_candidate": {
                            "stages": [
                                {
                                    "id": "main",
                                    "name": "Main",
                                    "messages": [{"role": "system", "pattern": "Classify the query.", "order": 0}],
                                }
                            ]
                        },
                        "num_candidates": 4,
                        "max_iterations": 2,
                    },
                }
            },
            backend_url="local://prompt-opt",
            api_key="local",
        )
        result = job.stream_until_complete(timeout=30.0, interval=0.05)
        self.assertEqual(result["status"], "succeeded")
        self.assertTrue(_RolloutHandler.requests)
        observed = _RolloutHandler.requests[-1]
        config = observed["policy"]["config"]
        self.assertIn("candidate_id", config)
        self.assertEqual(config["candidate"]["candidate_id"], config["candidate_id"])
        state = job.get_state_envelope()["state"]
        self.assertEqual(state["best_candidate_id"], result["best_candidate_id"])
        baseline_score = float(state["candidates"]["baseline"].get("avg_reward") or 0.0)
        best_score = float(result["best_reward"] or 0.0)
        self.assertGreater(best_score, baseline_score)

    def test_gepa_container_job_populates_candidate_and_event_payloads_and_improves(self) -> None:
        _RolloutHandler.requests = []
        job = PolicyOptimizationOfflineJob.create(
            kind="gepa_offline",
            system_name="gepa-contract",
            config={
                "prompt_learning": {
                    "algorithm": "gepa",
                    "execution_mode": "retrieved",
                    "container_url": self._base_url,
                    "task_data": {
                        "validation_examples": [{"seed": 0}, {"seed": 1}],
                        "train_examples": [{"seed": 0}, {"seed": 1}],
                    },
                    "gepa": {
                        "initial_candidate": {
                            "stages": [
                                {
                                    "id": "main",
                                    "name": "Main",
                                    "messages": [{"role": "system", "pattern": "Classify the query.", "order": 0}],
                                }
                            ]
                        },
                        "population": {"initial_size": 1, "num_generations": 1, "children_per_generation": 3},
                    },
                }
            },
            backend_url="local://prompt-opt",
            api_key="local",
        )
        result = job.stream_until_complete(timeout=30.0, interval=0.05)
        self.assertEqual(result["status"], "succeeded")
        events = job.events()["items"]
        event_types = {event["event_type"] for event in events}
        self.assertIn("prompt_learning.generation.started", event_types)
        self.assertIn("prompt_learning.generation.completed", event_types)
        state = job.get_state_envelope()["state"]
        candidates = state["candidates"]
        self.assertGreaterEqual(len(candidates), 2)
        self.assertTrue(any(str(item.get("avg_reward", 0.0)) != "None" for item in candidates.values()))
        baseline_score = float(candidates["baseline"].get("avg_reward") or 0.0)
        best_score = float(result["best_reward"] or 0.0)
        self.assertGreater(best_score, baseline_score)

    def test_gepa_does_not_repeat_same_stage_transform_in_best_candidate(self) -> None:
        _RolloutHandler.requests = []
        job = PolicyOptimizationOfflineJob.create(
            kind="gepa_offline",
            system_name="gepa-no-duplicate-transforms",
            config={
                "prompt_learning": {
                    "algorithm": "gepa",
                    "execution_mode": "retrieved",
                    "container_url": self._base_url,
                    "task_data": {
                        "validation_examples": [{"seed": 0}, {"seed": 1}],
                        "train_examples": [{"seed": 0}, {"seed": 1}],
                    },
                    "gepa": {
                        "initial_candidate": {
                            "stages": [
                                {
                                    "id": "main",
                                    "name": "Main",
                                    "messages": [{"role": "system", "pattern": "Classify the query.", "order": 0}],
                                }
                            ]
                        },
                        "population": {"initial_size": 1, "num_generations": 1, "children_per_generation": 8},
                    },
                }
            },
            backend_url="local://prompt-opt",
            api_key="local",
        )
        result = job.stream_until_complete(timeout=30.0, interval=0.05)
        self.assertEqual(result["status"], "succeeded")
        best_candidate = job.get_state_envelope()["state"]["candidates"][result["best_candidate_id"]]
        content = str(best_candidate.get("candidate_content", ""))
        self.assertLessEqual(content.count("Be concise and deterministic."), 1)
        self.assertLessEqual(content.count("Follow the requested output schema exactly."), 1)

    def test_mipro_container_rollouts_run_in_parallel_when_enabled(self) -> None:
        _RolloutHandler.requests = []
        _RolloutHandler.inflight = 0
        _RolloutHandler.max_inflight = 0
        _RolloutHandler.response_delay_seconds = 0.05
        try:
            job = PolicyOptimizationOfflineJob.create(
                kind="mipro_offline",
                system_name="mipro-parallel-rollouts",
                config={
                    "prompt_learning": {
                        "algorithm": "mipro",
                        "execution_mode": "retrieved",
                        "container_url": self._base_url,
                        "task_data": {
                            "validation_examples": [{"seed": idx, "input": f"Q{idx}", "answer": "x"} for idx in range(8)],
                            "train_examples": [{"seed": idx, "input": f"Q{idx}", "answer": "x"} for idx in range(8)],
                        },
                        "mipro": {
                            "initial_candidate": {
                                "stages": [
                                    {
                                        "id": "main",
                                        "name": "Main",
                                        "messages": [{"role": "system", "pattern": "Classify the query.", "order": 0}],
                                    }
                                ]
                            },
                            "num_candidates": 1,
                            "max_iterations": 1,
                            "parallel_batches": True,
                            "parallel_batch_size": 8,
                        },
                    }
                },
                backend_url="local://prompt-opt",
                api_key="local",
            )
            result = job.stream_until_complete(timeout=30.0, interval=0.05)
            self.assertEqual(result["status"], "succeeded")
            self.assertGreaterEqual(_RolloutHandler.max_inflight, 2)
        finally:
            _RolloutHandler.response_delay_seconds = 0.0


if __name__ == "__main__":
    unittest.main()
