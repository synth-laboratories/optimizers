from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SYNTH_ROOT = ROOT.parent / "synth-ai"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if SYNTH_ROOT.exists() and str(SYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(SYNTH_ROOT))

from prompt_opt.mipro import proposer_backends, run_mipro
from prompt_opt.sdk.optimization.internal.configs.prompt_learning import PromptLearningConfig
from prompt_opt.sdk.optimization.internal.prompt_learning import PromptLearningJob
from prompt_opt.sdk.optimization.policy.v1 import PolicyOptimizationOfflineJob


class MiproTests(unittest.TestCase):
    def test_proposer_backends_are_available(self) -> None:
        self.assertEqual(proposer_backends(), ["single_prompt", "rlm"])

    def test_prompt_learning_config_normalizes_proxied_to_retrieved(self) -> None:
        parsed = PromptLearningConfig.from_mapping(
            {
                "prompt_learning": {
                    "algorithm": "mipro",
                    "execution_mode": "proxied",
                    "task_data": {"examples": [{"input": "Q", "expected": "A"}]},
                    "mipro": {"initial_candidate": {"stages": []}},
                }
            }
        )
        self.assertEqual(parsed.execution_mode, "retrieved")
        self.assertIsNotNone(parsed.mipro)
        self.assertEqual(parsed.mipro.execution_mode, "retrieved")

    def test_run_mipro_routes_through_local_job_runtime(self) -> None:
        seen_prompts: list[str] = []

        def task_llm(prompt: str) -> str:
            seen_prompts.append(prompt)
            if "Return exactly one of" in prompt or "output schema exactly" in prompt:
                return "paris"
            return "london"

        result = run_mipro(
            config={
                "num_candidates": 4,
                "max_iterations": 3,
                "early_stop_rounds": 2,
                "min_improvement": 1e-6,
                "seed": 7,
                "proposer_backend": "single_prompt",
            },
            initial_policy={"template": "Answer the question."},
            dataset={
                "id": "cities",
                "examples": [
                    {"input": "Capital of France?", "expected": "paris", "metadata": {}},
                    {"input": "Capital of France?", "expected": "paris", "metadata": {}},
                ],
            },
            task_llm=task_llm,
        )

        self.assertTrue(str(result["run_id"]).startswith("pl_"))
        self.assertGreater(float(result["best_score"]), 0.0)
        self.assertIn("best_candidate_id", result["job_result"])
        self.assertIn("Capital of France?", "\n".join(seen_prompts))

    def test_offline_job_surface_matches_hosted_class(self) -> None:
        from synth_ai.sdk.optimization.policy.v1 import PolicyOptimizationOfflineJob as HostedOfflineJob

        required = {
            "create",
            "get",
            "list",
            "status",
            "events",
            "artifacts",
            "checkpoint",
            "submit_candidates",
            "get_state_baseline_info",
            "get_state_envelope",
            "pause",
            "resume",
            "cancel",
            "restart_from_checkpoint",
            "stream_until_complete",
        }
        hosted_methods = {
            name
            for name, member in inspect.getmembers(HostedOfflineJob)
            if callable(member) and not name.startswith("_")
        }
        local_methods = {
            name
            for name, member in inspect.getmembers(PolicyOptimizationOfflineJob)
            if callable(member) and not name.startswith("_")
        }
        self.assertTrue(required.issubset(hosted_methods))
        self.assertTrue(required.issubset(local_methods))

    def test_prompt_learning_job_exposes_candidate_crud(self) -> None:
        def task_llm(prompt: str) -> str:
            if "Return exactly one of" in prompt:
                return "paris"
            return "rome"

        job = PromptLearningJob.from_dict(
            {
                "prompt_learning": {
                    "algorithm": "mipro",
                    "execution_mode": "retrieved",
                    "task_data": {
                        "train_examples": [{"input": "Capital of France?", "answer": "paris"}],
                        "validation_examples": [{"input": "Capital of France?", "answer": "paris"}],
                    },
                    "mipro": {
                        "initial_candidate": {
                            "stages": [
                                {
                                    "id": "main",
                                    "name": "main",
                                    "messages": [{"role": "system", "pattern": "Answer the question.", "order": 0}],
                                }
                            ]
                        },
                        "num_candidates": 3,
                        "max_iterations": 2,
                    },
                    "local_runtime": {"task_model": task_llm},
                }
            },
            backend_url="local://prompt-opt",
            api_key="local",
        )
        job_id = job.submit()
        result = job.stream_until_complete(timeout=30.0, interval=0.05).to_dict()
        self.assertEqual(job_id, result["job_id"])
        status = job.get_status()
        self.assertEqual(status["status"], "succeeded")
        page = job.list_candidates()
        self.assertGreaterEqual(len(page["items"]), 1)
        best_candidate_id = result["best_candidate_id"]
        candidate = job.get_candidate(best_candidate_id)
        self.assertEqual(candidate["candidate_id"], best_candidate_id)
        state_envelope = job.get_state_envelope()
        self.assertEqual(state_envelope["state"]["best_candidate_id"], best_candidate_id)


if __name__ == "__main__":
    unittest.main()
