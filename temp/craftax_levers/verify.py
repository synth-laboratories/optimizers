#!/usr/bin/env python3
"""Smoke: code stack always; ReAct stack only with OPENAI_API_KEY."""

from __future__ import annotations

import json
import os

import httpx

from craftax_levers.seeds import GREEDY_POLICY, WOOD_PROMPT
from craftax_levers.stack import start_stack


def _schemas(body: dict) -> list[str]:
    return [str(item.get("schema_id")) for item in body.get("side_info") or []]


def _roll(url: str, payload: dict, timeout: float = 20.0) -> dict:
    return httpx.post(f"{url}/rollout", json=payload, timeout=timeout).json()


def main() -> None:
    rows: list[dict] = []
    code = start_stack("code")
    try:
        seed = _roll(code.orch_url, {"task_id": "train:0", "candidate": {}})
        greedy = _roll(code.orch_url, {"task_id": "train:0", "candidate": {"policy_script": GREEDY_POLICY}})
        rows.append({"stack": "code", "lever": "seed policy_script", "reward": seed["reward"], "asi": _schemas(seed)})
        rows.append({"stack": "code", "lever": "whole_file greedy", "reward": greedy["reward"], "asi": _schemas(greedy)})
        print("code orchestrator", code.orch_url)
    finally:
        code.stop()

    assert rows[0]["reward"] == 0.0
    assert rows[1]["reward"] == 2.0

    if not os.environ.get("OPENAI_API_KEY"):
        print(json.dumps(rows, indent=2))
        print("ok (code only; set OPENAI_API_KEY to run ReAct)")
        return

    react = start_stack("react")
    try:
        wander = _roll(
            react.orch_url,
            {"task_id": "train:0", "candidate": {"react_system_prompt": "Wander randomly."}},
            timeout=180.0,
        )
        wood = _roll(
            react.orch_url,
            {"task_id": "train:0", "candidate": {"react_system_prompt": WOOD_PROMPT}},
            timeout=180.0,
        )
        provider = next(
            item["summary"].get("llm_provider")
            for item in wander["side_info"]
            if item.get("schema_id") == "harness_v5_trace.v1"
        )
        rows.append(
            {
                "stack": "react",
                "lever": "ReAct wander overlay",
                "reward": wander["reward"],
                "llm_provider": provider,
                "asi": _schemas(wander),
            }
        )
        rows.append(
            {
                "stack": "react",
                "lever": "ReAct wood overlay",
                "reward": wood["reward"],
                "asi": _schemas(wood),
            }
        )
        print("react orchestrator", react.orch_url)
        assert wander["success_status"] == "succeeded"
        assert wood["success_status"] == "succeeded"
        assert provider in {"openai", "openrouter"}
        assert any(event.get("type") == "llm_request" for event in wander["trace"]["event_history"])
        assert any(event.get("type") == "llm_request" for event in wood["trace"]["event_history"])
    finally:
        react.stop()

    print(json.dumps(rows, indent=2))
    print("ok")


if __name__ == "__main__":
    main()
