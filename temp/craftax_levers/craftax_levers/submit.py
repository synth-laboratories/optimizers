"""Standard GEPA lever_bundle payloads for this container."""

from __future__ import annotations

from typing import Any

from craftax_levers.apply import sha256_text


def prompt_overlay(text: str) -> dict[str, Any]:
    return {"protocol_id": "prompt_overlay.v1", "content": text}


def whole_file(path: str, content: str, *, protocol_id: str = "whole_file.v1") -> dict[str, Any]:
    return {
        "protocol_id": protocol_id,
        "path": path,
        "content": content,
        "content_hash": sha256_text(content),
        "restart": protocol_id == "harness_restart.v1",
    }


def rollout_request(
    task_id: str,
    *,
    prompt: str | None = None,
    script: str | None = None,
    extra_values: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values: dict[str, Any] = dict(extra_values or {})
    if prompt is not None:
        values["react_system_prompt"] = prompt_overlay(prompt)
    if script is not None:
        values["harness_module"] = whole_file(
            "react_loop.py",
            script,
            protocol_id="harness_restart.v1",
        )
    return {
        "task_id": task_id,
        "submission_mode": "sync",
        "lever_bundle": {
            "schema_version": "lever_bundle.v1",
            "values": values,
        },
    }
