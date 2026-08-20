"""Harvest v0.7 operator ascope artifacts from a live GEPA episode directory."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _newest_state_dir(root: Path) -> Path | None:
    states = sorted(root.glob("**/proposer_workspaces/generation_*/state"), key=lambda p: str(p))
    return states[-1] if states else None


def _codex_has_mcp(root: Path, server: str) -> bool:
    # pathlib glob skips dot-directories, and storage compact may delete Codex
    # homes after the turn. Walk explicitly and also accept a durable receipt.
    needle = f"[mcp_servers.{server}]"
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = dirnames  # include dot dirs
        if "config.toml" not in filenames:
            continue
        path = Path(dirpath) / "config.toml"
        if path.parent.name in {".codex_home", ".codex_api_key_home"}:
            if needle in path.read_text(errors="replace"):
                return True
    receipt = root / "state" / "mcp_codex_receipt.json"
    if receipt.is_file():
        data = _read_json(receipt) or {}
        return bool(data.get("mcp_in_codex_config"))
    return False


def harvest_episode_dir(root: Path | str | None) -> dict[str, Any]:
    """Return a compact ascope receipt for scoring / comparison JSON."""
    if not root:
        return {"ok": False, "error": "no output_dir"}
    base = Path(root)
    if not base.exists():
        return {"ok": False, "error": f"missing {base}"}
    state = _newest_state_dir(base)
    if state is None:
        return {"ok": False, "error": "no proposer workspace state", "root": str(base)}
    workspace = state.parent
    operator = _read_json(state / "operator.json") or {}
    hypotheses = _read_json(state / "hypotheses.json") or {}
    inbox = _read_json(state / "manderqueue_inbox.json") or {}
    mcp = _read_json(state / "mcp_agent.json") or {}
    jesterky = _read_json(state / "jesterky_workflow_receipt.json") or _read_json(
        workspace / "jesterky_workflow_receipt.json"
    )
    scratchpad = state / "scratchpad.md"
    guidance = state / "guidance.md"
    open_hypos = hypotheses.get("open") if isinstance(hypotheses, dict) else None
    messages = inbox.get("messages") if isinstance(inbox, dict) else None
    mcp_server = str(mcp.get("server") or "workspace_fs") if isinstance(mcp, dict) else "workspace_fs"
    mcp_receipt = _read_json(state / "mcp_codex_receipt.json") or {}
    mutated: list[str] = []
    payload_fields: list[str] = []
    registry_paths = [base / "candidate_registry.json", *sorted(base.glob("**/candidate_registry.json"))]
    registry = None
    for path in registry_paths:
        registry = _read_json(path)
        if isinstance(registry, list) and registry:
            break
    if isinstance(registry, list):
        for candidate in registry:
            if not isinstance(candidate, dict):
                continue
            bundle = candidate.get("lever_bundle") if isinstance(candidate.get("lever_bundle"), dict) else {}
            for lever_id in bundle.get("mutated_lever_ids") or []:
                if str(lever_id) not in mutated:
                    mutated.append(str(lever_id))
            payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
            for field in payload:
                if str(field) not in payload_fields:
                    payload_fields.append(str(field))
    return {
        "ok": True,
        "workspace": str(workspace),
        "scratchpad": scratchpad.is_file(),
        "guidance": guidance.is_file(),
        "guidance_has_messages": bool(
            guidance.is_file()
            and "No new operator messages this turn." not in guidance.read_text(errors="replace")
        ),
        "hypotheses_file": isinstance(hypotheses, dict) and bool(hypotheses),
        "hypotheses_open": len(open_hypos) if isinstance(open_hypos, list) else 0,
        "manderqueue_ok": bool(inbox.get("ok")) if isinstance(inbox, dict) else False,
        "manderqueue_base_url": inbox.get("base_url") if isinstance(inbox, dict) else None,
        "manderqueue_messages": len(messages) if isinstance(messages, list) else 0,
        "mcp_enabled": bool(mcp.get("enabled")) if isinstance(mcp, dict) else False,
        "mcp_in_codex_config": bool(mcp_receipt.get("mcp_in_codex_config"))
        or _codex_has_mcp(base, mcp_server),
        "jesterky_annotated": int((jesterky or {}).get("annotated") or 0)
        if isinstance(jesterky, dict)
        else 0,
        "jesterky_fail_open": bool(
            isinstance(jesterky, dict) and jesterky.get("enabled") and not jesterky.get("annotated")
        ),
        "jesterky_context": (state / "jesterky_proposer_context.md").is_file(),
        "jesterky_themes": (state / "jesterky_theme_registry.json").is_file(),
        "jesterky_annotations": (state / "jesterky_trace_annotations.jsonl").is_file(),
        "levers": (operator.get("levers") if isinstance(operator, dict) else None),
        "mutated_lever_ids": mutated,
        "payload_fields": payload_fields,
        "code_lever_mutated": any(
            field in {"domain_policy", "code", "harness"} or "policy" in field
            for field in mutated + payload_fields
        ),
        "control": (operator.get("control") if isinstance(operator, dict) else None),
    }


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Harvest GEPA ascope artifacts from an episode dir")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    json.dump(harvest_episode_dir(args.root), sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
