"""Inspect a live-exec'd ReAct script: AST plus Python inspect of the namespace."""

from __future__ import annotations

import ast
import inspect
import os
from typing import Any

from gamebench_levers.apply import sha256_text


def _llm_call_sites(tree: ast.AST) -> int:
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "llm":
            count += 1
    return count


def _architecture(tree: ast.AST) -> str | None:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith(("react_llm_", "speedrunner_")):
                found.append(node.value)
    return found[-1] if found else None


def inspect_source(source: str, *, filename: str = "react_loop.py") -> dict[str, Any]:
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        # A candidate that does not parse must still be inspectable: the policy
        # service reports it as degraded instead of 500-ing its own /health.
        return {
            "filename": filename,
            "content_hash": sha256_text(source),
            "parse_ok": False,
            "syntax_error": f"{exc.msg} (line {exc.lineno})",
            "module_doc": None,
            "functions": [],
            "entrypoint": None,
            "requires_llm": False,
            "llm_call_sites": 0,
            "architecture": None,
            "react_llm": False,
            "source": source,
        }
    functions: list[dict[str, Any]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.append(
                {
                    "name": node.name,
                    "args": [arg.arg for arg in node.args.args],
                    "lineno": node.lineno,
                    "doc": ast.get_docstring(node),
                }
            )
    run = next((row for row in functions if row["name"] == "run_episode"), None)
    llm_sites = _llm_call_sites(tree)
    requires_llm = bool(run and "llm" in row_args(run))
    return {
        "filename": filename,
        "content_hash": sha256_text(source),
        "module_doc": ast.get_docstring(tree),
        "functions": functions,
        "entrypoint": "run_episode" if run else None,
        "requires_llm": requires_llm,
        "llm_call_sites": llm_sites,
        "architecture": _architecture(tree),
        "react_llm": requires_llm and llm_sites >= 1,
    }


def row_args(row: dict[str, Any]) -> list[str]:
    return list(row.get("args") or [])


def inspect_namespace(namespace: dict[str, Any] | None) -> list[dict[str, Any]]:
    live: list[dict[str, Any]] = []
    for name, obj in sorted((namespace or {}).items()):
        if name.startswith("_") or not inspect.isfunction(obj):
            continue
        live.append(
            {
                "name": name,
                "signature": str(inspect.signature(obj)),
                "qualname": getattr(obj, "__qualname__", name),
                "module": getattr(obj, "__module__", None),
                "doc": inspect.getdoc(obj),
                "callable": True,
            }
        )
    return live


def inspect_loaded(
    source: str,
    namespace: dict[str, Any] | None,
    *,
    filename: str = "react_loop.py",
    load_error: str | None = None,
) -> dict[str, Any]:
    static = inspect_source(source, filename=filename)
    run_fn = (namespace or {}).get("run_episode")
    return {
        "schema_id": "react_script_inspect.v1",
        "pid": os.getpid(),
        "in_process": True,
        "instantiated": callable(run_fn) and load_error is None,
        "load_error": load_error,
        "source": source,
        **static,
        "live_callables": inspect_namespace(namespace),
        "live_entrypoint": "run_episode" if callable(run_fn) else None,
    }


def inspect_summary(body: dict[str, Any]) -> dict[str, Any]:
    return {
        "architecture": body.get("architecture"),
        "entrypoint": body.get("entrypoint"),
        "requires_llm": body.get("requires_llm"),
        "llm_call_sites": body.get("llm_call_sites"),
        "react_llm": body.get("react_llm"),
        "functions": [row.get("name") for row in (body.get("functions") or [])],
        "pid": body.get("pid"),
        "instantiated": body.get("instantiated"),
        "content_hash": body.get("content_hash"),
    }
