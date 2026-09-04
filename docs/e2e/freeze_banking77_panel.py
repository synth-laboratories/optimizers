"""Freeze a deterministic, audited one-example-per-intent Banking77 panel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = "banking77.frozen_panel.v1"
TASK_PATTERN = re.compile(r"banking77/(?:train|heldout)/\d+")
UPSTREAM_TEST = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
AUDIT_SUFFIXES = frozenset({".json", ".jsonl", ".md", ".toml", ".txt"})


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def source_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".csv":
        text = raw.decode("utf-8")
        for index, row in enumerate(csv.DictReader(text.splitlines())):
            label = str(row.get("category") or row.get("label") or "").strip()
            if label:
                rows.append({"task_id": f"banking77/heldout/{index}", "seed": index, "label": label})
    else:
        payload = json.loads(raw)
        items = payload.get("rows", payload) if isinstance(payload, Mapping) else payload
        for index, row in enumerate(items):
            label = str(row.get("label") or row.get("category") or "").strip()
            seed = int(row.get("seed", index))
            task_id = str(row.get("task_id") or f"banking77/heldout/{seed}")
            if label:
                rows.append({"task_id": task_id, "seed": seed, "label": label})
    if not rows:
        raise ValueError(f"Banking77 source is empty: {path}")
    return rows, {"path": str(path.resolve()), "sha256": sha256_bytes(raw), "rows": len(rows)}


def audit_files(paths: Iterable[Path], *, skip: Path | None = None) -> list[Path]:
    files: set[Path] = set()
    skip_resolved = None if skip is None else skip.resolve()
    for path in paths:
        if path.is_dir():
            candidates = (item for item in path.rglob("*") if item.is_file())
        elif path.is_file():
            candidates = (path,)
        else:
            candidates = ()
        for item in candidates:
            if item.suffix.lower() not in AUDIT_SUFFIXES:
                continue
            resolved = item.resolve()
            if resolved != skip_resolved:
                files.add(resolved)
    return sorted(files, key=str)


def exclusions_from_files(files: Sequence[Path]) -> tuple[set[str], list[dict[str, Any]]]:
    excluded: set[str] = set()
    inventory: list[dict[str, Any]] = []
    for path in files:
        raw = path.read_bytes()
        ids = sorted(set(TASK_PATTERN.findall(raw.decode("utf-8", errors="replace"))))
        excluded.update(task_id for task_id in ids if "/heldout/" in task_id)
        # The inventory is an audit of files that contribute exclusions, not a
        # multi-megabyte list of every unrelated text file below a broad root.
        if ids:
            inventory.append(
                {"path": str(path), "sha256": sha256_bytes(raw), "task_ids_found": len(ids)}
            )
    return excluded, inventory


def freeze_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    excluded: set[str],
    panel_seed: str,
    source: Mapping[str, Any],
    exclusion_inventory: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    all_ids: set[str] = set()
    for row in rows:
        task_id, label = str(row["task_id"]), str(row["label"])
        if not task_id.startswith("banking77/heldout/"):
            raise ValueError(f"source row is not heldout: {task_id}")
        if task_id in all_ids:
            raise ValueError(f"source contains duplicate task id: {task_id}")
        all_ids.add(task_id)
        if task_id not in excluded:
            by_label[label].append(row)
    labels = sorted({str(row["label"]) for row in rows})
    if len(labels) != 77:
        raise ValueError(f"expected exactly 77 Banking77 labels, found {len(labels)}")
    missing = [label for label in labels if not by_label[label]]
    if missing:
        raise ValueError(f"exclusions leave no heldout candidate for labels: {missing}")

    selected: list[dict[str, Any]] = []
    for label in labels:
        ranked = sorted(
            by_label[label],
            key=lambda row: hashlib.sha256(
                f"{panel_seed}\0{label}\0{row['task_id']}".encode()
            ).hexdigest(),
        )
        row = ranked[0]
        selection_hash = sha256_bytes(f"{panel_seed}\0{label}\0{row['task_id']}".encode())
        selected.append(
            {"label": label, "task_id": str(row["task_id"]), "seed": int(row["seed"]), "selection_hash": selection_hash}
        )
    ids = [row["task_id"] for row in selected]
    if len(set(ids)) != 77 or set(ids) & excluded:
        raise AssertionError("panel uniqueness/exclusion invariant failed")
    exclusion_ids = sorted(excluded)
    panel_core = [{key: row[key] for key in ("label", "task_id", "seed")} for row in selected]
    return {
        "schema_version": SCHEMA_VERSION,
        "panel_seed": panel_seed,
        "selection_algorithm": "minimum sha256(panel_seed\\0label\\0task_id) among nonexcluded rows",
        "estimand": "macro intent accuracy: one deterministically selected heldout example per Banking77 intent",
        "source": dict(source),
        "exclusion_inventory": [dict(item) for item in exclusion_inventory],
        "excluded_task_ids": exclusion_ids,
        "exclusion_set_digest": sha256_bytes(json.dumps(exclusion_ids, separators=(",", ":")).encode()),
        "rows": selected,
        "task_ids": ids,
        "panel_digest": sha256_bytes(json.dumps(panel_core, sort_keys=True, separators=(",", ":")).encode()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Local canonical Banking77 test CSV or JSON")
    parser.add_argument("--download-hf", action="store_true", help="Fetch the canonical upstream test CSV")
    parser.add_argument("--cache", default="/tmp/banking77-cache/banking77-heldout.csv")
    parser.add_argument("--exclude", action="append", default=[], help="File or directory to inventory; repeatable")
    parser.add_argument("--panel-seed", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if bool(args.source) == bool(args.download_hf):
        parser.error("choose exactly one of --source or --download-hf")
    source_path = Path(args.source) if args.source else Path(args.cache)
    if args.download_hf and not source_path.is_file():
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(UPSTREAM_TEST, timeout=120) as response:
            source_path.write_bytes(response.read())
    rows, source = source_rows(source_path)
    output = Path(args.output)
    files = audit_files((Path(item) for item in args.exclude), skip=output)
    excluded, inventory = exclusions_from_files(files)
    payload = freeze_panel(
        rows,
        excluded=excluded,
        panel_seed=args.panel_seed,
        source=source,
        exclusion_inventory=inventory,
    )
    _atomic_json(output, payload)
    print(json.dumps({"output": str(output), "panel_digest": payload["panel_digest"], "rows": 77}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
