"""Assemble the immutable GSM8K eval-target build context.

Vendors ``gsm8k_world.py`` from a containers checkout and bakes the pinned
``openai/gsm8k`` rows next to it. The parquet files are fetched at the exact
revision the world module pins (``HF_REVISION``) and written through
``write_snapshot``, which refuses rows that do not reproduce the recorded split
digests — so a build context either carries the pinned dataset or does not
exist. No network is needed when the revision is already in the HF cache
(``HF_HUB_OFFLINE=1`` is honoured).

    python stage.py --containers-src <containers>/src --out <stage dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

HERE = Path(__file__).resolve().parent


def load_world(containers_src: Path) -> ModuleType:
    path = containers_src / "synth_containers" / "platform" / "gsm8k_world.py"
    if not path.is_file():
        raise SystemExit(f"no gsm8k_world.py at {path}")
    spec = importlib.util.spec_from_file_location("gsm8k_world", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: the module's dataclasses resolve their own
    # annotations through sys.modules under `from __future__ import annotations`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fetch_rows(world: ModuleType, split: str) -> tuple:
    from huggingface_hub import hf_hub_download

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - a build prerequisite
        raise SystemExit("stage.py needs pyarrow (pip install pyarrow huggingface_hub)") from exc

    pin = world.SPLIT_PINS[split]
    local = hf_hub_download(
        repo_id=world.HF_DATASET,
        repo_type="dataset",
        revision=world.HF_REVISION,
        filename=f"{world.HF_CONFIG}/{pin.hf_split}-00000-of-00001.parquet",
    )
    table = parquet.read_table(local, columns=["question", "answer"])
    return tuple(
        world.Gsm8kRow(question=str(question), answer_text=str(answer))
        for question, answer in zip(table.column("question").to_pylist(), table.column("answer").to_pylist())
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--containers-src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    world = load_world(args.containers_src.resolve())
    stage = args.out.resolve()
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    shutil.copy(HERE / "Dockerfile", stage / "Dockerfile")
    shutil.copy(HERE / "target.py", stage / "target.py")
    shutil.copy(Path(world.__file__), stage / "gsm8k_world.py")

    rows = {split: fetch_rows(world, split) for split in world.SPLIT_PINS}
    snapshot = world.write_snapshot(stage / "gsm8k", rows)  # verifies every digest first
    manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
    receipt = {
        "dataset": world.HF_DATASET,
        "config": world.HF_CONFIG,
        "revision": world.HF_REVISION,
        "splits": manifest["splits"],
        "shuffle_seed": manifest["shuffle_seed"],
        "world_module": str(Path(world.__file__).resolve()),
    }
    (stage / "stage-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
