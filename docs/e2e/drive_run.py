"""Drive the real executor against a container running in another process.

Every in-process test so far has had both halves in one interpreter. This is
the first time the contract crosses a socket, which is the only place a wire
disagreement can actually show up.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "/Users/joshuapurtell/GitHub/optimizers/src")
sys.path.insert(0, "/Users/joshuapurtell/GitHub/optimizers/tests")

from synth_optimizers.providers.tinker.fake import FakeTinkerProvider  # noqa: E402
from synth_optimizers.rl.config import load_config  # noqa: E402
from synth_optimizers.rl.executor import execute  # noqa: E402
from synth_optimizers.rl.plane import build_plane  # noqa: E402

CONFIG = """
schema_version = "cispo.container.v1"
run_id = "e2e_http_counter"

[container]
url = "{url}"

[taskset]
train_split = "train"
evaluation_split = "train"
train_ids = ["counter.default"]
evaluation_ids = []

[model]
provider = "fake"
id = "vendor/policy-20b"
family = "gpt_oss"
policy_kind = "counter"

[plan]
preset = "cispo"
group_size = 2
groups_per_step = 1
target_train_updates = 1
maximum_sampled_groups = 4

[pipeline]
max_execution_slots = 2
rollout_queue_capacity = 8
score_queue_capacity = 8
scored_result_queue_capacity = 8
train_ready_capacity = 1
maximum_policy_lag = 0
max_open_groups = 1
stale_disposition = "discard"

[topology]
expected_topology_id = "counter.reference.solo.v1"
trainable_teams = ["solo"]
partial_roster = "refuse"

[opponents]
match_set_revision = "match-set-0001"

[reward]
optimized_channel = "score"

[evaluation]
paired = false

[lifecycle]
resume_requires_rehandshake = true

[offline]
mode = "off"

[artifacts]
catalog = "checkpoints.sqlite3"
directory = "runs"
"""


def main() -> int:
    url = sys.argv[1]
    workdir = Path(sys.argv[2])
    workdir.mkdir(parents=True, exist_ok=True)
    path = workdir / "run.toml"
    path.write_text(CONFIG.format(url=url))

    config = load_config(path)
    print(f"config ok: run_id={config.run_id} plan={config.plan.preset}", flush=True)

    plane = build_plane(config, provider=FakeTinkerProvider())
    print("plane assembled against the live container", flush=True)
    try:
        result = execute(config, plane, receipts_dir=workdir / "receipts")
    finally:
        close = getattr(plane, "close", None)
        if callable(close):
            close()
    print(json.dumps({"summary": str(result)[:400]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
