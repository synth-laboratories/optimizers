from pathlib import Path
import re

from synth_optimizers import GepaConfig


def test_readme_quickstart_selects_nonempty_task_pools(tmp_path):
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    config_text = re.search(r"```toml\n(.*?)```", readme, re.DOTALL).group(1)
    config_path = tmp_path / "gepa.toml"
    config_path.write_text(config_text)
    config = GepaConfig.from_toml(config_path)
    assert config.taskset.train_ids == ["train:0", "train:1", "train:2", "train:3"]
    assert config.task_pools.pareto == config.taskset.train_ids
    assert config.task_pools.heldout == ["test:100", "test:101"]
