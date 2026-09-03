"""``synth-optimizers rl …``: the command surface, and what it refuses.

Two things are being held here. The family must be reachable through the
existing entrypoint rather than through a second CLI of its own, and every
command that resolves a policy must print the selector it was handed next to
the immutable id that selector resolved to. And a refusal must be legible: each
one exits non-zero with a message that names the thing that was wrong, so an
operator never has to read a stack trace to learn that an artifact digest
disagreed.

No network, no container runtime, no provider. The catalog and journal are
sqlite files in ``tmp_path``; the world builder is shared with the paired
evaluation suite.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from test_evaluation import World, build_world, make_record, sha

from synth_optimizers.cli import build_parser
from synth_optimizers.cli import main as umbrella_main
from synth_optimizers.rl.cli import dispatch, register
from synth_optimizers.rl.evaluation import PairedEvaluation
from synth_optimizers.rl.store import JournalStore, RunIdentity
from test_evaluation import (
    FakeBinder,
    FakeGateway,
    request_for,
    session_for,
)

RUN_ID = "run_a"


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture()
def world(tmp_path) -> World:
    home = tmp_path / "plane"
    home.mkdir()
    built = build_world(home)
    yield built
    built.catalog.close()


@pytest.fixture()
def catalog_path(tmp_path) -> str:
    return str(tmp_path / "plane" / "catalog.sqlite3")


@pytest.fixture()
def digests(tmp_path, world: World) -> str:
    path = tmp_path / "digests.json"
    path.write_text(json.dumps(world.probe.digests), encoding="utf-8")
    return str(path)


@pytest.fixture()
def pin_file(tmp_path) -> str:
    path = tmp_path / "pin.json"
    path.write_text(
        json.dumps(
            {
                "run_id": RUN_ID,
                "algorithm_plan_hash": "plan#alpha",
                "wire_api": "chat_completions",
                "sampling_transport": "text_in_text_out",
                "policy_kind": "lora",
                "model_family": "vendor/base-model-a",
                "container_image_digest": sha("image"),
                "container_contract_hash": sha("container_contract"),
                "task_family": "family_a",
            }
        ),
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture()
def journal(tmp_path) -> str:
    path = str(tmp_path / "journal.sqlite3")
    store = JournalStore(path)
    store.register_run(
        RunIdentity(
            run_id=RUN_ID,
            container_contract_hash=sha("container_contract"),
            container_image_digest=sha("image"),
            algorithm_plan_hash="plan#alpha",
            renderer_fingerprint=sha("renderer"),
            handshake_agreement_digest=sha("agreement"),
            capability_hash=sha("capability"),
        )
    )
    store.close()
    return path


def run_cli(argv: list[str]) -> int:
    """Through the umbrella parser, exactly as an operator reaches it."""

    args = build_parser().parse_args(argv)
    assert args.command == "rl"
    return dispatch(args)


def identity_file(tmp_path, name: str, **overrides: str) -> str:
    payload = {
        "run_id": RUN_ID,
        "container_contract_hash": sha("container_contract"),
        "container_image_digest": sha("image"),
        "algorithm_plan_hash": "plan#alpha",
        "renderer_fingerprint": sha("renderer"),
        "handshake_agreement_digest": sha("agreement"),
        "capability_hash": sha("capability"),
    }
    payload.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_rl_is_registered_on_the_existing_entrypoint() -> None:
    parser = build_parser()
    args = parser.parse_args(["rl", "catalog", "list", "--catalog", "x"])
    assert (args.command, args.rl_command, args.catalog_command) == ("rl", "catalog", "list")
    # The commands that were there before are untouched.
    assert parser.parse_args(["events", "replay", "--events", "x"]).command == "events"


def test_the_umbrella_main_routes_rl(world: World, catalog_path: str, capsys) -> None:
    assert umbrella_main(["rl", "catalog", "list", "--catalog", catalog_path]) == 0
    assert "ckpt_primary_u1" in capsys.readouterr().out


def test_the_family_declares_every_command(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["rl", "--help"])
    text = capsys.readouterr().out
    for command in ("run", "evaluate", "catalog", "receipt", "pause", "drain", "resume", "stop"):
        assert command in text


def test_register_is_additive_over_any_subparser_action() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    register(parser.add_subparsers(dest="command", required=True))
    assert parser.parse_args(["rl", "stop", "--journal", "j", "--run-id", "r"]).rl_command == "stop"


# --------------------------------------------------------------------------- #
# catalog list: every declared index
# --------------------------------------------------------------------------- #


@pytest.fixture()
def indexed(world: World) -> World:
    """A child checkpoint for the parent index, and a metric for the metric index."""

    child = replace(
        make_record(
            checkpoint_id="ckpt_primary_u2",
            policy_revision_id="pg_primary@2",
            update_id="update_0002",
            publication_status="published",
        ),
        parent_checkpoint_id="ckpt_primary_u1",
    )
    world.catalog.register_checkpoint(child)
    world.probe.digests[child.artifacts.sampler.ref] = child.artifacts.sampler.digest
    world.probe.digests[child.artifacts.resumable.ref] = child.artifacts.resumable.digest
    PairedEvaluation(
        world.resolver,
        session=session_for(),
        gateway=FakeGateway(),
        binder=FakeBinder(world.catalog),
    ).run(request_for(trained="ckpt_primary_u1", baseline="ckpt_baseline_primary"))
    return world


@pytest.mark.parametrize(
    ("flag", "value", "expected"),
    [
        ("--run", RUN_ID, "ckpt_primary_u1"),
        ("--policy-type", "type_second", "ckpt_second_u1"),
        ("--parameter-group", "pg_primary", "ckpt_primary_u1"),
        ("--update", "update_0001", "ckpt_primary_u1"),
        ("--parent", "ckpt_primary_u1", "ckpt_primary_u2"),
        ("--status", "published", "ckpt_primary_u1"),
        ("--status", "superseded", "ckpt_baseline_primary"),
        ("--metric", "mean_reward", "ckpt_primary_u1"),
        ("--policy-set", "set-trained", "ckpt_second_u1"),
        ("--train-call", "provider_train_ckpt_primary_u1", "ckpt_primary_u1"),
        ("--base-model", "vendor/base-model-a", "ckpt_primary_u1"),
    ],
)
def test_catalog_list_by_each_declared_index(
    indexed: World, catalog_path: str, capsys, flag: str, value: str, expected: str
) -> None:
    assert run_cli(["rl", "catalog", "list", "--catalog", catalog_path, flag, value]) == 0
    output = capsys.readouterr().out
    assert expected in output


def test_catalog_list_by_metric_ranks_and_reports_the_value(
    indexed: World, catalog_path: str, capsys
) -> None:
    code = run_cli(
        ["rl", "catalog", "list", "--catalog", catalog_path, "--metric", "mean_reward", "--json"]
    )
    assert code == 0
    rows = json.loads(capsys.readouterr().out)["checkpoints"]
    assert [row["checkpoint_id"] for row in rows] == ["ckpt_primary_u1", "ckpt_baseline_primary"]
    assert rows[0]["metric"] > rows[1]["metric"]


def test_the_status_index_partitions_rather_than_overlaps(
    indexed: World, catalog_path: str, capsys
) -> None:
    run_cli(["rl", "catalog", "list", "--catalog", catalog_path, "--status", "published"])
    published = capsys.readouterr().out
    run_cli(["rl", "catalog", "list", "--catalog", catalog_path, "--status", "superseded"])
    superseded = capsys.readouterr().out
    assert "ckpt_baseline_primary" not in published
    assert "ckpt_primary_u1  [" not in superseded


def test_catalog_list_reports_an_empty_index_rather_than_guessing(
    world: World, catalog_path: str, capsys
) -> None:
    assert run_cli(["rl", "catalog", "list", "--catalog", catalog_path, "--run", "run_z"]) == 0
    assert "no checkpoint matches" in capsys.readouterr().out


def test_catalog_list_refuses_an_unknown_publication_status(catalog_path: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["rl", "catalog", "list", "--catalog", catalog_path, "--status", "invented"]
        )


# --------------------------------------------------------------------------- #
# catalog describe
# --------------------------------------------------------------------------- #


def test_describe_an_immutable_id_prints_selector_and_resolution(
    world: World, catalog_path: str, capsys
) -> None:
    assert run_cli(["rl", "catalog", "describe", "--catalog", catalog_path, "set-trained"]) == 0
    output = capsys.readouterr().out
    assert "selector=set-trained resolved=set-trained" in output
    assert "policy_set" in output


def test_describe_an_alias_prints_the_immutable_id_it_resolved_to(
    world: World, catalog_path: str, digests: str, capsys
) -> None:
    code = run_cli(
        [
            "rl",
            "catalog",
            "describe",
            "--catalog",
            catalog_path,
            "champion",
            "--artifact-digests",
            digests,
        ]
    )
    assert code == 0
    output = capsys.readouterr().out
    assert "selector=champion resolved=checkpoint:ckpt_primary_u1 alias=champion" in output
    assert "provider://sampler/ckpt_primary_u1" in output


def test_describe_without_a_digest_source_refuses_to_resolve_an_alias(
    world: World, catalog_path: str, capsys
) -> None:
    assert run_cli(["rl", "catalog", "describe", "--catalog", catalog_path, "champion"]) == 1
    assert "--artifact-digests" in capsys.readouterr().err


def test_describe_an_unknown_selector_refuses(world: World, catalog_path: str, capsys) -> None:
    assert run_cli(["rl", "catalog", "describe", "--catalog", catalog_path, "ckpt_nope"]) == 1
    assert "neither an immutable id nor a registered alias" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #


def evaluate_argv(catalog_path: str, digests: str, pin_file: str, **overrides: str) -> list[str]:
    argv = [
        "rl",
        "evaluate",
        "--catalog",
        catalog_path,
        "--artifact-digests",
        digests,
        "--pin",
        pin_file,
        "--selector",
        overrides.get("selector", "ckpt_primary_u1"),
        "--baseline",
        overrides.get("baseline", "ckpt_baseline_primary"),
        "--seed",
        "row_0001=3",
        "--roster",
        "inst_a=pg_primary",
    ]
    return argv


def test_evaluate_resolve_only_prints_both_arms_and_runs_nothing(
    world: World, catalog_path: str, digests: str, pin_file: str, capsys
) -> None:
    argv = evaluate_argv(catalog_path, digests, pin_file) + ["--resolve-only"]
    assert run_cli(argv) == 0
    output = capsys.readouterr().out
    assert "trained: selector=ckpt_primary_u1 resolved=checkpoint:ckpt_primary_u1" in output
    assert (
        "baseline: selector=ckpt_baseline_primary resolved=checkpoint:ckpt_baseline_primary"
        in output
    )
    assert "no attempt was run" in output
    assert world.catalog.evaluations() == ()


def test_evaluate_resolve_only_json_carries_selector_and_immutable_id(
    world: World, catalog_path: str, digests: str, pin_file: str, capsys
) -> None:
    argv = evaluate_argv(catalog_path, digests, pin_file, selector="champion") + [
        "--resolve-only",
        "--json",
    ]
    assert run_cli(argv) == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{") :])
    trained = payload["arms"]["trained"]
    assert trained["requested_selector"] == "champion"
    assert trained["alias"] == "champion"
    assert trained["resolved_id"] == "ckpt_primary_u1"


def test_evaluate_refuses_a_mutable_selector(
    world: World, catalog_path: str, digests: str, pin_file: str, capsys
) -> None:
    argv = evaluate_argv(catalog_path, digests, pin_file, selector="latest") + ["--resolve-only"]
    assert run_cli(argv) == 1
    error = capsys.readouterr().err
    assert "not an identity" in error
    assert "latest" in error


def test_evaluate_refuses_a_digest_mismatch(
    world: World, catalog_path: str, tmp_path, pin_file: str, capsys
) -> None:
    tampered = dict(world.probe.digests)
    tampered["provider://sampler/ckpt_primary_u1"] = sha("someone_else")
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    argv = evaluate_argv(catalog_path, str(path), pin_file) + ["--resolve-only"]
    assert run_cli(argv) == 1
    error = capsys.readouterr().err
    assert "ckpt_primary_u1" in error
    assert "digests" in error


def test_evaluate_refuses_a_missing_artifact(
    world: World, catalog_path: str, tmp_path, pin_file: str, capsys
) -> None:
    partial = {
        ref: value
        for ref, value in world.probe.digests.items()
        if ref != "provider://sampler/ckpt_primary_u1"
    }
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(partial), encoding="utf-8")
    argv = evaluate_argv(catalog_path, str(path), pin_file) + ["--resolve-only"]
    assert run_cli(argv) == 1
    assert "does not exist" in capsys.readouterr().err


def test_evaluate_refuses_a_role_mismatch(
    world: World, catalog_path: str, digests: str, pin_file: str, tmp_path, capsys
) -> None:
    resumable_only = make_record(
        checkpoint_id="ckpt_state_only",
        policy_revision_id="pg_primary@2",
        update_id="update_0002",
        publication_status="published",
        sampler=False,
    )
    world.catalog.register_checkpoint(resumable_only)
    path = tmp_path / "with_state.json"
    payload = dict(world.probe.digests)
    payload[resumable_only.artifacts.resumable.ref] = resumable_only.artifacts.resumable.digest
    path.write_text(json.dumps(payload), encoding="utf-8")
    argv = evaluate_argv(catalog_path, str(path), pin_file, selector="ckpt_state_only") + [
        "--resolve-only"
    ]
    assert run_cli(argv) == 1
    assert "sampler_weights" in capsys.readouterr().err


def test_evaluate_requires_a_pin(world: World, catalog_path: str, digests: str) -> None:
    argv = [
        "rl",
        "evaluate",
        "--catalog",
        catalog_path,
        "--artifact-digests",
        digests,
        "--selector",
        "ckpt_primary_u1",
        "--baseline",
        "ckpt_baseline_primary",
        "--seed",
        "row_0001=3",
        "--roster",
        "inst_a=pg_primary",
    ]
    with pytest.raises(SystemExit) as raised:
        run_cli(argv)
    assert "--pin is required" in str(raised.value)


def test_evaluate_rejects_a_malformed_seed(
    world: World, catalog_path: str, digests: str, pin_file: str
) -> None:
    argv = evaluate_argv(catalog_path, digests, pin_file)
    argv[argv.index("row_0001=3")] = "row_0001"
    with pytest.raises(SystemExit) as raised:
        run_cli(argv)
    assert "TASK_ID=SEED" in str(raised.value)


def test_evaluate_without_a_live_plane_refuses_legibly(
    world: World, catalog_path: str, digests: str, pin_file: str
) -> None:
    try:
        import synth_optimizers.rl.session  # noqa: F401
    except ImportError:
        pass
    else:  # pragma: no cover - once the session seam lands this stops applying
        pytest.skip("the live session module landed; this refusal no longer applies")
    with pytest.raises(SystemExit) as raised:
        run_cli(evaluate_argv(catalog_path, digests, pin_file))
    assert "--resolve-only" in str(raised.value)


# --------------------------------------------------------------------------- #
# receipt
# --------------------------------------------------------------------------- #


def test_receipt_shows_the_run_binding_and_its_lifecycle(journal: str, capsys) -> None:
    assert run_cli(["rl", "receipt", "--journal", journal, "--run-id", RUN_ID]) == 0
    output = capsys.readouterr().out
    assert f"run {RUN_ID}" in output
    assert "admitting" in output
    assert "binding digest" in output


def test_receipt_includes_catalog_rows_and_evaluation_relations(
    indexed: World, journal: str, catalog_path: str, capsys
) -> None:
    code = run_cli(
        [
            "rl",
            "receipt",
            "--journal",
            journal,
            "--run-id",
            RUN_ID,
            "--catalog",
            catalog_path,
            "--json",
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    ids = {row["checkpoint_id"] for row in payload["checkpoints"]}
    assert {"ckpt_baseline_primary", "ckpt_primary_u1"} <= ids
    selectors = {row["requested_selector"] for row in payload["evaluations"]}
    resolved = {row["target_id"] for row in payload["evaluations"]}
    assert "ckpt_primary_u1" in selectors and "ckpt_primary_u1" in resolved


def test_receipt_refuses_an_unregistered_run(journal: str, capsys) -> None:
    assert run_cli(["rl", "receipt", "--journal", journal, "--run-id", "run_missing"]) == 1
    assert "not registered" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# lifecycle
# --------------------------------------------------------------------------- #


def test_pause_closes_admission(journal: str, capsys) -> None:
    assert run_cli(["rl", "pause", "--journal", journal, "--run-id", RUN_ID]) == 0
    assert "state=paused" in capsys.readouterr().out


def test_drain_then_finish_reaches_drained(journal: str, capsys) -> None:
    assert run_cli(["rl", "drain", "--journal", journal, "--run-id", RUN_ID]) == 0
    assert "state=draining" in capsys.readouterr().out
    assert run_cli(["rl", "drain", "--journal", journal, "--run-id", RUN_ID, "--finish"]) == 0
    assert "state=drained" in capsys.readouterr().out


def test_stop_is_terminal_and_refuses_a_second_stop(journal: str, capsys) -> None:
    assert run_cli(["rl", "stop", "--journal", journal, "--run-id", RUN_ID]) == 0
    assert "state=stopped" in capsys.readouterr().out
    assert run_cli(["rl", "stop", "--journal", journal, "--run-id", RUN_ID]) == 1
    assert "already stopped" in capsys.readouterr().err


def test_resume_without_a_rehandshake_is_refused(journal: str) -> None:
    assert run_cli(["rl", "pause", "--journal", journal, "--run-id", RUN_ID]) == 0
    with pytest.raises(SystemExit) as raised:
        run_cli(["rl", "resume", "--journal", journal, "--run-id", RUN_ID])
    assert "re-verified" in str(raised.value)


def test_resume_reopens_admission_when_the_binding_still_holds(
    journal: str, tmp_path, capsys
) -> None:
    assert run_cli(["rl", "pause", "--journal", journal, "--run-id", RUN_ID]) == 0
    capsys.readouterr()
    path = identity_file(tmp_path, "rehandshake.json")
    code = run_cli(
        ["rl", "resume", "--journal", journal, "--run-id", RUN_ID, "--rehandshake", path]
    )
    assert code == 0
    assert "state=admitting" in capsys.readouterr().out


def test_resume_refuses_a_changed_binding(journal: str, tmp_path, capsys) -> None:
    assert run_cli(["rl", "pause", "--journal", journal, "--run-id", RUN_ID]) == 0
    capsys.readouterr()
    path = identity_file(tmp_path, "drifted.json", container_image_digest=sha("other_image"))
    code = run_cli(
        ["rl", "resume", "--journal", journal, "--run-id", RUN_ID, "--rehandshake", path]
    )
    assert code == 1
    error = capsys.readouterr().err
    assert "container_image_digest" in error
    assert "new run" in error


def test_lifecycle_refuses_an_illegal_transition(journal: str, capsys) -> None:
    assert run_cli(["rl", "stop", "--journal", journal, "--run-id", RUN_ID]) == 0
    capsys.readouterr()
    assert run_cli(["rl", "pause", "--journal", journal, "--run-id", RUN_ID]) == 1
    assert "cannot be paused" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def test_run_refuses_a_config_that_is_not_there(tmp_path) -> None:
    with pytest.raises(SystemExit) as raised:
        run_cli(["rl", "run", "--config", str(tmp_path / "absent.toml")])
    assert "no such config file" in str(raised.value)


def test_run_without_an_executor_names_what_is_missing(tmp_path) -> None:
    try:
        import synth_optimizers.rl.executor  # noqa: F401
    except ImportError:
        pass
    else:  # pragma: no cover - once the executor lands this stops applying
        pytest.skip("the run executor landed; this refusal no longer applies")
    config = tmp_path / "run.toml"
    config.write_text("[run]\nrun_id = 'run_a'\n", encoding="utf-8")
    with pytest.raises(SystemExit) as raised:
        run_cli(["rl", "run", "--config", str(config)])
    assert "execute_run" in str(raised.value)
