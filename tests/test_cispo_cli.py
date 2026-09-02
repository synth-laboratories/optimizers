from __future__ import annotations

import pytest

from synth_optimizers.cispo_cli import (
    DEFAULT_BIND,
    DEFAULT_URL,
    TOKEN_ENV,
    URL_ENV,
    build_parser,
    follow_is_terminal,
    poll_follow,
)
from synth_optimizers.recipes.banking77 import cispo_recipe


def test_cispo_follow_treats_completed_as_terminal() -> None:
    assert follow_is_terminal("completed")
    assert follow_is_terminal("succeeded")
    assert follow_is_terminal("failed")
    assert follow_is_terminal("cancelled")
    assert not follow_is_terminal("running")
    assert not follow_is_terminal("queued")
    assert not follow_is_terminal("prepared")


def test_cispo_poll_follow_exits_on_completed() -> None:
    records = iter([{"status": "running"}, {"status": "completed", "run_id": "cispo_1"}])
    sleeps: list[float] = []
    lines: list[str] = []
    code = poll_follow(
        lambda: next(records),
        poll_seconds=0.25,
        sleep=sleeps.append,
        emit=lines.append,
    )
    assert code == 0
    assert sleeps == [0.25]
    assert lines == ["status=running", "status=completed"]


def test_cispo_poll_follow_failed_is_nonzero() -> None:
    code = poll_follow(
        lambda: {"status": "failed", "error": "boom"},
        poll_seconds=1.0,
        json_output=True,
        sleep=lambda _: None,
        emit=lambda _: None,
    )
    assert code == 1


def test_cispo_cli_service_defaults(monkeypatch) -> None:
    monkeypatch.delenv(URL_ENV, raising=False)
    args = build_parser().parse_args(["service"])
    assert args.bind == DEFAULT_BIND
    assert args.bind == "127.0.0.1:8880"
    assert args.db == ".cispo/service.sqlite"
    assert args.service_token_env == TOKEN_ENV
    assert args.fixture is False


def test_cispo_cli_submit_defaults_and_follow(monkeypatch) -> None:
    monkeypatch.delenv(URL_ENV, raising=False)
    args = build_parser().parse_args(
        ["submit", "--config", "cispo.json", "--follow", "--run-id", "cispo_hosted"]
    )
    assert args.service_url == DEFAULT_URL
    assert args.follow is True
    assert args.run_id == "cispo_hosted"
    assert args.poll_seconds == 1.0


def test_cispo_cli_has_no_sft_or_generic_is_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["service"]).cispo_command == "service"
    for unknown in ("sft", "go-ex", "is"):
        with pytest.raises(SystemExit):
            parser.parse_args([unknown])


def test_main_cli_does_not_register_cispo() -> None:
    from synth_optimizers.cli import build_parser as build_main_parser

    with pytest.raises(SystemExit):
        build_main_parser().parse_args(["cispo", "service"])


def test_learning_signal_recipe_is_a_cispo_request() -> None:
    request = cispo_recipe(mode="learning_signal").request
    assert request["algorithm_id"] == "cispo"
    assert request["implementation"] == "slime-reference"
    assert request["implementation_version"] == "cispo.slime.v1"
    assert request["schema_version"] == "cispo.request.v1"
    assert request["mode"] == "learning_signal"
