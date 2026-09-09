from __future__ import annotations

from synth_optimizers.sft_cli import follow_is_terminal, poll_follow


def test_sft_follow_treats_completed_as_terminal() -> None:
    assert follow_is_terminal("completed")
    assert follow_is_terminal("succeeded")
    assert follow_is_terminal("failed")
    assert follow_is_terminal("cancelled")
    assert not follow_is_terminal("running")
    assert not follow_is_terminal("queued")


def test_sft_poll_follow_exits_on_completed_without_tinker() -> None:
    records = iter(
        [
            {"status": "running"},
            {"status": "completed", "run_id": "sft_public_1"},
        ]
    )
    sleeps: list[float] = []
    lines: list[str] = []
    code = poll_follow(
        lambda: next(records),
        poll_seconds=0.5,
        sleep=sleeps.append,
        emit=lines.append,
    )
    assert code == 0
    assert sleeps == [0.5]
    assert lines == ["status=running", "status=completed"]


def test_sft_poll_follow_json_dumps_terminal_record() -> None:
    emitted: list[str] = []
    code = poll_follow(
        lambda: {"status": "completed", "run_id": "sft_json"},
        poll_seconds=1.0,
        json_output=True,
        sleep=lambda _: None,
        emit=emitted.append,
    )
    assert code == 0
    assert emitted[0] == "status=completed"
    assert '"run_id": "sft_json"' in emitted[1]
    assert '"status": "completed"' in emitted[1]


def test_sft_poll_follow_failed_returns_one() -> None:
    assert (
        poll_follow(
            lambda: {"status": "failed"},
            poll_seconds=1.0,
            sleep=lambda _: None,
            emit=lambda _: None,
        )
        == 1
    )
