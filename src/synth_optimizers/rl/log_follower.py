"""Disposable read-only CLI viewer; the scientific worker never awaits its IO."""
from __future__ import annotations

import asyncio
import os
import select
import signal
import subprocess
import sys
from pathlib import Path


class LogFollower:
    def __init__(self, path: Path, maximum: int):
        self.process: subprocess.Popen | None = None
        try:
            # A separate process may block on inherited stdout, but cannot block
            # the event loop. Never change flags on the shared stdout descriptor.
            self.process = subprocess.Popen(
                [sys.executable, '-I', str(Path(__file__).resolve()), str(path), str(maximum)],
                stdin=subprocess.PIPE, stdout=sys.stdout, stderr=subprocess.DEVNULL,
                start_new_session=True,
                env={key:value for key,value in os.environ.items()
                     if key in {"PATH", "LANG", "LC_ALL"}},
            )
        except Exception:
            pass  # The retained redacted log remains authoritative.

    async def close(self) -> None:
        process = self.process
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()  # EOF requests one final drain, not a replay.
            except OSError:
                pass
        deadline = asyncio.get_running_loop().time() + 1.0
        try:
            while process.poll() is None and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.025)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            # SIGKILLed viewer owns no grandchildren or remote resources.
            # Reap without a thread executor or blocking stdout writes.
            while process.poll() is None:
                await asyncio.sleep(0.01)


def main() -> None:
    path, maximum = Path(sys.argv[1]), int(sys.argv[2])
    offset = 0
    handle = None
    try:
        while offset < maximum:
            finished = bool(select.select([0], [], [], 0.05)[0])
            if handle is None:
                try:
                    handle = path.open('rb', buffering=0)
                except FileNotFoundError:
                    if finished:
                        return
                    continue
            while offset < maximum:
                chunk = handle.read(min(8192, maximum-offset))
                if not chunk:
                    break
                # Read precisely the already-redacted retained bytes, once.
                offset += len(chunk)
                view = memoryview(chunk)
                while view:
                    view = view[os.write(1, view):]
            if finished:
                return
    except (BrokenPipeError, OSError):
        return
    finally:
        if handle is not None:
            handle.close()


if __name__ == '__main__':
    main()
