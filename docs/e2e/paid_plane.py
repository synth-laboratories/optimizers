"""The real provider, assembled the way a production run assembles it.

Nothing is stubbed here. The only difference from `rl run` with no --plane is
that the credential is read from a named file rather than the ambient
environment, because it does not live in this shell.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

from synth_optimizers.rl.plane import build_plane

#: Where the Tinker credential lives when it is not already in the environment.
#: Override with SYNTH_TINKER_ENV_FILE; the default is where it happened to be
#: on the machine this was written on, which is not a promise about yours.
KEY_FILE = os.environ.get(
    "SYNTH_TINKER_ENV_FILE", "/Users/joshuapurtell/GitHub/frontend/.env.local"
)


def _load_credential() -> None:
    if os.environ.get("TINKER_API_KEY"):
        return
    for line in pathlib.Path(KEY_FILE).read_text().splitlines():
        if line.startswith("TINKER_API_KEY="):
            os.environ["TINKER_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
            return
    raise SystemExit(f"no TINKER_API_KEY in {KEY_FILE}")


def paid(config: Any = None, **kwargs: Any) -> Any:
    if config is None:
        raise SystemExit("this plane needs --config: it is assembled from one")
    _load_credential()
    return build_plane(config, **kwargs)
