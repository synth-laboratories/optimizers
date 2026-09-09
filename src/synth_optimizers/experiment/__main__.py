"""`python -m synth_optimizers.experiment` — the standalone entry point.

Mirrors `synth_optimizers.eval`: the app launches a module, not a shell command,
so nothing an agent can say reaches an executor invocation through here.
"""

from __future__ import annotations

import sys

from .commands import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
