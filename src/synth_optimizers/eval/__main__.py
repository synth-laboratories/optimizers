"""`python -m synth_optimizers.eval …` for app-owned worker launches."""

from __future__ import annotations

import sys

from .commands import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
