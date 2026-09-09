"""Reject new type debt while retaining the explicitly documented release baseline."""
from collections import Counter
from pathlib import Path
import re
import subprocess
import sys

result = subprocess.run(["uv", "run", "--locked", "--group", "dev", "ty", "check", "src", "--output-format", "concise"], text=True, capture_output=True)
output = result.stdout + result.stderr
print(output, end="")
if result.returncode not in (0, 1):
    sys.exit(result.returncode)
diagnostics = [re.sub(r":\d+:\d+: ", ": ", line) for line in output.splitlines() if re.match(r"src/.*:\d+:\d+: (error|warning)\[", line)]
if result.returncode and not diagnostics:
    sys.exit("Type checker failed without parseable diagnostics")
baseline = Counter(Path(__file__).with_name("ty-release-baseline.txt").read_text().splitlines())
added = Counter(diagnostics) - baseline
if added:
    sys.exit("New type diagnostics (release blocked):\n" + "\n".join(added.elements()))
print(f"Type-debt gate passed: {len(diagnostics)} existing diagnostics; no new diagnostic signatures.")
