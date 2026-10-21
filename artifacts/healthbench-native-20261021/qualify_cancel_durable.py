"""Zero-provider cancellation of a real native HealthBench OCI worker."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
image = json.loads((ROOT / "build-durable/artifact.json").read_text())["output"]["artifact"]["digest"]
fixture = ROOT / "cancel-fixture-durable"
fixture.mkdir(exist_ok=True)
(fixture / "boot.py").write_text('''import json,time
from pathlib import Path
import eval_adapter
from healthbench_chat import runtime
def blocked(config,messages):
    Path("/output/provider-free-fixture.json").write_text(json.dumps({"external_provider_calls":0,"blocked":True}))
    time.sleep(600)
    raise RuntimeError("fixture_expired")
runtime._chat=blocked
eval_adapter.execute()
''')
subprocess.run(["docker", "tag", image, "synth-healthbench-native-local:cancel-base-20261021"], check=True)
(fixture / "Dockerfile").write_text(f'FROM synth-healthbench-native-local:cancel-base-20261021@{image}\nCOPY boot.py /fixture.py\nCMD ["python", "/fixture.py"]\n')
subprocess.run(["docker", "build", "--iidfile", str(fixture / "image-id.txt"), str(fixture)], stdout=(fixture / "build.log").open("w"), stderr=subprocess.STDOUT, check=True)
pin = (fixture / "image-id.txt").read_text().strip()
environment = os.environ.copy()
environment.pop("PYTHONPATH", None)
environment["OPENROUTER_API_KEY"] = "provider-free-fixture-only"
home = ROOT / "cancel-home-durable"
worker = fixture / "worker.py"
worker.write_text(f'''from pathlib import Path
import json,sys
from synth_optimizers.eval.home import EvalHome
from synth_optimizers.eval.staging import CandidateSource,stage_candidate_set
from synth_optimizers.eval.runner import run_worker
from synth_optimizers.eval.healthbench import SharedOciExecutor,RECIPE,result_exit_code
from types import SimpleNamespace
home=EvalHome.open(Path({str(home)!r}))
(home.root/"pins.toml").write_text('[pins."'+RECIPE+'"]\\nimage_digest="{pin}"\\n')
candidates=stage_candidate_set(home,[CandidateSource("fixture",Path({str(ROOT / 'release-durable/prompt.json')!r}),"prompt.json","healthbench-prompt.v1",True)])
manifest=home.root/"worker.json"
manifest.write_text(json.dumps({{"schema_version":"eval.worker-manifest.v1","run_id":"native-cancel-fixture-durable","recipe_id":RECIPE,"home":str(home.root),"candidate_set_path":str(candidates.root/"candidate_set.json")}}))
code=run_worker(manifest,executor=SharedOciExecutor(),stream=sys.stdout)
raise SystemExit(code or result_exit_code(SimpleNamespace(home=home.root,run_id="native-cancel-fixture-durable")))
''')
with (fixture / "worker.log").open("w") as log:
    process = subprocess.Popen([str(ROOT / "venv/bin/python"), str(worker)], env=environment, stdout=log, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 60
        run = home / "runs/native-cancel-fixture-durable"
        while True:
            assert process.poll() is None, (fixture / "worker.log").read_text()[-2000:]
            outputs = list(run.glob("trials/*/output/provider-free-fixture.json"))
            if outputs and json.loads((outputs[0].parent / "operation.json").read_text())["state"] == "running":
                break
            assert time.monotonic() < deadline, "fixture did not reach native running"
            time.sleep(.1)
        cancel = subprocess.run([str(ROOT / "venv/bin/python"), "-m", "synth_optimizers.eval.healthbench", "cancel", "--home", str(home), "--run-id", "native-cancel-fixture-durable"], capture_output=True, text=True, env=environment, check=True)
        (fixture / "cancel-command.json").write_text(cancel.stdout)
        code = process.wait(timeout=30)
        assert code == 1
        jobs = list(run.glob("trials/*/job_result.json"))
        assert len(jobs) == 1
        result = json.loads(jobs[0].read_text())
        assert result["status"] == "cancelled" and not result["valid"]
        output = jobs[0].parent / "output"
        assert (output / "events.jsonl").exists() and (output / "native-cursor.json").exists()
        calls = [json.loads(line) for line in (output / "provider.jsonl").read_text().splitlines()]
        assert len(calls) == 1 and calls[0]["state"] == "reserved"
        trial_id = result["trial_id"]
        container = "synth-eval-" + trial_id[:40] + "-" + hashlib.sha256(trial_id.encode()).hexdigest()[:12]
        names = subprocess.check_output(["docker", "ps", "-a", "--format", "{{.Names}}"], text=True).splitlines()
        assert container not in names
        receipt = {"provider_free": True, "external_provider_calls": 0, "fixture_image": pin,
                   "parent_image": image, "supervisor_status": "cancelled", "native_last_observed_state": "running",
                   "native_terminal_cancel_not_fabricated": True, "partial_events_and_cursor_retained": True,
                   "unsettled_reservation_retained": True, "exact_container": container, "independently_confirmed_absent": True}
        (ROOT / "cancel-receipt-durable.json").write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=20)
