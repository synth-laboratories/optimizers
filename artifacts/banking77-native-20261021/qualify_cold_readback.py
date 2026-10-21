"""Read original completed native operations from fresh networkless processes."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "eval-home/runs/banking77-native-20261021"
image = json.loads((ROOT / "build-final/artifact.json").read_text())["output"]["artifact"]["digest"]
program = r'''
import json,sys,os
os.environ["SYNTH_BANKING77_SOURCE"]="hf"
os.environ["SYNTH_BANKING77_CACHE"]="/app/release/dataset"
from pathlib import Path
from fastapi.testclient import TestClient
from banking77_classify import runtime
from banking77_classify.targets import BANKING77_CLASSIFY
from synth_containers.platform import create_compat_app
from synth_containers.platform.seal import validate_rollout_seal
def forbidden(*args,**kwargs):
    raise RuntimeError("readback_attempted_provider_compute")
runtime._chat_completion=forbidden
app=create_compat_app(BANKING77_CLASSIFY,storage_root=Path("/state"))
client=TestClient(app)
child=sys.argv[1]
status=client.get("/rollouts/"+child)
assert status.status_code==200,status.text
state=status.json()
assert state["status"]=="completed" and state["terminated"]
seal=client.get("/rollouts/"+child+"/trace").json()
validate_rollout_seal(seal)
assert seal["content_digest"]==state["trace"]["content_digest"]
cursor=None
events=[]
while True:
    params={"scoped":True,"limit":3}
    if cursor:params["cursor"]=cursor
    response=client.get("/rollouts/"+child+"/events",params=params)
    assert response.status_code==200,response.text
    page=response.json()
    events.extend(e for e in page["events"] if e.get("sequence") is not None)
    cursor=page["cursor"]["resume"]
    if not page["cursor"]["has_more"]:
        assert page["cursor"]["closed"]
        break
assert [e["sequence"] for e in events]==list(range(1,seal["high_water"]+1))
request=json.loads(Path("/request.json").read_text())
replay=client.post("/rollouts",json=request)
assert replay.status_code==200 and replay.json().get("replayed"),replay.text
assert app.state.platform.step_calls==0 and app.state.platform.start_session_calls==0
print(json.dumps({"child_id":child,"status":state["status"],"trace_digest":seal["content_digest"],
                  "high_water":seal["high_water"],"start_replayed":True,"native_compute_calls":0}))
'''
receipts = []
for index, job in enumerate(sorted(RUN.glob("trials/*/job_result.json"))):
    value = json.loads(job.read_text())
    assert value["valid"], value
    output = job.parent / "output"
    child = json.loads((output / "operation.json").read_text())["child_id"]
    # Constructor-local metadata is allowed to initialize in an owned recovery
    # copy. The original result store remains immutable and is hashed below.
    before = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in output.rglob("*") if p.is_file()}
    for process in (1, 2):
        recovery = ROOT / f"cold-recovery-{index}-{process}"
        shutil.copytree(output / "native-storage", recovery)
        name = f"banking77-native-cold-{index}-{process}-20261021"
        command = ["docker", "run", "--rm", "--name", name, "--network", "none",
                   "--env", "PYTHONDONTWRITEBYTECODE=1", "--mount", f"type=bind,source={recovery},target=/state",
                   "--mount", f"type=bind,source={output / 'native-request.json'},target=/request.json,readonly",
                   "--entrypoint", "python", image, "-c", program, child]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        (ROOT / f"cold-readback-{index}-{process}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0, result.stderr[-2000:]
        receipt = json.loads(result.stdout)
        receipt.update(container_name=name, image=image, network="none", credentials_provided=False)
        receipts.append(receipt)
    assert before == {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in output.rglob("*") if p.is_file()}
assert len(receipts) == 4
names = subprocess.check_output(["docker", "ps", "-a", "--format", "{{.Names}}"], text=True).splitlines()
assert not {r["container_name"] for r in receipts} & set(names)
(ROOT / "cold-readback-receipt.json").write_text(json.dumps({"fresh_native_processes": receipts,
     "original_outputs_unchanged": True, "all_owned_readers_confirmed_absent": True}, indent=2))
print("Four fresh native processes recovered original operations without compute")
