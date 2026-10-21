"""Re-derive local witness receipts and exports; never starts provider compute."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]
RUN_ID = "healthbench-eval-accepted-20261021"
RUN = ROOT / "eval-home-accepted/runs" / RUN_ID


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def hash_file(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


descriptor = read(ROOT / "build-accepted/artifact.json")
image = descriptor["output"]["artifact"]["digest"]
rows, names = [], []
for job in sorted(RUN.glob("trials/*/job_result.json")):
    result = read(job)
    output = job.parent / "output"
    trial_id = result["trial_id"]
    assert result["valid"], result
    names.append("synth-eval-" + trial_id[:40] + "-" + hashlib.sha256(trial_id.encode()).hexdigest()[:12])
    calls = [json.loads(line) for line in (output / "provider.jsonl").read_text().splitlines()]
    completed = [call for call in calls if call["state"] == "completed"]
    assert len(completed) * 2 == len(calls)
    assert all(call["trainable"] is False for call in completed if call["role"] == "judge")
    agreement = read(output / "agreement.json")
    assert agreement["reference_protocol_equivalent"] is False
    assert agreement["info"]["target_id"] == "healthbench_chat"
    assert read(output / "cleanup.json")["http_server_stopped"]
    rows.append({"trial_id": trial_id, "valid": True, "reward": result["metrics"]["reward"],
                 "capability_digest": agreement["capability_digest"], "operation": read(output / "operation.json"),
                 "answer_calls": sum(call["role"] == "answer" for call in completed),
                 "judge_calls": sum(call["role"] == "judge" for call in completed)})
assert len(rows) == 2
existing = subprocess.check_output(["docker", "ps", "-a", "--format", "{{.Names}}"], text=True).splitlines()
assert not set(names) & set(existing)
save(ROOT / "cleanup-accepted.json", {"exact_owned_containers": names,
     "independently_confirmed_absent": True, "http_services_stopped": True,
     "cleanup_complete": True, "foreign_resources_untouched": True,
     "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat()})

# Two fresh installed readers and terminal reconnect must preserve the evidence.
environment = os.environ.copy()
environment.pop("PYTHONPATH", None)
before = {str(p.relative_to(RUN)): hash_file(p) for p in RUN.rglob("*") if p.is_file()}
for number in (1, 2):
    with (ROOT / f"accepted-readback-process{number}.json").open("w") as stream:
        subprocess.run([str(ROOT / "venv/bin/python"), "-m", "synth_optimizers.eval.healthbench", "read",
                        "--home", str(ROOT / "eval-home-accepted"), "--run-id", RUN_ID],
                       cwd=PROJECT.parent.parent, env=environment, stdout=stream, check=True)
with (ROOT / "accepted-reconnect.json").open("w") as stream:
    subprocess.run(read(ROOT / "accepted-command.json")["argv"], cwd=PROJECT.parent.parent,
                   env=environment, stdout=stream, check=True)
assert before == {str(p.relative_to(RUN)): hash_file(p) for p in RUN.rglob("*") if p.is_file()}
save(ROOT / "reconnect-accepted.json", {"two_process_readback": True, "artifact_digests_verified": True,
     "original_files_unchanged": len(before), "duplicate_compute": 0, "run_id": RUN_ID})

exports = [ROOT / "healthbench2-eval-1.0.0-linux-arm64-accepted.tar",
           ROOT / "healthbench2-eval-1.0.0-startup.tar.gz",
           ROOT / "installed-optimizer-development-snapshot.zip"]
if not exports[0].exists():
    subprocess.run(["docker", "save", "--output", str(exports[0]), image], check=True)
if not exports[1].exists():
    with tarfile.open(exports[1], "w:gz") as archive:
        archive.add(ROOT / "release-accepted", arcname="healthbench2-eval-1.0.0")
installed = ROOT / "venv/lib/python3.12/site-packages/synth_optimizers"
files = {str(p.relative_to(installed)): hash_file(p) for p in installed.rglob("*")
         if p.is_file() and "__pycache__" not in p.parts}
save(ROOT / "installed-accepted.json", {"kind": "development_package_install",
     "host_profile": "macos-arm64-python3.12", "native_build": False,
     "native_used_for_evaluation": False, "files": files})
if not exports[2].exists():
    with zipfile.ZipFile(exports[2], "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(installed / name, "synth_optimizers/" + name)
charges = read(ROOT / "provider-charges-accepted.json")
assert charges["charge_final"] and len(charges["generations"]) == 31
sources = {name: subprocess.check_output(["git", "-C", str(PROJECT.parent / name), "rev-parse", "HEAD"], text=True).strip()
           for name in ("optimizers", "evals")}
if (ROOT / "cohort.json").exists():
    retained = read(ROOT / "cohort.json")
    sources = retained["sources"]
    for artifact in retained["exports"]:
        assert hash_file(ROOT / artifact["path"]) == artifact["digest"]
cohort = {"profile": "local-linux-arm64-command-http-adapter", "image": image,
          "build_input_digest": descriptor["plan"]["input_digest"],
          "protocol_digest": rows[0]["operation"]["protocol_digest"], "run_id": RUN_ID,
          "trials": rows, "sources": sources, "charge_final": True,
          "provider_reported_usd": charges["total_cost_usd"], "cleanup_complete": True,
          "scope": "execution_support_only", "final_panel_evaluated": False,
          "proposer_calls": 0, "reference_protocol_equivalent": False,
          "exports": [{"path": p.name, "digest": hash_file(p), "bytes": p.stat().st_size} for p in exports]}
cohort_digest = "sha256:" + hashlib.sha256(json.dumps(cohort, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
save(ROOT / "cohort.json", {"cohort_digest": cohort_digest, **cohort})
print(json.dumps({"cohort": cohort_digest, "provider_reported_usd": charges["total_cost_usd"], "image": image}))
