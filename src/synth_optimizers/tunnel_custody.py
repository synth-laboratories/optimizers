"""Caller custody protocol; canonical reads resolve retained uncertain effects.

See backend notes/specifications/tanha/current/systems/tunnels/v2_catalog.md.
A local receipt does not confer authority and never triggers effect replay.
"""

from __future__ import annotations

import re
import hashlib
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from .tunnel_custody_journal import FileLeaseCustodyJournal, TunnelCustodyInDoubt
from .tunnels import TunnelError


class CallerLeaseCustody:
    def __init__(self, client, journal: FileLeaseCustodyJournal, receipt: dict):
        self.client = client
        self.journal = journal
        self.receipt = receipt
        self.connector_path = f"/api/v1/synthtunnel/v2/connectors/{receipt['connector']}"
        self.lease_path = f"{self.connector_path}/leases/{receipt['lease']}"

    def save(self, **changes):
        self.receipt = self.journal.save({**self.receipt, **changes})

    def lookup(self, path: str, phase: str) -> dict:
        try:
            result = self.client._json_request("GET", path)
        except Exception as error:
            raise TunnelCustodyInDoubt(self.journal.path, phase) from error
        if not isinstance(result, dict):
            raise TunnelError("custody lookup response invalid")
        return result

    def command(self, action: str) -> dict:
        request = self.receipt[action + "_request"]
        result = self.lookup(f"{self.connector_path}/commands/{request['command_id']}", action)
        if (
            result.get("command") != request["command_id"]
            or result.get("connector") != self.receipt["connector"]
        ):
            raise TunnelError("custody command receipt identity mismatch")
        return result

    def current(self) -> dict:
        result = self.lookup(self.lease_path, "lease_lookup")
        if (
            result.get("lease") != self.receipt["lease"]
            or result.get("connector") != self.receipt["connector"]
        ):
            raise TunnelError("custody lease receipt identity mismatch")
        if type(result.get("revision")) is not int or result["revision"] < 1:
            raise TunnelError("custody lease receipt revision invalid")
        return result

    def settle_grant(self):
        if self.receipt["phase"] == "grant_in_doubt":
            self.command("grant")
            current = self.current()
            if current.get("owner_binding") != self.receipt["grant_request"]["owner_binding"]:
                raise TunnelError("grant caller custody mismatch")
            self.save(phase="granted")
        current = self.current()
        if current.get("policy") != "granted" or datetime.fromisoformat(
            current["expires_at"]
        ) <= datetime.now(UTC):
            raise TunnelError("shared lease authority expired or revoked")
        return current

    def handoff_status(self) -> dict:
        request = self.receipt.get("offer_request")
        if request is None:
            raise TunnelError("shared lease has no prepared handoff")
        result = self.lookup(f"{self.lease_path}/handoffs/{request['offer']}", "offer")
        expected = {
            "offer_id": request["offer"],
            "connector_id": self.receipt["connector"],
            "lease_id": self.receipt["lease"],
            "job_id": request["job_id"],
            "caller_binding": request["caller_binding"],
            "expected_lease_revision": request["expected_lease_revision"],
        }
        if any(result.get(key) != value for key, value in expected.items()):
            raise TunnelError("custody handoff receipt binding mismatch")
        return result

    def prepare_handoff(self, job: str | None) -> dict:
        if self.receipt["phase"] in {"closed", "detached", "close_in_doubt"}:
            raise TunnelError("shared tunnel caller custody is closed or unresolved")
        saved = self.receipt.get("offer_request")
        job = job or (saved["job_id"] if saved else "tunnel_" + uuid4().hex)
        if not isinstance(job, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", job):
            raise TunnelError("invalid shared tunnel job identity")
        if saved:
            if saved["job_id"] != job:
                raise TunnelError("shared lease already bound to another job")
            result = self.handoff_status()  # Never repeat an uncertain POST.
        else:
            current = self.settle_grant()
            binding = self.receipt["grant_request"]["owner_binding"]
            if current["owner_binding"] != binding or current["revision"] != 1:
                raise TunnelError("shared lease caller custody is stale")
            request = {
                "offer": str(uuid4()),
                "expected_lease_revision": 1,
                "caller_binding": binding,
                "job_id": job,
                "deadline_at": (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
            }
            self.save(phase="offer_in_doubt", offer_request=request)
            try:
                self.client._json_request("POST", self.lease_path + "/handoffs", request)
            except Exception as error:
                raise TunnelCustodyInDoubt(self.journal.path, "offer") from error
            result = self.handoff_status()
        if result.get("state") not in {"offered", "accepted"}:
            raise TunnelError("shared tunnel handoff is no longer live")
        self.save(phase="offered", offer_receipt=result)
        return result

    def close(self):
        if self.receipt["phase"] in {"closed", "detached"}:
            return
        if self.receipt.get("offer_request"):
            handoff = self.handoff_status()
            if handoff.get("state") == "accepted":
                current = self.current()
                if (
                    current["owner_binding"] != handoff["offer_id"]
                    or current["revision"] < handoff["lease_revision"]
                    or not handoff.get("executor_instance")
                ):
                    raise TunnelError("accepted custody lineage mismatch")
                self.save(phase="detached", offer_receipt=handoff)
                return
        if self.receipt.get("close_request"):
            self.command("close")
            current = self.current()
            if current["policy"] != "revoked":
                raise TunnelCustodyInDoubt(self.journal.path, "close")
            self.save(phase="closed")
            return
        current = self.settle_grant()
        if (
            current["owner_binding"] != self.receipt["grant_request"]["owner_binding"]
            or current["revision"] != 1
        ):
            raise TunnelError("shared lease caller custody is stale")
        status = self.lookup(self.connector_path, "connector_lookup")
        request = {
            "command_id": str(uuid4()),
            "expected_revision": status["revision"],
            "deadline_at": (datetime.now(UTC) + timedelta(seconds=30)).isoformat(),
            "expected_lease_revision": 1,
        }
        self.save(phase="close_in_doubt", close_request=request)
        try:
            self.client._json_request("POST", self.lease_path + "/revoke", request)
        except Exception as error:
            # A concurrent acceptance is resolved on the next close by lookup.
            raise TunnelCustodyInDoubt(self.journal.path, "close") from error
        self.command("close")
        self.save(phase="closed")

    def submit_once(self, payload: dict) -> dict:
        offer = self.receipt.get("offer_request")
        if offer is None or payload.get("run_id") != offer["job_id"]:
            raise TunnelError("shared tunnel submit lacks exact prepared job custody")
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        saved = self.receipt.get("submission")
        if saved:
            if saved["digest"] != digest:
                raise TunnelError("shared tunnel submission identity reused with different input")
            if saved.get("response"):
                return saved["response"]
            raise TunnelCustodyInDoubt(self.journal.path, "submission")
        self.save(submission={"job": offer["job_id"], "digest": digest, "state": "in_doubt"})
        try:
            response = self.client._json_request("POST", "/api/v1/optimizers/runs", payload)
        except Exception as error:
            raise TunnelCustodyInDoubt(self.journal.path, "submission") from error
        if response.get("run_id") != offer["job_id"]:
            raise TunnelCustodyInDoubt(self.journal.path, "submission_identity")
        retained = {
            key: response[key]
            for key in (
                "run_id",
                "status",
                "events_url",
                "status_url",
                "artifact_base_url",
                "algorithm",
                "implementation",
                "implementation_version",
                "attempt_id",
            )
            if key in response
        }
        self.save(
            submission={
                "job": offer["job_id"],
                "digest": digest,
                "state": "acknowledged",
                "response": retained,
            }
        )
        return response

    def submission_status(self) -> dict:
        submission = self.receipt.get("submission")
        if submission is None:
            raise TunnelError("shared tunnel has no retained submission")
        return self.lookup("/api/v1/optimizers/runs/" + submission["job"], "submission")
