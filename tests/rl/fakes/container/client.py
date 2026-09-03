"""The executor-side driver.

The client reads the declared route table from ``/metadata`` and calls nothing
that is not in it, so a container that omits a mandatory route fails loudly
rather than silently falling back to a guessed path.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from synth_optimizers.contracts.rl_clauses import HANDSHAKE_SCHEMA_VERSION, MANDATORY_CLAUSES
from synth_optimizers.contracts.rl_identity import RolloutReceipt
from synth_optimizers.contracts.rl_records import (
    EvidenceError,
    InferenceCall,
    RewardRecord,
    TrainableEpisode,
    TrainableSegment,
)

from .codecs import (
    inference_call_from_payload,
    reward_record_from_payload,
    rollout_receipt_from_payload,
    segment_from_payload,
    trainable_episode_from_payload,
)
from .config import ContainerError


@dataclass(frozen=True, slots=True)
class AttemptResult:
    """One attempt driven end to end through the declared routes."""

    rollout_id: str
    submit: Mapping[str, Any]
    states: tuple[Mapping[str, Any], ...]
    events: tuple[Mapping[str, Any], ...]
    finalize: Mapping[str, Any]
    trace: Mapping[str, Any]
    artifacts: Mapping[str, Any]
    reward_payload: Mapping[str, Any] | None
    calls: tuple[InferenceCall, ...]
    episodes: tuple[TrainableEpisode, ...]
    #: Foreign-authored spans -- opponent, other instance, verifier, judge --
    #: recorded as untrainable context with their author named.
    context_segments: tuple[TrainableSegment, ...] = ()

    @property
    def trainable_calls(self) -> tuple[InferenceCall, ...]:
        return tuple(call for call in self.calls if call.trainable)

    @property
    def receipt(self) -> RolloutReceipt:
        """The rollout receipt from the terminal transition."""

        payload = self.finalize.get("receipt")
        if payload is None:
            raise EvidenceError(f"rollout {self.rollout_id} sealed no receipt")
        return rollout_receipt_from_payload(payload)

    def context_segments_by(self, author_kind: str) -> tuple[TrainableSegment, ...]:
        return tuple(
            segment for segment in self.context_segments if segment.author_kind == author_kind
        )

    @property
    def reward(self) -> RewardRecord:
        if self.reward_payload is None:
            raise EvidenceError(
                f"rollout {self.rollout_id} produced no reward record; absent is not zero"
            )
        return reward_record_from_payload(self.reward_payload)

    @property
    def trace_digest(self) -> str:
        return str(self.trace.get("trace_digest") or "")

    def calls_for(self, agent_instance_id: str) -> tuple[InferenceCall, ...]:
        return tuple(
            call for call in self.calls if call.agent_instance_id == agent_instance_id
        )

    def episode_for(self, agent_instance_id: str) -> TrainableEpisode:
        for episode in self.episodes:
            if episode.agent_instance_id == agent_instance_id:
                return episode
        raise EvidenceError(
            f"rollout {self.rollout_id} has no trajectory for instance {agent_instance_id!r}"
        )


class ContainerClient:
    """Calls only routes the container declared in ``/metadata``."""

    def __init__(self, base_url: str, *, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.routes: dict[str, str] = {}
        self.handshake_id: str = ""
        self.agreement_digest: str = ""
        self._load_routes()

    # -- transport ------------------------------------------------------ #

    def request(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> tuple[int, Any]:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as error:
            payload = json.loads(error.read() or b"{}")
            return error.code, payload

    def call(
        self, method: str, path: str, body: Mapping[str, Any] | None = None
    ) -> Any:
        status, payload = self.request(method, path, body)
        if status >= 400:
            raise ContainerError(status, payload if isinstance(payload, dict) else {})
        return payload

    def route(self, key: str, **params: Any) -> str:
        if key not in self.routes:
            raise ContainerError(404, {"error": "route_not_declared", "reason": key})
        return self.routes[key].format(**params)

    def _load_routes(self) -> None:
        payload = self.call("GET", "/metadata")
        contract = payload["metadata"]["optimizer_contracts"]["cispo"]
        self.contract_version = contract["version"]
        self.routes = {key: value for key, value in contract.items() if key.endswith("_route")}

    # -- declared routes ------------------------------------------------ #

    def health(self) -> Mapping[str, Any]:
        return self.call("GET", self.route("health_route"))

    def capabilities(self) -> Mapping[str, Any]:
        return self.call("GET", self.route("capabilities_route"))

    def handshake(self, document: Mapping[str, Any]) -> Mapping[str, Any]:
        payload = self.call("POST", self.route("handshake_route"), document)
        if payload.get("accepted"):
            self.handshake_id = payload["handshake_id"]
            self.agreement_digest = payload["agreement_digest"]
        return payload

    def taskset(self) -> Mapping[str, Any]:
        return self.call("GET", self.route("taskset_route"))

    def taskset_tasks(self, ids: Iterable[str]) -> Mapping[str, Any]:
        query = urllib.parse.urlencode({"ids": ",".join(ids)})
        return self.call("GET", f"{self.route('taskset_tasks_route')}?{query}")

    def topology(self, topology_id: str) -> Mapping[str, Any]:
        return self.call("GET", self.route("topology_route", topology_id=topology_id))

    def bind_policy(self, **body: Any) -> Mapping[str, Any]:
        return self.call("POST", self.route("policy_bind_route"), body)

    def bind_policy_set(self, **body: Any) -> Mapping[str, Any]:
        return self.call("POST", self.route("policy_set_bind_route"), body)

    def submit(self, **body: Any) -> Mapping[str, Any]:
        body.setdefault("handshake_id", self.handshake_id)
        body.setdefault("agreement_digest", self.agreement_digest)
        return self.call("POST", self.route("rollout_route"), body)

    def state(self, rollout_id: str) -> Mapping[str, Any]:
        return self.call("GET", self.route("rollout_state_route", rollout_id=rollout_id))

    def events(self, rollout_id: str, cursor: int = 0) -> Mapping[str, Any]:
        path = self.route("rollout_events_route", rollout_id=rollout_id)
        return self.call("GET", f"{path}?cursor={cursor}")

    def renew(self, rollout_id: str) -> Mapping[str, Any]:
        return self.call("POST", self.route("rollout_renew_route", rollout_id=rollout_id), {})

    def finalize(self, rollout_id: str) -> Mapping[str, Any]:
        return self.call("POST", self.route("rollout_finalize_route", rollout_id=rollout_id), {})

    def terminate(self, rollout_id: str, reason: str = "cancelled") -> Mapping[str, Any]:
        path = self.route("rollout_terminate_route", rollout_id=rollout_id)
        return self.call("POST", path, {"reason": reason})

    def trace(self, rollout_id: str) -> Mapping[str, Any]:
        payload = self.call("GET", self.route("trace_route", rollout_id=rollout_id))
        if payload.get("inline"):
            return payload
        body = self.call("GET", str(payload["trace_ref"]))
        if body["trace_digest"] != payload["trace_digest"]:
            raise EvidenceError("trace reference digest does not match its inventory entry")
        return body

    def artifacts(self, rollout_id: str) -> Mapping[str, Any]:
        return self.call("GET", self.route("artifacts_route", rollout_id=rollout_id))

    def reward(self, rollout_id: str) -> tuple[int, Any]:
        return self.request("GET", f"{self.route('reward_route')}?rollout_id={rollout_id}")

    # -- convenience drivers -------------------------------------------- #

    def task_ids(self) -> tuple[str, ...]:
        rows = self.taskset_tasks(())["rows"]
        return tuple(str(row["task_id"]) for row in rows)

    def requirement_document(self, **overrides: Any) -> dict[str, Any]:
        capabilities = self.capabilities()["capabilities"]
        document: dict[str, Any] = {
            "schema_version": HANDSHAKE_SCHEMA_VERSION,
            "run_id": "run_fake",
            "optimizer": {"name": "synth_optimizers.cispo", "version": "0.0.0-test"},
            "policy": {
                "provider": "fake",
                "model_id": capabilities["renderer_profile"]["tokenizer_id"],
                "transport": "message_in_capture_out",
            },
            "renderer_profile": {
                "profile_id": capabilities["renderer_profile"]["profile_id"],
                "config_digest": capabilities["renderer_profile"]["config_digest"],
                "tokenizer_digest": capabilities["renderer_profile"]["tokenizer_digest"],
            },
            "requirements": list(MANDATORY_CLAUSES),
            "topology": {
                "expected_topology_id": capabilities["topology"]["topology_id"],
                "trainable_teams": [
                    team["team_id"]
                    for team in capabilities["topology"]["teams"]
                    if team["trainable"]
                ],
                "partial_roster": capabilities["topology"]["partial_roster_disposition"],
            },
            "run_plan": {
                "group_size": 2,
                "groups_per_step": 1,
                "max_execution_slots": 2,
                "maximum_policy_lag": 1,
                "target_train_updates": 1,
                "expected_horizon_seconds": capabilities["horizon"]["value"],
            },
            "taskset": {"taskset_id": self.taskset()["taskset_id"], "split": "train"},
            "clock": {"executor_time": "2026-09-02T12:00:00+00:00"},
        }
        for key, value in overrides.items():
            if isinstance(value, Mapping) and isinstance(document.get(key), dict):
                document[key] = {**document[key], **value}
            else:
                document[key] = value
        return document

    def preflight(self, **overrides: Any) -> Mapping[str, Any]:
        """Health, metadata, capabilities, handshake -- in the declared order."""

        self.health()
        self.capabilities()
        return self.handshake(self.requirement_document(**overrides))

    def negotiate(self, **overrides: Any) -> tuple[Mapping[str, Any], ...]:
        """Preflight, then re-handshake once per degradation. Never assume.

        A degraded concurrency clause is satisfied by *lowering the run plan*
        and re-handshaking; any other degradation is re-handshaked with the
        fallback named explicitly in ``accept_degraded``. Returns every
        handshake exchange in order, so a receipt can name them all.
        """

        exchanges = [self.preflight(**overrides)]
        latest = exchanges[-1]
        if latest.get("accepted"):
            return tuple(exchanges)
        if latest["rejected_mandatory_clauses"]:
            return tuple(exchanges)
        degraded = list(latest["unaccepted_degraded_clauses"])
        lowered = dict(overrides)
        if "lifecycle.concurrency" in degraded:
            ceiling = int(latest["obligations"]["max_concurrency"])
            plan = dict(lowered.get("run_plan") or {})
            plan.update({"max_execution_slots": ceiling, "group_size": ceiling})
            lowered["run_plan"] = plan
            degraded = [clause for clause in degraded if clause != "lifecycle.concurrency"]
        lowered["accept_degraded"] = degraded
        exchanges.append(self.handshake(self.requirement_document(**lowered)))
        return tuple(exchanges)

    def bind(
        self, *, probe: bool = False, policy_revision: int = 0, **extra: Any
    ) -> Mapping[str, Any]:
        """Bind one policy, or a whole roster when the topology is joint."""

        capabilities = self.capabilities()["capabilities"]
        instances = capabilities["topology"]["agent_instances"]
        kind = "probe" if probe else "trainable"
        if len(instances) < 2:
            return self.bind_policy(kind=kind, policy_revision=policy_revision, **extra)
        return self.bind_policy_set(
            kind=kind,
            policy_revision=policy_revision,
            policy_set_revision_id=extra.pop("policy_set_revision_id", "policy-set-1"),
            bindings=[
                {
                    "agent_instance_id": instance["agent_instance_id"],
                    "policy_ref": instance["policy_ref"] or f"ckpt::rev{policy_revision}",
                }
                for instance in instances
            ],
            **extra,
        )

    def run_attempt(
        self,
        *,
        task_id: str,
        idempotency_key: str | None = None,
        correlation: Mapping[str, Any] | None = None,
        binding: Mapping[str, Any] | None = None,
        probe: bool = False,
        polls: int = 4,
        renew: bool = True,
    ) -> AttemptResult:
        """Submit, poll, renew, finalize, read trace and reward. One attempt."""

        if not self.handshake_id:
            self.preflight()
        record = binding or self.bind(probe=probe)
        key = idempotency_key or f"key::{task_id}::{record['config_id']}"
        payload = dict(correlation or {})
        payload.setdefault("run_id", "run_fake")
        payload.setdefault("group_id", "group_fake")
        payload.setdefault("sample_index", 0)
        payload.setdefault("seed", 7)
        payload.setdefault("policy_revision", int(record.get("policy_revision") or 0))
        if "policy_set_revision_id" in record:
            payload.setdefault("policy_set_revision", record["policy_set_revision_id"])
        submit = self.submit(
            task_id=task_id,
            idempotency_key=key,
            policy_config_id=record["config_id"],
            correlation=payload,
        )
        rollout_id = str(submit["rollout_id"])
        if renew:
            self.renew(rollout_id)
        seen: list[Mapping[str, Any]] = []
        for _ in range(polls):
            snapshot = self.state(rollout_id)
            seen.append(snapshot)
            if snapshot["state"] in {"scored", "awaiting_score"} or snapshot["terminal"]:
                break
        finalize = self.finalize(rollout_id)
        trace = self.trace(rollout_id)
        artifacts = self.artifacts(rollout_id)
        status, reward_payload = self.reward(rollout_id)
        events = self.events(rollout_id)["events"]
        return AttemptResult(
            rollout_id=rollout_id,
            submit=submit,
            states=tuple(seen),
            events=tuple(events),
            finalize=finalize,
            trace=trace,
            artifacts=artifacts,
            reward_payload=reward_payload if status == 200 else None,
            calls=tuple(inference_call_from_payload(row) for row in trace["calls"]),
            episodes=tuple(
                trainable_episode_from_payload(row) for row in trace.get("episodes") or ()
            ),
            context_segments=tuple(
                segment_from_payload(row) for row in trace.get("context_segments") or ()
            ),
        )
