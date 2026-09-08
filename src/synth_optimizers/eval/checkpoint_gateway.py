"""Eval-only immutable sampler binding over the shared sampling gateway."""
from dataclasses import dataclass, asdict
import secrets

from ..contracts.rl_records import RendererProfile
from ..rl.gateway import SamplerGatewayService, GatewayServer
from ..rl.plane import build_renderer
from ..rl.ports import PolicyRevision
from ..runtime.jobs import digest_payload


@dataclass(frozen=True)
class EvaluationPin:
    # Evaluation needs no optimizer loss/logprob handshake or training group.
    group_id: str
    run_id: str
    behavior_fingerprint: str
    policy_revision: int
    policy_revision_id: str
    policy_kind: str = "immutable_checkpoint"
    wire_api: str = "chat_completions"
    sampling_transport: str = "message_in_capture_out"
    policy_set_revision_id: str | None = None

    @property
    def pin_digest(self):
        return digest_payload(asdict(self))


class EvaluationGatewayService(SamplerGatewayService):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.provider_failures = []

    def handle(self, proxy_request_id, payload, **kwargs):
        route = self._locked_route(proxy_request_id)
        requested = payload.get("policy_snapshot_id")
        if requested is not None and requested != route.revision.checkpoint_id:
            raise ValueError("requested checkpoint differs from immutable route binding")
        try:
            result = super().handle(proxy_request_id, payload, **kwargs)
        except Exception as exc:
            from ..providers.protocols import ProviderError
            if isinstance(exc, ProviderError):
                self.provider_failures.append(exc)
            raise
        return {**result, "synth": {"policy_snapshot_id": route.revision.checkpoint_id,
                                    "proxy_request_ids": [proxy_request_id],
                                    "sampler_reference": route.revision.sampler_reference}}


class CheckpointGateway:
    def __init__(self, provider, checkpoint, *, run_id, renderer_profile, bind_host="127.0.0.1",
                 advertised_host="127.0.0.1", ttl_seconds=600):
        self.checkpoint, self.run_id = checkpoint, run_id
        profile = RendererProfile(**renderer_profile)
        renderer = build_renderer(provider, profile, wire_api="chat_completions")
        self.gateway = EvaluationGatewayService(renderer, provider, credential_salt=secrets.token_hex(32),
                                             origin_ttl_seconds=ttl_seconds)
        self.server = GatewayServer(self.gateway, host=bind_host)
        self.advertised_host = advertised_host
        identity = digest_payload({"checkpoint": checkpoint, "renderer": renderer_profile})
        self.revision = PolicyRevision(
            revision=int(checkpoint["step"]), revision_id=checkpoint["checkpoint_id"],
            checkpoint_id=checkpoint["checkpoint_id"], parameter_group_id=run_id,
            sampler_reference=checkpoint["provider_reference"], behavior_fingerprint=identity,
            metadata={"sampler_digest": checkpoint["digest"]})

    def __enter__(self):
        self.server.start()
        port = self.server.base_url.rsplit(":", 1)[1]
        self.gateway.set_origin_root(f"http://{self.advertised_host}:{port}")
        return self

    def bind(self, trial_id, sample_index=0):
        pin = EvaluationPin(trial_id, self.run_id, self.revision.behavior_fingerprint,
                            self.revision.revision, self.revision.revision_id)
        return self.gateway.bind(self.revision, pin=pin, sample_index=sample_index,
                                 proxy_request_id=trial_id)

    def evidence(self, trial_id):
        from ..runtime.operations import encode
        calls = self.gateway.calls(trial_id)
        return {"schema_version": "eval.checkpoint_binding.v1", "checkpoint": self.checkpoint,
                "actual_sampler_reference": self.revision.sampler_reference,
                "verification": "gateway_immutable_reference", "calls": [encode(call) for call in calls]}

    def __exit__(self, *exc):
        self.server.close()
