"""Versioned schemas for the local `eval` algorithm.

`eval` scores an immutable set of policy candidates against any container that
conforms to `eval.target.v1`.  Every type here is a wire contract shared with
Workshop and with target containers, so each one carries an explicit
`schema_version` and refuses partially-formed input rather than defaulting it.

Nothing in this module imports or executes candidate code.  Candidates are
opaque content-addressed directories that only ever reach a container mount.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

EVAL_ALGORITHM_ID = "eval"
EVAL_ALGORITHM_VERSION = "1"

TARGET_MANIFEST_SCHEMA = "eval.target.v1"
POLICY_CANDIDATE_SCHEMA = "optimizer.policy-candidate.v1"
CANDIDATE_SET_SCHEMA = "optimizer.policy-candidate-set.v1"
TRIAL_MANIFEST_SCHEMA = "eval.trial.v1"
CONTAINER_RESULT_SCHEMA = "eval.container-result.v1"
RUN_MANIFEST_SCHEMA = "eval.run-manifest.v1"
SEED_LEDGER_SCHEMA = "eval.seed-ledger.v1"
TRIAL_RECORD_SCHEMA = "eval.trial-result.v1"
SCORECARD_SCHEMA = "eval.scorecard.v1"
SELECTION_SCHEMA = "eval.selection.v1"
WORKER_MANIFEST_SCHEMA = "eval.worker-manifest.v1"
WORKER_EVENT_SCHEMA = "eval.worker-event.v1"

#: Selection outcomes.  `completed` orchestration never implies a winner.
SELECTION_STATUSES = ("promoted", "no_champion", "inconclusive", "invalid_evidence")
#: Container-reported rig health.  `failed` means the trial was not evaluated.
CONTAINER_STATUSES = ("evaluated", "failed")
#: Container-reported outcome of the policy it did evaluate.
BENCHMARK_STATUSES = ("passed", "failed", "invalid")
#: Runner-side terminal states for one trial.
TRIAL_STATUSES = ("evaluated", "failed", "timeout", "cancelled")

#: A trained MLX LoRA adapter, or the adapter-free base it is measured against.
#: The adapter bytes live *in* the candidate, so `artifact_digest` is the
#: digest of the adapter itself and "base vs checkpoint-20 vs final" is one
#: content-addressed `CandidateSet` with a declared baseline scored on shared
#: seeds — a paired difference, not two runs compared by hand.
MLX_LORA_POLICY_KIND = "mlx-lora.v1"
MLX_LORA_POLICY_SCHEMA = "eval.mlx-lora-policy.v1"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

#: The only hosts a cleartext `http://` route may name.  Every one of them
#: resolves to this machine — the loopback spellings, plus the name Docker
#: gives the host from inside a bridge network — so a plaintext bearer token
#: never crosses a wire.  Any other `http://` origin is refused, including
#: private ranges: "it is on my LAN" is not the same claim as "it cannot
#: leave this machine", and widening this set is how a candidate reaches an
#: endpoint the product never named.
LOCAL_HTTP_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "host.docker.internal"})

#: Both OpenAI API families are first-class.  A route that names neither is
#: not a route this product knows how to bill, meter, or proxy.
ROUTE_PATH_SUFFIXES = ("/v1/chat/completions", "/v1/responses")


class EvalContractError(ValueError):
    """Input that does not satisfy an `eval` schema."""


def _object(value: Any, *, context: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    raise EvalContractError(f"{context} must be a JSON object")


def _text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvalContractError(f"{field_name} must be a non-empty string")
    return value.strip()


def _identifier(value: Any, *, field_name: str) -> str:
    text = _text(value, field_name=field_name)
    if not _ID_PATTERN.match(text):
        raise EvalContractError(f"{field_name} is not a valid identifier: {text!r}")
    return text


def _digest(value: Any, *, field_name: str) -> str:
    text = _text(value, field_name=field_name)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", text):
        raise EvalContractError(f"{field_name} must be sha256:<64 hex chars>")
    return text


def _positive_int(value: Any, *, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise EvalContractError(f"{field_name} must be a positive integer")
    return value


def _seed_tuple(value: Any, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvalContractError(f"{field_name} must be a list of integers")
    seeds: list[int] = []
    for item in value:
        if not isinstance(item, int) or isinstance(item, bool):
            raise EvalContractError(f"{field_name} must contain only integers")
        seeds.append(item)
    if len(set(seeds)) != len(seeds):
        raise EvalContractError(f"{field_name} must not repeat a seed")
    return tuple(seeds)


def _text_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise EvalContractError(f"{field_name} must be a list of strings")
    return tuple(_identifier(item, field_name=field_name) for item in value)


def canonical_json(value: Any) -> str:
    """Stable JSON used wherever a digest must be reproducible."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest_of(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_of_tree(root: Path) -> str:
    """Content address a candidate directory: sorted relative path + bytes."""

    if root.is_file():
        entries = [(root.name, root)]
    else:
        entries = sorted(
            ((str(p.relative_to(root)), p) for p in root.rglob("*") if p.is_file()),
            key=lambda item: item[0],
        )
    hasher = hashlib.sha256()
    for relative, path in entries:
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class MlxLoraPolicy:
    """What `policy.json` in an `mlx-lora.v1` candidate directory declares.

    `adapter = false` is the base model entering the same candidate set as an
    adapter-free member, which is what makes the base-vs-LoRA question a paired
    difference rather than two runs compared by hand.  It is declared rather
    than inferred from which files happen to be present: a checkpoint whose
    adapter failed to copy would otherwise be silently scored as its own
    baseline, and the whole comparison would report a lift of zero.
    """

    base_model: str
    adapter: bool
    chat_template_digest: str
    thinking_mode: str
    rank: int | None

    @classmethod
    def from_mapping(cls, value: Any) -> MlxLoraPolicy:
        data = _object(value, context="policy.json")
        schema = data.get("schema_version", MLX_LORA_POLICY_SCHEMA)
        if schema != MLX_LORA_POLICY_SCHEMA:
            raise EvalContractError(f"unsupported mlx-lora policy schema {schema!r}")
        adapter = data.get("adapter")
        if not isinstance(adapter, bool):
            raise EvalContractError("policy.json must declare adapter as true or false")
        thinking_mode = _text(data.get("thinking_mode"), field_name="policy.thinking_mode")
        if thinking_mode not in {"off", "on"}:
            raise EvalContractError("policy.thinking_mode must be off or on")
        rank = data.get("rank")
        if adapter:
            rank = _positive_int(rank, field_name="policy.rank")
        elif rank is not None:
            raise EvalContractError("an adapter-free policy must not declare a rank")
        return cls(
            base_model=_text(data.get("base_model"), field_name="policy.base_model"),
            adapter=adapter,
            chat_template_digest=_digest(
                data.get("chat_template_digest"), field_name="policy.chat_template_digest"
            ),
            thinking_mode=thinking_mode,
            rank=rank,
        )

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": MLX_LORA_POLICY_SCHEMA,
            "base_model": self.base_model,
            "adapter": self.adapter,
            "chat_template_digest": self.chat_template_digest,
            "thinking_mode": self.thinking_mode,
        }
        if self.rank is not None:
            payload["rank"] = self.rank
        return payload


#: Files an adapter-carrying `mlx-lora.v1` candidate must contain.
MLX_LORA_ADAPTER_FILES = ("adapter_config.json", "adapters.safetensors")


def read_mlx_lora_policy(root: Path) -> MlxLoraPolicy:
    """Refuse an `mlx-lora.v1` candidate directory that is not what it claims.

    Checked wherever a candidate is staged *and* again before a run starts, so
    a candidate set assembled by Workshop is held to the same shape as one the
    reference staging path produced.
    """

    if not root.is_dir():
        raise EvalContractError(f"{MLX_LORA_POLICY_KIND} candidate must be a directory: {root}")
    manifest = root / "policy.json"
    if not manifest.is_file():
        raise EvalContractError(f"{MLX_LORA_POLICY_KIND} candidate is missing policy.json")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvalContractError(f"{manifest} is not readable JSON") from exc
    policy = MlxLoraPolicy.from_mapping(payload)
    present = [name for name in MLX_LORA_ADAPTER_FILES if (root / name).is_file()]
    if policy.adapter:
        missing = [name for name in MLX_LORA_ADAPTER_FILES if name not in present]
        if missing:
            raise EvalContractError(
                f"{MLX_LORA_POLICY_KIND} candidate is missing {', '.join(missing)}"
            )
    elif present:
        raise EvalContractError(
            f"{MLX_LORA_POLICY_KIND} candidate declares adapter=false but ships "
            f"{', '.join(present)}"
        )
    return policy


@dataclass(frozen=True, slots=True)
class MetricSpec:
    id: str
    direction: str

    @classmethod
    def from_mapping(cls, value: Any) -> MetricSpec:
        data = _object(value, context="metric")
        direction = _text(data.get("direction"), field_name="metric.direction")
        if direction not in {"maximize", "minimize"}:
            raise EvalContractError("metric.direction must be maximize or minimize")
        return cls(id=_identifier(data.get("id"), field_name="metric.id"), direction=direction)

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "direction": self.direction}

    def signed(self, value: float) -> float:
        """Higher is always better once a metric is signed."""

        return value if self.direction == "maximize" else -value


@dataclass(frozen=True, slots=True)
class TargetManifest:
    """What a conforming evaluation container promises.

    The manifest lives in the trusted recipe catalog for v1; a container never
    gets to widen its own contract at runtime.
    """

    policy_kinds: tuple[str, ...]
    trial_mode: str
    metrics: tuple[MetricSpec, ...]
    required_gates: tuple[str, ...]
    required_artifacts: tuple[str, ...]
    supports_live_events: bool
    network: str

    @classmethod
    def from_mapping(cls, value: Any) -> TargetManifest:
        data = _object(value, context="target manifest")
        schema = data.get("schema_version", TARGET_MANIFEST_SCHEMA)
        if schema != TARGET_MANIFEST_SCHEMA:
            raise EvalContractError(f"unsupported target manifest schema {schema!r}")
        trial_mode = _text(data.get("trial_mode"), field_name="trial_mode")
        if trial_mode != "one-policy-one-seed":
            raise EvalContractError("v1 targets must declare trial_mode=one-policy-one-seed")
        network = _text(data.get("network", "none"), field_name="network")
        if network not in {"none", "bridge"}:
            raise EvalContractError("target network must be none or bridge")
        metrics = data.get("metrics")
        if not isinstance(metrics, Sequence) or not metrics:
            raise EvalContractError("target manifest must declare at least one metric")
        policy_kinds = _text_tuple(data.get("policy_kinds"), field_name="policy_kinds")
        if not policy_kinds:
            raise EvalContractError("target manifest must declare at least one policy kind")
        return cls(
            policy_kinds=policy_kinds,
            trial_mode=trial_mode,
            metrics=tuple(MetricSpec.from_mapping(item) for item in metrics),
            required_gates=_text_tuple(data.get("required_gates", []), field_name="required_gates"),
            required_artifacts=_text_tuple(
                data.get("required_artifacts", []), field_name="required_artifacts"
            ),
            supports_live_events=bool(data.get("supports_live_events", False)),
            network=network,
        )

    def metric(self, metric_id: str) -> MetricSpec:
        for metric in self.metrics:
            if metric.id == metric_id:
                return metric
        raise EvalContractError(f"target does not declare metric {metric_id!r}")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": TARGET_MANIFEST_SCHEMA,
            "policy_kinds": list(self.policy_kinds),
            "trial_mode": self.trial_mode,
            "metrics": [metric.to_json() for metric in self.metrics],
            "required_gates": list(self.required_gates),
            "required_artifacts": list(self.required_artifacts),
            "supports_live_events": self.supports_live_events,
            "network": self.network,
        }


def _model_route(value: Any) -> str:
    """A route a recipe is allowed to name, normalized and re-rendered.

    `https://` is unrestricted.  `http://` is permitted only for a host that
    cannot be anywhere but this machine (`LOCAL_HTTP_HOSTS`), which is what a
    local inference proxy needs and what nothing else has any business being.
    Userinfo, query strings, and fragments are refused outright: a route is an
    endpoint, and credentials or per-request parameters smuggled into one are
    how a "route" quietly becomes a request the recipe never described.

    Mirrors `_validate_remote_checkpoint_endpoint` in the containers Banking77
    runtime, minus its environment-variable allowlist — the recipe catalog is
    already the allowlist here, and a second one read from the environment
    would be an escape hatch around it.
    """

    text = _text(value, field_name="model.route")
    try:
        parsed = urlsplit(text)
        port = parsed.port
    except ValueError as exc:
        raise EvalContractError(f"model.route is not a valid URL: {text!r}") from exc
    if parsed.scheme not in {"http", "https"}:
        raise EvalContractError("model.route must be an https endpoint, or a local http one")
    if not parsed.hostname:
        raise EvalContractError("model.route must name a host")
    if parsed.username is not None or parsed.password is not None:
        raise EvalContractError("model.route must not carry credentials in its userinfo")
    if parsed.query or parsed.fragment:
        raise EvalContractError("model.route must not carry a query string or fragment")
    if not parsed.path.startswith("/") or ".." in parsed.path.split("/"):
        raise EvalContractError(f"model.route path is not absolute: {text!r}")
    if not parsed.path.endswith(ROUTE_PATH_SUFFIXES):
        raise EvalContractError("model.route must end in " + " or ".join(ROUTE_PATH_SUFFIXES))
    host = parsed.hostname.lower()
    if parsed.scheme == "http" and host not in LOCAL_HTTP_HOSTS:
        raise EvalContractError(
            f"model.route may only use http:// for a local host "
            f"({', '.join(sorted(LOCAL_HTTP_HOSTS))}), got {host!r}"
        )
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc += f":{port}"
    return f"{parsed.scheme}://{netloc}{parsed.path}"


@dataclass(frozen=True, slots=True)
class ModelRoute:
    """One model a recipe permits, with the route and the price of using it.

    Both live in the recipe rather than in the candidate: an agent picks a model
    from an allowlist, it does not get to name an endpoint, and the dollars a
    run reports are computed from a rate the product declared and dated, not
    from something a container made up.
    """

    id: str
    request_model: str | None
    route: str
    secret: str
    efforts: tuple[str, ...]
    usd_per_1m_input: float
    usd_per_1m_output: float
    usd_per_1m_cached_input: float
    price_source: str
    price_as_of: str

    @classmethod
    def from_mapping(cls, value: Any) -> ModelRoute:
        data = _object(value, context="model route")
        route = _model_route(data.get("route"))

        def rate(field_name: str) -> float:
            raw = data.get(field_name)
            if not isinstance(raw, (int, float)) or isinstance(raw, bool) or raw < 0:
                raise EvalContractError(f"model.{field_name} must be a non-negative number")
            return float(raw)

        return cls(
            id=_text(data.get("id"), field_name="model.id"),
            request_model=(
                _text(data.get("request_model"), field_name="model.request_model")
                if data.get("request_model") is not None
                else None
            ),
            route=route,
            secret=_identifier(data.get("secret"), field_name="model.secret"),
            efforts=_text_tuple(data.get("efforts", []), field_name="model.efforts"),
            usd_per_1m_input=rate("usd_per_1m_input"),
            usd_per_1m_output=rate("usd_per_1m_output"),
            usd_per_1m_cached_input=rate("usd_per_1m_cached_input"),
            price_source=_text(data.get("price_source"), field_name="model.price_source"),
            price_as_of=_text(data.get("price_as_of"), field_name="model.price_as_of"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            **({"request_model": self.request_model} if self.request_model else {}),
            "route": self.route,
            "secret": self.secret,
            "efforts": list(self.efforts),
            "usd_per_1m_input": self.usd_per_1m_input,
            "usd_per_1m_output": self.usd_per_1m_output,
            "usd_per_1m_cached_input": self.usd_per_1m_cached_input,
            "price_source": self.price_source,
            "price_as_of": self.price_as_of,
        }


@dataclass(frozen=True, slots=True)
class TrialBudget:
    """Per-trial caps on a paid policy. A cap reached is not an error.

    A trial that stops on its budget still produced the rollout it paid for, so
    it stays valid evidence; what it must never do is spend past the ceiling and
    report the number anyway.
    """

    max_llm_calls: int
    max_usd: float

    @classmethod
    def from_mapping(cls, value: Any) -> TrialBudget:
        data = _object(value, context="budget")
        max_usd = data.get("max_usd")
        if not isinstance(max_usd, (int, float)) or isinstance(max_usd, bool) or max_usd <= 0:
            raise EvalContractError("budget.max_usd must be a positive number")
        return cls(
            max_llm_calls=_positive_int(
                data.get("max_llm_calls"), field_name="budget.max_llm_calls"
            ),
            max_usd=float(max_usd),
        )

    def to_json(self) -> dict[str, Any]:
        return {"max_llm_calls": self.max_llm_calls, "max_usd": self.max_usd}


@dataclass(frozen=True, slots=True)
class PolicyCandidate:
    """One immutable, content-addressed policy artifact.

    `label` is presentation only.  `id` and `artifact_digest` are the identity
    that scoring, evidence, and promotion decisions key off.
    """

    id: str
    label: str
    kind: str
    artifact_uri: str
    artifact_digest: str
    entrypoint: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Any) -> PolicyCandidate:
        data = _object(value, context="policy candidate")
        schema = data.get("schema_version", POLICY_CANDIDATE_SCHEMA)
        if schema != POLICY_CANDIDATE_SCHEMA:
            raise EvalContractError(f"unsupported policy candidate schema {schema!r}")
        artifact = _object(data.get("artifact"), context="candidate.artifact")
        return cls(
            id=_identifier(data.get("id"), field_name="candidate.id"),
            label=_text(data.get("label"), field_name="candidate.label"),
            kind=_identifier(data.get("kind"), field_name="candidate.kind"),
            artifact_uri=_text(artifact.get("uri"), field_name="candidate.artifact.uri"),
            artifact_digest=_digest(artifact.get("digest"), field_name="candidate.artifact.digest"),
            entrypoint=_text(data.get("entrypoint"), field_name="candidate.entrypoint"),
            metadata=_object(data.get("metadata", {}), context="candidate.metadata"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": POLICY_CANDIDATE_SCHEMA,
            "id": self.id,
            "label": self.label,
            "kind": self.kind,
            "artifact": {"uri": self.artifact_uri, "digest": self.artifact_digest},
            "entrypoint": self.entrypoint,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """The frozen comparison group.  Ordering and baseline are part of it."""

    id: str
    candidates: tuple[PolicyCandidate, ...]
    baseline_id: str | None
    created_at: str
    root: Path | None = None

    @classmethod
    def from_mapping(cls, value: Any, *, root: Path | None = None) -> CandidateSet:
        data = _object(value, context="candidate set")
        schema = data.get("schema_version", CANDIDATE_SET_SCHEMA)
        if schema != CANDIDATE_SET_SCHEMA:
            raise EvalContractError(f"unsupported candidate set schema {schema!r}")
        raw = data.get("candidates")
        if not isinstance(raw, Sequence) or not raw:
            raise EvalContractError("candidate set must contain at least one candidate")
        candidates = tuple(PolicyCandidate.from_mapping(item) for item in raw)
        ids = [candidate.id for candidate in candidates]
        if len(set(ids)) != len(ids):
            raise EvalContractError("candidate ids must be unique inside a set")
        baseline_id = data.get("baseline_id")
        if baseline_id is not None:
            baseline_id = _identifier(baseline_id, field_name="baseline_id")
            if baseline_id not in ids:
                raise EvalContractError("baseline_id is not a member of the candidate set")
        return cls(
            id=_identifier(data.get("id"), field_name="candidate_set.id"),
            candidates=candidates,
            baseline_id=baseline_id,
            created_at=_text(data.get("created_at"), field_name="candidate_set.created_at"),
            root=root,
        )

    @classmethod
    def load(cls, path: Path) -> CandidateSet:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_mapping(payload, root=path.parent)

    def candidate(self, candidate_id: str) -> PolicyCandidate:
        for candidate in self.candidates:
            if candidate.id == candidate_id:
                return candidate
        raise EvalContractError(f"unknown candidate {candidate_id!r}")

    @property
    def baseline(self) -> PolicyCandidate | None:
        return self.candidate(self.baseline_id) if self.baseline_id else None

    def artifact_path(self, candidate: PolicyCandidate) -> Path:
        """Resolve a staged candidate to its on-disk, read-only directory."""

        prefix = "local-artifact://sha256/"
        if not candidate.artifact_uri.startswith(prefix):
            raise EvalContractError(
                f"candidate {candidate.id} artifact uri must use {prefix}, "
                f"got {candidate.artifact_uri!r}"
            )
        if self.root is None:
            raise EvalContractError("candidate set has no staging root")
        digest_hex = candidate.artifact_uri[len(prefix) :]
        path = (self.root / "artifacts" / digest_hex).resolve()
        if not path.is_dir():
            raise EvalContractError(f"staged artifact missing for candidate {candidate.id}")
        return path

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": CANDIDATE_SET_SCHEMA,
            "id": self.id,
            "created_at": self.created_at,
            "baseline_id": self.baseline_id,
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }

    def digest(self) -> str:
        return digest_of(self.to_json())


@dataclass(frozen=True, slots=True)
class SeedLedger:
    """Explicit integer seeds, generated once and never regenerated.

    Confirmation seeds are disjoint from screening seeds by construction, and
    every candidate in a comparison group sees the same seeds (common random
    numbers), so a lift is a paired difference rather than a lucky draw.
    """

    screening: tuple[int, ...]
    confirmation: tuple[int, ...]
    scenarios: tuple[str, ...]
    sealed_at: str

    def __post_init__(self) -> None:
        overlap = set(self.screening) & set(self.confirmation)
        if overlap:
            raise EvalContractError(
                f"confirmation seeds must be disjoint from screening seeds: {sorted(overlap)}"
            )
        if not self.screening:
            raise EvalContractError("seed ledger requires at least one screening seed")
        if not self.scenarios:
            raise EvalContractError("seed ledger requires at least one scenario")

    @classmethod
    def from_mapping(cls, value: Any) -> SeedLedger:
        data = _object(value, context="seed ledger")
        schema = data.get("schema_version", SEED_LEDGER_SCHEMA)
        if schema != SEED_LEDGER_SCHEMA:
            raise EvalContractError(f"unsupported seed ledger schema {schema!r}")
        return cls(
            screening=_seed_tuple(data.get("screening"), field_name="screening"),
            confirmation=_seed_tuple(data.get("confirmation", []), field_name="confirmation"),
            scenarios=_text_tuple(data.get("scenarios"), field_name="scenarios"),
            sealed_at=_text(data.get("sealed_at"), field_name="sealed_at"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SEED_LEDGER_SCHEMA,
            "screening": list(self.screening),
            "confirmation": list(self.confirmation),
            "scenarios": list(self.scenarios),
            "sealed_at": self.sealed_at,
        }


@dataclass(frozen=True, slots=True)
class EliminationRule:
    """The only way a candidate may leave the comparison after screening."""

    kind: str
    value: float | None = None

    @classmethod
    def from_mapping(cls, value: Any) -> EliminationRule:
        data = _object(value, context="elimination rule")
        kind = _text(data.get("kind", "none"), field_name="elimination.kind")
        if kind not in {"none", "keep_top_k", "min_primary_mean"}:
            raise EvalContractError(f"unsupported elimination rule {kind!r}")
        raw = data.get("value")
        if kind == "none":
            return cls(kind=kind, value=None)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise EvalContractError(f"elimination rule {kind} requires a numeric value")
        return cls(kind=kind, value=float(raw))

    def to_json(self) -> dict[str, Any]:
        return {"kind": self.kind, "value": self.value}


@dataclass(frozen=True, slots=True)
class SelectionSpec:
    """Recipe data, never agent prose."""

    primary_metric: str
    min_lift: float
    min_valid_trials: int
    decision_mode: str
    elimination: EliminationRule

    @classmethod
    def from_mapping(cls, value: Any) -> SelectionSpec:
        data = _object(value, context="selection")
        decision_mode = _text(data.get("decision_mode"), field_name="selection.decision_mode")
        if decision_mode not in {"report_only", "promote"}:
            raise EvalContractError("selection.decision_mode must be report_only or promote")
        min_lift = data.get("min_lift", 0.0)
        if not isinstance(min_lift, (int, float)) or isinstance(min_lift, bool):
            raise EvalContractError("selection.min_lift must be numeric")
        return cls(
            primary_metric=_identifier(
                data.get("primary_metric"), field_name="selection.primary_metric"
            ),
            min_lift=float(min_lift),
            min_valid_trials=_positive_int(
                data.get("min_valid_trials", 1), field_name="selection.min_valid_trials"
            ),
            decision_mode=decision_mode,
            elimination=EliminationRule.from_mapping(data.get("elimination", {"kind": "none"})),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "primary_metric": self.primary_metric,
            "min_lift": self.min_lift,
            "min_valid_trials": self.min_valid_trials,
            "decision_mode": self.decision_mode,
            "elimination": self.elimination.to_json(),
        }


@dataclass(frozen=True, slots=True)
class TrialLimits:
    """Hard per-trial resource ceilings enforced at the container boundary."""

    max_parallel_trials: int
    timeout_seconds: int
    cpus: float
    memory_mb: int
    max_output_bytes: int

    @classmethod
    def from_mapping(cls, value: Any) -> TrialLimits:
        data = _object(value, context="limits")
        cpus = data.get("cpus", 1.0)
        if not isinstance(cpus, (int, float)) or isinstance(cpus, bool) or cpus <= 0:
            raise EvalContractError("limits.cpus must be a positive number")
        return cls(
            max_parallel_trials=_positive_int(
                data.get("max_parallel_trials", 1), field_name="limits.max_parallel_trials"
            ),
            timeout_seconds=_positive_int(
                data.get("timeout_seconds"), field_name="limits.timeout_seconds"
            ),
            cpus=float(cpus),
            memory_mb=_positive_int(data.get("memory_mb"), field_name="limits.memory_mb"),
            max_output_bytes=_positive_int(
                data.get("max_output_bytes", 256 * 1024 * 1024),
                field_name="limits.max_output_bytes",
            ),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "max_parallel_trials": self.max_parallel_trials,
            "timeout_seconds": self.timeout_seconds,
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "max_output_bytes": self.max_output_bytes,
        }


@dataclass(frozen=True, slots=True)
class TrialKey:
    """The unit of evidence: one candidate, one seed, one scenario, one stage."""

    candidate_id: str
    seed: int
    scenario: str
    stage: str

    @property
    def trial_id(self) -> str:
        return f"trial_{self.stage}_{self.candidate_id}_{self.scenario}_{self.seed}"

    def to_json(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "seed": self.seed,
            "scenario": self.scenario,
            "stage": self.stage,
        }


@dataclass(frozen=True, slots=True)
class ContainerResult:
    """`/output/result.json`, exactly as the container wrote it.

    `status` is rig health; `benchmark_status` is what the policy did.  A
    missing metric stays missing: it is never coerced into a zero score.
    """

    trial_id: str
    status: str
    benchmark_status: str | None
    metrics: dict[str, float]
    gates: tuple[tuple[str, bool], ...]
    usage: dict[str, Any]
    artifacts: tuple[dict[str, Any], ...]
    error: str | None

    @classmethod
    def from_mapping(cls, value: Any) -> ContainerResult:
        data = _object(value, context="container result")
        schema = data.get("schema_version")
        if schema != CONTAINER_RESULT_SCHEMA:
            raise EvalContractError(
                f"container result schema must be {CONTAINER_RESULT_SCHEMA}, got {schema!r}"
            )
        status = _text(data.get("status"), field_name="result.status")
        if status not in CONTAINER_STATUSES:
            raise EvalContractError(f"result.status must be one of {CONTAINER_STATUSES}")
        benchmark_status = data.get("benchmark_status")
        if benchmark_status is not None:
            benchmark_status = _text(benchmark_status, field_name="result.benchmark_status")
            if benchmark_status not in BENCHMARK_STATUSES:
                raise EvalContractError(
                    f"result.benchmark_status must be one of {BENCHMARK_STATUSES}"
                )
        elif status == "evaluated":
            raise EvalContractError("an evaluated trial must report benchmark_status")
        raw_metrics = _object(data.get("metrics", {}), context="result.metrics")
        metrics: dict[str, float] = {}
        for key, metric_value in raw_metrics.items():
            if metric_value is None:
                continue
            if not isinstance(metric_value, (int, float)) or isinstance(metric_value, bool):
                raise EvalContractError(f"result.metrics.{key} must be numeric or null")
            metrics[key] = float(metric_value)
        gates: list[tuple[str, bool]] = []
        for entry in data.get("gates", []) or []:
            gate = _object(entry, context="result.gates[]")
            gates.append(
                (_identifier(gate.get("id"), field_name="gate.id"), bool(gate.get("passed")))
            )
        artifacts = tuple(
            _object(entry, context="result.artifacts[]")
            for entry in data.get("artifacts", []) or []
        )
        return cls(
            trial_id=_text(data.get("trial_id"), field_name="result.trial_id"),
            status=status,
            benchmark_status=benchmark_status,
            metrics=metrics,
            gates=tuple(gates),
            usage=_object(data.get("usage", {}), context="result.usage"),
            artifacts=artifacts,
            error=data.get("error") if isinstance(data.get("error"), str) else None,
        )

    def gate_map(self) -> dict[str, bool]:
        return dict(self.gates)


@dataclass(frozen=True, slots=True)
class TrialRecord:
    """`job_result.json`: the durable terminal record for one trial."""

    key: TrialKey
    trial_id: str
    status: str
    benchmark_status: str | None
    metrics: dict[str, float]
    gates: dict[str, bool]
    missing_gates: tuple[str, ...]
    missing_artifacts: tuple[str, ...]
    usage: dict[str, Any]
    artifacts: tuple[dict[str, Any], ...]
    started_at: str
    finished_at: str
    exit_code: int | None
    error: str | None
    evidence_dir: str

    @property
    def valid(self) -> bool:
        """Evidence is usable for scoring only when the rig and gates held."""

        return (
            self.status == "evaluated"
            and not self.missing_gates
            and not self.missing_artifacts
            and all(self.gates.values())
        )

    @classmethod
    def from_mapping(cls, value: Any) -> TrialRecord:
        data = _object(value, context="trial record")
        schema = data.get("schema_version")
        if schema != TRIAL_RECORD_SCHEMA:
            raise EvalContractError(f"unsupported trial record schema {schema!r}")
        key = _object(data.get("key"), context="trial record key")
        return cls(
            key=TrialKey(
                candidate_id=_identifier(key.get("candidate_id"), field_name="key.candidate_id"),
                seed=int(key["seed"]),
                scenario=_identifier(key.get("scenario"), field_name="key.scenario"),
                stage=_text(key.get("stage"), field_name="key.stage"),
            ),
            trial_id=_text(data.get("trial_id"), field_name="trial_id"),
            status=_text(data.get("status"), field_name="status"),
            benchmark_status=data.get("benchmark_status"),
            metrics={k: float(v) for k, v in (data.get("metrics") or {}).items()},
            gates={k: bool(v) for k, v in (data.get("gates") or {}).items()},
            missing_gates=tuple(data.get("missing_gates") or ()),
            missing_artifacts=tuple(data.get("missing_artifacts") or ()),
            usage=_object(data.get("usage", {}), context="usage"),
            artifacts=tuple(data.get("artifacts") or ()),
            started_at=_text(data.get("started_at"), field_name="started_at"),
            finished_at=_text(data.get("finished_at"), field_name="finished_at"),
            exit_code=data.get("exit_code"),
            error=data.get("error"),
            evidence_dir=_text(data.get("evidence_dir"), field_name="evidence_dir"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": TRIAL_RECORD_SCHEMA,
            "trial_id": self.trial_id,
            "key": self.key.to_json(),
            "status": self.status,
            "benchmark_status": self.benchmark_status,
            "valid": self.valid,
            "metrics": self.metrics,
            "gates": self.gates,
            "missing_gates": list(self.missing_gates),
            "missing_artifacts": list(self.missing_artifacts),
            "usage": self.usage,
            "artifacts": list(self.artifacts),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_code": self.exit_code,
            "error": self.error,
            "evidence_dir": self.evidence_dir,
        }


@dataclass(frozen=True, slots=True)
class MetricSummary:
    metric_id: str
    mean: float | None
    minimum: float | None
    maximum: float | None
    count: int

    def to_json(self) -> dict[str, Any]:
        return {
            "metric": self.metric_id,
            "mean": self.mean,
            "min": self.minimum,
            "max": self.maximum,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class CandidateScorecard:
    """Every candidate is scored separately; nothing is aggregated away."""

    candidate_id: str
    label: str
    stage: str
    is_baseline: bool
    trials_total: int
    trials_valid: int
    trials_failed: int
    metrics: tuple[MetricSummary, ...]
    gate_failures: dict[str, int]
    paired_lift: float | None
    paired_trials: int
    eliminated_at: str | None
    elimination_reason: str | None
    cost_usd: float | None
    # How much of the scored episodes this candidate's own policy actually
    # chose. A budget-exhausted LLM policy does not stop playing — it returns a
    # fallback action for every remaining step — so without this a mean over
    # model plays and filler is indistinguishable from a mean over model plays.
    # `None` means no trial reported coverage, the ordinary case for a code
    # policy with no budget to exhaust.
    budget_exhausted_trials: int = 0
    policy_step_fraction: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SCORECARD_SCHEMA,
            "candidate_id": self.candidate_id,
            "label": self.label,
            "stage": self.stage,
            "is_baseline": self.is_baseline,
            "trials": {
                "total": self.trials_total,
                "valid": self.trials_valid,
                "failed": self.trials_failed,
                "budget_exhausted": self.budget_exhausted_trials,
            },
            "metrics": [summary.to_json() for summary in self.metrics],
            "gate_failures": self.gate_failures,
            "paired_lift": self.paired_lift,
            "paired_trials": self.paired_trials,
            "eliminated_at": self.eliminated_at,
            "elimination_reason": self.elimination_reason,
            "cost_usd": self.cost_usd,
            "policy_step_fraction": self.policy_step_fraction,
        }


@dataclass(frozen=True, slots=True)
class SelectionDecision:
    status: str
    winner_id: str | None
    baseline_id: str | None
    primary_metric: str
    lift: float | None
    min_lift: float
    reason: str

    def __post_init__(self) -> None:
        if self.status not in SELECTION_STATUSES:
            raise EvalContractError(f"selection status must be one of {SELECTION_STATUSES}")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": SELECTION_SCHEMA,
            "status": self.status,
            "winner_id": self.winner_id,
            "baseline_id": self.baseline_id,
            "primary_metric": self.primary_metric,
            "lift": self.lift,
            "min_lift": self.min_lift,
            "reason": self.reason,
        }


def write_json(path: Path, payload: Any) -> None:
    """Atomic write: evidence is either the old file or the whole new one."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), "utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row))
            handle.write("\n")
