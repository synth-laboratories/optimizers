"""A plane whose provider is stubbed, so the cross-process run costs nothing.

Everything else is real: the container is another process reached over a
socket, the client is the shipped one, and the gateway, binder and catalog are
the ones a paid run would use.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from synth_optimizers.providers.protocols import (
    ForwardRequest,
    ForwardResult,
    ProviderCapabilities,
    ProviderCheckpoint,
    ProviderError,
    ProviderSession,
    ProviderUsage,
    SampleRequest,
    SampleResult,
    TrainingStepRequest,
    TrainingStepResult,
)
from synth_optimizers.rl.plane import build_plane

# The container's own renderer rule, restated here so the one renderer in the
# run is the one the container declares. Copied rather than imported because
# this process is the optimizer's virtualenv and ``synth_containers`` is not in
# it; the values are pinned by ``cispo_target.reference_renderer_profile``.
RENDER_VOCAB_BASE = 100_000
RENDER_VOCAB_SIZE = 50_000
RENDER_STOP_TOKEN_IDS = (200_002, 199_999)

#: The shell commands the mini-SWE stand-in draws from. Read-only and bounded:
#: this is a stand-in policy, not an attempt to solve the trial.
SHELL_COMMANDS = (
    "ls -a",
    "pwd",
    "cat instruction.md",
    "ls -l",
    "echo probe > notes.txt",
    "wc -l notes.txt",
    "find . -maxdepth 2 -type f",
    "head -n 5 notes.txt",
)

#: What the stand-in says when the prompt declares no vocabulary of its own.
#: A fixed bag of clinical-advice terms -- it reads no rubric and knows no gold;
#: all it has to do is answer differently for different samples so a group has
#: something to compare.
ADVICE_VOCABULARY = (
    "seek urgent care emergency physician doctor symptoms advises recommends evaluation "
    "hospital medication dose monitor blood pressure pain chest breathing fever infection "
    "antibiotics allergy pregnancy child dehydration hydration rest follow appointment "
    "specialist referral test results treatment risk warning signs immediately safety "
    "history exam clinic nurse dizziness nausea vomiting bleeding swelling"
).split()


def render_tokens(text: str) -> tuple[int, ...]:
    words = text.split() or [""]
    return tuple(
        RENDER_VOCAB_BASE
        + int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16) % RENDER_VOCAB_SIZE
        for word in words
    )


class UnpaidProvider:
    """A ``TrainingProvider`` that reaches no provider and buys nothing.

    It is the seam ``build_plane(provider=...)`` exists for. Every method a
    paid run would call is here, and none of them leaves the process.
    """

    def __init__(self) -> None:
        self.artifacts: dict[str, str] = {}
        self.sessions: list[ProviderSession] = []
        self.train_calls: list[TrainingStepRequest] = []
        self.step = 0
        # What each rendered prompt said, and how many turns deep it is, keyed
        # by the tokens it rendered to. A real provider reads the prompt; this
        # one only ever sees token ids, because rendering is its own job, so it
        # keeps what it rendered.
        self._prompts: dict[tuple[int, ...], str] = {}
        self._turns: dict[tuple[int, ...], int] = {}

    # -- the renderer surface the plane requires by name ------------------ #

    def tokenize_chat(
        self, messages: Sequence[Mapping[str, Any]], *, add_generation_prompt: bool = False
    ) -> dict[str, Any]:
        text = "\n".join(str(row.get("content") or "") for row in messages)
        tokens = render_tokens(text)
        self._prompts[tokens] = text
        self._turns[tokens] = 0
        return {
            "prompt_token_ids": list(tokens),
            "stop_token_ids": list(RENDER_STOP_TOKEN_IDS),
        }

    def bridge_chat(
        self,
        previous_prompt_token_ids: Sequence[int],
        previous_completion_token_ids: Sequence[int],
        messages: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        """The turn-to-turn bridge a paid provider has, so this path is the same.

        This renderer is word-wise, so the tokens a new turn adds are exactly
        the tokens of its own text; the sampled ids are carried through rather
        than re-derived from the assistant text the container echoed back.
        """

        if not messages:
            return None
        added: list[int] = []
        for row in messages:
            added.extend(render_tokens(str(row.get("content") or "")))
        tokens = (
            tuple(previous_prompt_token_ids) + tuple(previous_completion_token_ids) + tuple(added)
        )
        # The whole conversation is kept, not only the turn just added: the
        # rules a prompt states -- what a legal action is, what shape a reply
        # takes -- are stated once, in the opening turn, and a stand-in that
        # only ever saw the latest observation would forget them at turn two.
        previous = tuple(previous_prompt_token_ids)
        added_text = "\n".join(str(row.get("content") or "") for row in messages)
        self._prompts[tokens] = (self._prompts.get(previous, "") + "\n" + added_text).strip()
        self._turns[tokens] = self._turns.get(previous, 0) + 1
        return {
            "prompt_token_ids": list(tokens),
            "stop_token_ids": list(RENDER_STOP_TOKEN_IDS),
        }

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return " ".join(str(int(token)) for token in token_ids)

    # -- sessions --------------------------------------------------------- #

    def discover_capabilities(self, model_id: str) -> ProviderCapabilities:
        from synth_optimizers.providers.protocols import CISPO_REQUIRED_CAPABILITIES

        return ProviderCapabilities(
            provider="unpaid",
            model_id=model_id,
            capabilities=frozenset(CISPO_REQUIRED_CAPABILITIES),
            validated={name: True for name in CISPO_REQUIRED_CAPABILITIES},
            spend_free=True,
        )

    def resolve_model(self, model_id: str) -> str:
        return model_id

    def create_session(
        self, model_id: str, *, rank: int, seed: int, request_id: str
    ) -> ProviderSession:
        session = ProviderSession(
            provider="unpaid",
            session_id=f"session_{len(self.sessions) + 1}",
            model_id=model_id,
            request_id=request_id,
        )
        self.sessions.append(session)
        return session

    def restore_session(
        self, checkpoint: ProviderCheckpoint, *, request_id: str
    ) -> ProviderSession:
        session = ProviderSession(
            provider="unpaid",
            session_id=checkpoint.resume_token or checkpoint.checkpoint_id,
            model_id="unpaid/model",
            request_id=request_id,
        )
        self.sessions.append(session)
        return session

    # -- sampling --------------------------------------------------------- #

    # -- what the stand-in says ------------------------------------------- #

    def _choose(self, prompt: str, request_id: str, turn: int = 0) -> str:
        """Answer the prompt the way a policy would: from what it was offered.

        Two samples of one task have to be allowed to differ, or every group
        scores identically, CISPO skips it for zero advantage -- correctly --
        and no training step is ever reached. So the choice is spread over what
        the prompt itself declares legal, deterministically, by the per-call
        request id. It reaches no network and buys nothing.
        """

        draw = int(hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:8], 16)

        marker = "valid_actions="
        at = prompt.rfind(marker)
        if at >= 0:
            raw = prompt[at + len(marker) :].strip().splitlines()[0]
            actions = json.loads(raw)
            if isinstance(actions, list) and actions:
                cap = re.search(r"(\d+) to (\d+) action names", prompt)
                if cap is None:
                    return str(actions[draw % len(actions)])
                # The prompt asks for a plan, not a move, and says how long a
                # plan may be. Two samples of one state have to be allowed to
                # plan differently, so both which actions and how many are the
                # draw's -- inside the bounds the prompt itself declares.
                low, high = int(cap.group(1)), int(cap.group(2))
                span = low + draw % max(1, high - low + 1)
                return json.dumps(
                    [str(actions[(draw + offset) % len(actions)]) for offset in range(span)]
                )

        if "fenced bash block" in prompt:
            # The prompt is the mini-SWE harness's, and it says exactly what a
            # reply may be: one fenced bash block holding one shell command.
            # Which command, and when to stop, is the draw's; the shape is the
            # harness's.
            command = SHELL_COMMANDS[draw % len(SHELL_COMMANDS)]
            if turn >= 2 + draw % 4:
                # The harness's own way to stop. A stand-in that never stopped
                # would run the whole fifty-command horizon every time, and the
                # trace it sealed would be the same length every time.
                command = "echo MINI_SWE_DONE"
            return f"```bash\n{command}\n```"

        head, sep, tail = prompt.partition("Allowed labels")
        if sep:
            # Only the query itself, never the instructions above it: the system
            # turn is the same for every row, so counting its words would rank
            # every row's labels the same way.
            _, _, query_text = head.rpartition("Customer query:")
            query = set(re.findall(r"[a-z]+", (query_text or head).lower()))
            labels = [line.strip() for line in tail.splitlines()[1:] if line.strip()]
            if labels:
                order = {label: index for index, label in enumerate(labels)}
                ranked = sorted(
                    labels,
                    key=lambda label: (
                        -len(set(re.findall(r"[a-z]+", label.lower())) & query),
                        order[label],
                    ),
                )
                # The four closest by word overlap. A wider draw would answer
                # at random and never score; a narrower one would answer the
                # same label every time and never vary.
                shortlist = ranked[:4]
                return shortlist[draw % len(shortlist)]

        # Nothing in the prompt names what may be said, so the stand-in answers
        # in prose, from a fixed clinical-advice vocabulary it brought with it.
        # The slice it takes is the draw's, so two samples of one conversation
        # differ -- which is the only property this stand-in has to have.
        span = 6 + draw % (len(ADVICE_VOCABULARY) - 6)
        start = draw % len(ADVICE_VOCABULARY)
        return " ".join(
            ADVICE_VOCABULARY[(start + offset) % len(ADVICE_VOCABULARY)]
            for offset in range(span)
        )

    def _sampled(self, request: SampleRequest) -> SampleResult:
        key = tuple(request.prompt_token_ids)
        prompt = self._prompts.get(key, "")
        text = self._choose(prompt, request.request_id, self._turns.get(key, 0))
        tokens = render_tokens(text)
        logprobs = tuple(
            round(-0.05 - ((int(token) + index) % 97) / 500.0, 6)
            for index, token in enumerate(tokens)
        )
        return SampleResult(
            request_id=request.request_id,
            token_ids=tokens,
            logprobs=logprobs,
            text=text,
            finish_reason="stop",
            usage=ProviderUsage(
                input_tokens=len(request.prompt_token_ids),
                output_tokens=len(tokens),
                cost_usd=0.0,
                cost_missing=False,
            ),
        )

    def sample(self, session: ProviderSession, request: SampleRequest) -> SampleResult:
        return self._sampled(request)

    def sample_checkpoint(
        self, checkpoint: ProviderCheckpoint, request: SampleRequest
    ) -> SampleResult:
        return self._sampled(request)

    # -- training --------------------------------------------------------- #

    def forward(self, session: ProviderSession, request: ForwardRequest) -> ForwardResult:
        rows = tuple(
            tuple(-0.25 if flag else 0.0 for flag in mask[: len(tokens)])
            for tokens, mask in zip(request.token_ids, request.response_masks, strict=True)
        )
        return ForwardResult(
            request_id=request.request_id,
            logprobs=rows,
            usage=ProviderUsage(
                training_tokens=sum(sum(1 for flag in mask if flag) for mask in request.response_masks),
                cost_usd=0.0,
                cost_missing=False,
            ),
        )

    def train_step(
        self, session: ProviderSession, request: TrainingStepRequest
    ) -> TrainingStepResult:
        self.train_calls.append(request)
        self.step += 1
        return TrainingStepResult(
            request_id=request.request_id,
            step=self.step,
            metrics={"loss": 1.0 / self.step},
            usage=ProviderUsage(
                training_tokens=17 * max(1, len(request.data)),
                cost_usd=0.0,
                cost_missing=False,
            ),
        )

    def save_checkpoint(
        self, session: ProviderSession, *, step: int, kind: str, request_id: str
    ) -> ProviderCheckpoint:
        reference = f"unpaid://{session.session_id}/{kind}/{step}"
        digest = "sha256:" + hashlib.sha256(reference.encode()).hexdigest()
        self.artifacts[reference] = digest
        return ProviderCheckpoint(
            checkpoint_id=f"{session.session_id}-{kind}-{step}",
            provider_reference=reference,
            step=step,
            digest=digest,
            kind=kind,
            resume_token=f"resume:{session.session_id}:{step}",
        )

    def cancel(self, session: ProviderSession) -> None:
        return None

    def classify_error(self, error: BaseException) -> ProviderError:
        return ProviderError("unpaid_error", str(error))


def unpaid(config: Any = None, **kwargs: Any) -> Any:
    if config is None:
        raise SystemExit("this plane needs --config: it is assembled from one")
    return build_plane(config, provider=UnpaidProvider(), **kwargs)


# --------------------------------------------------------------------------- #
# No shims.
# --------------------------------------------------------------------------- #
#
# Two used to live here: one that re-sent the sampler origin as an object
# because ``ContractContainerSession.bind`` flattened it to a bare URL, and one
# that reordered ``SamplerGatewayService.bind`` so the route existed before its
# attempt facts were declared. Both are fixed in the optimizer now -- ``bind``
# sends ``sampler_origin`` and ``behavior_fingerprint`` itself, and the gateway
# declares after registering the route, marked provisional so the container's
# own rollout id can settle it after submission. Keeping either shim is now
# actively harmful: the gateway one declared non-provisional facts, so the
# post-submission settle failed with "already recorded calls".
