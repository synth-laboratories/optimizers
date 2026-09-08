"""Training admission over the shared durable optimizer budget ledger."""
from decimal import Decimal, InvalidOperation

from ..contracts.training_schemas import SchemaError
from ..providers.tinker.fake import FakeTinkerProvider
from ..rl.budget import ExperimentBudget, micros

TOKEN_RATES = ("input_usd_per_million", "output_usd_per_million", "training_usd_per_million")
FEES = ("session_usd", "save_usd", "restore_usd")


def resolve_budget(config, provider):
    value = config.get("budget")
    if value is None and isinstance(getattr(provider, "_transport", None), FakeTinkerProvider):
        return {"max_cost_usd": 1, "pricing_version": "fixture.zero.v1",
                "pricing": {key: 0 for key in (*TOKEN_RATES, *FEES)}}
    if not isinstance(value, dict) or not value.get("pricing_version"):
        raise SchemaError("training requires an aggregate budget and versioned pricing including checkpoint fees")
    try:
        if micros(value["max_cost_usd"]) <= 0:
            raise ValueError("empty cap")
        for field in (*TOKEN_RATES, *FEES):
            micros(value["pricing"][field])
    except (KeyError, ValueError, TypeError, InvalidOperation) as exc:
        raise SchemaError("budget requires a positive cap and explicit nonnegative token/session/save/restore prices") from exc
    return value


class TrainingBudget:
    def __init__(self, store, job_id, plan):
        self.ledger = ExperimentBudget(store.path, plan.get("experiment_id") or job_id, plan["max_cost_usd"])
        self.prices = {key: Decimal(str(value)) for key, value in plan["pricing"].items()}

    def ceiling(self, kind, request=None):
        if kind in {"session", "save", "restore"}:
            return self.prices[f"{kind}_usd"]
        if kind in {"sample", "sample_checkpoint"}:
            return (len(request.prompt_token_ids) * self.prices["input_usd_per_million"] +
                    request.max_tokens * self.prices["output_usd_per_million"]) / 1_000_000
        if kind == "forward":
            tokens = sum(map(len, request.token_ids))
        elif kind == "train":
            tokens = sum(len(row.get("input_ids") or row.get("token_ids") or ()) +
                         len(row.get("prompt_token_ids") or ()) for row in request.data)
        else:
            raise ValueError(f"no pricing contract for operation {kind}")
        return tokens * self.prices["training_usd_per_million"] / 1_000_000

    def reserve(self, request_id, kind, request=None):
        self.ledger.reserve(request_id, kind, self.ceiling(kind, request))

    def settle(self, request_id, result):
        usage = getattr(result, "usage", None)
        # Missing billing remains conservative; it is not an estimated invoice.
        cost = None if usage is None or usage.cost_missing else usage.cost_usd
        self.ledger.settle(request_id, cost)
