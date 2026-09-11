"""Public eval compatibility adapter for the shared containers OCI executor.

Recipe selection and scoring stay in optimizers. Resource lifecycle and limit
actuation live in synth_containers. This module preserves existing import names.
"""

from synth_containers.oci_trial import (
    ContainerRuntimeError,
    ContainerStopUnconfirmed,
    ExecutionContractError,
    OciTrialExecutor as SharedOciTrialExecutor,
    StopFailureReason,
    TrialExecution,
    TrialExecutor,
    TrialRunRequest,
    _output_bytes,
    _tail_events,
    _tail_text,
    shutil,
    subprocess,
)
from .models import EvalContractError

__all__ = [
    "ContainerRuntimeError",
    "ContainerStopUnconfirmed",
    "StopFailureReason",
    "TrialExecution",
    "TrialExecutor",
    "TrialRunRequest",
    "OciTrialExecutor",
    "_output_bytes",
    "_tail_events",
    "_tail_text",
    "shutil",
    "subprocess",
]


class OciTrialExecutor(SharedOciTrialExecutor):
    """Keep the public facade's existing invalid-runtime error type."""

    def __init__(self, runtime: str = "docker") -> None:
        try:
            super().__init__(runtime)
        except ExecutionContractError as error:
            raise EvalContractError(str(error)) from error
