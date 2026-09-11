"""The optimizer facade delegates execution without changing public error types."""
import pytest
from synth_containers import oci_trial
from synth_optimizers.eval import executor
from synth_optimizers.eval.models import EvalContractError


def test_shared_lifecycle_implementation():
    assert executor.OciTrialExecutor.run is oci_trial.OciTrialExecutor.run
    assert executor.TrialRunRequest is oci_trial.TrialRunRequest
    assert executor.ContainerStopUnconfirmed is oci_trial.ContainerStopUnconfirmed


def test_invalid_runtime_preserves_eval_error():
    with pytest.raises(EvalContractError):
        executor.OciTrialExecutor("unsupported")
