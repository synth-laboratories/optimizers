"""Public Synth optimizer package.

The hosted optimizer client is intentionally importable without the native GEPA
extension so cloud-only callers can submit runs with just a Synth API key.
Local GEPA execution and service exports are available when the native runtime
package is installed.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .hosted import (
    HostedGepaConfig,
    HostedOptimizerAuthError,
    HostedOptimizerClient,
    HostedOptimizerError,
    HostedOptimizerHTTPError,
)
from .o11y import (
    LiveProgress,
    RegistryRecord,
    RunBoard,
    RunFailure,
    RunState,
    RunStatus,
    RunUsage,
    project_run_events,
)
from .sdk import OptimizerConfig, OptimizerRun

try:
    __version__ = version("synth-optimizers")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "HostedGepaConfig",
    "HostedOptimizerAuthError",
    "HostedOptimizerClient",
    "HostedOptimizerError",
    "HostedOptimizerHTTPError",
    "LiveProgress",
    "OptimizerConfig",
    "OptimizerRun",
    "RegistryRecord",
    "RunBoard",
    "RunFailure",
    "RunState",
    "RunStatus",
    "RunUsage",
    "project_run_events",
    "__version__",
]

try:
    from ._synth_optimizers import (
        BudgetExceededError,
        CacheCorruptError,
        CacheFullError,
        CacheMissError,
        CancelledError,
        ConfigError,
        ContainerContractError,
        EventCompareError,
        GepaRunResult,
        InvariantError,
        OptimizerDiskBudgetError,
        OptimizerHttpError,
        OptimizerIoError,
        OptimizerJsonError,
        OptimizerSqliteError,
        OptimizerTomlDecodeError,
        ProposerError,
        RunFailedError,
        StateTransitionError,
        SynthOptimizerError,
        __version__ as __native_version__,
        events_compare,
        events_replay,
        gepa_compact_run_storage,
        gepa_delete_run_storage,
        gepa_serve,
    )
    from .gepa import (
        BudgetConfig,
        CacheConfig,
        GepaBudgetConfig,
        GepaConfig,
        GepaDefaults,
        GepaPipeline,
        GepaPipelineMode,
        GepaRun,
        GepaStalenessPolicy,
        GepaTaskPools,
        ObjectiveConfig,
        OutputConfig,
        PolicyConfig,
        PolicyType,
        ProposerConfig,
        ProposerDefaults,
        ProposerPromptConfig,
        RunSettings,
        TasksetSelection,
    )

    __version__ = __native_version__
    __all__ += [
        "BudgetExceededError",
        "CacheCorruptError",
        "CacheFullError",
        "CacheMissError",
        "CancelledError",
        "ConfigError",
        "ContainerContractError",
        "EventCompareError",
        "BudgetConfig",
        "CacheConfig",
        "GepaBudgetConfig",
        "GepaConfig",
        "GepaDefaults",
        "GepaPipeline",
        "GepaPipelineMode",
        "GepaRun",
        "GepaRunResult",
        "GepaStalenessPolicy",
        "GepaTaskPools",
        "InvariantError",
        "ObjectiveConfig",
        "OptimizerDiskBudgetError",
        "OptimizerHttpError",
        "OptimizerIoError",
        "OptimizerJsonError",
        "OptimizerSqliteError",
        "OptimizerTomlDecodeError",
        "OutputConfig",
        "PolicyConfig",
        "PolicyType",
        "ProposerConfig",
        "ProposerDefaults",
        "ProposerError",
        "ProposerPromptConfig",
        "RunFailedError",
        "RunSettings",
        "StateTransitionError",
        "SynthOptimizerError",
        "TasksetSelection",
        "events_compare",
        "events_replay",
        "gepa_compact_run_storage",
        "gepa_delete_run_storage",
        "gepa_serve",
    ]
except ModuleNotFoundError as exc:
    if exc.name not in {
        "_synth_optimizers",
        "synth_optimizers._synth_optimizers",
        "synth_containers",
    }:
        raise
