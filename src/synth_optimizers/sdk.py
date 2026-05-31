from __future__ import annotations

from typing import Generic, Protocol, TypeVar


ResultT = TypeVar("ResultT")


class OptimizerConfig(Protocol[ResultT]):
    def execute(self) -> ResultT: ...


class OptimizerRun(Generic[ResultT]):
    def __init__(self, config: OptimizerConfig[ResultT]) -> None:
        self.config = config

    def execute(self) -> ResultT:
        return self.config.execute()
