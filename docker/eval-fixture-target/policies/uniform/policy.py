"""Baseline: ignore the observation and always step right."""


class Policy:
    def act(self, observation: dict) -> str:
        return "right"
