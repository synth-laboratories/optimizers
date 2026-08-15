"""Remember which squares have been visited and sweep outward from the start."""


class Policy:
    def __init__(self) -> None:
        self._seen: set[int] = set()
        self._direction = "right"

    def act(self, observation: dict) -> str:
        position = observation["position"]
        corridor = observation["corridor"]
        if position in self._seen and len(self._seen) < corridor:
            self._direction = "left" if self._direction == "right" else "right"
        self._seen.add(position)
        return self._direction
