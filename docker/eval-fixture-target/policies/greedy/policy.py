"""Walk toward the seed-determined target by the shorter direction."""


class Policy:
    def act(self, observation: dict) -> str:
        corridor = observation["corridor"]
        position = observation["position"]
        target = observation["target"]
        forward = (target - position) % corridor
        return "right" if forward <= corridor - forward else "left"
