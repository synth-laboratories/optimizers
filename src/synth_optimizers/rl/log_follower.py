"""Shared, bounded viewer for the authoritative retained worker log."""

from synth_containers.retained_log import LogFollower, main

__all__ = ["LogFollower"]

if __name__ == "__main__":
    main()
