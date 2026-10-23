"""Dataset containers and blocked time-series splits."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Block:
    start: int
    stop: int
