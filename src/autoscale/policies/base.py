"""Policy protocol."""

from typing import Protocol

from autoscale.env.base import Observation


class Policy(Protocol):
    def decide(self, observation: Observation) -> int: ...
