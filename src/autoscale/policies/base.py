"""Policy interface. decide(obs) returns the change to apply to the TOTAL pool (attached + booting)."""
from __future__ import annotations
from abc import ABC, abstractmethod
from ..env.base import Observation


class Policy(ABC):
    name = "policy"

    def reset(self):
        pass

    @abstractmethod
    def decide(self, obs: Observation) -> int: ...
