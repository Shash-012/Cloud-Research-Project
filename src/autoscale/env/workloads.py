"""Workload generators used by the experiment matrix."""

from typing import Protocol


class Workload(Protocol):
    def rate(self, timestamp_s: float) -> float: ...

    def complexity(self, timestamp_s: float) -> float: ...
