"""Stable environment boundary shared by SimPy and future CloudSim adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Observation:
    """Controller-visible state at one monitoring interval."""

    timestamp_s: float
    vm_count: int
    request_rate: float
    response_time_ms: float
    cpu_util: float
    metrics: tuple[float, ...] = ()


class Environment(Protocol):
    """Minimum simulator contract required by policies and the orchestrator."""

    def reset(self, seed: int | None = None) -> Observation: ...

    def observe(self) -> Observation: ...

    def set_vms(self, count: int) -> None: ...

    def step(self) -> Observation: ...
