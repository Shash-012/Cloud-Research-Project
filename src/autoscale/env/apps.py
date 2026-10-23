"""Application service-demand profiles."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AppProfile:
    name: str
    base_service_demand_s: float
    resource_mix: dict[str, float]


CPU_BOUND = AppProfile(
    "cpu_bound", 0.020, {"cpu": 0.75, "memory": 0.10, "disk": 0.05, "network": 0.10}
)
DB_IO = AppProfile("db_io", 0.025, {"cpu": 0.20, "memory": 0.15, "disk": 0.50, "network": 0.15})
ML_BATCH = AppProfile(
    "ml_batch", 0.030, {"cpu": 0.35, "memory": 0.30, "disk": 0.15, "network": 0.20}
)
