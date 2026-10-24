"""Config loading: merges configs/default.yaml with apps/*.yaml and workloads/*.yaml."""
from __future__ import annotations
from pathlib import Path
import copy
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
RESULTS = ROOT / "results"


def _read(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def app_capacity_per_vm(app: dict) -> float:
    """Max sustainable request rate of one VM (req/s): the tightest of CPU, disk, thread limits."""
    cpu_s, io_s = app["cpu_ms"] / 1000.0, app["io_ms"] / 1000.0
    caps = [app["vcpu"] / cpu_s]
    if io_s > 0 and app["io_cap"] > 0:
        caps.append(app["io_cap"] / io_s)
    if app["slots"] < 500:
        caps.append(app["slots"] / (cpu_s + io_s))
    return min(caps)


def load_config(path: str | Path | None = None) -> dict:
    cfg = _read(Path(path) if path else CONFIG_DIR / "default.yaml")
    cfg["apps"] = {}
    for p in sorted((CONFIG_DIR / "apps").glob("*.yaml")):
        a = _read(p)
        a["capacity_per_vm"] = app_capacity_per_vm(a)
        cfg["apps"][a["name"]] = a
    cfg["workloads"] = {}
    for p in sorted((CONFIG_DIR / "workloads").glob("*.yaml")):
        w = _read(p)
        cfg["workloads"][w["name"]] = w
    return copy.deepcopy(cfg)
