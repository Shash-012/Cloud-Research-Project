"""YAML configuration loading and dotted access."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def load_config(path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get(cfg: dict, key: str, default=None):
    """Dotted access: get(cfg, "a.b.c")."""
    node = cfg
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def save_config(cfg: dict, path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def merge_cloudsim(proto: dict, cs: dict, exp: dict | None = None) -> dict:
    """Config for the Phase 2 Python modules running against CloudSim Plus.

    Model, propagation and selection settings come from prototype.yaml; the scenario (time, pool limits,
    actions, cost, power, splits) comes from cloudsim.yaml. The SLO is the effective one, rt_ms / capacity_scale
    (Step 3.1, "Run time"). The original cloudsim.yaml is kept under "cloudsim".
    """
    import copy
    cfg = copy.deepcopy(proto)
    s = float(cs["requests"]["capacity_scale"])
    cfg["source"] = "cloudsim"
    cfg["capacity_scale"] = s
    cfg["time"] = dict(cs["time"])
    cfg["system"] = dict(cfg["system"], vm_min=cs["cluster"]["vm_min"], vm_max=cs["cluster"]["vm_max"],
                         initial_vms=cs["cluster"]["initial_vms"],
                         service_rate_per_vm=s / cs["requests"]["mean_service_s"])
    cfg["slo"] = dict(cfg["slo"], **{k: v for k, v in cs["slo"].items() if k != "rt_ms"},
                      rt_ms=cs["slo"]["rt_ms"] / s)
    for key in ("actions", "cost", "power", "splits"):
        cfg[key] = copy.deepcopy(cs[key])
    cfg["cloudsim"] = copy.deepcopy(cs)
    if exp is not None:
        cfg["baselines"] = copy.deepcopy(exp["baselines"])
    return cfg


def java_config(cs: dict, exp: dict, port: int | None = None) -> dict:
    """JSON handed to the Java simulator: cloudsim.yaml plus the baseline thresholds (and the bridge port)."""
    import copy
    out = copy.deepcopy(cs)
    out["baselines"] = copy.deepcopy(exp["baselines"])
    if port is not None:
        out["bridge"] = dict(out["bridge"], port=port)
    return out


def default_cloudsim_config() -> dict:
    return load_config(CONFIG_DIR / "cloudsim.yaml")


def default_prototype_config() -> dict:
    return load_config(CONFIG_DIR / "prototype.yaml")


def default_experiments_config() -> dict:
    return load_config(CONFIG_DIR / "experiments.yaml")
