from .base import Policy
from .rt_threshold import RTThreshold
from .cpu_threshold import CPUThreshold
from .opt_proactive import OPTProactive
from .opt_reactive import OPTReactive

DEFAULT_THRESH = {"rt": dict(upper_frac=0.9, lower_frac=0.4), "cpu": dict(upper=75.0, lower=35.0)}


def make_policy(name: str, cfg: dict, app: str, bundle=None, tuned: dict | None = None) -> Policy:
    qos = cfg["apps"][app]["qos_ms"]; cool = cfg["cooldown_s"]
    tuned = tuned or {}
    if name == "rt":
        t = {**DEFAULT_THRESH["rt"], **tuned.get("rt", {})}
        return RTThreshold(t["upper_frac"] * qos, t["lower_frac"] * qos, cool)
    if name == "cpu":
        t = {**DEFAULT_THRESH["cpu"], **tuned.get("cpu", {})}
        return CPUThreshold(t["upper"], t["lower"], cool)
    if name in ("opt_reactive", "opt_proactive"):
        frac = tuned.get(name, {}).get("target_frac", cfg["opt_target_frac"])
        return (OPTReactive if name == "opt_reactive" else OPTProactive)(bundle, cfg, frac)
    raise ValueError(name)
