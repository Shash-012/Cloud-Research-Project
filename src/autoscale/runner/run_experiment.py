"""One run = env x policy x workload x seed. Writes nothing; returns KPIs (+ optional interval/request tables)."""
from __future__ import annotations
from pathlib import Path
import yaml
from ..config import load_config, RESULTS
from ..env.simpy_env import make_env
from ..models.bundle import ModelBundle
from ..policies import make_policy
from ..analysis.kpis import compute_kpis

_BUNDLES: dict = {}
_CFG: dict = {}


def get_bundle(app: str, models_dir: Path | None = None) -> ModelBundle:
    key = (app, str(models_dir))
    if key not in _BUNDLES:
        _BUNDLES[key] = ModelBundle.load((models_dir or RESULTS / "models") / app)
    return _BUNDLES[key]


def load_tuned(path: Path | None = None) -> dict:
    p = path or (RESULTS / "tables" / "tuned_baselines.yaml")
    return yaml.safe_load(open(p)) if p.exists() else {}


def run_one(app: str, workload: str, policy: str, seed: int, cfg: dict | None = None, bundle=None, tuned=None,
            duration_s: float | None = None, keep: bool = False, init_vms: int | None = None) -> dict:
    cfg = cfg or load_config()
    if bundle is None and policy.startswith("opt"):
        bundle = get_bundle(app)
    tuned_ap = (tuned if tuned is not None else load_tuned()).get(f"{app}/{workload}", {})
    pol = make_policy(policy, cfg, app, bundle, tuned_ap)
    pol.reset()
    env = make_env(cfg, app, workload, seed, duration_s)
    obs = env.reset(init_vms)
    while not env.done:
        obs = env.step(pol.decide(obs))
    iv, rq = env.finalize()
    k = compute_kpis(iv, rq, cfg["apps"][app]["qos_ms"], cfg["max_vms"])
    k.update(app=app, workload=workload, policy=policy, seed=seed)
    return (k, iv, rq) if keep else k
