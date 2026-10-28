"""Scaling policies (Step 2.11). Each implements decide(obs, state) -> (k, info)."""
from pathlib import Path

import numpy as np

from ..models.metric_estimator import MetricEstimator
from ..models.quantile import QuantileHGB, QuantileNetModel
from ..models.rt_ensemble import RTEnsemble
from ..schema import DECISION_CANDIDATE_FIELDS
from ..uncertainty.conformal import bound_z
from ..uncertainty.propagation import candidate_bounds, point_prediction, quantile_candidate_bounds
from .objectives import OBJECTIVES, objective_matrix
from .selection import select


def valid_actions(w: int, cfg: dict) -> np.ndarray:
    """Actions in [k_min, k_max] whose pool stays within [vm_min, vm_max]."""
    a, s = cfg["actions"], cfg["system"]
    ks = np.arange(a["k_min"], a["k_max"] + 1)
    return ks[(w + ks >= s["vm_min"]) & (w + ks <= s["vm_max"])]


def observed_metrics(obs: dict, metrics) -> dict:
    out = {m: obs[m] for m in metrics}
    out["rate"] = obs.get("rate")
    return out


def candidate_records(ks, n, b, F, sel, u_hat) -> list[dict]:
    recs = []
    for i in range(len(ks)):
        recs.append({"k": int(ks[i]), "n": int(n[i]), "mean_log": float(b["mean_log"][i]),
                     "sigma": float(b["sigma"][i]), "U": float(F[i, 0]), "cost": float(F[i, 1]),
                     "energy": float(F[i, 2]), "u_hat": float(u_hat[i]),
                     "feasible": bool(sel["feasible"][i]), "on_front": bool(sel["on_front"][i]),
                     "score": None if np.isnan(sel["score"][i]) else float(sel["score"][i])})
    assert list(recs[0]) == DECISION_CANDIDATE_FIELDS
    return recs


class Policy:
    name = "policy"

    def reset(self, rng: np.random.Generator) -> None:
        self.rng = rng

    def decide(self, obs: dict, state: dict):
        raise NotImplementedError


class _Threshold(Policy):
    def __init__(self, cfg, upper, lower, cooldown):
        self.cfg, self.upper, self.lower, self.cooldown = cfg, upper, lower, cooldown

    def signal(self, obs):
        raise NotImplementedError

    def decide(self, obs, state):
        if state.get("cooldown", 0) > 0:
            state["cooldown"] -= 1
            return 0, {"fallback": False}
        x = self.signal(obs)
        k = 1 if x > self.upper else (-1 if x < self.lower else 0)
        w = obs["n_running"] + obs["n_booting"]
        if k not in valid_actions(w, self.cfg):
            k = 0
        if k != 0:
            state["cooldown"] = self.cooldown
        return k, {"fallback": False, "signal": float(x)}


class CpuThreshold(_Threshold):
    name = "cpu_threshold"

    def signal(self, obs):
        return obs["cpu_util"]


class RtThreshold(_Threshold):
    name = "rt_threshold"

    def __init__(self, cfg, upper_frac, lower_frac, cooldown):
        slo = cfg["slo"]["rt_ms"]
        super().__init__(cfg, upper_frac * slo, lower_frac * slo, cooldown)

    def signal(self, obs):
        return obs["rt_mean_ms"]


class Opt(Policy):
    """Base paper: paper_linear stage 1 + MSE network; first k (from k_min up) predicted under the SLO."""
    name = "opt"

    def __init__(self, cfg, stage1, net):
        self.cfg, self.stage1, self.net = cfg, stage1, net

    def decide(self, obs, state):
        w = obs["n_running"] + obs["n_booting"]
        ks = valid_actions(w, self.cfg)
        b = point_prediction(self.stage1, self.net, observed_metrics(obs, self.stage1.metrics), w, ks)
        ok = b["U"] < self.cfg["slo"]["rt_ms"]
        fallback = not ok.any()
        i = int(np.argmax(ok)) if ok.any() else len(ks) - 1
        n = b["n"]
        F = objective_matrix(b["U"], n, np.zeros(len(ks)), self.cfg)
        sel = {"feasible": ok, "on_front": np.zeros(len(ks), bool), "score": np.full(len(ks), np.nan)}
        return int(ks[i]), {"fallback": fallback, "chosen_U": float(b["U"][i]),
                            "chosen_mean_log": float(b["mean_log"][i]),
                            "candidates": candidate_records(ks, n, b, F, sel, np.zeros(len(ks)))}


class Proposed(Policy):
    """Stage 1 -> ensemble with propagation -> chance constraint -> Pareto selection."""

    def __init__(self, cfg, stage1, ensemble=None, mode="full", rule=None, weights=None, objectives=OBJECTIVES,
                 z=None, qmodel=None, name="proposed"):
        self.cfg, self.stage1, self.ensemble, self.mode = cfg, stage1, ensemble, mode
        self.rule = rule or cfg["selection"]["rule"]
        self.weights = weights or cfg["selection"]["weights"]
        self.rho = cfg["selection"]["rho"]
        self.objectives = tuple(objectives)
        self.z = cfg["slo"]["z"] if z is None else z
        self.qmodel = qmodel
        self.name = name

    def u_hat(self, b, before, w, ks):
        if "cpu_util" in self.stage1.metrics:
            cpu = b["m_hat"][:, self.stage1.metrics.index("cpu_util")]
        else:
            cpu = self.stage1.predict(before, w, ks, metrics=["cpu_util"])[:, 0]
        return np.clip(cpu / 100.0, 0.0, 1.0)

    def decide(self, obs, state):
        w = obs["n_running"] + obs["n_booting"]
        ks = valid_actions(w, self.cfg)
        before = {m: obs[m] for m in self.stage1.models} | {"rate": obs.get("rate")}
        if self.qmodel is not None:
            b = quantile_candidate_bounds(self.stage1, self.qmodel, before, w, ks)
        else:
            b = candidate_bounds(self.stage1, self.ensemble, before, w, ks, self.z,
                                 self.cfg["propagation"]["draws"], self.rng, mode=self.mode,
                                 resample=self.cfg["propagation"]["resample"])
        n = np.asarray(b["n"], float)
        u_hat = self.u_hat(b, before, w, ks)
        F = objective_matrix(b["U"], n, u_hat, self.cfg)
        sel = select(ks, n, F, self.cfg["slo"]["rt_ms"], self.rule, self.weights, self.rho, self.objectives)
        i = sel["index"]
        return int(ks[i]), {"fallback": sel["fallback"], "chosen_U": float(b["U"][i]),
                            "chosen_mean_log": float(b["mean_log"][i]),
                            "candidates": candidate_records(ks, n, b, F, sel, u_hat)}


# ---------------------------------------------------------------------------------------------
ABLATIONS = ["abl_stage1_linear", "abl_single_member", "abl_no_uncertainty", "abl_epistemic_only",
             "abl_no_propagation", "abl_quantile_nn", "abl_quantile_hgb", "abl_min_cost",
             "abl_two_objective", "abl_with_rate"]


def load_models(model_dir) -> dict:
    md = Path(model_dir)
    models = {
        "stage1": MetricEstimator.load(md / "stage1.joblib"),
        "stage1_linear": MetricEstimator.load(md / "stage1_paper_linear.joblib"),
        "ensemble": RTEnsemble.load(md / "rt_ensemble"),
        "mse": RTEnsemble.load(md / "rt_mse"),
        "conformal_path": md / "conformal.json",
    }
    optional = {"quantile_nn": (QuantileNetModel.load, md / "quantile_nn"),
                "quantile_hgb": (QuantileHGB.load, md / "quantile_hgb.joblib"),
                "ensemble_rate": (RTEnsemble.load, md / "with_rate" / "rt_ensemble")}
    for key, (loader, path) in optional.items():
        if path.exists():
            models[key] = loader(path)
    return models


def make_policy(name: str, cfg: dict, exp_cfg: dict, models: dict | None = None) -> Policy:
    b = exp_cfg["baselines"]
    if name == "cpu_threshold":
        c = b["cpu_threshold"]
        return CpuThreshold(cfg, c["upper"], c["lower"], c["cooldown_steps"])
    if name == "rt_threshold":
        c = b["rt_threshold"]
        return RtThreshold(cfg, c["upper_frac_slo"], c["lower_frac_slo"], c["cooldown_steps"])
    if models is None:
        raise ValueError(f"policy {name} needs trained models")
    if name == "opt":
        return Opt(cfg, models["stage1_linear"], models["mse"])

    zf = lambda mode: bound_z(cfg, mode, models.get("conformal_path"))   # noqa: E731
    s1, ens = models["stage1"], models["ensemble"]
    if name == "proposed":
        return Proposed(cfg, s1, ens, "full", z=zf("full"))
    if name.startswith("proposed_w"):                      # weight sweep, e.g. proposed_w311
        weights = [float(c) for c in name[len("proposed_w"):]]
        return Proposed(cfg, s1, ens, "full", weights=weights, z=zf("full"), name=name)
    table = {
        "abl_stage1_linear": lambda: Proposed(cfg, models["stage1_linear"], ens, "full", z=zf("full"), name=name),
        "abl_single_member": lambda: Proposed(cfg, s1, ens.subset(1), "full", z=zf("full"), name=name),
        "abl_no_uncertainty": lambda: Proposed(cfg, s1, ens, "none", z=0.0, name=name),
        "abl_epistemic_only": lambda: Proposed(cfg, s1, ens, "epistemic", z=zf("epistemic"), name=name),
        "abl_no_propagation": lambda: Proposed(cfg, s1, ens, "ale_epi", z=zf("ale_epi"), name=name),
        "abl_quantile_nn": lambda: Proposed(cfg, s1, qmodel=models["quantile_nn"], name=name),
        "abl_quantile_hgb": lambda: Proposed(cfg, s1, qmodel=models["quantile_hgb"], name=name),
        "abl_min_cost": lambda: Proposed(cfg, s1, ens, "full", rule="min_cost", z=zf("full"), name=name),
        "abl_two_objective": lambda: Proposed(cfg, s1, ens, "full", objectives=("cost", "energy"),
                                              z=zf("full"), name=name),
        "abl_with_rate": lambda: Proposed(cfg, s1, models["ensemble_rate"], "full", z=zf("full"), name=name),
    }
    if name not in table:
        raise ValueError(f"unknown policy {name}")
    return table[name]()
