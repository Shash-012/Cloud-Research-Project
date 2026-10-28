"""Counterfactual check of the predictor (Step 3.16).

Picks random (seed, step) pairs from held-out HoldPolicy runs (seeds that are neither calibration nor evaluation
seeds). For each pair and each k, the simulation is re-run from the start with the same seed and
--force-action STEP:k; runs are identical up to STEP, so they differ only in the action. The measured mean response
time in interval STEP+2 is compared with the model's exp(mean_log) and U_k from the interval-STEP observation.

HoldPolicy keeps the initial pool, so each pair's initial pool is chosen for it: large enough that the run is not
overloaded before STEP (u <= 0.95 at the highest rate up to STEP+2) and giving a target utilisation u0 ~ U(0.5, 0.9)
at STEP. Runs stop at STEP+3.
"""
import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autoscaler.config import load_config, merge_cloudsim
from autoscaler.decision.policies import load_models
from autoscaler.logging_utils import run_dir
from autoscaler.sim.cloudsim import read_run_log, run_cloudsim, workload_csv
from autoscaler.uncertainty.conformal import bound_z
from autoscaler.uncertainty.propagation import candidate_bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cloudsim.yaml")
    ap.add_argument("--prototype", default="configs/prototype.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--model-dir", default="models/cloudsim")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--pairs", type=int, default=None)
    ap.add_argument("--max-step", type=int, default=None, help="limit STEP (shorter runs)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reuse", default=None, help="analyse an existing p3_04 run directory without re-running")
    args = ap.parse_args()
    cs, proto, exp = load_config(args.config), load_config(args.prototype), load_config(args.experiments)
    cf = cs["counterfactual"]
    cfg = merge_cloudsim(proto, cs, exp)
    rd = Path(args.reuse) if args.reuse else run_dir(args.out, "p3_04_counterfactual", cfg)
    rng = np.random.default_rng(cf["seed"])
    steps = cs["time"]["steps_per_run"]
    max_step = min(args.max_step or steps - 3, steps - 3)
    mu = cfg["system"]["service_rate_per_vm"] / cfg["capacity_scale"]      # unscaled rates below
    n_pairs = args.pairs or cf["pairs"]

    pairs = []
    for i in range(n_pairs):
        seed = int(rng.choice(cf["holdout_seeds"]))
        step = int(rng.integers(cf["min_step"], max_step + 1))
        rates = pd.read_csv(workload_csv(cf["workload"], seed, steps, proto["workloads"]))["rate"].to_numpy()
        u0 = rng.uniform(0.5, 0.9)
        n0 = max(math.ceil(rates[step] / (u0 * mu)), math.ceil(rates[: step + 3].max() / (0.95 * mu)),
                 1 - min(cf["ks"]))
        n0 = min(n0, cs["cluster"]["vm_max"] - max(cf["ks"]))
        pairs.append({"pair": i, "seed": seed, "step": step, "n0": int(n0), "u0": float(u0)})

    jobs = [(p, k) for p in pairs for k in cf["ks"]]

    def one(job):
        p, k = job
        folder = rd / "runs" / f"pair{p['pair']:02d}_k{k:+d}"
        r = run_cloudsim("hold", cf["workload"], p["seed"], folder, proto, cs, exp, steps=p["step"] + 3,
                         force_action=(p["step"], k), initial_vms=p["n0"],
                         run_id=f"cf_pair{p['pair']}_k{k}")
        return p, k, r

    if not args.reuse:
        with ThreadPoolExecutor(args.workers) as ex:
            done = list(ex.map(one, jobs))
        failed = [r for _, _, r in done if r["status"] != "ok"]
        if failed:
            raise SystemExit(f"{len(failed)} counterfactual runs failed, e.g. {failed[0]}")

    models = load_models(args.model_dir)
    stage1, ens = models["stage1"], models["ensemble"]
    z = bound_z(cfg, "full", models.get("conformal_path"))
    rows = []
    for p in pairs:
        logs = {k: read_run_log(rd / "runs" / f"pair{p['pair']:02d}_k{k:+d}") for k in cf["ks"]}
        base = logs[0].set_index("step")
        # identical history up to STEP in every run (determinism + forced action only at STEP)
        hist = lambda k: logs[k].drop(columns=["run_id"]).set_index("step").loc[: p["step"] - 1]   # noqa: E731
        same = all(hist(k).equals(hist(0)) for k in cf["ks"])                                      # run_id differs by k
        obs = base.loc[p["step"]]
        w = int(obs["n_running"] + obs["n_booting"])
        before = {m: float(obs[m]) for m in stage1.models} | {"rate": float(obs["rate"])}
        b = candidate_bounds(stage1, ens, before, w, np.array(cf["ks"]), z, cfg["propagation"]["draws"],
                             np.random.default_rng(cf["seed"] + p["pair"]), mode="full")
        for i, k in enumerate(cf["ks"]):
            lk = logs[k].set_index("step")
            measured = float(lk.loc[p["step"] + 2, "rt_mean_ms"])
            rows.append({**p, "k": k, "w": w, "n_after": int(lk.loc[p["step"] + 2, "n_running"]),
                         "applied_k": int(lk.loc[p["step"], "k"]), "history_identical": same,
                         "measured_rt_ms": measured, "pred_rt_ms": float(np.exp(b["mean_log"][i])),
                         "U_ms": float(b["U"][i]), "sigma": float(b["sigma"][i]),
                         "covered": bool(measured <= b["U"][i])})
    df = pd.DataFrame(rows)
    tables, figs = Path(args.tables), Path(args.figures)
    df.to_csv(tables / "counterfactual.csv", index=False)
    df.to_csv(rd / "counterfactual.csv", index=False)
    by_k = df.groupby("k").apply(lambda g: pd.Series({
        "pairs": len(g), "coverage_U": g.covered.mean(),
        "median_abs_err_ms": float(np.median(np.abs(g.pred_rt_ms - g.measured_rt_ms))),
        "median_rel_err": float(np.median(np.abs(np.log(g.pred_rt_ms / g.measured_rt_ms)))),
        "mean_measured_ms": g.measured_rt_ms.mean(), "mean_pred_ms": g.pred_rt_ms.mean()}),
        include_groups=False).reset_index()
    by_k.to_csv(tables / "counterfactual_by_k.csv", index=False)

    fig, axes = plt.subplots(1, len(cf["ks"]), figsize=(2.6 * len(cf["ks"]), 3), sharex=True, sharey=True)
    for ax, k in zip(axes, cf["ks"]):
        g = df[df.k == k]
        ax.scatter(g.measured_rt_ms, g.pred_rt_ms, s=10, label="exp(mean)")
        ax.scatter(g.measured_rt_ms, g.U_ms, s=10, marker="^", label="U_k")
        lim = [max(1, df[["measured_rt_ms", "pred_rt_ms"]].min().min()), df[["measured_rt_ms", "U_ms"]].max().max()]
        ax.plot(lim, lim, color="0.5", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"k = {k:+d}  cov {g.covered.mean():.2f}", fontsize=9)
        ax.set_xlabel("measured RT (ms)")
    axes[0].set_ylabel("predicted (ms)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "counterfactual.png", dpi=200)
    fig.savefig(figs / "counterfactual.pdf")
    (rd / "summary.json").write_text(json.dumps({"z": z, "pairs": len(pairs), "runs": len(jobs)}, indent=2))
    pd.set_option("display.width", 200)
    print(by_k.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("history identical before STEP in every pair:", bool(df.history_identical.all()))


if __name__ == "__main__":
    main()
