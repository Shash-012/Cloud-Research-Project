"""End-to-end calibration of every uncertainty mode and both quantile models (Step 2.9)."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autoscaler.data.dataset import load_transitions
from autoscaler.data.sources import add_source_args, resolve
from autoscaler.data.splits import split_from_config
from autoscaler.logging_utils import run_dir
from autoscaler.models.metric_estimator import MetricEstimator
from autoscaler.models.quantile import QuantileHGB, QuantileNetModel
from autoscaler.models.rt_ensemble import RTEnsemble
from autoscaler.uncertainty.calibration_metrics import (conditional_coverage, coverage, end_to_end,
                                                         end_to_end_quantile, gaussian_nll, mean_margin,
                                                         pinball_loss, reliability)
from autoscaler.uncertainty.conformal import conformal_z, save_conformal
from autoscaler.uncertainty.propagation import MODES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prototype.yaml")
    ap.add_argument("--data", default=None, help="transition table (default depends on --source)")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--out", default="results/runs")
    add_source_args(ap)
    args = ap.parse_args()
    src = resolve(args)
    cfg = src.cfg
    rd = run_dir(args.out, f"p2_04_calibration_report{src.suffix}", cfg)
    md, tables, figs = Path(args.model_dir), Path(args.tables), Path(args.figures)
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    _, calib, test = split_from_config(load_transitions(src.data, src.name), cfg)
    stage1 = MetricEstimator.load(md / "stage1.joblib")
    stage1_lin = MetricEstimator.load(md / "stage1_paper_linear.joblib")
    ens = RTEnsemble.load(md / "rt_ensemble")
    z = cfg["slo"]["z"]
    S = cfg["propagation"]["draws"]
    delta = cfg["slo"]["delta"]
    slo = cfg["slo"]["rt_ms"]
    levels = cfg["calibration_report"]["levels"]
    seed = cfg["calibration_report"]["seed"]
    resample = cfg["propagation"]["resample"]

    variants = [(mode, stage1, ens, mode) for mode in MODES]
    variants += [("full_single_member", stage1, ens.subset(1), "full"),
                 ("full_stage1_paper_linear", stage1_lin, ens, "full")]

    rows, rel, cond_tables, z_cal = [], {}, [], {}
    for i, (name, s1, e, mode) in enumerate(variants):
        rc = end_to_end(s1, e, calib, z, S, np.random.default_rng(seed + 2 * i), mode, resample)
        rt = end_to_end(s1, e, test, z, S, np.random.default_rng(seed + 2 * i + 1), mode, resample)
        zc = conformal_z(rc["y"], rc["mean_log"], rc["sigma"], delta) if np.all(rc["sigma"] > 0) else np.nan
        if name in MODES:
            z_cal[name] = zc
        cond = conditional_coverage(rt["y"], rt["U"], rt["mean_log"], cfg["calibration_report"]["bins"], slo)
        cond.insert(0, "variant", name)
        cond_tables.append(cond)
        U_conf = np.exp(rt["mean_log"] + zc * rt["sigma"]) if np.isfinite(zc) else np.full_like(rt["U"], np.nan)
        rows.append({
            "variant": name, "mode": mode, "z": z, "coverage": coverage(rt["y"], rt["U"]),
            "coverage_slo_bin": float(cond.loc[cond.contains_slo, "coverage"].mean()),
            "mean_margin_ms": mean_margin(rt["U"], rt["mean_log"]),
            "nll": gaussian_nll(rt["y"], rt["mean_log"], rt["sigma"]),
            "mean_sigma": float(np.mean(rt["sigma"])), "mean_sigma_ale": float(np.mean(rt["sigma_ale"])),
            "mean_sigma_epi": float(np.mean(rt["sigma_epi"])),
            "mean_sigma_stage1": float(np.mean(rt["sigma_stage1"])),
            "z_cal": zc, "coverage_conformal": coverage(rt["y"], U_conf) if np.isfinite(zc) else np.nan,
            "pinball": np.nan, "rows": len(test)})
        rel[name] = reliability(rt["y"], rt["mean_log"], rt["sigma"], levels)

    tau = cfg["quantile"]["tau"]
    for name, mdl in (("quantile_nn", QuantileNetModel.load(md / "quantile_nn")),
                      ("quantile_hgb", QuantileHGB.load(md / "quantile_hgb.joblib"))):
        rt = end_to_end_quantile(stage1, mdl, test)
        cond = conditional_coverage(rt["y"], rt["U"], rt["mean_log"], cfg["calibration_report"]["bins"], slo)
        cond.insert(0, "variant", name)
        cond_tables.append(cond)
        rows.append({"variant": name, "mode": "quantile", "z": np.nan, "coverage": coverage(rt["y"], rt["U"]),
                     "coverage_slo_bin": float(cond.loc[cond.contains_slo, "coverage"].mean()),
                     "mean_margin_ms": np.nan, "nll": np.nan, "mean_sigma": np.nan, "mean_sigma_ale": np.nan,
                     "mean_sigma_epi": np.nan, "mean_sigma_stage1": np.nan, "z_cal": np.nan,
                     "coverage_conformal": np.nan, "pinball": pinball_loss(rt["y"], rt["mean_log"], tau),
                     "rows": len(test)})
        rel[name] = {tau: coverage(rt["y"], rt["U"])}

    table = pd.DataFrame(rows)
    table.to_csv(tables / f"calibration{src.suffix}.csv", index=False)
    table.to_csv(rd / "calibration.csv", index=False)
    rel_rows = [{"variant": v, "level": float(lv), "coverage": float(c)}
                for v, vals in rel.items()
                for lv, c in (vals.items() if isinstance(vals, dict) else zip(levels, vals))]
    pd.DataFrame(rel_rows).to_csv(tables / f"calibration_reliability{src.suffix}.csv", index=False)
    cond_all = pd.concat(cond_tables)
    cond_all.to_csv(tables / f"calibration_conditional{src.suffix}.csv", index=False)
    save_conformal(z_cal, md / "conformal.json")
    save_conformal(z_cal, rd / "conformal.json")

    colours = dict(zip(list(rel), plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot([0.45, 1], [0.45, 1], color="0.6", lw=1, ls="--", label="ideal")
    for name, vals in rel.items():
        if isinstance(vals, dict):
            ax[0].scatter(list(vals), list(vals.values()), marker="s", label=name, zorder=3,
                          color=colours[name])
        else:
            ax[0].plot(levels, vals, marker="o", label=name, color=colours[name])
    ax[0].axhspan(0.93, 0.97, xmin=0, xmax=1, color="0.9", zorder=0)
    ax[0].set(xlabel="nominal coverage", ylabel="observed coverage (test, end to end)",
              title="Reliability")
    ax[0].legend(fontsize=7)
    for name in ["full", "ale_epi", "epistemic", "quantile_nn", "quantile_hgb"]:
        c = cond_all[cond_all.variant == name]
        ax[1].plot(range(len(c)), c.coverage, marker="o", label=name, color=colours[name])
    ax[1].axhline(0.95, color="0.6", ls="--", lw=1)
    ax[1].set(xlabel="bin of predicted RT (quantiles, low to high)", ylabel="coverage at 0.95",
              title="Conditional coverage")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / f"calibration_reliability{src.suffix}.png", dpi=200)
    fig.savefig(figs / f"calibration_reliability{src.suffix}.pdf")

    pd.set_option("display.width", 250)
    print(table[["variant", "coverage", "coverage_slo_bin", "mean_margin_ms", "nll", "mean_sigma",
                 "mean_sigma_ale", "mean_sigma_epi", "mean_sigma_stage1", "z_cal", "coverage_conformal",
                 "pinball"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(cond_all[cond_all.variant == "full"].to_string(index=False))


if __name__ == "__main__":
    main()
