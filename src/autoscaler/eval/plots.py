"""Figures of Step 5.3. Every function saves a PNG at 200 dpi and a PDF next to it."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAIN_ORDER = ["cpu_threshold", "rt_threshold", "opt", "proposed_w111"]


def save(fig, path) -> list[Path]:
    path = Path(path).with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return [path, path.with_suffix(".pdf")]


def workloads(series: dict, path):
    """1. Workload shapes: {shape: rates}."""
    fig, axes = plt.subplots(len(series), 1, figsize=(9, 1.8 * len(series)), sharex=True, squeeze=False)
    for ax, (shape, rates) in zip(axes[:, 0], series.items()):
        ax.plot(rates, lw=0.8, color="#2f5d8a")
        ax.set_title(shape, fontsize=9, loc="left")
        ax.set_ylabel("req/s")
    axes[-1, 0].set_xlabel("step")
    return save(fig, path)


def time_series(logs: dict, slo_ms: float, path, title: str):
    """2. One seed of one workload: rate, pool per policy, response time against the SLO. logs: {policy: df}."""
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    first = next(iter(logs.values()))
    ax[0].plot(first.step, first.rate, color="0.4", lw=0.8)
    ax[0].set_ylabel("rate (req/s)")
    ax[0].set_title(title, fontsize=10, loc="left")
    for pol, log in logs.items():
        pool = log.n_running + log.n_booting + (log.n_draining if "n_draining" in log else 0)
        ax[1].step(log.step, pool, where="post", lw=0.9, label=pol)
        ax[2].plot(log.step, log.rt_mean_ms, lw=0.7, label=pol)
    ax[1].set_ylabel("VMs")
    ax[1].legend(fontsize=7, ncol=4)
    ax[2].axhline(slo_ms, color="#b33", ls="--", lw=1, label="SLO")
    ax[2].set_yscale("log")
    ax[2].set_ylabel("mean RT (ms)")
    ax[2].set_xlabel("step")
    return save(fig, path)


def bars_ci(summary: pd.DataFrame, path, order=None, title=""):
    """3. Violation rate, cost and energy per policy per workload with 95% intervals."""
    order = order or [p for p in MAIN_ORDER if p in set(summary.policy)] + \
        sorted(set(summary.policy) - set(MAIN_ORDER))
    wls = list(dict.fromkeys(summary.workload))
    fig, axes = plt.subplots(3, 1, figsize=(max(6, 0.9 * len(order) + 3), 8), sharex=True)
    width = 0.8 / max(len(wls), 1)
    x = np.arange(len(order))
    for ax, m in zip(axes, ["violation_rate", "cost_usd", "energy_kwh"]):
        for j, wl in enumerate(wls):
            s = summary[summary.workload == wl].set_index("policy").reindex(order)
            ax.bar(x + (j - (len(wls) - 1) / 2) * width, s[m], width, yerr=s[f"{m}_ci"].fillna(0), capsize=2,
                   label=wl, error_kw={"lw": 0.6})
        ax.set_ylabel(m)
    axes[0].legend(fontsize=7, ncol=len(wls))
    axes[0].set_title(title, fontsize=10, loc="left")
    axes[-1].set_xticks(x, order, rotation=35, ha="right", fontsize=8)
    return save(fig, path)


def tradeoff(summary: pd.DataFrame, path, title=""):
    """4. Cost against violation rate, colour = energy; the proposed weight points are joined."""
    wls = list(dict.fromkeys(summary.workload))
    fig, axes = plt.subplots(1, len(wls), figsize=(5 * len(wls), 4), squeeze=False)
    markers = dict(zip(MAIN_ORDER[:3], ["s", "D", "^"]))
    vmin, vmax = summary.energy_kwh.min(), summary.energy_kwh.max()
    for ax, wl in zip(axes[0], wls):
        g = summary[summary.workload == wl]
        prop = g[g.policy.str.startswith("proposed_w")].sort_values("cost_usd")
        ax.plot(prop.violation_rate, prop.cost_usd, color="0.6", lw=0.8, zorder=1)
        sc = ax.scatter(prop.violation_rate, prop.cost_usd, c=prop.energy_kwh, vmin=vmin, vmax=vmax, marker="o",
                        label="proposed (weights)", zorder=2)
        for r in prop.itertuples():
            ax.annotate(r.policy.replace("proposed_w", "w"), (r.violation_rate, r.cost_usd), fontsize=6)
        for pol, mk in markers.items():
            b = g[g.policy == pol]
            if len(b):
                ax.scatter(b.violation_rate, b.cost_usd, c=b.energy_kwh, vmin=vmin, vmax=vmax, marker=mk,
                           edgecolors="k", s=60, label=pol, zorder=3)
        ax.set(xlabel="violation rate", ylabel="cost (USD / run)", title=f"{title} {wl}")
        ax.legend(fontsize=7)
        fig.colorbar(sc, ax=ax, label="energy (kWh)")
    return save(fig, path)


def ablation_deltas(abl: pd.DataFrame, path, title=""):
    """5. Change in each metric relative to proposed (weights 1,1,1), per ablation."""
    metrics = ["violation_rate", "cost_usd", "energy_kwh"]
    g = abl[abl.policy != "proposed_w111"]
    wls = list(dict.fromkeys(g.workload))
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 0.35 * max(len(set(g.policy)), 4) + 2), sharey=True)
    pols = sorted(set(g.policy))
    y = np.arange(len(pols))
    h = 0.8 / max(len(wls), 1)
    for ax, m in zip(axes, metrics):
        for j, wl in enumerate(wls):
            s = g[g.workload == wl].set_index("policy").reindex(pols)
            ax.barh(y + (j - (len(wls) - 1) / 2) * h, s[f"{m}_delta_vs_proposed"], h, label=wl)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"{m}: ablation - proposed", fontsize=9)
    axes[0].set_yticks(y, pols, fontsize=8)
    axes[0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10, x=0.01, ha="left")
    return save(fig, path)


def calibration(rel: dict, cond: dict, path):
    """6. Reliability (nominal vs observed coverage) per variant, and conditional coverage by predicted-RT bin.
    rel/cond: {env: DataFrame}."""
    envs = list(rel)
    fig, axes = plt.subplots(len(envs), 2, figsize=(11, 4.2 * len(envs)), squeeze=False)
    for row, env in enumerate(envs):
        r, c = rel[env], cond[env]
        ax = axes[row, 0]
        ax.plot([0.45, 1], [0.45, 1], color="0.6", ls="--", lw=1)
        ax.axhspan(0.93, 0.97, color="0.92", zorder=0)
        for v, g in r.groupby("variant", sort=False):
            if len(g) > 1:
                ax.plot(g.level, g.coverage, marker="o", label=v)
            else:
                ax.scatter(g.level, g.coverage, marker="s", label=v)
        ax.set(xlabel="nominal coverage", ylabel="observed (end to end)", title=f"{env}: reliability")
        ax.legend(fontsize=6)
        ax = axes[row, 1]
        for v, g in c.groupby("variant", sort=False):
            if v in ("full", "ale_epi", "epistemic", "quantile_nn", "quantile_hgb"):
                ax.plot(g.bin, g.coverage, marker="o", label=v)
        ax.axhline(0.95, color="0.6", ls="--", lw=1)
        ax.set(xlabel="predicted-RT bin (low to high)", ylabel="coverage at 0.95", title=f"{env}: conditional")
        ax.legend(fontsize=6)
    return save(fig, path)


def uncertainty_decomposition(tables: dict, path):
    """7. Mean sigma_ale, sigma_epi, sigma_stage1 against predicted utilisation. tables: {env: DataFrame}."""
    fig, axes = plt.subplots(1, len(tables), figsize=(5.5 * len(tables), 4), squeeze=False)
    for ax, (env, t) in zip(axes[0], tables.items()):
        for col, label in (("sigma_ale", "aleatoric"), ("sigma_epi", "epistemic"), ("sigma_stage1", "stage 1")):
            ax.plot(t.u_mid, t[col], marker="o", label=label)
        ax.set(xlabel="predicted post-scaling utilisation", ylabel="mean sigma (log RT)", title=env)
        ax.legend(fontsize=8)
    return save(fig, path)


def counterfactual(df: pd.DataFrame, path):
    """8. Predicted (exp(mean) and U_k) vs measured response time, by k."""
    ks = sorted(df.k.unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(2.6 * len(ks), 3.2), sharex=True, sharey=True, squeeze=False)
    lo = max(1.0, min(df.measured_rt_ms.min(), df.pred_rt_ms.min()))
    hi = max(df.measured_rt_ms.max(), df.U_ms.max())
    for ax, k in zip(axes[0], ks):
        g = df[df.k == k]
        ax.scatter(g.measured_rt_ms, g.pred_rt_ms, s=9, label="exp(mean)")
        ax.scatter(g.measured_rt_ms, g.U_ms, s=9, marker="^", label="U_k")
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"k={k:+d} cov={g.covered.mean():.2f}", fontsize=8)
        ax.set_xlabel("measured (ms)", fontsize=8)
    axes[0, 0].set_ylabel("predicted (ms)")
    axes[0, 0].legend(fontsize=6)
    return save(fig, path)


def power_sensitivity(df: pd.DataFrame, path, threshold: float = 1 / 0.84 ** 2):
    """9. Share of decisions with a multi-point cost-energy front against the power ratio."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for (env, model), g in df.groupby(["env", "model"]):
        g = g.sort_values("ratio")
        ax.plot(g.ratio, g.share_multi_point, marker="o" if model == "square" else "s", label=f"{env} {model}")
    ax.axvline(threshold, color="#b33", ls="--", lw=1, label=f"threshold 1/u_slo^2 = {threshold:.2f}")
    ax.set(xlabel="(P_max - P_idle) / P_idle", ylabel="share of decisions with >1 front point")
    ax.legend(fontsize=7)
    return save(fig, path)
