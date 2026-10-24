"""Figures. Reads data/tables only; every figure is regenerable from results/."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper")
POLICY_ORDER = ["rt", "cpu", "opt_reactive", "opt_proactive"]
POLICY_LABEL = {"rt": "RT", "cpu": "CPU", "opt_reactive": "OPT baseline", "opt_proactive": "OPT proact"}


def _save(fig, out):
    fig.tight_layout(); fig.savefig(out, dpi=200); plt.close(fig)


def plot_delta_boxplots(ds, metric_names, out, title=""):
    """Paper Fig 1/2: delta-m after scale-up vs scale-down for chosen metrics."""
    dm = ds.m_after - ds.m_before
    rows = []
    for n in metric_names:
        j = ds.names.index(n); z = dm[:, j]; sd = z.std() or 1.0
        for k, v in zip(ds.k, z / sd):
            if k != 0:
                rows.append(dict(metric=n, change="up" if k > 0 else "down", delta=v))
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(1.3 * len(metric_names) + 2, 3.6))
    sns.boxplot(data=df, x="metric", y="delta", hue="change", ax=ax, fliersize=2)
    ax.set_ylabel("normalised delta-m"); ax.set_xlabel(""); ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, out)


def plot_pca(datasets: dict, out):
    """Paper Fig 3: PCA of raw metric values and of delta-metrics, coloured by application."""
    from sklearn.decomposition import PCA
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, kind in zip(axs, ["raw metric values", "delta metrics"]):
        X = {a: (d.m_after if kind.startswith("raw") else d.m_after - d.m_before) for a, d in datasets.items()}
        allx = np.vstack(list(X.values())); mu, sd = allx.mean(0), allx.std(0); sd[sd == 0] = 1
        pca = PCA(2).fit((allx - mu) / sd)
        for a, x in X.items():
            z = pca.transform((x - mu) / sd); ax.scatter(z[:, 0], z[:, 1], s=8, alpha=0.6, label=a)
        ax.set_title(f"PCA of {kind}"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    axs[0].legend(); _save(fig, out)


def plot_r2_bars(sel_df, thr, out, title=""):
    d = sel_df[sel_df["dynamic"]].sort_values("r2", ascending=False)
    fig, ax = plt.subplots(figsize=(max(6, 0.25 * len(d)), 3.4))
    ax.bar(range(len(d)), d["r2"].clip(lower=-0.5), color=np.where(d["r2"] > thr, "#2A7F8E", "#C0504D"))
    ax.axhline(thr, color="k", ls="--", lw=1); ax.set_ylabel("held-out R2 of Eq. 1"); ax.set_xticks([])
    ax.set_title(title or "R2 of Eq. 1 for metrics that passed the t-test"); _save(fig, out)


def plot_pred_vs_actual(pred: pd.DataFrame, out):
    """Paper Fig 5: predicted vs actual RT for the five models."""
    models = list(pred["model"].unique())
    fig, axs = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.4), squeeze=False)
    for ax, m in zip(axs[0], models):
        d = pred[pred["model"] == m]; y, p = d["y_true"].to_numpy(), d["y_pred"].to_numpy()
        r2 = 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum(); mae = np.abs(p - y).mean()
        ax.scatter(y, p, s=6, alpha=0.6); lim = [0, max(y.max(), p.max())]; ax.plot(lim, lim, "b-", lw=1)
        ax.set_title(f"{m}\nMAE={mae:.0f} ms  r2={r2:.2f}", fontsize=9); ax.set_xlabel("actual RT (ms)")
    axs[0][0].set_ylabel("predicted RT (ms)"); _save(fig, out)


def plot_rt_by_vm(ds, out):
    """Paper Fig 6 (top right): RT box plot per VM count -> no direct VM-RT link in the training data."""
    df = pd.DataFrame(dict(vm=(ds.w_before + ds.k).astype(int), rt=ds.rt_after))
    fig, ax = plt.subplots(figsize=(5, 3.4)); sns.boxplot(data=df, x="vm", y="rt", ax=ax, fliersize=2)
    ax.set_xlabel("number of VMs"); ax.set_ylabel("RT (ms)"); _save(fig, out)


def plot_timeseries(runs: dict, qos_ms: float, out, title=""):
    """Paper Fig 4: load, RT and VM count for two policies on the same workload. runs: label -> interval df."""
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    first = next(iter(runs.values()))
    ax2 = axs[0].twinx(); ax2.plot(first["t"] / 60, first["rr"], "g--", lw=0.8, label="request rate"); ax2.set_ylabel("req/s")
    ax2.grid(False)
    for lab, iv in runs.items():
        axs[0].plot(iv["t"] / 60, iv["rt_mean_ms"], lw=1, label=lab)
        axs[1].step(iv["t"] / 60, iv["w"], where="post", lw=1.2, label=lab)
    axs[0].axhline(qos_ms, color="r", ls=":", lw=1); axs[0].set_ylabel("RT (ms)"); axs[0].set_yscale("log")
    axs[1].set_ylabel("VMs"); axs[1].set_xlabel("time (min)"); axs[0].legend(loc="upper left"); axs[0].set_title(title)
    _save(fig, out)


def plot_policy_boxes(runs: pd.DataFrame, value: str, out, ylabel=None, title=""):
    d = runs.copy(); d["policy"] = d["policy"].map(POLICY_LABEL)
    order = [POLICY_LABEL[p] for p in POLICY_ORDER if POLICY_LABEL[p] in set(d["policy"])]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    sns.boxplot(data=d, x="app", y=value, hue="policy", hue_order=order, ax=ax, fliersize=2)
    ax.set_ylabel(ylabel or value); ax.set_xlabel(""); ax.set_title(title); _save(fig, out)
