"""Aggregate a sweep's run summaries into tables and figures (Phase 5 subset used by Step 2.13).

Writes results/tables/<env>_run_summaries.csv, <env>_main_results.csv, <env>_paired_tests.csv
and results/figures/<env>_policy_bars.png.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autoscaler.config import load_config
from autoscaler.eval.stats import paired_table, summary_table

METRICS = ["violation_rate", "cost_usd", "energy_kwh", "rt_mean_ms", "rt_p95_ms", "vm_hours", "mean_pool",
           "churn", "mean_abs_k", "fallback_rate", "realised_coverage", "mean_margin_ms", "wall_clock_s"]


def bars(summary: pd.DataFrame, order, path):
    workloads = list(dict.fromkeys(summary.workload))
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    width = 0.8 / len(workloads)
    x = np.arange(len(order))
    for ax, (m, label) in zip(axes, [("violation_rate", "violation rate"), ("cost_usd", "cost (USD / day)"),
                                     ("energy_kwh", "energy (kWh / day)")]):
        for j, wl in enumerate(workloads):
            s = summary[summary.workload == wl].set_index("policy").reindex(order)
            ax.bar(x + (j - (len(workloads) - 1) / 2) * width, s[m], width, yerr=s[f"{m}_ci"], label=wl,
                   capsize=2, error_kw={"lw": 0.6})
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8, ncol=len(workloads))
    axes[-1].set_xticks(x, order, rotation=40, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="synthetic")
    ap.add_argument("--sweep", required=True, help="sweep run directory with run_summaries.csv")
    ap.add_argument("--config", default="configs/prototype.yaml")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--figures", default="results/figures")
    args = ap.parse_args()
    load_config(args.config)
    tables, figs = Path(args.tables), Path(args.figures)
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(args.sweep) / "run_summaries.csv")
    df = df[df.status == "ok"]
    df.to_csv(tables / f"{args.env}_run_summaries.csv", index=False)
    summ = summary_table(df, METRICS)
    summ.to_csv(tables / f"{args.env}_main_results.csv", index=False)
    paired = paired_table(df, "proposed", ["violation_rate", "cost_usd", "energy_kwh"])
    paired.to_csv(tables / f"{args.env}_paired_tests.csv", index=False)

    order = [p for p in ["cpu_threshold", "rt_threshold", "opt", "proposed"] if p in set(df.policy)]
    order += sorted(set(df.policy) - set(order))
    bars(summ, order, figs / f"{args.env}_policy_bars.png")

    pd.set_option("display.width", 250)
    view = summ.pivot(index="policy", columns="workload", values="violation_rate").reindex(order)
    print("violation rate (mean over seeds):")
    print(view.to_string(float_format=lambda v: f"{v:.4f}"))
    print("wrote tables to", tables)


if __name__ == "__main__":
    main()
