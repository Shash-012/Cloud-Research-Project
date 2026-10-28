"""Per-run summaries (Step 5.1).

Grid mode (default): reads results/runs/manifest.csv and every listed run folder, computes summarise() after the
warm-up and writes results/tables/run_summaries.csv (one row per run, all environments).
Sweep mode (--sweep DIR): the Phase 2 prototype sweep, which already holds run_summaries.csv; writes
<env>_run_summaries.csv, <env>_main_results.csv, <env>_paired_tests.csv and <env>_policy_bars.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autoscaler.config import load_config
from autoscaler.eval.metrics import summarise
from autoscaler.eval.stats import paired_table, summary_table
from autoscaler.sim.loop import load_decisions

METRICS = ["violation_rate", "cost_usd", "energy_kwh", "rt_mean_ms", "rt_p95_ms", "vm_hours", "mean_pool",
           "churn", "mean_abs_k", "fallback_rate", "realised_coverage", "mean_margin_ms", "wall_clock_s"]
LAG = {"synthetic": 1, "cloudsim": 2}      # intervals between a decision and the interval it affects fully


def bars(summary: pd.DataFrame, order, path):
    workloads = list(dict.fromkeys(summary.workload))
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    width = 0.8 / len(workloads)
    x = np.arange(len(order))
    for ax, (m, label) in zip(axes, [("violation_rate", "violation rate"), ("cost_usd", "cost (USD / run)"),
                                     ("energy_kwh", "energy (kWh / run)")]):
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


def env_config(env: str, proto_path: str, cs_path: str) -> dict:
    return load_config(proto_path) if env == "synthetic" else load_config(cs_path)


def summarise_grid(manifest: Path, proto_path: str, cs_path: str) -> pd.DataFrame:
    man = pd.read_csv(manifest)
    man = man.drop_duplicates("run_id", keep="last")           # a resumed run's last attempt counts
    rows = []
    cfgs = {}
    for _, m in man.iterrows():
        row = {k: m[k] for k in ("run_id", "env", "part", "policy", "workload", "seed", "power_model",
                                 "power_ratio", "steps", "status")}
        row["wall_clock_s"] = m["wall_clock_s"]
        folder = Path(m["run_dir"])
        if m["status"] != "ok" or not (folder / "run_log.parquet").exists():
            rows.append(row)
            continue
        cfg = cfgs.setdefault(m["env"], env_config(m["env"], proto_path, cs_path))
        c = json.loads(json.dumps(cfg))
        c["time"]["warmup_steps"] = min(c["time"]["warmup_steps"], int(m["steps"]) // 4)
        log = pd.read_parquet(folder / "run_log.parquet")
        dec_path = folder / "decisions.jsonl.gz"
        dec = load_decisions(dec_path) if dec_path.exists() else []
        s = summarise(log, dec, c, lag=LAG[m["env"]])
        s.pop("wall_clock_s", None)
        rows.append({**row, **s})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default=None, help="sweep mode: environment label")
    ap.add_argument("--sweep", default=None, help="sweep mode: prototype sweep directory with run_summaries.csv")
    ap.add_argument("--manifest", default="results/runs/manifest.csv")
    ap.add_argument("--config", default="configs/prototype.yaml")
    ap.add_argument("--cloudsim-config", default="configs/cloudsim.yaml")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--prefix", default="", help="grid mode: prefix for the output table (e.g. smoke_)")
    args = ap.parse_args()
    tables, figs = Path(args.tables), Path(args.figures)
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)

    if args.sweep is None:
        df = summarise_grid(Path(args.manifest), args.config, args.cloudsim_config)
        out = tables / f"{args.prefix}run_summaries.csv"
        df.to_csv(out, index=False)
        ok = df[df.status == "ok"]
        print(f"{len(df)} runs in manifest, {len(ok)} ok -> {out}")
        print(ok.groupby(["env", "part", "policy"])[["violation_rate", "cost_usd", "energy_kwh"]].mean()
              .to_string(float_format=lambda v: f"{v:.4f}"))
        return

    env = args.env or "synthetic"
    df = pd.read_csv(Path(args.sweep) / "run_summaries.csv")
    df = df[df.status == "ok"]
    df.to_csv(tables / f"{env}_run_summaries.csv", index=False)
    summ = summary_table(df, METRICS)
    summ.to_csv(tables / f"{env}_main_results.csv", index=False)
    paired = paired_table(df, "proposed", ["violation_rate", "cost_usd", "energy_kwh"])
    paired.to_csv(tables / f"{env}_paired_tests.csv", index=False)
    order = [p for p in ["cpu_threshold", "rt_threshold", "opt", "proposed"] if p in set(df.policy)]
    order += sorted(set(df.policy) - set(order))
    bars(summ, order, figs / f"{env}_policy_bars.png")
    view = summ.pivot(index="policy", columns="workload", values="violation_rate").reindex(order)
    print("violation rate (mean over seeds):")
    print(view.to_string(float_format=lambda v: f"{v:.4f}"))
    print("wrote tables to", tables)


if __name__ == "__main__":
    main()
