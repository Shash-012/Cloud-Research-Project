"""Metric filters and stage-1 estimators (Steps 2.4, 2.5; CloudSim Plus data with --source cloudsim, Step 3.15)."""
import argparse
import json
from pathlib import Path

import pandas as pd

from autoscaler.data.dataset import load_transitions
from autoscaler.data.splits import split_from_config
from autoscaler.logging_utils import run_dir
from autoscaler.models.metric_estimator import MetricEstimator
from autoscaler.models.metric_filters import final_metric_set, r2_filter, save_selected, ttest_filter
from autoscaler.data.sources import add_source_args, resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prototype.yaml")
    ap.add_argument("--data", default=None, help="transition table (default depends on --source)")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--jobs", type=int, default=5)
    add_source_args(ap)
    args = ap.parse_args()
    src = resolve(args)
    cfg, METRICS = src.cfg, src.metrics
    rd = run_dir(args.out, f"p2_02_fit_stage1{src.suffix}", cfg)
    model_dir, tables = Path(args.model_dir), Path(args.tables)
    tables.mkdir(parents=True, exist_ok=True)

    train, _, _ = split_from_config(load_transitions(src.data, src.name), cfg)
    f = cfg["metric_filter"]

    tt = ttest_filter(train, METRICS, f["ttest_alpha"])
    # Fit every metric so the table is complete; the selection uses both filters.
    est = MetricEstimator(cfg).fit(train, METRICS, n_jobs=args.jobs, group_col=src.group_col)
    r2_keep = r2_filter(est.scores, f["r2_threshold"])
    selected = final_metric_set(tt, r2_keep, METRICS)
    if not selected:
        raise SystemExit("no metric passed both filters")
    est.restrict(selected)
    est.save(model_dir / "stage1.joblib")

    # base-paper estimator for OPT and abl_stage1_linear (same metric set)
    lin = MetricEstimator(cfg, candidates=["paper_linear"]).fit(train, selected, n_jobs=args.jobs,
                                                                    group_col=src.group_col)
    lin.save(model_dir / "stage1_paper_linear.joblib")

    save_selected(selected, model_dir / "selected_metrics.json",
                  {"ttest": tt, "r2_pass": r2_keep, "chosen": est.chosen})

    rows = []
    for m in METRICS:
        row = {"metric": m, **{c: round(est.scores[m][c], 4) for c in est.candidates},
               "chosen": est.chosen[m], "ttest_p": tt[m]["p"], "ttest_keep": tt[m]["keep"],
               "r2_keep": r2_keep[m], "selected": m in selected}
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(tables / f"stage1_r2{src.suffix}.csv", index=False)
    table.to_csv(rd / f"stage1_r2{src.suffix}.csv", index=False)
    (rd / "summary.json").write_text(json.dumps({"selected": selected, "chosen": est.chosen}, indent=2))
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))
    print("selected metrics:", selected)
    print("residual covariance diag:", est.residual_cov.diagonal().round(3))


if __name__ == "__main__":
    main()
