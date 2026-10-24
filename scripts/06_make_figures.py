"""Regenerate every results figure/table from results/raw (E3-E6)."""
import _common  # noqa
import pandas as pd
from autoscale.config import load_config, RESULTS
from autoscale.data.dataset import TransitionDataset
from autoscale.analysis import plots
from autoscale.analysis.tables import kpi_table, save_table

cfg = load_config(); raw, fig, tab = RESULTS / "raw", RESULTS / "figures", RESULTS / "tables"
fig.mkdir(exist_ok=True, parents=True); tab.mkdir(exist_ok=True, parents=True)
if (raw / "e3_predictions.csv").exists():
    plots.plot_pred_vs_actual(pd.read_csv(raw / "e3_predictions.csv"), fig / "E3_pred_vs_actual.png")
    for p in raw.glob("dataset_constrr_*.npz"):
        plots.plot_rt_by_vm(TransitionDataset.load(p), fig / f"E3_rt_by_vm_{p.stem.split('_')[-2]}_{p.stem.split('_')[-1]}.png")
names = {"waves": "E4", "webshop": "E5", "surge": "E6"}
for wl, eid in names.items():
    f = raw / f"runs_{wl}.csv"
    if not f.exists(): continue
    runs = pd.read_csv(f)
    save_table(kpi_table(runs), tab / f"{eid}_{wl}")
    plots.plot_policy_boxes(runs, "qos_viol_pct", fig / f"{eid}_{wl}_violations.png", "QoS violations (%)", f"{wl} load")
    plots.plot_policy_boxes(runs, "vm_used_pct", fig / f"{eid}_{wl}_vm_used.png", "VM used (% of max)", f"{wl} load")
    for app in runs.app.unique():
        ts = {}
        for pol, lab in (("opt_reactive", "reactive (OPT baseline)"), ("opt_proactive", "proactive (OPT proact)"), ("cpu", "CPU"), ("rt", "RT")):
            p = raw / f"ts_{app}_{wl}_{pol}.csv"
            if p.exists(): ts[lab] = pd.read_csv(p)
        if ts:
            plots.plot_timeseries(ts, cfg["apps"][app]["qos_ms"], fig / f"{eid}_{wl}_{app}_timeseries.png", f"{app}, {wl}")
print("figures ->", fig, "| tables ->", tab)
