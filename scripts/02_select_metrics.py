"""E1/E2: metric selection per app (t-test + R2 filter), stability over 200/400/720 observations, figures."""
import _common  # noqa
import json
import numpy as np, pandas as pd
from autoscale.config import load_config, RESULTS
from autoscale.data.dataset import TransitionDataset
from autoscale.selection.accuracy import select_metrics
from autoscale.env.metrics_gen import build_catalog
from autoscale.analysis import plots

cfg = load_config(); cat = build_catalog()
out_t, out_f = RESULTS / "tables", RESULTS / "figures"
out_t.mkdir(parents=True, exist_ok=True); out_f.mkdir(parents=True, exist_ok=True)
data, chosen, rows = {}, {}, []
for app in cfg["apps"]:
    ds = TransitionDataset.load(RESULTS / "raw" / f"dataset_{app}.npz"); data[app] = ds
    sets = {}
    for n in (200, 400, 720):
        if n <= len(ds):
            s = select_metrics(ds.head(n), cfg["alpha"], cfg["r2_threshold"], cfg["cohen_d_min"], cfg["select_train_frac"])
            sets[n] = set(s[s.selected].metric)
    full = select_metrics(ds, cfg["alpha"], cfg["r2_threshold"], cfg["cohen_d_min"], cfg["select_train_frac"])
    full = full.merge(cat[["kind", "group", "driver"]], left_on="idx", right_index=True)
    full.to_csv(out_t / f"E1_selection_{app}.csv", index=False)
    chosen[app] = sorted(full[full.selected].metric)
    ns = sorted(sets)
    jac = [len(sets[a] & sets[b]) / max(1, len(sets[a] | sets[b])) for a in ns for b in ns if a < b]
    rows.append(dict(app=app, n_selected=len(chosen[app]), n_dynamic=int(full.dynamic.sum()),
                     n_predictable=int(full.predictable.sum()),
                     false_pos_on_irrelevant=int(full[full.kind.isin(["const", "drift", "noise", "total", "slow"]) & full.selected].shape[0]),
                     stability_min_jaccard=min(jac) if jac else np.nan))
    top = full[full.selected].sort_values("p").head(6).metric.tolist()
    plots.plot_delta_boxplots(ds, top, out_f / f"E1_delta_boxplot_{app}.png", f"{app}: selected metrics")
    plots.plot_r2_bars(full, cfg["r2_threshold"], out_f / f"E2_r2_{app}.png", f"{app}: R2 of Eq. 1")
    print(app, "selected", len(chosen[app]), "| dynamic", int(full.dynamic.sum()), "| after R2 filter", int(full.selected.sum()))
plots.plot_pca(data, out_f / "E1_pca.png")
summary = pd.DataFrame(rows); summary.to_csv(out_t / "E1_summary.csv", index=False); print(summary.to_string(index=False))
pairs = {(a, b): len(set(chosen[a]) & set(chosen[b])) for a in chosen for b in chosen if a < b}
print("overlap between apps:", pairs, "-> subsets differ" if any(v < min(len(chosen[a]), len(chosen[b])) for (a, b), v in pairs.items()) else "")
json.dump(chosen, open(out_t / "selected_metrics.json", "w"), indent=1)
