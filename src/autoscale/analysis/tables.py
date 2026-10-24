"""KPI tables in the paper's layout (mean +/- std over seeds)."""
from __future__ import annotations
import pandas as pd
from .plots import POLICY_ORDER, POLICY_LABEL

COLS = [("qos_viol_pct", "QoS viol. (%)", 2), ("rt_avg_ms", "RTavg (ms)", 0), ("vm_used_pct", "VM used (% of max)", 1)]


def kpi_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (app, pol), g in runs.groupby(["app", "policy"]):
        r = dict(app=app, policy=POLICY_LABEL.get(pol, pol), _o=POLICY_ORDER.index(pol) if pol in POLICY_ORDER else 9)
        for c, lab, nd in COLS:
            r[lab] = f"{g[c].mean():.{nd}f} +/- {g[c].std():.{nd}f}"
        r["n_seeds"] = len(g)
        rows.append(r)
    return pd.DataFrame(rows).sort_values(["app", "_o"]).drop(columns="_o")


def save_table(df: pd.DataFrame, stem):
    df.to_csv(f"{stem}.csv", index=False)
    with open(f"{stem}.md", "w") as f:
        f.write(df.to_markdown(index=False))
    with open(f"{stem}.tex", "w") as f:
        f.write(df.to_latex(index=False, escape=True))
