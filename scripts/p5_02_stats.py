"""Statistics over the grid (Step 5.2).

Writes main_results.csv, ablations.csv, sensitivity.csv, paired_tests.csv and hypervolume.csv:
mean and 95% t interval per (env, workload, policy, metric); paired Wilcoxon tests of proposed (weights 1,1,1)
against each baseline and each ablation per workload and metric, Holm-corrected within each table (primary, as specified) and, as a secondary family, within each (env, workload,
metric); hypervolume of
each policy's points in min-max normalised (violation rate, cost, energy) space, reference point (1.1, 1.1, 1.1).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from autoscaler.eval.stats import holm, hypervolume, mean_ci, minmax_normalise, paired_wilcoxon

METRICS = ["violation_rate", "cost_usd", "energy_kwh", "rt_mean_ms", "rt_p95_ms", "vm_hours", "churn",
           "mean_abs_k", "fallback_rate", "realised_coverage", "mean_margin_ms"]
TESTED = ["violation_rate", "cost_usd", "energy_kwh"]
REFERENCE = "proposed_w111"
HV_OBJECTIVES = ["violation_rate", "cost_usd", "energy_kwh"]


def ci_table(df: pd.DataFrame, keys) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for m in METRICS:
            if m in g:
                row[m], row[f"{m}_ci"] = mean_ci(g[m])
        row["runs"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, reference: pd.DataFrame, others: pd.DataFrame, table: str) -> pd.DataFrame:
    rows = []
    for (env, wl), g in others.groupby(["env", "workload"]):
        ref = reference[(reference.env == env) & (reference.workload == wl)].set_index("seed")
        for pol, gp in g.groupby("policy"):
            gp = gp.set_index("seed")
            common = ref.index.intersection(gp.index)
            for m in TESTED:
                med, p = paired_wilcoxon(ref.loc[common, m], gp.loc[common, m]) if len(common) else (np.nan, np.nan)
                mean_d, ci_d = mean_ci(ref.loc[common, m] - gp.loc[common, m]) if len(common) else (np.nan, np.nan)
                rows.append({"table": table, "env": env, "workload": wl, "reference": REFERENCE, "policy": pol,
                             "metric": m, "pairs": len(common), "median_diff_ref_minus_policy": med,
                             "mean_diff_ref_minus_policy": mean_d, "mean_diff_ci": ci_d, "p": p})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = holm(out.p.fillna(1.0).to_numpy())        # Holm across all comparisons of this table
        out["p_holm_family"] = np.nan                             # secondary: Holm within (env, workload, metric)
        for _, g in out.groupby(["env", "workload", "metric"]):
            out.loc[g.index, "p_holm_family"] = holm(g.p.fillna(1.0).to_numpy())
        # smallest p the exact two-sided signed-rank test can give with this many pairs, and the smallest
        # Holm-adjusted value it allows in each family: a comparison can only be significant if this is < alpha
        out["p_min_possible"] = np.minimum(1.0, 2.0 / 2.0 ** out.pairs)
        out["p_holm_min_possible"] = np.minimum(1.0, out.p_min_possible * len(out))
        out["p_holm_family_min_possible"] = np.minimum(1.0, out.p_min_possible * out.groupby(
            ["env", "workload", "metric"]).metric.transform("size"))
    return out


def hypervolumes(main: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (env, wl), g in main.groupby(["env", "workload"]):
        means = g.groupby("policy")[HV_OBJECTIVES].mean()
        if means.isna().any().any() or len(means) == 0:
            continue
        ref_set = means.to_numpy()
        families = {}
        for pol in means.index:
            fam = "proposed" if pol.startswith("proposed_w") else pol
            families.setdefault(fam, []).append(pol)
        for fam, pols in families.items():
            pts = minmax_normalise(means.loc[pols].to_numpy(), ref_set)
            rows.append({"env": env, "workload": wl, "policy": fam, "points": len(pols),
                         "hypervolume": hypervolume(pts)})
    return pd.DataFrame(rows)


def counterfactual_band(by_k: Path, lo: float = 0.90, hi: float = 0.98) -> pd.DataFrame | None:
    """Probability that a bound with true coverage c lands inside [lo, hi] for one k and for every k at once,
    given the number of pairs per k in counterfactual_by_k.csv (acceptance check 3)."""
    from scipy.stats import binom
    if not by_k.exists():
        return None
    cf = pd.read_csv(by_k)
    n, n_k = int(cf.pairs.min()), len(cf)
    k_lo, k_hi = int(np.ceil(lo * n - 1e-9)), int(np.floor(hi * n + 1e-9))
    rows = []
    for c in (0.94, 0.95, 0.96):
        one = float(binom.cdf(k_hi, n, c) - binom.cdf(k_lo - 1, n, c))
        rows.append({"true_coverage": c, "pairs_per_k": n, "ks": n_k, "band_lo": lo, "band_hi": hi,
                     "covered_lo": k_lo, "covered_hi": k_hi, "p_pass_one_k": one, "p_pass_all_k": one ** n_k})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", default="results/tables/run_summaries.csv")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--model-tables", default="results/tables", help="where counterfactual_by_k.csv is")
    args = ap.parse_args()
    tables = Path(args.tables)
    tables.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summaries)
    df = df[df.status == "ok"]

    main_df = df[df.part == "main"]
    abl_df = df[df.part == "ablations"]
    sens_df = df[df.part == "sensitivity"]
    reference = main_df[main_df.policy == REFERENCE]

    ci_table(main_df, ["env", "workload", "policy"]).to_csv(tables / "main_results.csv", index=False)
    abl = ci_table(pd.concat([abl_df, reference[reference.workload.isin(abl_df.workload.unique())]]),
                   ["env", "workload", "policy"])
    if len(abl):
        for m in TESTED:                      # change relative to proposed (weights 1,1,1)
            base = abl[abl.policy == REFERENCE].set_index(["env", "workload"])[m]
            abl[f"{m}_delta_vs_proposed"] = [v - base.get((e, w), np.nan)
                                             for v, e, w in zip(abl[m], abl.env, abl.workload)]
    abl.to_csv(tables / "ablations.csv", index=False)
    ci_table(sens_df, ["env", "policy", "power_model", "power_ratio"]).to_csv(tables / "sensitivity.csv", index=False)

    tests = pd.concat([paired(df, reference, main_df[main_df.policy.isin(["cpu_threshold", "rt_threshold", "opt"])],
                              "main"),
                       paired(df, reference, abl_df, "ablations")], ignore_index=True)
    tests.to_csv(tables / "paired_tests.csv", index=False)
    hv = hypervolumes(main_df)
    hv.to_csv(tables / "hypervolume.csv", index=False)
    band = counterfactual_band(Path(args.model_tables) / "counterfactual_by_k.csv")
    if band is not None:
        band.to_csv(tables / "counterfactual_band_probability.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"{len(df)} ok runs; main {len(main_df)}, ablations {len(abl_df)}, sensitivity {len(sens_df)}")
    print(tests.head(12).to_string(index=False))
    print(hv.to_string(index=False))


if __name__ == "__main__":
    main()
