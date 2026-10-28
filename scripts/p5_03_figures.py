"""All figures of Step 5.3 (PNG at 200 dpi and PDF), from the grid summaries, run logs and model tables."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from autoscaler.config import load_config, merge_cloudsim
from autoscaler.decision.objectives import candidate_cost, candidate_energy
from autoscaler.decision.pareto import pareto_mask
from autoscaler.env.workloads import SHAPES, make_workload
from autoscaler.eval import plots
from autoscaler.seeds import STREAM_WORKLOAD, child_rngs
from autoscaler.sim.loop import load_decisions


def decomposition_table(env: str, proto: dict, cs: dict, model_dir: Path, bins: int = 8) -> pd.DataFrame | None:
    """Mean sigma components on the env's test split, binned by predicted post-scaling cpu utilisation."""
    from autoscaler.data.dataset import CLOUDSIM_TRANSITIONS, SYNTHETIC_TRANSITIONS, load_transitions
    from autoscaler.data.splits import split_from_config
    from autoscaler.models.metric_estimator import MetricEstimator
    from autoscaler.models.rt_ensemble import RTEnsemble
    from autoscaler.uncertainty.calibration_metrics import end_to_end
    data = CLOUDSIM_TRANSITIONS if env == "cloudsim" else SYNTHETIC_TRANSITIONS
    if not Path(data).exists() or not (model_dir / "stage1.joblib").exists():
        return None
    cfg = merge_cloudsim(proto, cs) if env == "cloudsim" else proto
    _, _, test = split_from_config(load_transitions(data, env), cfg)
    s1 = MetricEstimator.load(model_dir / "stage1.joblib")
    ens = RTEnsemble.load(model_dir / "rt_ensemble")
    out = end_to_end(s1, ens, test, cfg["slo"]["z"], cfg["propagation"]["draws"], np.random.default_rng(5300))
    u = np.clip(s1.predict_frame(test, ["cpu_util"])[:, 0] / 100.0, 0, 1)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(u, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        if sel.sum() >= 10:
            rows.append({"u_mid": (edges[b] + edges[b + 1]) / 2, "rows": int(sel.sum()),
                         **{c: float(out[c][sel].mean()) for c in ("sigma_ale", "sigma_epi", "sigma_stage1")}})
    return pd.DataFrame(rows)


def power_shares(summ: pd.DataFrame, proto: dict, cs: dict) -> pd.DataFrame:
    """Share of non-fallback proposed decisions whose feasible (cost, energy) front has > 1 point, per setting."""
    rows = []
    sens = summ[(summ.part == "sensitivity") & (summ.status == "ok") & (summ.policy == "proposed")]
    for (env, model, ratio), g in sens.groupby(["env", "power_model", "power_ratio"]):
        cfg = proto if env == "synthetic" else merge_cloudsim(proto, cs)
        p_max = cfg["power"]["p_max_w"]
        pcfg = {"model": model, "p_idle_w": p_max / (1 + ratio), "p_max_w": p_max}
        sizes = []
        for run_dir in g.run_dir if "run_dir" in g else []:
            f = Path(run_dir) / "decisions.jsonl.gz"
            if not f.exists():
                continue
            for d in load_decisions(f):
                feas = [c for c in d["candidates"] if c["feasible"]]
                if d["fallback"] or not feas:
                    continue
                n = np.array([c["n"] for c in feas], float)
                u = np.array([c["u_hat"] for c in feas], float)
                F = np.column_stack([candidate_cost(n, cfg), candidate_energy(n, u, cfg, pcfg)])
                sizes.append(int(pareto_mask(F).sum()))
        if sizes:
            rows.append({"env": env, "model": model, "ratio": ratio, "decisions": len(sizes),
                         "share_multi_point": float(np.mean(np.array(sizes) > 1))})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--summaries", default=None, help="default: <tables>/run_summaries.csv")
    ap.add_argument("--model-tables", default="results/tables", help="where calibration/counterfactual tables are")
    ap.add_argument("--manifest", default="results/runs/manifest.csv")
    ap.add_argument("--config", default="configs/prototype.yaml")
    ap.add_argument("--cloudsim-config", default="configs/cloudsim.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    args = ap.parse_args()
    proto, cs, exp = load_config(args.config), load_config(args.cloudsim_config), load_config(args.experiments)
    tables, figs, mt = Path(args.tables), Path(args.figures), Path(args.model_tables)
    figs.mkdir(parents=True, exist_ok=True)
    made = []

    # 1. workloads (synthetic length and CloudSim Plus length)
    for env, steps in (("synthetic", proto["time"]["steps_per_run"]), ("cloudsim", cs["time"]["steps_per_run"])):
        series = {s: make_workload(s, steps, child_rngs(1, 3)[STREAM_WORKLOAD], proto["workloads"]) for s in SHAPES}
        made += plots.workloads(series, figs / f"fig1_workloads_{env}")

    summ_path = Path(args.summaries or tables / "run_summaries.csv")
    summ = pd.read_csv(summ_path) if summ_path.exists() else pd.DataFrame()
    if len(summ):
        man = pd.read_csv(args.manifest).drop_duplicates("run_id", keep="last")
        summ = summ.merge(man[["run_id", "run_dir"]], on="run_id", how="left")
    main_res = pd.read_csv(tables / "main_results.csv") if (tables / "main_results.csv").exists() else pd.DataFrame()
    abl = pd.read_csv(tables / "ablations.csv") if (tables / "ablations.csv").exists() else pd.DataFrame()

    for env in sorted(set(summ.env)) if len(summ) else []:
        slo = proto["slo"]["rt_ms"] if env == "synthetic" else merge_cloudsim(proto, cs)["slo"]["rt_ms"]
        ok = summ[(summ.env == env) & (summ.status == "ok") & (summ.part == "main")]
        # 2. example time series, one seed per workload
        for wl in sorted(set(ok.workload)):
            g = ok[ok.workload == wl]
            seed = int(g.seed.min())
            logs = {r.policy: pd.read_parquet(Path(r.run_dir) / "run_log.parquet")
                    for r in g[g.seed == seed].itertuples() if r.policy in plots.MAIN_ORDER or
                    r.policy == "proposed_w111"}
            if logs:
                made += plots.time_series(logs, slo, figs / f"fig2_timeseries_{env}_{wl}", f"{env}, {wl}, seed {seed}")
        m = main_res[main_res.env == env] if len(main_res) else main_res
        if len(m):
            made += plots.bars_ci(m, figs / f"fig3_bars_{env}", title=env)          # 3.
            made += plots.tradeoff(m, figs / f"fig4_tradeoff_{env}", title=env)     # 4.
        a = abl[abl.env == env] if len(abl) else abl
        if len(a) and "violation_rate_delta_vs_proposed" in a and (a.policy != "proposed_w111").any():
            made += plots.ablation_deltas(a, figs / f"fig5_ablations_{env}", title=env)   # 5.

    # 6. calibration, end to end, both environments where available
    rel, cond = {}, {}
    for env, suffix in (("synthetic", ""), ("cloudsim", "_cloudsim")):
        r, c = mt / f"calibration_reliability{suffix}.csv", mt / f"calibration_conditional{suffix}.csv"
        if r.exists() and c.exists():
            rel[env], cond[env] = pd.read_csv(r), pd.read_csv(c)
    if rel:
        made += plots.calibration(rel, cond, figs / "fig6_calibration")
    # 7. uncertainty decomposition
    dec = {env: t for env, md in (("synthetic", Path("models")), ("cloudsim", Path("models/cloudsim")))
           if (t := decomposition_table(env, proto, cs, md)) is not None and len(t)}
    if dec:
        made += plots.uncertainty_decomposition(dec, figs / "fig7_uncertainty_decomposition")
    # 8. counterfactual
    if (mt / "counterfactual.csv").exists():
        made += plots.counterfactual(pd.read_csv(mt / "counterfactual.csv"), figs / "fig8_counterfactual")
    # 9. power sensitivity from the sensitivity runs' decision logs
    if len(summ):
        shares = power_shares(summ, proto, cs)
        if len(shares):
            shares.to_csv(tables / "power_sensitivity_grid.csv", index=False)
            made += plots.power_sensitivity(shares, figs / "fig9_power_sensitivity")
    pngs = [p for p in made if p.suffix == ".png"]
    print(f"{len(pngs)} figures (PNG + PDF) in {figs}:")
    for p in pngs:
        print("  ", p.name)


if __name__ == "__main__":
    main()
