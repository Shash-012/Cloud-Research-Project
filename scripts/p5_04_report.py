"""Build the PDF report (Step 5.4) from the tables of Steps 5.1-5.2, the model tables and the figures.

python scripts/p5_04_report.py                      -> results/reports/final_report.pdf
python scripts/p5_04_report.py --smoke ...          -> a smoke-test report proving the chain works
"""
import argparse
import json
import platform
from pathlib import Path

import pandas as pd

from autoscaler.config import ROOT, load_config
from autoscaler.eval.report import Report, git_head, java_version, read

KEY = ["violation_rate", "violation_rate_ci", "cost_usd", "cost_usd_ci", "energy_kwh", "energy_kwh_ci", "runs"]


def pick(df, cols):
    return None if df is None else df[[c for c in cols if c in df.columns]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--model-tables", default="results/tables")
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--out", default="results/reports/final_report.pdf")
    ap.add_argument("--title", default="Uncertainty-aware, Pareto-optimal autoscaling: final report")
    ap.add_argument("--smoke", action="store_true", help="label the report as a smoke test of the pipeline")
    ap.add_argument("--note", default="", help="extra paragraph for the setup section")
    args = ap.parse_args()
    T, M, F = Path(args.tables), Path(args.model_tables), Path(args.figures)
    cs, proto, exp = (load_config(ROOT / f"configs/{n}.yaml") for n in ("cloudsim", "prototype", "experiments"))
    r = Report(args.title)

    # 1. setup
    r.h("1. Setup")
    lock = (ROOT / "requirements.lock.txt").read_text(encoding="utf-16" if (ROOT / "requirements.lock.txt")
                                                     .read_bytes()[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8")
    pins = [l for l in lock.splitlines() if l.split("==")[0].lower() in
            ("numpy", "pandas", "scipy", "scikit-learn", "torch", "matplotlib", "reportlab")]
    speed = read(M / "cloudsim_speed.csv")
    r.p(f"Environments: synthetic formula system ({proto['time']['steps_per_run']} one-minute steps) and CloudSim "
        f"Plus 8.5.7 ({cs['time']['steps_per_run']} steps, {cs['cluster']['hosts']} single-PE hosts, one VM per host, "
        f"VM start-up {cs['cluster']['vm_startup_delay_s']} s). capacity_scale s = {cs['requests']['capacity_scale']}; "
        f"effective SLO {cs['slo']['rt_ms'] / cs['requests']['capacity_scale']:.0f} ms. Evaluation seeds "
        f"{exp['eval_seeds'][0]}..{exp['eval_seeds'][-1]}, calibration seeds {exp['calibration_seeds'][0]}.."
        f"{exp['calibration_seeds'][-1]}. Git commit {git_head(ROOT)[:12]}. Python {platform.python_version()}; "
        f"{java_version()}; packages: {', '.join(pins)}.")
    if args.smoke:
        r.p("<b>Smoke test.</b> This report was built from a deliberately small grid (2 policies, 1 workload, "
            "2 seeds, shortened runs) to show that the runner, the bridge, the analysis and the report generator "
            "work end to end. Its numbers are not results.")
    if args.note:
        r.p(args.note)
    r.table(speed, "Run time of one 360-step CloudSim Plus run (Step 3.13)")

    # 2. model quality
    r.h("2. Model quality")
    for env, suf in (("synthetic", ""), ("CloudSim Plus", "_cloudsim")):
        r.h3(env)
        r.table(read(M / f"stage1_r2{suf}.csv"), f"Stage-1 cross-validated R2 and selected metrics ({env})")
        r.table(read(M / f"stage2_accuracy{suf}.csv"), f"Stage-2 accuracy with true post-scaling metrics ({env})")
        cal = read(M / f"calibration{suf}.csv")
        r.table(pick(cal, ["variant", "coverage", "coverage_slo_bin", "mean_margin_ms", "nll", "mean_sigma",
                           "z_cal", "coverage_conformal", "pinball"]),
                f"End-to-end calibration at the 0.95 level ({env})")
    r.figure(F / "fig6_calibration.png", "Reliability and conditional coverage, end to end")
    r.figure(F / "fig7_uncertainty_decomposition.png", "Uncertainty decomposition against predicted utilisation")

    # 3. main results
    r.h("3. Main results")
    main_res = read(T / "main_results.csv")
    r.table(pick(main_res, ["env", "workload", "policy"] + KEY), "Mean and 95% interval per policy and workload")
    tests = read(T / "paired_tests.csv")
    r.table(None if tests is None else tests[tests.table == "main"].drop(columns=["table"]),
            "Wilcoxon signed-rank tests, proposed (1,1,1) vs baselines, Holm-corrected")
    r.table(read(T / "hypervolume.csv"), "Hypervolume in normalised (violation, cost, energy) space, ref (1.1,1.1,1.1)")
    for f in sorted(F.glob("fig3_bars_*.png")) + sorted(F.glob("fig4_tradeoff_*.png")):
        r.figure(f, f.stem)
    for f in sorted(F.glob("fig2_timeseries_*.png"))[:4]:
        r.figure(f, f.stem)

    # 4. ablations
    r.h("4. Ablations")
    abl = read(T / "ablations.csv")
    r.table(pick(abl, ["env", "workload", "policy", "violation_rate", "cost_usd", "energy_kwh",
                       "violation_rate_delta_vs_proposed", "cost_usd_delta_vs_proposed",
                       "energy_kwh_delta_vs_proposed", "runs"]), "Ablations relative to proposed (1,1,1)")
    r.table(None if tests is None else tests[tests.table == "ablations"].drop(columns=["table"]),
            "Wilcoxon tests, proposed vs ablations, Holm-corrected")
    for f in sorted(F.glob("fig5_ablations_*.png")):
        r.figure(f, f.stem)

    # 5. power-model sensitivity
    r.h("5. Power-model sensitivity")
    r.table(pick(read(T / "sensitivity.csv"), ["env", "policy", "power_model", "power_ratio"] + KEY),
            "Sensitivity runs")
    r.table(read(T / "power_sensitivity_grid.csv"), "Share of decisions with a multi-point cost-energy front")
    r.figure(F / "fig9_power_sensitivity.png", "Cost-energy trade-off against (P_max - P_idle)/P_idle")

    # 6. counterfactual
    r.h("6. Counterfactual check (CloudSim Plus)")
    r.table(read(M / "counterfactual_by_k.csv"), "Prediction error and coverage of U_k by forced action k")
    r.figure(F / "fig8_counterfactual.png", "Predicted vs measured response time, by k")

    # 7. synthetic vs CloudSim Plus
    r.h("7. Synthetic vs CloudSim Plus")
    if main_res is not None and set(main_res.env) >= {"synthetic", "cloudsim"}:
        piv = main_res.pivot_table(index=["workload", "policy"], columns="env",
                                   values=["violation_rate", "cost_usd", "energy_kwh"]).reset_index()
        piv.columns = ["_".join(c for c in col if c) for col in piv.columns]
        r.table(piv, "Same policy and workload in both environments")
    else:
        r.p("<i>Needs main-grid results from both environments.</i>")

    # 8. limitations
    r.h("8. Limitations and threats to validity")
    lim = []
    cal_cs = read(M / "calibration_cloudsim.csv")
    if cal_cs is not None:
        full = cal_cs.set_index("variant").loc["full"]
        lim.append(f"End-to-end coverage of the full mode on CloudSim Plus data is {full.coverage:.3f} "
                   f"(nominal 0.95; conformal z gives {full.coverage_conformal:.3f}).")
    cf = read(M / "counterfactual_by_k.csv")
    if cf is not None:
        out_of_band = cf[(cf.coverage_U < 0.90) | (cf.coverage_U > 0.98)]
        lim.append("Counterfactual coverage of U_k ranges from "
                   f"{cf.coverage_U.min():.2f} to {cf.coverage_U.max():.2f} across k.")
        if len(out_of_band):
            ks = ", ".join(f"{int(k):+d}" for k in out_of_band.k)
            lim.append(f"<b>Known limitation:</b> for k = {ks} the coverage lies outside the 0.90-0.98 band of "
                       "acceptance check 3 (over-coverage). These actions leave the pool lightly loaded, where outcomes "
                       "are nearly deterministic and U_k is far below the SLO, so the bound is conservative and cannot "
                       "change a feasibility decision. With 50 pairs per k even a perfectly calibrated bound passes all "
                       "seven k with probability about 0.43. Accepted as a documented limitation "
                       "(docs/implementation_notes.md, sections 4.1 and 4.2).")
    if main_res is not None and "churn" in main_res:
        hi = main_res.loc[main_res.churn.idxmax()]
        lim.append(f"Highest churn: {hi.policy} on {hi.workload} ({hi.env}), {hi.churn:.0f} actions per run.")
    lim.append("mem_util in CloudSim Plus is a modelling assumption (30 + 45 u plus noise), not measured memory.")
    lim.append("One VM per host: no consolidation; pool size equals machines drawing power.")
    for line in lim:
        r.p("- " + line)
    out = r.build(args.out)
    print("wrote", out)


if __name__ == "__main__":
    main()
