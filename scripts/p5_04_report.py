"""Build the PDF report (Step 5.4) from the tables of Steps 5.1-5.2, the model tables and the figures.

python scripts/p5_04_report.py                      -> results/reports/final_report.pdf
python scripts/p5_04_report.py --smoke ...          -> a smoke-test report proving the chain works

Before building, it writes the derived tables it renders (setup.csv, manifest_summary.csv, main_results_<env>.csv,
findings.csv, env_agreement.csv) to --tables, so every number in the PDF is in a CSV produced by a script.
"""
import argparse
import platform
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from autoscaler.config import ROOT, load_config
from autoscaler.eval.grid import expand
from autoscaler.eval.report import Report, fmt, git_head, java_version, read

KEY = ["violation_rate", "violation_rate_ci", "cost_usd", "cost_usd_ci", "energy_kwh", "energy_kwh_ci", "runs"]
REPORTED = [("violation_rate", "violation rate"), ("rt_p95_ms", "p95 RT (ms)"), ("cost_usd", "cost (USD)"),
            ("energy_kwh", "energy (kWh)"), ("churn", "churn (actions)"), ("fallback_rate", "fallback rate"),
            ("realised_coverage", "realised coverage")]
TESTED = [("violation_rate", "violation rate"), ("cost_usd", "cost"), ("energy_kwh", "energy")]
MAIN = ["cpu_threshold", "rt_threshold", "opt", "proposed_w111"]
ENV_NAME = {"synthetic": "synthetic", "cloudsim": "CloudSim Plus"}
ALPHA = 0.05
QUESTIONS = {
    "E1": ("non-linear and ensemble prediction",
           {"abl_stage1_linear": "linear stage 1 (paper_linear)", "abl_single_member": "single network (M = 1)",
            "abl_with_rate": "request rate added as a stage-2 input"}),
    "E2": ("uncertainty quantification and the chance constraint",
           {"abl_no_uncertainty": "no uncertainty (U = exp(mean))", "abl_epistemic_only": "epistemic only",
            "abl_no_propagation": "no stage-1 propagation (ale_epi)", "abl_quantile_nn": "QuantileNet bound",
            "abl_quantile_hgb": "QuantileHGB bound"}),
    "E3": ("the Pareto decision layer",
           {"abl_min_cost": "min-cost selection instead of the Pareto layer",
            "abl_two_objective": "two objectives (cost, energy) only"}),
}


def pick(df, cols):
    return None if df is None else df[[c for c in cols if c in df.columns]]


def pm(mean, ci, digits=3):
    """'mean ± ci' with the given significant digits; blank when the mean is missing."""
    if pd.isna(mean):
        return ""
    m = fmt(float(mean), digits)
    return m if pd.isna(ci) else f"{m} ± {fmt(float(ci), 2)}"


def lockfile_text() -> str:
    raw = (ROOT / "requirements.lock.txt").read_bytes()
    return raw.decode("utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8")


def tool_version(cmd, pattern) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        m = re.search(pattern, (r.stdout or "") + (r.stderr or ""))
        return m.group(1) if m else "unknown"
    except OSError:
        return "unknown"


def cloudsim_version() -> str:
    m = re.search(r"<artifactId>cloudsimplus</artifactId>\s*<version>([^<]+)</version>",
                  (ROOT / "cloudsim" / "pom.xml").read_text(encoding="utf-8"))
    return m.group(1) if m else "unknown"


# ---------------------------------------------------------------- derived tables (written before rendering)

def setup_table(cs, proto, exp) -> pd.DataFrame:
    pins = {l.split("==")[0].lower(): l.split("==")[1] for l in lockfile_text().splitlines() if "==" in l}
    s = cs["requests"]["capacity_scale"]
    rows = [
        ("git commit", git_head(ROOT)), ("Python", platform.python_version()), ("Java", java_version()),
        ("Maven", tool_version("mvn -v", r"Apache Maven (\S+)")), ("CloudSim Plus", cloudsim_version()),
        *[(pkg, pins[pkg]) for pkg in ("numpy", "pandas", "scipy", "scikit-learn", "torch", "matplotlib",
                                         "reportlab") if pkg in pins],
        ("capacity_scale s", s),
        ("eval seeds", f"{exp['eval_seeds'][0]}..{exp['eval_seeds'][-1]} ({len(exp['eval_seeds'])})"),
        ("calibration seeds", f"{exp['calibration_seeds'][0]}..{exp['calibration_seeds'][-1]}"),
        ("workloads", ", ".join(exp["workloads"])),
        ("synthetic: steps per run / warm-up", f"{proto['time']['steps_per_run']} / {proto['time']['warmup_steps']}"),
        ("synthetic: SLO (ms)", proto["slo"]["rt_ms"]),
        ("CloudSim Plus: steps per run / warm-up / interval (s)",
         f"{cs['time']['steps_per_run']} / {cs['time']['warmup_steps']} / {cs['time']['interval_s']}"),
        ("CloudSim Plus: hosts, VM limits, initial VMs",
         f"{cs['cluster']['hosts']} hosts (1 PE, {cs['cluster']['host_mips']} MIPS), "
         f"VMs {cs['cluster']['vm_min']}..{cs['cluster']['vm_max']}, initial {cs['cluster']['initial_vms']}"),
        ("CloudSim Plus: VM start-up delay (s)", cs["cluster"]["vm_startup_delay_s"]),
        ("CloudSim Plus: mean service time (s), length distribution",
         f"{cs['requests']['mean_service_s']}, {cs['requests']['length_dist']}"),
        ("CloudSim Plus: SLO (ms), effective rt_ms / s", f"{cs['slo']['rt_ms']}, {cs['slo']['rt_ms'] / s:g}"),
        ("chance constraint delta, z", f"{cs['slo']['delta']}, {cs['slo']['z']:.4f}"),
        ("actions k", f"{cs['actions']['k_min']}..{cs['actions']['k_max']}"),
        ("price (USD per VM-hour)", cs["cost"]["price_per_vm_hour"]),
        ("power model (main grid)", f"{cs['power']['model']}, P_idle {cs['power']['p_idle_w']} W, "
                                    f"P_max {cs['power']['p_max_w']} W"),
        ("baseline cpu_threshold", str(exp["baselines"]["cpu_threshold"])),
        ("baseline rt_threshold", str(exp["baselines"]["rt_threshold"])),
        ("proposed weight vectors", "; ".join(str(tuple(w)) for w in exp["weights_sweep"])),
        ("mem_util model (CloudSim Plus)", f"{cs['telemetry']['mem_util']['offset']} + "
                                           f"{cs['telemetry']['mem_util']['slope']} u + N(0, "
                                           f"{cs['telemetry']['mem_util']['noise_std']}^2)"),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def failed_runs_listed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set(re.findall(r"`((?:synthetic|cloudsim)__[^`]+)`", path.read_text(encoding="utf-8")))


def manifest_summary(exp, manifest: Path, failed_md: Path, summaries) -> pd.DataFrame | None:
    if not manifest.exists():
        return None
    man = pd.read_csv(manifest).drop_duplicates("run_id", keep="last").set_index("run_id")
    listed = failed_runs_listed(failed_md)
    rows = []
    for env in ("synthetic", "cloudsim"):
        for part in ("main", "ablations", "sensitivity"):
            ids = [j.run_id for j in expand(exp, env, part)]
            status = man.status.reindex(ids)
            ok = int((status == "ok").sum())
            failed = [i for i in ids if status.get(i) != "ok"]
            rows.append({"env": env, "part": part, "planned": len(ids), "ok": ok, "not_ok": len(failed),
                         "not_ok_listed_in_failed_runs_md": sum(i in listed for i in failed),
                         "attempts_in_manifest": int(pd.read_csv(manifest).run_id.isin(ids).sum())})
    return pd.DataFrame(rows)


def main_tables(main_res: pd.DataFrame, exp) -> dict:
    """{env: DataFrame} with 'mean ± ci' strings for the reported metrics, per workload and policy."""
    order = MAIN + [f"proposed_w{''.join(str(int(x)) for x in w)}" for w in exp["weights_sweep"]][1:]
    out = {}
    for env, g in main_res.groupby("env"):
        rows = []
        for wl in exp["workloads"]:
            gw = g[g.workload == wl].set_index("policy")
            for pol in [p for p in order if p in gw.index]:
                r = gw.loc[pol]
                rows.append({"workload": wl, "policy": pol, "runs": int(r.runs),
                             **{label: pm(r.get(m), r.get(f"{m}_ci")) for m, label in REPORTED}})
        out[env] = pd.DataFrame(rows)
    return out


def verdict(row) -> str:
    """Direction and significance of reference (proposed) minus policy; all tested metrics: lower is better."""
    if pd.isna(row.p):
        return "no data"
    if row.p_holm < ALPHA:
        level = "significant"
    elif row.p_holm_family < ALPHA:
        level = "significant (secondary family only)"
    else:
        return "no detectable difference"
    return f"proposed {'better' if row.mean_diff_ref_minus_policy < 0 else 'worse'}, {level}"


def findings_table(tests: pd.DataFrame) -> pd.DataFrame:
    rows = []
    t = tests.copy()
    t["verdict"] = [verdict(r) for r in t.itertuples()]
    for _, r in t.iterrows():
        rq = next((q for q, (_, abl) in QUESTIONS.items() if r.policy in abl), "baselines")
        rows.append({"question": rq, "env": r.env, "workload": r.workload, "comparison": r.policy,
                     "metric": r.metric, "mean_diff_proposed_minus_other": r.mean_diff_ref_minus_policy,
                     "ci95": r.mean_diff_ci, "median_diff": r.median_diff_ref_minus_policy, "p": r.p,
                     "p_holm": r.p_holm, "p_holm_family": r.p_holm_family, "verdict": r.verdict})
    return pd.DataFrame(rows)


def env_agreement(main_res: pd.DataFrame) -> pd.DataFrame | None:
    """Sign of (proposed (1,1,1) - baseline) per workload and metric in each environment, and whether they agree."""
    if not {"synthetic", "cloudsim"} <= set(main_res.env):
        return None
    rows = []
    for wl in sorted(set(main_res.workload)):
        for base in MAIN[:3]:
            for m, label in TESTED:
                d = {}
                for env in ("synthetic", "cloudsim"):
                    g = main_res[(main_res.env == env) & (main_res.workload == wl)].set_index("policy")
                    if {"proposed_w111", base} <= set(g.index):
                        d[env] = g.loc["proposed_w111", m] - g.loc[base, m]
                if len(d) == 2:
                    sign = {e: int(np.sign(round(v, 12))) for e, v in d.items()}
                    rows.append({"workload": wl, "baseline": base, "metric": m,
                                 "diff_synthetic": d["synthetic"], "diff_cloudsim": d["cloudsim"],
                                 "agree": sign["synthetic"] == sign["cloudsim"]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- text generated from the tables

LABEL = dict(TESTED)


def effect_phrase(r) -> str:
    return (f"{ENV_NAME[r.env]} {r.workload} vs {r.comparison}: {LABEL[r.metric]} "
            f"{fmt(float(r.mean_diff_proposed_minus_other), 3)} ± {fmt(float(r.ci95), 2)}")


def metric_summary(f) -> tuple[str, list[str]]:
    """'violation rate 0/0/4, ...' (better/worse/no difference) and the largest significant effect per metric."""
    counts, effects = [], []
    for m, label in TESTED:
        g = f[f.metric == m]
        if not len(g):
            continue
        b = g[g.verdict.str.startswith("proposed better")]
        w = g[g.verdict.str.startswith("proposed worse")]
        counts.append(f"{label} {len(b)}/{len(w)}/{len(g) - len(b) - len(w)}")
        sig = pd.concat([b, w])
        if len(sig):
            top = sig.loc[sig.mean_diff_proposed_minus_other.abs().idxmax()]
            effects.append(effect_phrase(top))
    return ", ".join(counts), effects


def question_paragraph(q, title, ablations, fnd) -> str:
    f = fnd[fnd.comparison.isin(ablations)]
    if not len(f):
        return f"<b>{q} ({title}).</b> <i>No ablation results available.</i>"
    counts, effects = metric_summary(f)
    text = [f"<b>{q} ({title}).</b> {len(f)} paired comparisons of the full method against "
            f"{', '.join(ablations[a] for a in ablations if a in set(f.comparison))} (environment x workload x metric, "
            f"10 seeds). Significant under the specified table-wide Holm correction: {int((f.p_holm < ALPHA).sum())}. "
            f"Secondary family, full method better / worse / no detectable difference: {counts}."]
    if effects:
        text.append("Largest significant effect per metric (full method minus ablation, mean ± 95% CI; negative = "
                    "lower with the component): " + "; ".join(effects) + ".")
    better = f[f.verdict.str.startswith("proposed better")]
    never = [ablations[a] for a in ablations if a in set(f.comparison) and a not in set(better.comparison)]
    no_viol = better[better.metric == "violation_rate"]
    if never:
        text.append("<b>No detectable benefit on any metric</b> against: " + ", ".join(never) + ".")
    if not len(no_viol):
        text.append("<b>The components did not reduce the violation rate detectably in any comparison.</b>")
    worse = f[f.verdict.str.startswith("proposed worse")]
    if len(worse):
        text.append(f"In {len(worse)} comparisons the full method was significantly worse, on "
                    + ", ".join(sorted({LABEL[m] for m in worse.metric})) + ".")
    return " ".join(text)


def baselines_paragraph(fnd, main_res) -> str:
    f = fnd[fnd.question == "baselines"]
    if not len(f):
        return "<b>Proposed vs baselines.</b> <i>No main-grid tests available.</i>"
    parts = []
    for (env, base), g in f.groupby(["env", "comparison"], sort=False):
        counts, _ = metric_summary(g)
        parts.append(f"vs {base} ({ENV_NAME[env]}): {counts}")
    counts, effects = metric_summary(f)
    text = (f"<b>Proposed (1,1,1) vs baselines.</b> {len(f)} comparisons (environment x workload x metric); "
            f"{int((f.p_holm < ALPHA).sum())} significant under the specified table-wide Holm correction. Secondary "
            f"family, proposed better / worse / no detectable difference over the four workloads: " +
            "; ".join(parts) + ".")
    if effects:
        text += (" Largest significant effect per metric (proposed minus baseline): " + "; ".join(effects) + ".")
    return text


def limitations(M, T, main_res, tests, man_sum, sens_grid, agree, cs) -> list[str]:
    lim = []
    if tests is not None and len(tests):
        lim.append(
            f"<b>Statistical power.</b> With {int(tests.pairs.max())} seeds the exact two-sided Wilcoxon test cannot "
            f"give p below {fmt(float(tests.p_min_possible.min()), 3)}. Holm across a whole table "
            f"({len(tests[tests.table == 'main'])} main and {len(tests[tests.table == 'ablations'])} ablation "
            f"comparisons) therefore cannot go below {fmt(float(tests[tests.table == 'main'].p_min_possible.min() * len(tests[tests.table == 'main'])), 3)}"
            f" and {fmt(float(tests[tests.table == 'ablations'].p_min_possible.min() * len(tests[tests.table == 'ablations'])), 3)}"
            ": no comparison can be significant under the specified correction, whatever the effect size. The "
            "secondary family (Holm within environment, workload and metric) is reported to show direction and "
            "size; it controls the error rate per outcome, not per table. More seeds would be needed for "
            "table-wide claims.")
    cal_cs = read(M / "calibration_cloudsim.csv")
    if cal_cs is not None:
        full = cal_cs.set_index("variant").loc["full"]
        lim.append(f"End-to-end coverage of the full mode on CloudSim Plus test data is {full.coverage:.3f} "
                   f"(nominal 0.95), {full.coverage_slo_bin:.3f} in the predicted-RT bin containing the SLO; the "
                   f"conformal z would give {full.coverage_conformal:.3f}.")
    if main_res is not None and "realised_coverage" in main_res:
        p = main_res[main_res.policy.str.startswith("proposed_w")].dropna(subset=["realised_coverage"])
        if len(p):
            lo, hi = p.loc[p.realised_coverage.idxmin()], p.loc[p.realised_coverage.idxmax()]
            lim.append(f"Realised coverage of the chosen action's bound in the grid ranges from "
                       f"{lo.realised_coverage:.3f} ({lo.policy}, {lo.workload}, {ENV_NAME[lo.env]}) to "
                       f"{hi.realised_coverage:.3f} ({hi.policy}, {hi.workload}, {ENV_NAME[hi.env]}), "
                       "against a nominal 0.95. Realised coverage is measured only on the chosen action, so it is "
                       "conditional on the selection and not the same quantity as the test-set coverage.")
        f = main_res[main_res.policy.str.startswith("proposed_w")].dropna(subset=["fallback_rate"])
        if len(f):
            hi = f.loc[f.fallback_rate.idxmax()]
            lim.append(f"Highest fallback rate (no feasible candidate): {hi.fallback_rate:.3f} ({hi.policy}, "
                       f"{hi.workload}, {ENV_NAME[hi.env]}).")
    cf, band = read(M / "counterfactual_by_k.csv"), read(T / "counterfactual_band_probability.csv")
    if cf is not None:
        out_of_band = cf[(cf.coverage_U < 0.90) | (cf.coverage_U > 0.98)]
        txt = (f"Counterfactual coverage of U_k ranges from {cf.coverage_U.min():.2f} to "
               f"{cf.coverage_U.max():.2f} across k ({int(cf.pairs.min())} pairs per k).")
        if len(out_of_band):
            ks = ", ".join(f"{int(k):+d}" for k in out_of_band.k)
            txt += (f" <b>Known limitation:</b> for k = {ks} it lies outside the 0.90-0.98 band of acceptance check 3"
                    " (over-coverage). These actions leave the pool lightly loaded, where outcomes are nearly "
                    "deterministic and U_k is far below the SLO, so the bound is conservative there.")
        if band is not None:
            b = band.set_index("true_coverage")
            txt += (f" Even a perfectly calibrated bound (true coverage 0.95) passes all {int(b.ks.iloc[0])} k with "
                    f"probability {b.loc[0.95, 'p_pass_all_k']:.2f} at this sample size "
                    f"({b.loc[0.94, 'p_pass_all_k']:.2f} at 0.94, {b.loc[0.96, 'p_pass_all_k']:.2f} at 0.96).")
        lim.append(txt + " Accepted as a documented limitation (docs/implementation_notes.md, 4.1-4.2).")
    if sens_grid is not None and len(sens_grid):
        lo, hi = int(sens_grid.decisions.min()), int(sens_grid.decisions.max())
        lim.append(f"The power-threshold analysis rests on {lo if lo == hi else f'{lo} to {hi}'} non-fallback "
                   "decisions per setting and environment, pooled over seeds; decisions within a run are not "
                   "independent.")
    if agree is not None and len(agree):
        lim.append(f"Synthetic and CloudSim Plus agree on the sign of proposed minus baseline in "
                   f"{int(agree.agree.sum())} of {len(agree)} (workload, baseline, metric) cells; conclusions drawn "
                   "from one environment alone should be read with that in mind.")
    if main_res is not None and (main_res.policy == "opt").any():
        for env, o in main_res[main_res.policy == "opt"].groupby("env"):
            txt = (f"opt in {ENV_NAME[env]}: fallback rate {o.fallback_rate.min():.3f} to {o.fallback_rate.max():.3f}, "
                   f"churn {o.churn.min():.0f} to {o.churn.max():.0f} actions per run, violation rate "
                   f"{o.violation_rate.min():.3f} to {o.violation_rate.max():.3f} across workloads.")
            if o.fallback_rate.min() > 0.2:
                txt += (" It oscillates between k_min and the k_max fallback: its linear stage 1 predicts nearly "
                        "the same CPU for every action (docs/prototype_findings.md, finding 1), so gains over opt "
                        "here largely reflect that weakness of the baseline as specified.")
            lim.append(txt)
    if main_res is not None and "churn" in main_res:
        hi = main_res.loc[main_res.churn.idxmax()]
        lim.append(f"Highest churn: {hi.policy} on {hi.workload} ({ENV_NAME[hi.env]}), {hi.churn:.0f} actions per "
                   "run; scaling actions are free apart from the VM start-up delay, so churn has no cost here.")
    if man_sum is not None:
        bad, listed = int(man_sum.not_ok.sum()), int(man_sum.not_ok_listed_in_failed_runs_md.sum())
        lim.append(f"Grid completeness: {int(man_sum.ok.sum())} of {int(man_sum.planned.sum())} planned runs ok"
                   + (f"; {bad} are missing ({listed} of them listed in docs/failed_runs.md) and are excluded from "
                      "the means and the paired tests." if bad else "."))
    mu = cs["telemetry"]["mem_util"]
    txt = f"mem_util in CloudSim Plus is a modelling assumption ({mu['offset']} + {mu['slope']} u plus noise), not measured memory"
    r2 = read(M / "stage1_r2_cloudsim.csv")
    if r2 is not None and "mem_util" in set(r2.metric):
        row = r2.set_index("metric").loc["mem_util"]
        txt += (f"; its best stage-1 R2 is {row[row.chosen]:.3f}, and it is "
                + ("selected" if str(row.selected) == "True" else "not selected") + " as a model input")
    lim.append(txt + ".")
    lim.append(f"One VM per host ({cs['cluster']['hosts']} hosts): no consolidation, so every VM is a machine drawing "
               "idle power; energy results do not transfer to packed hosts.")
    lim.append("The threshold baselines use the roadmap's placeholder thresholds (configs/experiments.yaml), not "
               "the base paper's values, which were not available.")
    return lim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--model-tables", default="results/tables")
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--manifest", default="results/runs/manifest.csv")
    ap.add_argument("--failed-runs", default="docs/failed_runs.md")
    ap.add_argument("--out", default="results/reports/final_report.pdf")
    ap.add_argument("--title", default="Uncertainty-aware, Pareto-optimal autoscaling: final report")
    ap.add_argument("--smoke", action="store_true", help="label the report as a smoke test of the pipeline")
    ap.add_argument("--note", default="", help="extra paragraph for the setup section")
    args = ap.parse_args()
    T, M, F = Path(args.tables), Path(args.model_tables), Path(args.figures)
    cs, proto, exp = (load_config(ROOT / f"configs/{n}.yaml") for n in ("cloudsim", "prototype", "experiments"))

    # derived tables, written first so that the PDF only renders CSVs
    setup = setup_table(cs, proto, exp)
    setup.to_csv(T / "setup.csv", index=False)
    summaries = read(T / "run_summaries.csv")
    man_sum = manifest_summary(exp, Path(args.manifest), Path(args.failed_runs), summaries)
    if man_sum is not None:
        man_sum.to_csv(T / "manifest_summary.csv", index=False)
    main_res, tests, abl = read(T / "main_results.csv"), read(T / "paired_tests.csv"), read(T / "ablations.csv")
    mtabs = main_tables(main_res, exp) if main_res is not None else {}
    for env, t in mtabs.items():
        t.to_csv(T / f"main_results_{env}_formatted.csv", index=False)
    fnd = findings_table(tests) if tests is not None and len(tests) else None
    if fnd is not None:
        fnd.to_csv(T / "findings.csv", index=False)
    agree = env_agreement(main_res) if main_res is not None else None
    if agree is not None:
        agree.to_csv(T / "env_agreement.csv", index=False)
    sens_grid = read(T / "power_sensitivity_grid.csv")
    repro = read(T / "reproducibility.csv")

    r = Report(args.title)
    # a. setup
    r.h("1. Setup")
    r.p(f"Two environments: a synthetic formula system and CloudSim Plus {cloudsim_version()} (one VM per host, "
        "Java policies in-process, model-based policies through a TCP bridge to Python). Policies: cpu_threshold, "
        "rt_threshold, opt (the base paper's method: linear stage 1 and one MSE-trained network, smallest k whose "
        "predicted response time is below the SLO) and proposed with seven weight vectors; ten ablations on {', '.join(exp['ablation_workloads'])}; power-model sensitivity on "
        f"{exp['sensitivity']['workload']}. Every metric is computed after the warm-up.")
    if args.smoke:
        r.p("<b>Smoke test.</b> This report was built from a deliberately small grid (2 policies, 1 workload, "
            "2 seeds, shortened runs) to show that the runner, the bridge, the analysis and the report generator "
            "work end to end. Its numbers are not results.")
    if args.note:
        r.p(args.note)
    r.table(setup, "Scenario, seeds and software versions (setup.csv)", col_width=[6.5 * 28.35, 11.3 * 28.35])
    r.table(man_sum, "Grid runs per environment and part (manifest_summary.csv)")
    r.table(repro, "Five randomly chosen runs re-run from scratch (reproducibility.csv)")
    r.table(read(M / "cloudsim_speed.csv"), "Run time of one 360-step CloudSim Plus run (Step 3.13)")

    # b. model quality
    r.h("2. Model quality")
    for env, suf in (("synthetic", ""), ("CloudSim Plus", "_cloudsim")):
        r.h3(env)
        r.table(read(M / f"stage1_r2{suf}.csv"), f"Stage-1 cross-validated R2 per metric and candidate, and the "
                                                 f"selected metrics ({env})")
        r.table(read(M / f"stage2_accuracy{suf}.csv"), f"Stage-2 accuracy with true post-scaling metrics ({env})")
        cal = read(M / f"calibration{suf}.csv")
        r.table(pick(cal, ["variant", "mode", "coverage", "coverage_slo_bin", "mean_margin_ms", "nll", "mean_sigma",
                           "z_cal", "coverage_conformal", "pinball", "rows"]),
                f"End-to-end coverage at the 0.95 level per uncertainty mode and quantile model; coverage_slo_bin is "
                f"the conditional coverage in the predicted-RT bin containing the SLO ({env})")
        cond = read(M / f"calibration_conditional{suf}.csv")
        if cond is not None and "contains_slo" in cond:
            near = cond[cond.contains_slo.astype(str) == "True"]
            r.table(pick(near, ["variant", "bin", "pred_lo_ms", "pred_hi_ms", "rows", "coverage"]),
                    f"Conditional coverage in the bin containing the SLO ({env})")
    r.figure(F / "fig6_calibration.png", "Figure 6. Reliability diagram and conditional coverage, end to end")
    r.figure(F / "fig7_uncertainty_decomposition.png", "Figure 7. Uncertainty decomposition against predicted "
                                                       "utilisation")

    # c. main results
    r.page_break()
    r.h("3. Main results")
    r.p("Mean ± 95% t-interval half-width over the 10 seeds. Violation rate and fallback rate are shares of "
        "intervals and decisions; cost is USD per run, energy kWh per run, churn the number of actions with k != 0 "
        "per run. Threshold policies make no model-based decisions, so their fallback rate is 0 and realised "
        "coverage is undefined.")
    for env, t in mtabs.items():
        r.h3(ENV_NAME[env])
        for wl in exp["workloads"]:
            r.table(t[t.workload == wl].drop(columns=["workload"]), f"{ENV_NAME[env]}, {wl}", digits=3)
    if tests is not None:
        r.table(pick(tests[tests.table == "main"], ["env", "workload", "policy", "metric", "pairs",
                                                    "mean_diff_ref_minus_policy", "mean_diff_ci",
                                                    "median_diff_ref_minus_policy", "p", "p_holm", "p_holm_family"]),
                "Paired Wilcoxon signed-rank tests, proposed (1,1,1) minus baseline (negative = proposed lower). "
                "p_holm: Holm over the whole table (as specified); p_holm_family: Holm within env, workload and "
                "metric (secondary)", max_rows=200, digits=3)
    r.table(read(T / "hypervolume.csv"), "Hypervolume in min-max normalised (violation, cost, energy) space, "
                                         "reference (1.1, 1.1, 1.1); proposed = its seven weight points")
    for f in sorted(F.glob("fig4_tradeoff_*.png")):
        r.figure(f, f"Figure 4. Cost-violation-energy trade-off, seven weight settings joined ({f.stem[14:]})")
    for f in sorted(F.glob("fig3_bars_*.png")):
        r.figure(f, f"Figure 3. Violation rate, cost and energy with 95% intervals ({f.stem[10:]})")
    for f in sorted(F.glob("fig2_timeseries_*.png")):
        r.figure(f, f"Figure 2. Example time series ({f.stem[16:]})")
    r.figure(F / "fig1_workloads_cloudsim.png", "Figure 1. Workload shapes (seed 1)")

    # d. ablations
    r.page_break()
    r.h("4. Ablations")
    r.table(pick(abl, ["env", "workload", "policy", "violation_rate", "cost_usd", "energy_kwh",
                       "violation_rate_delta_vs_proposed", "cost_usd_delta_vs_proposed",
                       "energy_kwh_delta_vs_proposed", "runs"]), "Ablations: means and change relative to proposed "
                                                                 "(1,1,1)", max_rows=60, digits=3)
    if tests is not None:
        r.table(pick(tests[tests.table == "ablations"], ["env", "workload", "policy", "metric",
                                                         "mean_diff_ref_minus_policy", "mean_diff_ci", "p", "p_holm",
                                                         "p_holm_family"]),
                "Paired Wilcoxon tests, proposed minus ablation (negative = the full method is lower)",
                max_rows=200, digits=3)
    for f in sorted(F.glob("fig5_ablations_*.png")):
        r.figure(f, f"Figure 5. Ablations relative to proposed ({f.stem[15:]})")

    # e. power-model sensitivity
    r.h("5. Power-model sensitivity")
    r.table(pick(read(T / "sensitivity.csv"), ["env", "policy", "power_model", "power_ratio"] + KEY),
            "Sensitivity runs on daily_ramp_burst, mean ± 95% interval half-width", max_rows=60, digits=3)
    r.table(sens_grid, "Share of non-fallback proposed decisions whose feasible (cost, energy) front has more than "
                       "one point (power_sensitivity_grid.csv)")
    r.figure(F / "fig9_power_sensitivity.png", "Figure 9. Multi-point front share against (P_max - P_idle)/P_idle; "
                                               "the dashed line is the analytic threshold 1/u_slo^2")

    # f. counterfactual
    r.h("6. Counterfactual check by action k (CloudSim Plus)")
    r.table(read(M / "counterfactual_by_k.csv"), "Prediction error and coverage of U_k by forced action k")
    r.table(read(T / "counterfactual_band_probability.csv"),
            "Probability that a bound of given true coverage lands in the 0.90-0.98 band for one k and for all k")
    r.figure(F / "fig8_counterfactual.png", "Figure 8. Predicted vs measured response time, by k")

    # g. synthetic vs CloudSim Plus
    r.h("7. Synthetic vs CloudSim Plus")
    if main_res is not None and set(main_res.env) >= {"synthetic", "cloudsim"}:
        piv = main_res[main_res.policy.isin(MAIN)].pivot_table(
            index=["workload", "policy"], columns="env", values=["violation_rate", "cost_usd", "energy_kwh"]
        ).reset_index()
        piv.columns = ["_".join(c for c in col if c) for col in piv.columns]
        r.table(piv, "Same policy and workload in both environments (means)", digits=3)
        if agree is not None and len(agree):
            dis = agree[~agree.agree]
            r.p(f"The sign of proposed (1,1,1) minus each baseline agrees between the environments in "
                f"{int(agree.agree.sum())} of {len(agree)} (workload, baseline, metric) cells." +
                (" Disagreements: " + "; ".join(f"{d.workload}, vs {d.baseline}, {d.metric}" for d in dis.itertuples())
                 + "." if len(dis) else ""))
            r.table(agree, "Proposed minus baseline, per environment (env_agreement.csv)", max_rows=60, digits=3)
    else:
        r.p("<i>Needs main-grid results from both environments.</i>")

    # h. findings
    r.page_break()
    r.h("8. Findings")
    if fnd is not None:
        r.p(baselines_paragraph(fnd, main_res))
        for q, (title, ablations) in QUESTIONS.items():
            r.p(question_paragraph(q, title, ablations, fnd))
        r.p("Full list of verdicts: results/tables/findings.csv.")
    else:
        r.p("<i>Needs the paired tests.</i>")

    # i. limitations
    r.h("9. Limitations and threats to validity")
    for line in limitations(M, T, main_res, tests, man_sum, sens_grid, agree, cs):
        r.p("- " + line)
    out = r.build(args.out)
    print("wrote", out)


if __name__ == "__main__":
    main()
