"""Acceptance check 2 (end of prototype) from the sweep, the calibration table and a determinism re-run."""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from autoscaler.config import load_config
from autoscaler.decision.objectives import candidate_cost, candidate_energy
from autoscaler.decision.pareto import pareto_mask
from autoscaler.decision.power import idle_for_ratio
from autoscaler.sim.loop import load_decisions

SENSITIVITY = [("linear", 4.0), ("square", 4.0), ("cubic", 4.0), ("square", 1.0), ("square", 1.42),
               ("square", 2.0)]


def sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def front_sizes(decisions, cfg, model, ratio, warm):
    p_max = cfg["power"]["p_max_w"]
    pcfg = {"model": model, "p_idle_w": idle_for_ratio(p_max, ratio), "p_max_w": p_max}
    sizes = []
    for d in decisions:
        if d["step"] < warm or d["fallback"]:
            continue
        feas = [c for c in d["candidates"] if c["feasible"]]
        n = np.array([c["n"] for c in feas], float)
        u = np.array([c["u_hat"] for c in feas], float)
        F = np.column_stack([candidate_cost(n, cfg), candidate_energy(n, u, cfg, pcfg)])
        sizes.append(int(pareto_mask(F).sum()))
    return np.array(sizes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prototype.yaml")
    ap.add_argument("--sweep", default=None, help="sweep dir (default: results/runs/latest_synthetic_sweep.txt)")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--out", default="results/runs")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tables, figs = Path(args.tables), Path(args.figures)
    sweep = Path(args.sweep or (Path(args.out) / "latest_synthetic_sweep.txt").read_text().strip())
    runs = sweep / "runs"
    warm = cfg["time"]["warmup_steps"]
    results = []

    # 1. proposed violation rate below opt on >= 3 of 4 workloads
    main_res = pd.read_csv(tables / "synthetic_main_results.csv")
    v = main_res.pivot(index="workload", columns="policy", values="violation_rate")
    wins = int((v["proposed"] < v["opt"]).sum())
    detail = "; ".join(f"{w}: proposed {v.loc[w, 'proposed']:.4f} vs opt {v.loc[w, 'opt']:.4f}" for w in v.index)
    results.append({"item": 1, "check": "proposed violation rate < opt on >= 3 of 4 workloads",
                    "value": f"{wins} of {len(v)}", "pass": wins >= 3, "detail": detail})

    # 2. abl_min_cost chooses the smallest feasible pool in every non-fallback decision
    total = bad = 0
    for f in sorted(runs.glob("synthetic__abl_min_cost__*/decisions.jsonl.gz")):
        for d in load_decisions(f):
            if d["fallback"]:
                continue
            feas = [c["n"] for c in d["candidates"] if c["feasible"]]
            chosen = next(c["n"] for c in d["candidates"] if c["k"] == d["k"])
            total += 1
            bad += chosen != min(feas)
    results.append({"item": 2, "check": "abl_min_cost picks the smallest feasible pool (non-fallback decisions)",
                    "value": f"{total - bad} of {total}", "pass": total > 0 and bad == 0,
                    "detail": f"{bad} decisions picked a larger pool"})

    # 3. end-to-end coverage of full mode in [0.93, 0.97]
    cal = pd.read_csv(tables / "calibration.csv").set_index("variant")
    cov = float(cal.loc["full", "coverage"])
    cov_c = float(cal.loc["full", "coverage_conformal"])
    use_conf = bool(cfg["slo"].get("use_conformal"))
    eff = cov_c if use_conf else cov
    results.append({"item": 3, "check": "end-to-end coverage of full in [0.93, 0.97]",
                    "value": f"{eff:.4f}", "pass": 0.93 <= eff <= 0.97,
                    "detail": f"z=1.6449: {cov:.4f}; conformal z={cal.loc['full', 'z_cal']:.4f}: {cov_c:.4f}; "
                              f"controller uses {'conformal' if use_conf else 'Gaussian'} z"})

    # 4. power-model sensitivity of the (cost, energy) front of feasible candidates
    dec_files = sorted(runs.glob("synthetic__proposed__*/decisions.jsonl.gz"))
    decisions = [d for f in dec_files for d in load_decisions(f)]
    sens = []
    for model, ratio in SENSITIVITY:
        s = front_sizes(decisions, cfg, model, ratio, warm)
        sens.append({"model": model, "ratio": ratio, "decisions": len(s), "share_multi_point": float(np.mean(s > 1)),
                     "share_single_point": float(np.mean(s == 1)), "mean_front_size": float(s.mean())})
    sens = pd.DataFrame(sens)
    sens.to_csv(tables / "power_sensitivity.csv", index=False)
    sq4 = sens[(sens.model == "square") & (sens.ratio == 4.0)].iloc[0]
    sq1 = sens[(sens.model == "square") & (sens.ratio == 1.0)].iloc[0]
    results.append({"item": 4, "check": "square, ratio 4: multi-point front in many decisions; ratio 1: almost always one",
                    "value": f"ratio 4: {sq4.share_multi_point:.3f} multi; ratio 1: {sq1.share_single_point:.3f} single",
                    "pass": bool(sq4.share_multi_point >= 0.2 and sq1.share_single_point >= 0.95),
                    "detail": "thresholds used for 'many' / 'almost always': >= 20% / >= 95% of decisions"})

    fig, ax = plt.subplots(figsize=(6, 4))
    sq = sens[sens.model == "square"].sort_values("ratio")
    ax.plot(sq.ratio, sq.share_multi_point, marker="o", label="square")
    for m in ("linear", "cubic"):
        r = sens[sens.model == m]
        ax.scatter(r.ratio, r.share_multi_point, marker="s", label=m)
    ax.axvline(1 / 0.84 ** 2, color="#b33", ls="--", lw=1, label="threshold 1/u_slo^2 = 1.42")
    ax.set(xlabel="(P_max - P_idle) / P_idle", ylabel="share of decisions with multi-point front",
           title="Cost-energy trade-off vs power model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / "power_sensitivity.png", dpi=200)
    fig.savefig(figs / "power_sensitivity.pdf")

    # 5. two runs with the same seed give identical logs (pytest is run separately)
    hashes = []
    for _ in range(2):
        out = subprocess.run([sys.executable, "scripts/p2_05_run_prototype.py", "--policy", "proposed",
                              "--workload", "flash_crowd", "--seed", "3", "--out", args.out],
                             capture_output=True, text=True, check=True).stdout
        rd = Path(out.strip().splitlines()[-1].split("run dir:")[1].strip())
        hashes.append((sha(rd / "run_log.parquet"), sha(rd / "decisions.jsonl.gz")))
    sweep_rd = runs / "synthetic__proposed__flash_crowd__s3"
    sweep_hash = (sha(sweep_rd / "run_log.parquet"), sha(sweep_rd / "decisions.jsonl.gz"))
    same = hashes[0] == hashes[1]
    run_log_matches_sweep = hashes[0][0] == sweep_hash[0]
    results.append({"item": 5, "check": "two runs with the same seed produce identical logs (file hashes)",
                    "value": "identical" if same else "different", "pass": same,
                    "detail": f"run_log sha256 {hashes[0][0][:16]}..., decisions {hashes[0][1][:16]}...; "
                              f"run log identical to the sweep's copy: {run_log_matches_sweep}"})

    res = pd.DataFrame(results)
    res.to_csv(tables / "prototype_acceptance.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 120)
    print(res.to_string(index=False))
    print(sens.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
