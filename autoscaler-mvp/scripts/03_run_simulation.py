"""Step 3: run the closed control loop over the synthetic workload and plot it.

Writes results/logs/simulation.csv plus the three figures the prototype exists
to produce:
  03  response time over time, SLO marked
  04  VM count over time, against the workload that drove it
  05  one Pareto-front snapshot: cost vs energy, feasible and infeasible marked

Success criterion: the loop tracks the workload, holds response time under the
SLO outside the burst, and the Pareto stage has more than one point to choose
between at least some of the time. If the front is a single point everywhere,
the extension is doing nothing and the energy model needs revisiting.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import joblib  # noqa: E402
from _bootstrap import FIGURES, LOGS, MODELS  # noqa: E402

from autoscaler import config as C  # noqa: E402
from autoscaler.environment import generate_workload  # noqa: E402
from autoscaler.loop import run  # noqa: E402
from autoscaler.predictor import EnsembleRTPredictor  # noqa: E402


def main():
    estimator = joblib.load(MODELS / "metric_estimator.joblib")
    predictor = EnsembleRTPredictor.load(MODELS / "rt_ensemble.joblib")

    workload = generate_workload()
    log, decisions = run(workload, predictor, estimator)

    log.to_csv(LOGS / "simulation.csv", index=False)
    with open(LOGS / "decisions.json", "w") as f:
        json.dump(decisions, f, indent=2)

    # Plot the decision with the largest Pareto front. Peak load is the *least*
    # informative choice -- there almost everything is infeasible and the
    # fallback fires, so the figure would show the mechanism not working. This
    # picks the timestep where the trade-off is widest, and says so on the axis
    # title. Every other decision is in decisions.json to re-plot if wanted.
    snapshot = max(
        decisions,
        key=lambda d: (sum(c["pareto"] for c in d["candidates"]), d["request_rate"]),
    )

    ticks = log[log["decision_made"]]
    print(f"steps simulated       {len(log)}")
    print(f"decisions taken       {len(ticks)}")
    print(f"VM count range        {log['vm_count'].min()} - {log['vm_count'].max()}")
    print(f"mean VM count         {log['vm_count'].mean():.2f}")
    print(f"SLO violation rate    {log['slo_violated'].mean():.2%}")
    print(f"  outside the burst   {log[~log['step'].between(C.SPIKE_CENTER - 3 * C.SPIKE_WIDTH, C.SPIKE_CENTER + 3 * C.SPIKE_WIDTH)]['slo_violated'].mean():.2%}")
    print(f"fallback invoked      {ticks['used_fallback'].sum()} of {len(ticks)} decisions")
    print(f"p95 response time     {log['response_time_ms'].quantile(0.95):.0f} ms")
    print(f"total cost            ${log['cost'].sum():.4f}")
    print(f"total energy          {log['energy_wh'].sum() / 1000:.2f} kWh")

    warm = log[log["step"] >= 200]
    print(f"\nafter warm-up (step 200+)")
    print(f"  median RT           {warm['response_time_ms'].median():.0f} ms")
    print(f"  p95 RT              {warm['response_time_ms'].quantile(0.95):.0f} ms")
    print(f"  SLO violation rate  {warm['slo_violated'].mean():.2%}")
    print("  NOTE: the chance constraint is calibrated for ~5% breach, and the")
    print("  predictor's held-out coverage confirms that. The realised rate is")
    print("  higher because the unobserved complexity driver is autocorrelated")
    print("  over ~%d steps while the calibration assumes iid residuals, and" % C.COMPLEXITY_PERIOD)
    print("  because an action is held for the whole control interval. Both are")
    print("  open items, not bugs -- see README, Limitations.")

    n_pareto = sum(c["pareto"] for c in snapshot["candidates"])
    n_feasible = sum(c["feasible"] for c in snapshot["candidates"])
    print(f"\nsnapshot at step {snapshot['step']} ({snapshot['request_rate']:.0f} req/s)")
    print(f"  candidates {len(snapshot['candidates'])}, feasible {n_feasible}, on front {n_pareto}")

    # ------------------------------------------------------- 03 response time
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(log["step"], log["response_time_ms"], lw=0.9, color="#2b6cb0", label="response time")
    ax.axhline(C.RT_SLO_MS, color="#c53030", ls="--", lw=1.2, label=f"SLO {C.RT_SLO_MS:.0f} ms")
    viol = log[log["slo_violated"]]
    ax.scatter(viol["step"], viol["response_time_ms"], s=5, color="#c53030", zorder=3,
               label=f"violations ({len(viol)})")
    ax.set_yscale("log")
    ax.set_xlabel("control step (1 min)")
    ax.set_ylabel("response time (ms, log)")
    ax.set_title("Response time under Pareto-aware autoscaling")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "03_response_time.png", dpi=150)

    # ------------------------------------------------------------ 04 VM count
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.step(log["step"], log["vm_count"], where="post", lw=1.3, color="#2f855a", label="VM count")
    ax.set_xlabel("control step (1 min)")
    ax.set_ylabel("VMs", color="#2f855a")
    ax.set_ylim(0, C.VM_MAX + 1)
    ax2 = ax.twinx()
    ax2.plot(log["step"], log["request_rate"], lw=0.9, color="#a0aec0", alpha=0.9,
             label="request rate")
    ax2.set_ylabel("requests / s", color="#718096")
    fb = ticks[ticks["used_fallback"]]
    if len(fb):
        ax.scatter(fb["step"], fb["vm_count"], s=14, color="#c53030", zorder=4,
                   label=f"SLO infeasible, fallback ({len(fb)})")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, fontsize=8, loc="upper left")
    ax.set_title("Pool size tracking the workload")
    fig.tight_layout()
    fig.savefig(FIGURES / "04_vm_count.png", dpi=150)

    # -------------------------------------------------------- 05 Pareto front
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    cands = snapshot["candidates"]
    infeas = [c for c in cands if not c["feasible"]]
    feas = [c for c in cands if c["feasible"] and not c["pareto"]]
    front = sorted([c for c in cands if c["pareto"]], key=lambda c: c["cost"])

    if infeas:
        ax.scatter([c["cost"] for c in infeas], [c["energy"] for c in infeas],
                   s=90, marker="x", color="#c53030",
                   label=f"SLO infeasible (mean + {C.Z_SCORE}σ > SLO)")
    if feas:
        ax.scatter([c["cost"] for c in feas], [c["energy"] for c in feas],
                   s=80, facecolors="none", edgecolors="#4a5568", label="feasible, dominated")
    if front:
        ax.plot([c["cost"] for c in front], [c["energy"] for c in front],
                "-", lw=1.0, color="#6b46c1", alpha=0.6)
        ax.scatter([c["cost"] for c in front], [c["energy"] for c in front],
                   s=110, color="#6b46c1", label="Pareto front")

    chosen = min(front, key=lambda c: c["cost"]) if front else None
    if chosen:
        ax.scatter([chosen["cost"]], [chosen["energy"]], s=260, marker="*",
                   color="#d69e2e", edgecolors="black", linewidths=0.6, zorder=5,
                   label="chosen (min cost on front)")

    for c in cands:
        ax.annotate(f"{c['vm_after']} VM", (c["cost"], c["energy"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=8)

    ax.set_xlabel("cost over interval (USD)")
    ax.set_ylabel("energy over interval (Wh)")
    ax.set_title(f"Widest trade-off in the run: step {snapshot['step']}\n"
                 f"{snapshot['request_rate']:.0f} req/s, currently {snapshot['vm_current']} VMs")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "05_pareto_front.png", dpi=150)

    for name in ("03_response_time", "04_vm_count", "05_pareto_front"):
        print(f"written: {FIGURES / (name + '.png')}")
    print(f"written: {LOGS / 'simulation.csv'}")

    if n_pareto <= 1:
        print("\nWARNING: front collapsed to a single point -- the Pareto stage is inert here.")


if __name__ == "__main__":
    main()
