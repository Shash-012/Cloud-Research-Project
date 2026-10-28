"""Run one closed-loop episode in the synthetic environment (Step 2.12)."""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from autoscaler.config import load_config, save_config
from autoscaler.decision.policies import load_models
from autoscaler.eval.metrics import summarise
from autoscaler.logging_utils import run_dir
from autoscaler.sim.loop import episode, save_logs


def plot_episode(log, slo, path, title):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(log.step, log.rate, color="0.6", lw=0.8, label="request rate (req/s)")
    ax0b = ax[0].twinx()
    ax0b.step(log.step, log.n_running + log.n_booting, where="post", color="#2f5d8a", lw=1.0, label="pool size")
    ax[0].set_ylabel("req/s")
    ax0b.set_ylabel("VMs")
    ax[0].set_title(title, fontsize=10, loc="left")
    h1, l1 = ax[0].get_legend_handles_labels()
    h2, l2 = ax0b.get_legend_handles_labels()
    ax[0].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax[1].plot(log.step, log.rt_mean_ms, color="#2f5d8a", lw=0.7, label="mean response time")
    ax[1].axhline(slo, color="#b33", ls="--", lw=1, label=f"SLO {slo:.0f} ms")
    ax[1].set_ylim(0, max(3 * slo, 1))
    ax[1].set_ylabel("ms")
    ax[1].set_xlabel("step (1 min)")
    ax[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prototype.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--policy", default="proposed")
    ap.add_argument("--workload", default="daily_ramp_burst")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--figure", default=None, help="also save the episode plot here")
    args = ap.parse_args()
    torch.set_num_threads(1)
    cfg, exp = load_config(args.config), load_config(args.experiments)
    rd = run_dir(args.out, f"p2_05_{args.policy}_{args.workload}_s{args.seed}", cfg)
    save_config(exp, rd / "experiments.yaml")

    models = None if args.policy in ("cpu_threshold", "rt_threshold") else load_models(args.model_dir)
    t0 = time.perf_counter()
    log, dec = episode(args.policy, args.workload, args.seed, cfg, exp, models)
    elapsed = time.perf_counter() - t0
    save_logs(log, dec, rd)
    summary = summarise(log, dec, cfg) | {"wall_clock_s": elapsed}
    (rd / "summary.json").write_text(json.dumps(summary, indent=2))
    title = f"{args.policy} on {args.workload}, seed {args.seed}"
    plot_episode(log, cfg["slo"]["rt_ms"], rd / "episode.png", title)
    if args.figure:
        plot_episode(log, cfg["slo"]["rt_ms"], args.figure, title)
    print(json.dumps(summary, indent=2))
    print("run dir:", rd)


if __name__ == "__main__":
    main()
