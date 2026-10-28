"""Plot every workload shape (Step 2.1) to results/figures/workloads.png."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autoscaler.config import load_config, save_config
from autoscaler.env.workloads import SHAPES, make_workload
from autoscaler.seeds import STREAM_WORKLOAD, child_rngs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/prototype.yaml")
    ap.add_argument("--out", default="results/figures")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = cfg["time"]["steps_per_run"]

    fig, axes = plt.subplots(len(SHAPES), 1, figsize=(9, 2.0 * len(SHAPES)), sharex=True)
    for ax, shape in zip(axes, SHAPES):
        rng = child_rngs(args.seed, 3)[STREAM_WORKLOAD]
        ax.plot(make_workload(shape, steps, rng, cfg["workloads"]), lw=0.8, color="#2f5d8a")
        ax.set_ylabel("req/s")
        ax.set_title(shape, fontsize=9, loc="left")
        ax.set_ylim(0, 480)
    axes[-1].set_xlabel("step (1 min)")
    fig.tight_layout()
    fig.savefig(out / "workloads.png", dpi=200)
    fig.savefig(out / "workloads.pdf")
    save_config(cfg, out / "workloads_config.yaml")
    print("wrote", out / "workloads.png")


if __name__ == "__main__":
    main()
