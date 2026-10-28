"""Export workload CSVs for CloudSim Plus (Step 3.4): data/raw/workloads/<shape>_seed<seed>.csv."""
import argparse

from autoscaler.config import load_config
from autoscaler.sim.cloudsim import WORKLOAD_DIR, workload_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cloudsim.yaml")
    ap.add_argument("--prototype", default="configs/prototype.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--out", default=str(WORKLOAD_DIR))
    args = ap.parse_args()
    cs, proto, exp = load_config(args.config), load_config(args.prototype), load_config(args.experiments)
    steps = cs["time"]["steps_per_run"]
    shapes = list(dict.fromkeys(exp["workloads"] + cs["calibration_runs"]["workloads"]))
    seeds = exp["eval_seeds"] + exp["calibration_seeds"] + cs["counterfactual"]["holdout_seeds"]
    n = 0
    for shape in shapes:
        for seed in seeds:
            workload_csv(shape, seed, steps, proto["workloads"], args.out)
            n += 1
    print(f"{n} workload files ({len(shapes)} shapes x {len(seeds)} seeds, {steps} steps) in {args.out}")


if __name__ == "__main__":
    main()
