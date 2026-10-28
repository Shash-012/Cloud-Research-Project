"""Run one CloudSim Plus simulation (Step 3.12); Python policies go through the bridge."""
import argparse
import json

from autoscaler.config import load_config, save_config
from autoscaler.logging_utils import run_dir
from autoscaler.sim.cloudsim import run_cloudsim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cloudsim.yaml")
    ap.add_argument("--prototype", default="configs/prototype.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--model-dir", default="models/cloudsim")
    ap.add_argument("--policy", default="proposed")
    ap.add_argument("--workload", default="step")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--force-action", default=None, help="STEP:K")
    ap.add_argument("--initial-vms", type=int, default=None)
    ap.add_argument("--energy-crosscheck", action="store_true")
    ap.add_argument("--out", default="results/runs")
    args = ap.parse_args()
    cs, proto, exp = load_config(args.config), load_config(args.prototype), load_config(args.experiments)
    rd = run_dir(args.out, f"p3_02_{args.policy}_{args.workload}_s{args.seed}", cs)
    save_config(exp, rd / "experiments.yaml")
    force = tuple(int(x) for x in args.force_action.split(":")) if args.force_action else None
    res = run_cloudsim(args.policy, args.workload, args.seed, rd, proto, cs, exp, args.model_dir,
                       steps=args.steps, force_action=force, initial_vms=args.initial_vms,
                       energy_crosscheck=args.energy_crosscheck)
    print(json.dumps(res, indent=2))
    if res["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
