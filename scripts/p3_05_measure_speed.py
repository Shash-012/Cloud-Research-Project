"""Time 360-step CloudSim Plus runs (Step 3.13 item 7) and write results/tables/cloudsim_speed.csv.

The roadmap times a HoldPolicy run on daily_ramp_burst. With the default 4 initial VMs that run is overloaded
from the first interval (queues grow without bound), so the Hold run uses a pool sized for the peak (20 VMs,
u <= 0.8); a cpu_threshold run shows a controlled pool. If a run exceeds max_minutes, set
requests.capacity_scale (Step 3.1) and re-run the validation tests.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from autoscaler.config import load_config
from autoscaler.sim.cloudsim import run_cloudsim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cloudsim.yaml")
    ap.add_argument("--prototype", default="configs/prototype.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--out", default="results/runs/speed")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--reuse", action="store_true", help="read existing run folders instead of re-running")
    ap.add_argument("--max-minutes", type=float, default=10.0)
    args = ap.parse_args()
    cs, proto, exp = load_config(args.config), load_config(args.prototype), load_config(args.experiments)
    base = Path(args.out)
    jobs = [("hold", 20), ("cpu_threshold", None)]
    if not args.reuse:
        with ThreadPoolExecutor(len(jobs)) as ex:
            list(ex.map(lambda j: run_cloudsim(j[0], "daily_ramp_burst", 1, base / j[0], proto, cs, exp,
                                               initial_vms=j[1]), jobs))
    rows = []
    for pol, n0 in jobs:
        s = json.loads((base / pol / "summary.json").read_text())
        rows.append({"policy": pol, "initial_vms": n0 or cs["cluster"]["initial_vms"], "workload": "daily_ramp_burst",
                     "steps": s["steps"], "requests": s["requests"], "capacity_scale": s["capacity_scale"],
                     "java_wall_clock_s": round(s["wall_clock_s"], 1), "minutes": round(s["wall_clock_s"] / 60, 2),
                     "concurrent_runs": len(jobs), "status": s["status"]})
    df = pd.DataFrame(rows)
    df.to_csv(Path(args.tables) / "cloudsim_speed.csv", index=False)
    print(df.to_string(index=False))
    worst = df.minutes.max()
    print(f"slowest run {worst:.1f} min ->",
          "keep capacity_scale" if worst <= args.max_minutes else "lower capacity_scale (Step 3.1)")


if __name__ == "__main__":
    main()
