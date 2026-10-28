"""Calibration data inside CloudSim Plus (Step 3.14).

--collect runs RandomTargetPolicy on the calibration workloads x calibration seeds (1001..1010) into
data/raw/cloudsim_calibration/; the run logs are then turned into the transition table of Step 1.3:
an action taken at the end of interval t has its before-metrics in interval t and its after-metrics and target in
interval t+2, the first interval in which booted VMs serve for the whole interval. Rows whose rate drifted by more
than 15% between t and t+2 are kept (both rates are recorded) and flagged.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from autoscaler.config import load_config
from autoscaler.data.calibration import extract_cloudsim_transitions as extract
from autoscaler.data.dataset import CLOUDSIM_TRANSITIONS, save_transitions
from autoscaler.sim.cloudsim import ensure_jar, read_run_log, run_cloudsim

RAW = Path("data/raw/cloudsim_calibration")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cloudsim.yaml")
    ap.add_argument("--prototype", default="configs/prototype.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--raw", default=str(RAW))
    ap.add_argument("--out", default=str(CLOUDSIM_TRANSITIONS))
    ap.add_argument("--collect", action="store_true", help="run the calibration simulations first")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cs, proto, exp = load_config(args.config), load_config(args.prototype), load_config(args.experiments)
    raw = Path(args.raw)

    jobs = [(w, s) for w in cs["calibration_runs"]["workloads"] for s in exp["calibration_seeds"]]
    if args.collect:
        ensure_jar()                                   # build once, before the worker threads start
        t0 = time.perf_counter()

        def one(job):
            w, s = job
            folder = raw / f"cloudsim__random_target__{w}__s{s}"
            if (folder / "run_log.parquet").exists():
                return {"run_id": folder.name, "status": "ok (existing)", "wall_clock_s": 0.0}
            return run_cloudsim("random_target", w, s, folder, proto, cs, exp)

        with ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(one, jobs))
        bad = [r for r in results if not r["status"].startswith("ok")]
        print(f"{len(results)} calibration runs, {len(bad)} failed, {time.perf_counter() - t0:.0f}s")
        for r in results:
            print(f"  {r['run_id']}: {r['status']} {r['wall_clock_s']:.0f}s")
        if bad:
            raise SystemExit("calibration runs failed")

    tc = cs["transitions"]
    frames = []
    for w, s in jobs:
        folder = raw / f"cloudsim__random_target__{w}__s{s}"
        frames.append(extract(read_run_log(folder), tc["after_offset"], tc["rate_drift_flag"]))
    df = pd.concat(frames, ignore_index=True)
    save_transitions(df, args.out, source="cloudsim")
    summary = {"rows": len(df), "runs": df.run_id.nunique(), "missing": int(df.isna().sum().sum()),
               "k_range": [int(df.k.min()), int(df.k.max())],
               "share_rt_above_slo": float((df.rt_after_ms > cs["slo"]["rt_ms"] / cs["requests"]["capacity_scale"]).mean()),
               "share_rate_drift_flag": float(df.rate_drift_flag.mean()),
               "k_counts": {int(k): int(v) for k, v in df.k.value_counts().sort_index().items()}}
    Path(args.out).with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
